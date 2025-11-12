from datasets import Dataset
from datetime import datetime
from peft import LoraConfig
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from trl import SFTConfig, SFTTrainer
from utils import set_seed, format_data_vlm
import argparse
import os
import pandas as pd
import torch
import wandb
import yaml


def collate(processor):
    tok = processor.tokenizer
    pad_id = processor.tokenizer.pad_token_id or tok.eos_token_id

    # --- Utility: subsequence search (robust to tokenizer variants) ---
    def find_subseq(a, b, start=0):
        L1, L2 = len(a), len(b)
        if L2 == 0 or L1 < L2:
            return -1
        i = start
        while i <= L1 - L2:
            if a[i:i+L2] == b:
                return i
            i += 1
        return -1

    # --- Assistant section boundaries and visual placeholder tokens ---
    # Qwen-style assistant boundaries
    QWEN_ASSIST_START = tok.encode("<|im_start|>assistant", add_special_tokens=False)
    QWEN_IM_END       = tok.encode("<|im_end|>", add_special_tokens=False)

    # Llama-style assistant boundaries (several possible header variants)
    LLAMA_ASSIST_HEADS = [
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<start_header_id>assistant<end_header_id>\n\n",
        "<start_header_id>assistant<end_header_id>\n",
        "<start_header_id>assistant<end_header_id>",
    ]
    LLAMA_ASSIST_HEADS = [tok.encode(s, add_special_tokens=False) for s in LLAMA_ASSIST_HEADS]
    LLAMA_EOT_CANDS    = [
        tok.encode("<|eot_id|>", add_special_tokens=False),
        tok.encode("<eot_id>",   add_special_tokens=False),
    ]
    LLAMA_EOT_IDS = next((e for e in LLAMA_EOT_CANDS if len(e) > 0), [])

    # Visual placeholders for both Qwen and Llama models
    VISION_TOKENS = [
        tok.encode("<|vision_start|>", add_special_tokens=False),
        tok.encode("<|vision_end|>",   add_special_tokens=False),
        tok.encode("<|image_pad|>",    add_special_tokens=False),
        tok.encode("<image>",          add_special_tokens=False),  # Llama
    ]

    def collate_fn(examples):
        # 1) Expand chat templates
        texts = [tok.apply_chat_template(ex["messages"], tokenize=False) for ex in examples]

        # 2) Detect whether this batch contains any images
        has_images = any(bool(ex.get("images")) for ex in examples)

        # 3) Tokenise (with or without image input)
        if has_images:
            images = [ex.get("images", []) for ex in examples]
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            batch = tok(texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = batch["input_ids"]
        attn      = batch["attention_mask"]

        # 4) Supervise assistant-only sections: all else is masked with -100
        labels = torch.full_like(input_ids, -100)
        B = input_ids.size(0)

        for b in range(B):
            ids = input_ids[b].tolist()
            valid = int(attn[b].sum().item())
            matched = False

            # 4.1 Qwen: <|im_start|>assistant ... <|im_end|>
            pos = 0
            while True:
                j = find_subseq(ids[:valid], QWEN_ASSIST_START, start=pos)
                if j == -1:
                    break
                k = j + len(QWEN_ASSIST_START)
                end = find_subseq(ids[:valid], QWEN_IM_END, start=k)
                if end == -1:
                    end = valid
                if k < end:
                    labels[b, k:end] = input_ids[b, k:end]
                    matched = True
                pos = end + len(QWEN_IM_END) if end != -1 else valid

            # 4.2 Llama: <start_header_id>assistant<end_header_id> ... <eot_id>
            if not matched:
                for head in LLAMA_ASSIST_HEADS:
                    if len(head) == 0:
                        continue
                    pos = 0
                    while True:
                        j = find_subseq(ids[:valid], head, start=pos)
                        if j == -1:
                            break
                        k = j + len(head)  # start of assistant text
                        end = find_subseq(ids[:valid], LLAMA_EOT_IDS, start=k) if len(LLAMA_EOT_IDS) > 0 else -1
                        if end == -1:
                            end = valid
                        if k < end:
                            labels[b, k:end] = input_ids[b, k:end]
                            matched = True
                        pos = end + (len(LLAMA_EOT_IDS) if len(LLAMA_EOT_IDS) > 0 else 0)

        # 5) Mask padding
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        # 6) Mask visual placeholder tokens (Qwen & Llama)
        if has_images:
            for vt in VISION_TOKENS:
                if len(vt) == 0:
                    continue
                for b in range(B):
                    ids = input_ids[b].tolist()
                    valid = int(attn[b].sum().item())
                    pos = 0
                    while True:
                        s = find_subseq(ids[:valid], vt, start=pos)
                        if s == -1:
                            break
                        labels[b, s:s+len(vt)] = -100
                        pos = s + len(vt)

        batch["labels"] = labels
        return batch
    return collate_fn

# def collate(processor):
#     def collate_fn(examples):
#         texts = [processor.tokenizer.apply_chat_template(example['messages'], tokenize=False) for example in examples]
        
#         if examples[0]['images']:
#             images = [example['images'] for example in examples]
#             batch = processor(text=texts, images=images, return_tensors='pt', padding=True)
#         else:
#             batch = processor(text=texts, return_tensors='pt', padding=True)

#         labels = batch['input_ids'].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100

#         if examples[0]['images']:
#             if isinstance(processor, Qwen2_5_VLProcessor):
#                 image_tokens = [151652, 151653, 151655]
#             else:
#                 image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
#             for image_token_id in image_tokens:
#                 labels[labels == image_token_id] = -100

#         batch['labels'] = labels
#         return batch
#     return collate_fn

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(42)
    wandb.init(
        entity='shef_aire',
        project='vlm-post-ocr',
        name=f"{os.path.basename(config['settings']['adapter'])}_{datetime.now():%d-%b-%I:%M%p}",
        config=config,
    )

    train = pd.read_csv(os.path.join(config['settings']['data'], 'train.csv'))
    train = Dataset.from_pandas(train)
    train = [format_data_vlm(sample, config['settings']['data'], ocr=config['settings']['use_ocr'], img=config['settings']['use_img']) for sample in train]
    val = pd.read_csv(os.path.join(config['settings']['data'], 'val.csv'))
    val = Dataset.from_pandas(val)
    val = [format_data_vlm(sample, config['settings']['data'], ocr=config['settings']['use_ocr'], img=config['settings']['use_img']) for sample in val]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    peft_config = LoraConfig(**config['lora'])
    train_args = SFTConfig(
        output_dir=config['settings']['adapter'],
        **config['train'],
    )

    model = AutoModelForVision2Seq.from_pretrained(
        config['settings']['model'],
        torch_dtype=torch.bfloat16,
        device_map='auto',
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(config['settings']['model'])
    processor.tokenizer.chat_template = config['settings'].get('chat_template', processor.tokenizer.chat_template)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        peft_config=peft_config,
        args=train_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collate(processor),
    )

    if config['settings']['freeze_img']:
        for name, param in model.named_parameters():
            if 'lora' in name and ('vision_model' in name or 'visual' in name):
                param.requires_grad = False

    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning base models for post-OCR correction')
    parser.add_argument('--config', type=str, help='Path to config (YAML)')
    main(parser.parse_args())
