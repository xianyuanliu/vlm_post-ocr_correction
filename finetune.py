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
    def collate_fn(examples):
        texts = [processor.tokenizer.apply_chat_template(example['messages'], tokenize=False) for example in examples]
        
        if examples[0]['images']:
            images = [example['images'] for example in examples]
            batch = processor(text=texts, images=images, return_tensors='pt', padding=True)
        else:
            batch = processor(text=texts, return_tensors='pt', padding=True)

        labels = batch['input_ids'].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        if examples[0]['images']:
            if isinstance(processor, Qwen2_5_VLProcessor):
                image_tokens = [151652, 151653, 151655]
            else:
                image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

        batch['labels'] = labels
        return batch
    return collate_fn


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['train']['seed']
    set_seed(seed)
    
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
