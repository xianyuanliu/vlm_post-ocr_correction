from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from utils import set_seed, format_data_vlm, compute_results
import argparse
import os
import pandas as pd
import re
import torch
import yaml
import datetime


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    test = pd.read_csv(os.path.join(config['settings']['data'], 'test.csv'))
    test = Dataset.from_pandas(test)
    test = [format_data_vlm(sample, config['settings']['data'], ocr=config['settings']['use_ocr'], img=config['settings']['use_img']) for sample in test]
    preds = []

    model = AutoModelForVision2Seq.from_pretrained(
        config['settings']['model'],
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model.load_adapter(config['settings']['adapter'])
    processor = AutoProcessor.from_pretrained(config['settings']['model'])
    processor.tokenizer = AutoTokenizer.from_pretrained(config['settings']['adapter'])

    for sample in tqdm(test):
        text = processor.tokenizer.apply_chat_template(sample['messages'][:2], tokenize=False, add_generation_prompt=True)
        if sample['images']:
            inputs = processor(text=text, images=sample['images'], return_tensors='pt', padding=True).to(model.device)
        else:
            inputs = processor(text=text, return_tensors='pt', padding=True).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        generated = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        pred = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = re.sub(r'(\b.+?\b)(?:\s+\1\b)+', r'\1', pred, flags=re.DOTALL)
        preds.append(pred)

    results = compute_results(pd.read_csv(os.path.join(config['settings']['data'], 'test.csv')), preds)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f"results/{os.path.basename(config['settings']['adapter'])}-{args.seed}-{timestamp}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating fine-tuned models for post-OCR correction')
    parser.add_argument('--config', type=str, help='Path to config (YAML)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    main(parser.parse_args())
