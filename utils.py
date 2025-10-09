from PIL import Image
from transformers import set_seed as hf_seed
import evaluate
import numpy as np
import os
import random
import torch


# Helper function to preprocess text
def preprocess(text):
    text = text.str.replace("‘", "'", regex=False)
    text = text.str.replace("’", "'", regex=False)
    text = text.str.replace("“", '"', regex=False)
    text = text.str.replace("”", '"', regex=False)
    text = text.str.replace("—", "-", regex=False)
    text = text.str.replace(r'\s+', ' ', regex=True)
    text = text.str.strip()
    return text


# Calculate the bounding box, encapsulating a series of word coordinates
def calculate_bbox(coords):
    min_x1 = min_y1 = float('inf')
    max_x2 = max_y2 = float('-inf')

    for coord in coords:
        x1, y1, x2, y2 = map(int, coord.split(','))
        min_x1 = min(min_x1, x1)
        min_y1 = min(min_y1, y1)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)

    return min_x1, min_y1, max_x2, max_y2


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    hf_seed(seed)


# Convert dataset sample into format for fine-tuning VLMs
def format_data_vlm(sample, folder, ocr=True, img=True):
    if not (ocr or img):
        raise ValueError("OCR text or source image must be provided for fine-tuning VLMs")

    system_message = "You are a helpful assistant, designed to correct errors in OCR text. Generate the corrected text based on the provided input."
    messages = [{"role": "system", "content": [{"type": "text", "text": system_message}]}]

    user = []
    if img:
        user.append({"type": "image"})
    if ocr:
        user.append({"type": "text", "text": sample['OCR Text']})

    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": sample['Ground Truth']}]})

    if img:
        image = Image.open(os.path.join(folder, 'patch', sample['Image Patch'])).convert('RGB')
        return {"messages": messages, "images": [image]}
    else:
        return {"messages": messages, "images": []}

# Updates dataframe with model corrections and computed metrics
def compute_results(df, preds):
    df['Model Correction'] = preds
    cer = evaluate.load('cer')
    wer = evaluate.load('wer')
    df = df.rename(columns={'CER': 'CER_init'})
    df = df.rename(columns={'WER': 'WER_init'})
    df['CER_post'] = df.apply(lambda row: cer.compute(predictions=[row['Model Correction']], references=[row['Ground Truth']]), axis=1)
    df['WER_post'] = df.apply(lambda row: wer.compute(predictions=[row['Model Correction']], references=[row['Ground Truth']]), axis=1)
    df['CER Reduction (%)'] = ((df['CER_init'] - df['CER_post']) / df['CER_init']) * 100
    df['WER Reduction (%)'] = ((df['WER_init'] - df['WER_post']) / df['WER_init']) * 100
    return df
