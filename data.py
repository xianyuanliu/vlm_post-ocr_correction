from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import preprocess, calculate_bbox
import evaluate
import json
import pandas as pd
import os
import xml.etree.ElementTree as ET


# Builds dataframe storing BLN600 sequences
def build_seq(metadata):

    with open(metadata, 'r') as f:
        bln600 = json.load(f)

    sample, date, publication, ocr, gt = [], [], [], [], []

    for doc in tqdm(bln600):
        with open(os.path.join('data/seq', f"{doc['short_id']}.txt"), 'r') as f:
            lines = f.readlines()

        ocr_text, ground_truth = '', ''
        for line in lines:
            if line.startswith('OCR Text: '):
                ocr_text = line.replace('OCR Text: ', '').strip()
            elif line.startswith('Ground Truth: '):
                ground_truth = line.replace('Ground Truth: ', '').strip()
            if ocr_text and ground_truth:
                sample.append(doc['short_id'])
                date.append(f"{doc['date'][:4]}-{doc['date'][4:6]}-{doc['date'][6:]}")
                publication.append(doc['publication'])
                ocr.append(ocr_text)
                gt.append(ground_truth)
                ocr_text, ground_truth = '', ''

    df = pd.DataFrame({'Sample': sample, 'Date': date, 'Publication': publication, 'OCR Text': ocr, 'Ground Truth': gt})
    df['Ground Truth'] = preprocess(df['Ground Truth'])

    return df


# Build dataframe, merging multiple sequences with a window
def merge_seq(df, window=1):
    rows = []

    for short_id in df['Sample'].unique():
        sample = df[df['Sample'] == short_id].reset_index(drop=True)

        for i in range(0, len(sample), window):
            subset = sample.iloc[i:i + window]
            rows.append({
                'Sample': short_id,
                'Date': sample['Date'].iloc[0],
                'Publication': sample['Publication'].iloc[0],
                'OCR Text': ' '.join(subset['OCR Text'].astype(str)),
                'Ground Truth': ' '.join(subset['Ground Truth'].astype(str))
            })

    df = pd.DataFrame(rows)
    return df


# Crop image patches using sequences to create paired data {IMG, OCR, GT}
def crop_patch(metadata, sequences, window=1, mask=False):

    with open(metadata, 'r') as f:
        bln600 = json.load(f)

    dataset = pd.read_csv(sequences)
    dataset = merge_seq(dataset, window=window)
    os.makedirs('data/patch', exist_ok=True)

    for doc in tqdm(bln600):
        tree = ET.parse(os.path.join('data/xml', os.path.basename(doc['xml'])))
        img = Image.open(os.path.join('data/img', os.path.basename(doc['img'])))

        sample = dataset[dataset['Sample']==int(doc['short_id'])].copy()
        files = []
        count = 1

        for ocr in sample['OCR Text']:
            words = ocr.split()
            positions = []
            i = 0
            for word in tree.iter('wd'):
                if word.text and word.text.strip() == words[i]:
                    positions.append(word.attrib['pos'])
                    i += 1
                    if i == len(words):
                        break
                else:
                    i = 0
                    positions.clear()

            bbox = calculate_bbox(positions)
            try:
                if mask:
                    patch = img.crop(bbox)

                    x_min, y_min, x_max, y_max = bbox
                    mask = Image.new('1', (x_max - x_min, y_max - y_min), 0)
                    draw = ImageDraw.Draw(mask)
                    for pos in positions:
                        x1, y1, x2, y2 = map(int, pos.split(','))
                        draw.rectangle([x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min], fill=1)
                    patch = Image.composite(patch, Image.new('1', patch.size, 1), mask)

                    patch.save(os.path.join('data/patch', f"{doc['short_id']}-{count}.tif"))
                    files.append(f"{doc['short_id']}-{count}.tif")
                else:
                    patch = img.crop(bbox)
                    patch.save(os.path.join('data/patch', f"{doc['short_id']}-{count}.tif"))
                    files.append(f"{doc['short_id']}-{count}.tif")
            except ValueError:
                files.append(None)
            count += 1

        dataset.loc[sample.index, 'Image Patch'] = files

    cer = evaluate.load('cer')
    wer = evaluate.load('wer')
    dataset['CER'] = dataset.apply(lambda row: cer.compute(predictions=[row['OCR Text']], references=[row['Ground Truth']]), axis=1)
    dataset['WER'] = dataset.apply(lambda row: wer.compute(predictions=[row['OCR Text']], references=[row['Ground Truth']]), axis=1)

    return dataset


# Split dataset into training, validation and testing sets
def split(dataset, seed=42):
    dataset = dataset.dropna()
    train_ids, test_ids = train_test_split(dataset['Sample'].unique(), test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=seed)
    train_set = dataset[dataset['Sample'].isin(train_ids)]
    val_set = dataset[dataset['Sample'].isin(val_ids)]
    test_set = dataset[dataset['Sample'].isin(test_ids)]
    return train_set, val_set, test_set


if __name__ == '__main__':
    seq = build_seq('data/metadata.json')
    seq.to_csv('data/sequences.csv', index=False)

    data = crop_patch('data/metadata.json', 'data/sequences.csv', window=1, mask=False)
    train, val, test = split(data)

    data.to_csv('data/dataset.csv', index=False)
    train.to_csv('data/train.csv', index=False)
    val.to_csv('data/val.csv', index=False)
    test.to_csv('data/test.csv', index=False)
