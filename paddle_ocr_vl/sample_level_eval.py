import argparse
import os
import pandas as pd
import evaluate


def main(input_csv):
    df = pd.read_csv(input_csv)
    cer = evaluate.load('cer')
    wer = evaluate.load('wer')

    results = []
    for sample, group in df.groupby('name'):
        ocr_combined = ' '.join(group['ocr_text'].astype(str))
        correction_combined = ' '.join(group['paddleocrvl_text'].astype(str))
        gt_combined = ' '.join(group['ground_truth'].astype(str))

        cer_init = cer.compute(predictions=[ocr_combined], references=[gt_combined])
        wer_init = wer.compute(predictions=[ocr_combined], references=[gt_combined])
        cer_post = cer.compute(predictions=[correction_combined], references=[gt_combined])
        wer_post = wer.compute(predictions=[correction_combined], references=[gt_combined])

        cer_reduction = ((cer_init - cer_post) / cer_init) * 100 if cer_init > 0 else 0
        wer_reduction = ((wer_init - wer_post) / wer_init) * 100 if wer_init > 0 else 0

        results.append({
            'name': sample,
            'ocr_text': ocr_combined,
            'ground_truth': gt_combined,
            'CER_init': cer_init,
            'WER_init': wer_init,
            'paddleocrvl_text': correction_combined,
            'CER_post': cer_post,
            'WER_post': wer_post,
            'CER_reduction': cer_reduction,
            'WER_reduction': wer_reduction
        })

    out_df = pd.DataFrame(results)
    out_path = os.path.basename(input_csv).replace('.csv', '_sample_level.csv')
    out_df.to_csv(out_path, index=False)

    print(f'Saved to {out_path}')
    print(f'Samples: {len(out_df)}')
    print(f'CER_init: {out_df["CER_init"].mean():.4f} CER_post: {out_df["CER_post"].mean():.4f} CER %: {out_df["CER_reduction"].mean():.2f}')
    print(f'WER_init: {out_df["WER_init"].mean():.4f} WER_post: {out_df["WER_post"].mean():.4f} WER %: {out_df["WER_reduction"].mean():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OCR corrections at sample level')
    parser.add_argument('input', help='Input CSV file from results/')
    args = parser.parse_args()
    main(args.input)