from __future__ import annotations

from pathlib import Path
import csv
import statistics

import evaluate

BASE_DIR = Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return " ".join(text.split())


def collect_texts(dir_path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in dir_path.rglob("*.txt"):
        if path.is_file():
            texts[path.stem] = read_text(path)
    return texts


def resolve_ocr_dir() -> Path:
    ocr_dir = BASE_DIR / "BLN600/OCR Text"
    if ocr_dir.exists():
        return ocr_dir
    fallback = BASE_DIR / "BLN600/OCT Text"
    if fallback.exists():
        print("Using BLN600/OCT Text (OCR Text not found).")
        return fallback
    return ocr_dir


def load_test_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()

    stems: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set()

        if "Sample" in reader.fieldnames:
            for row in reader:
                value = (row.get("Sample") or "").strip()
                if value:
                    stems.add(value)
        elif "Image Patch" in reader.fieldnames:
            for row in reader:
                value = (row.get("Image Patch") or "").strip()
                if value:
                    stems.add(Path(value).stem)

    return stems


def safe_compute(metric, pred: str | None, ref: str | None) -> float | None:
    if not pred or not ref:
        return None
    return metric.compute(predictions=[pred], references=[ref])


def main(use_test_only: bool = False) -> None:

    if use_test_only:
        test_stems = load_test_stems(BASE_DIR / "data/test.csv")
        out_path = BASE_DIR / "cer_wer_summary_test.csv"
    else:
        test_stems = None
        out_path = BASE_DIR / "cer_wer_summary.csv"



    gt_dir = BASE_DIR / "BLN600/Ground Truth"
    ocr_dir = resolve_ocr_dir()
    paddle_dir = BASE_DIR / "paddleocr_output/PaddleOCRVL_Text"

    gt_texts = collect_texts(gt_dir)
    ocr_texts = collect_texts(ocr_dir)
    paddle_texts = collect_texts(paddle_dir)

    if not gt_texts:
        print(f"No ground truth files found in {gt_dir}")
        return

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    fieldnames = [
        "name",
        "ground_truth",
        "ocr_text",
        "paddleocrvl_text",
        "cer_ocr",
        "wer_ocr",
        "cer_paddle",
        "wer_paddle",
    ]

    cer_ocr_vals = []
    wer_ocr_vals = []
    cer_paddle_vals = []
    wer_paddle_vals = []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if use_test_only and test_stems:
            stems = sorted(gt_texts.keys() & test_stems)
        else:
            stems = sorted(gt_texts.keys())

        for stem in stems:
            gt = gt_texts.get(stem, "")
            ocr = ocr_texts.get(stem, "")
            paddle = paddle_texts.get(stem, "")

            cer_ocr = safe_compute(cer_metric, ocr, gt)
            wer_ocr = safe_compute(wer_metric, ocr, gt)
            cer_paddle = safe_compute(cer_metric, paddle, gt)
            wer_paddle = safe_compute(wer_metric, paddle, gt)

            if cer_ocr is not None:
                cer_ocr_vals.append(cer_ocr)
            if wer_ocr is not None:
                wer_ocr_vals.append(wer_ocr)
            if cer_paddle is not None:
                cer_paddle_vals.append(cer_paddle)
            if wer_paddle is not None:
                wer_paddle_vals.append(wer_paddle)

            writer.writerow(
                {
                    "name": stem,
                    "ground_truth": gt,
                    "ocr_text": ocr,
                    "paddleocrvl_text": paddle,
                    "cer_ocr": "" if cer_ocr is None else f"{cer_ocr:.6f}",
                    "wer_ocr": "" if wer_ocr is None else f"{wer_ocr:.6f}",
                    "cer_paddle": "" if cer_paddle is None else f"{cer_paddle:.6f}",
                    "wer_paddle": "" if wer_paddle is None else f"{wer_paddle:.6f}",
                }
            )

    print(f"Saved CSV to {out_path}")
    print(f"Ground truth files: {len(gt_texts)}")
    if use_test_only:
        print(f"Test.csv stems: {len(test_stems)}")
        print(f"Filtered by test.csv: {len(stems)}")
    print(f"OCR matched: {len(cer_ocr_vals)}")
    print(f"PaddleOCRVL matched: {len(cer_paddle_vals)}")

    if cer_ocr_vals:
        print(f"Mean CER (OCR): {statistics.mean(cer_ocr_vals):.6f}")
        print(f"Mean WER (OCR): {statistics.mean(wer_ocr_vals):.6f}")
    if cer_paddle_vals:
        print(f"Mean CER (PaddleOCRVL): {statistics.mean(cer_paddle_vals):.6f}")
        print(f"Mean WER (PaddleOCRVL): {statistics.mean(wer_paddle_vals):.6f}")


if __name__ == "__main__":
    main(use_test_only=True)
