import datetime
import json
import os
import tempfile
from pathlib import Path

import evaluate
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCRVL


def prepare_input(path_str: str) -> tuple[str, str | None]:
    """
    Convert TIFF to temporary PNG for PaddleOCR-VL.
    Return (prepared_path, temp_dir).
    """
    path = Path(path_str)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return path_str, None

    temp_dir = tempfile.mkdtemp(prefix="paddleocr_")
    out_path = Path(temp_dir) / f"{path.stem}.png"

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.save(out_path, format="PNG")

    return str(out_path), temp_dir


def extract_text_vl(output) -> str:
    """
    Extract plain text from PaddleOCR-VL output
    by concatenating block_content fields.
    """
    if not output:
        return ""

    texts = []
    for res in output:
        j = getattr(res, "json", None)
        if not isinstance(j, dict):
            continue
        for blk in j.get("parsing_res_list", []):
            content = blk.get("block_content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())

    return " ".join(texts)


def cleanup_temp_dir(temp_dir: str | None) -> None:
    """Remove temporary directory if it exists."""
    if temp_dir and os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)


# Initialize PaddleOCR-VL once
pipeline = PaddleOCRVL()

input_csv = Path("./data/test_paddle.csv")
patch_dir = Path("data/patch")

output_dir = Path("paddleocr_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Optional: save per-image VL JSON outputs
per_img_dir = output_dir / "per_image_json"
per_img_dir.mkdir(parents=True, exist_ok=True)

# Load CSV and metrics
df = pd.read_csv(input_csv)
cer = evaluate.load("cer")
wer = evaluate.load("wer")

preds = []

for _, row in df.iterrows():
    image_name = row.get("Image Patch")
    if not isinstance(image_name, str):
        preds.append("")
        continue

    img_path = patch_dir / image_name
    if not img_path.exists():
        preds.append("")
        continue

    prepared_path, temp_dir = prepare_input(str(img_path))
    try:
        output = pipeline.predict(prepared_path)

        # Save raw PaddleOCR-VL JSON output per image (optional)
        stem = Path(image_name).stem
        for i, res in enumerate(output):
            j = getattr(res, "json", None)
            if isinstance(j, dict):
                (per_img_dir / f"{stem}_{i:02d}.json").write_text(
                    json.dumps(j, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        # Extract plain text for evaluation
        preds.append(extract_text_vl(output))

    finally:
        cleanup_temp_dir(temp_dir)

# Add predictions and metrics to CSV
df["PaddleOCR Text"] = preds
df["PaddleOCR CER"] = df.apply(
    lambda r: cer.compute(
        predictions=[str(r.get("PaddleOCR Text", ""))],
        references=[str(r.get("Ground Truth", ""))],
    ),
    axis=1,
)
df["PaddleOCR WER"] = df.apply(
    lambda r: wer.compute(
        predictions=[str(r.get("PaddleOCR Text", ""))],
        references=[str(r.get("Ground Truth", ""))],
    ),
    axis=1,
)

# Save a single CSV (overwrite each run)
out_csv = output_dir / "paddleocr_vl.csv"
df.to_csv(out_csv, index=False)
print(f"Saved results to {out_csv}")
