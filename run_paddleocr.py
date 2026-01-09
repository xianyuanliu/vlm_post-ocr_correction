import datetime
import os
import tempfile
from pathlib import Path

import evaluate
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCRVL


def prepare_input(path_str: str) -> tuple[str, str | None]:
    path = Path(path_str)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return path_str, None

    temp_dir = tempfile.mkdtemp(prefix="paddleocr_")
    out_path = Path(temp_dir) / f"{path.stem}.png"

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.save(out_path, format="PNG")

    return str(out_path), temp_dir


def collect_inputs(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_dir():
        allowed = {".tif", ".tiff", ".jpg", ".jpeg"}
        return sorted(p for p in path.iterdir() if p.suffix.lower() in allowed)
    return [path]


def extract_text_vl(output) -> str:
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
    if temp_dir and os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)


pipeline = PaddleOCRVL()

input_dir = Path("./data/test_paddle.csv")
output_dir = Path("paddleocr_output")
output_dir.mkdir(parents=True, exist_ok=True)

if input_dir.suffix.lower() == ".csv":
    df = pd.read_csv(input_dir)
    patch_dir = Path("data/patch")
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")
    preds = []
    for _, row in df.iterrows():
        image_name = row.get("Image Patch")
        if not isinstance(image_name, str):
            preds.append("")
            continue
        img_path = patch_dir / image_name
        prepared_path, temp_dir = prepare_input(str(img_path))
        try:
            output = pipeline.predict(prepared_path)
            preds.append(extract_text_vl(output))
        finally:
            cleanup_temp_dir(temp_dir)
    df["PaddleOCR Text"] = preds
    df["PaddleOCR CER"] = df.apply(
        lambda row: cer.compute(predictions=[row["PaddleOCR Text"]], references=[row["Ground Truth"]]),
        axis=1,
    )
    df["PaddleOCR WER"] = df.apply(
        lambda row: wer.compute(predictions=[row["PaddleOCR Text"]], references=[row["Ground Truth"]]),
        axis=1,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"paddleocr-{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
else:
    for input_file in collect_inputs(str(input_dir)):
        prepared_path, temp_dir = prepare_input(str(input_file))
        try:
            result = pipeline.ocr(prepared_path, cls=True)
            out_txt = output_dir / f"{input_file.stem}.txt"
            out_txt.write_text(extract_text(result), encoding="utf-8")
        finally:
            cleanup_temp_dir(temp_dir)
