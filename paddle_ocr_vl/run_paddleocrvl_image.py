import shutil
import tempfile
from pathlib import Path

from PIL import Image
from paddleocr import PaddleOCRVL

BASE_DIR = Path(__file__).resolve().parents[1]


def prepare_input(path_str: str) -> tuple[str, str | None]:
    """
    Convert TIFF/IMG to temporary PNG for PaddleOCR-VL.
    Return (prepared_path, temp_dir).
    """
    path = Path(path_str)
    if path.suffix.lower() not in {".tif", ".tiff", ".img"}:
        return path_str, None

    temp_dir = tempfile.mkdtemp(prefix="paddleocr_")
    out_path = Path(temp_dir) / f"{path.stem}.png"

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.save(out_path, format="PNG")

    return str(out_path), temp_dir


def extract_text_vl(output) -> str:
    # Extract plain text from PaddleOCR-VL output (your version wraps payload under j["res"])
    if not output:
        return ""

    texts = []
    for res in output:
        j = getattr(res, "json", None)

        # res.json might be a method
        if callable(j):
            j = j()

        if not isinstance(j, dict) or "res" not in j:
            continue

        payload = j["res"]

        for blk in payload.get("parsing_res_list", []):
            content = blk.get("block_content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())

    return " ".join(texts)


def cleanup_temp_dir(temp_dir: str | None) -> None:
    """Remove temporary directory if it exists."""
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


def iter_images(input_dir: Path) -> list[Path]:
    exts = {".tif", ".tiff", ".img", ".jpg", ".jpeg"}
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def save_text(out_dir: Path, image_path: Path, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    input_dir = BASE_DIR / "BLN600/Images"
    output_dir = BASE_DIR / "paddleocr_output/PaddleOCRVL_Text"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = iter_images(input_dir)
    if not images:
        print(f"No images found under {input_dir}")
        return

    # Initialize PaddleOCR-VL once
    pipeline = PaddleOCRVL()

    for img_path in images:
        prepared_path, temp_dir = prepare_input(str(img_path))
        try:
            output = pipeline.predict(prepared_path)
            text = extract_text_vl(output)
            save_text(output_dir, img_path, text)
        finally:
            cleanup_temp_dir(temp_dir)

    print(f"Saved {len(images)} text files to {output_dir}")


if __name__ == "__main__":
    main()
