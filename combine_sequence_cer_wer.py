from __future__ import annotations

from pathlib import Path
import csv
import re
import statistics

import evaluate


TARGET_SETTINGS = {
    "llama-3.2-11b-vision-img-only",
    "llama-3.2-11b-vision-ocr-only",
    "qwen2.5-vl-7b-img-only",
    "qwen2.5-vl-7b-ocr-only",
}


def normalize(text: str) -> str:
    return " ".join(text.split())


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No headers found in {path}")

        required = {"Sample", "OCR Text", "Ground Truth", "Model Correction"}
        missing = required - set(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {path}: {missing_str}")

        return list(reader)


def combine_by_sample(
    rows: list[dict[str, str]]
) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    combined: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []

    for row in rows:
        sample = (row.get("Sample") or "").strip()
        if not sample:
            continue

        if sample not in combined:
            combined[sample] = {"ocr": [], "gt": [], "post": []}
            order.append(sample)

        combined[sample]["ocr"].append(normalize(row.get("OCR Text", "")))
        combined[sample]["gt"].append(normalize(row.get("Ground Truth", "")))
        combined[sample]["post"].append(normalize(row.get("Model Correction", "")))

    return order, combined


def safe_compute(metric, pred: str, ref: str) -> float | None:
    if not pred or not ref:
        return None
    return metric.compute(predictions=[pred], references=[ref])


def extract_setting(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(.*?)-\d+(?:-\d+)?-\d{8}_\d{6}$", stem)
    return match.group(1) if match else stem


def process_file(
    input_path: Path,
    cer_metric,
    wer_metric,
) -> tuple[str, dict[str, list[float]], int]:
    rows = load_rows(input_path)
    order, combined = combine_by_sample(rows)

    cer_init_vals = []
    wer_init_vals = []
    cer_post_vals = []
    wer_post_vals = []
    cer_red_vals = []
    wer_red_vals = []

    fieldnames = [
        "Sample",
        "Ground Truth",
        "OCR Text",
        "Model Correction",
        "CER_init",
        "WER_init",
        "CER_post",
        "WER_post",
        "CER_reduction_pct",
        "WER_reduction_pct",
    ]

    output_path = input_path.with_name(f"{input_path.stem}_combined_cer_wer.csv")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in order:
            ocr = " ".join(combined[sample]["ocr"]).strip()
            gt = " ".join(combined[sample]["gt"]).strip()
            post = " ".join(combined[sample]["post"]).strip()

            cer_init = safe_compute(cer_metric, ocr, gt)
            wer_init = safe_compute(wer_metric, ocr, gt)
            cer_post = safe_compute(cer_metric, post, gt)
            wer_post = safe_compute(wer_metric, post, gt)
            cer_red = None
            wer_red = None
            if cer_init not in (None, 0.0) and cer_post is not None:
                cer_red = (cer_init - cer_post) / cer_init * 100.0
            if wer_init not in (None, 0.0) and wer_post is not None:
                wer_red = (wer_init - wer_post) / wer_init * 100.0

            if cer_init is not None:
                cer_init_vals.append(cer_init)
            if wer_init is not None:
                wer_init_vals.append(wer_init)
            if cer_post is not None:
                cer_post_vals.append(cer_post)
            if wer_post is not None:
                wer_post_vals.append(wer_post)
            if cer_red is not None:
                cer_red_vals.append(cer_red)
            if wer_red is not None:
                wer_red_vals.append(wer_red)

            writer.writerow(
                {
                    "Sample": sample,
                    "Ground Truth": gt,
                    "OCR Text": ocr,
                    "Model Correction": post,
                    "CER_init": "" if cer_init is None else f"{cer_init:.6f}",
                    "WER_init": "" if wer_init is None else f"{wer_init:.6f}",
                    "CER_post": "" if cer_post is None else f"{cer_post:.6f}",
                    "WER_post": "" if wer_post is None else f"{wer_post:.6f}",
                    "CER_reduction_pct": "" if cer_red is None else f"{cer_red:.6f}",
                    "WER_reduction_pct": "" if wer_red is None else f"{wer_red:.6f}",
                }
            )

    metrics = {
        "cer_init": cer_init_vals,
        "wer_init": wer_init_vals,
        "cer_post": cer_post_vals,
        "wer_post": wer_post_vals,
        "cer_red": cer_red_vals,
        "wer_red": wer_red_vals,
    }
    return extract_setting(input_path), metrics, len(order)


def main(use_all_results: bool = False) -> None:
    if use_all_results:
        input_paths = sorted(Path("results").glob("*.csv"))
    else:
        input_paths = [
            Path("results/llama-3.2-11b-vision-img-only-1-20251104_020411.csv")
        ]

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    setting_metrics: dict[str, dict[str, list[float]]] = {}
    setting_samples: dict[str, int] = {}
    setting_run_means: dict[str, list[dict[str, float]]] = {}

    for input_path in input_paths:
        if input_path.stem.endswith("_combined_cer_wer"):
            continue
        try:
            setting, metrics, samples = process_file(
                input_path, cer_metric, wer_metric
            )
        except ValueError as exc:
            print(f"Skipping {input_path}: {exc}")
            continue

        if setting not in setting_metrics:
            setting_metrics[setting] = {
                "cer_init": [],
                "wer_init": [],
                "cer_post": [],
                "wer_post": [],
                "cer_red": [],
                "wer_red": [],
            }
            setting_samples[setting] = 0
            setting_run_means[setting] = []

        for key, values in metrics.items():
            setting_metrics[setting][key].extend(values)
        setting_samples[setting] += samples

        run_means = {}
        if metrics["cer_init"]:
            run_means["cer_init"] = statistics.mean(metrics["cer_init"])
            run_means["wer_init"] = statistics.mean(metrics["wer_init"])
        if metrics["cer_post"]:
            run_means["cer_post"] = statistics.mean(metrics["cer_post"])
            run_means["wer_post"] = statistics.mean(metrics["wer_post"])
        if metrics["cer_init"] and metrics["cer_post"]:
            mean_cer_init = statistics.mean(metrics["cer_init"])
            mean_cer_post = statistics.mean(metrics["cer_post"])
            if mean_cer_init != 0.0:
                run_means["cer_red"] = (
                    (mean_cer_init - mean_cer_post) / mean_cer_init * 100.0
                )
        if metrics["wer_init"] and metrics["wer_post"]:
            mean_wer_init = statistics.mean(metrics["wer_init"])
            mean_wer_post = statistics.mean(metrics["wer_post"])
            if mean_wer_init != 0.0:
                run_means["wer_red"] = (
                    (mean_wer_init - mean_wer_post) / mean_wer_init * 100.0
                )
        if run_means:
            setting_run_means[setting].append(run_means)

        print(f"Saved combined CER/WER to {input_path.stem}_combined_cer_wer.csv")
        print(f"Samples: {samples}")
        if metrics["cer_init"]:
            print(f"Mean CER_init: {statistics.mean(metrics['cer_init']):.6f}")
            print(f"Mean WER_init: {statistics.mean(metrics['wer_init']):.6f}")
        if metrics["cer_post"]:
            print(f"Mean CER_post: {statistics.mean(metrics['cer_post']):.6f}")
            print(f"Mean WER_post: {statistics.mean(metrics['wer_post']):.6f}")
        if metrics["cer_init"] and metrics["cer_post"]:
            mean_cer_init = statistics.mean(metrics["cer_init"])
            mean_cer_post = statistics.mean(metrics["cer_post"])
            if mean_cer_init != 0.0:
                mean_cer_red = (mean_cer_init - mean_cer_post) / mean_cer_init * 100.0
                print(f"Mean CER_reduction: {mean_cer_red:.6f}")
        if metrics["wer_init"] and metrics["wer_post"]:
            mean_wer_init = statistics.mean(metrics["wer_init"])
            mean_wer_post = statistics.mean(metrics["wer_post"])
            if mean_wer_init != 0.0:
                mean_wer_red = (mean_wer_init - mean_wer_post) / mean_wer_init * 100.0
                print(f"Mean WER_reduction: {mean_wer_red:.6f}")
        print()

    if use_all_results:
        print("Means by setting:")
        for setting in sorted(setting_metrics.keys()):
            metrics = setting_metrics[setting]
            print(f"{setting} (samples: {setting_samples[setting]})")
            if metrics["cer_init"]:
                print(f"Mean CER_init: {statistics.mean(metrics['cer_init']):.6f}")
                print(f"Mean WER_init: {statistics.mean(metrics['wer_init']):.6f}")
            if metrics["cer_post"]:
                print(f"Mean CER_post: {statistics.mean(metrics['cer_post']):.6f}")
                print(f"Mean WER_post: {statistics.mean(metrics['wer_post']):.6f}")
            if metrics["cer_init"] and metrics["cer_post"]:
                mean_cer_init = statistics.mean(metrics["cer_init"])
                mean_cer_post = statistics.mean(metrics["cer_post"])
                if mean_cer_init != 0.0:
                    mean_cer_red = (
                        (mean_cer_init - mean_cer_post) / mean_cer_init * 100.0
                    )
                    print(f"Mean CER_reduction: {mean_cer_red:.6f}")
            if metrics["wer_init"] and metrics["wer_post"]:
                mean_wer_init = statistics.mean(metrics["wer_init"])
                mean_wer_post = statistics.mean(metrics["wer_post"])
                if mean_wer_init != 0.0:
                    mean_wer_red = (
                        (mean_wer_init - mean_wer_post) / mean_wer_init * 100.0
                    )
                    print(f"Mean WER_reduction: {mean_wer_red:.6f}")
            print()

        print("5-run means for target settings:")
        for setting in sorted(TARGET_SETTINGS):
            runs = setting_run_means.get(setting, [])
            if not runs:
                print(f"{setting}: no runs found")
                continue
            cer_init_vals = [r["cer_init"] for r in runs if "cer_init" in r]
            wer_init_vals = [r["wer_init"] for r in runs if "wer_init" in r]
            cer_post_vals = [r["cer_post"] for r in runs if "cer_post" in r]
            wer_post_vals = [r["wer_post"] for r in runs if "wer_post" in r]
            cer_red_vals = [r["cer_red"] for r in runs if "cer_red" in r]
            wer_red_vals = [r["wer_red"] for r in runs if "wer_red" in r]
            print(f"{setting} (runs: {len(runs)})")
            if cer_init_vals:
                print(f"Mean CER_init (runs): {statistics.mean(cer_init_vals):.6f}")
                print(f"Mean WER_init (runs): {statistics.mean(wer_init_vals):.6f}")
            if cer_post_vals:
                print(f"Mean CER_post (runs): {statistics.mean(cer_post_vals):.6f}")
                print(f"Mean WER_post (runs): {statistics.mean(wer_post_vals):.6f}")
            if cer_init_vals and cer_post_vals:
                mean_cer_init = statistics.mean(cer_init_vals)
                mean_cer_post = statistics.mean(cer_post_vals)
                if mean_cer_init != 0.0:
                    mean_cer_red = (
                        (mean_cer_init - mean_cer_post) / mean_cer_init * 100.0
                    )
                    print(f"Mean CER_reduction (runs): {mean_cer_red:.6f}")
            if wer_init_vals and wer_post_vals:
                mean_wer_init = statistics.mean(wer_init_vals)
                mean_wer_post = statistics.mean(wer_post_vals)
                if mean_wer_init != 0.0:
                    mean_wer_red = (
                        (mean_wer_init - mean_wer_post) / mean_wer_init * 100.0
                    )
                    print(f"Mean WER_reduction (runs): {mean_wer_red:.6f}")
            print()


if __name__ == "__main__":
    main(use_all_results=True)
