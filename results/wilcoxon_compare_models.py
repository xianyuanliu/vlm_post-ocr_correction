import argparse
import csv
from statistics import mean, median
from scipy.stats import wilcoxon


def read_metric_by_key(path: str, key_col: str, metric_col: str):
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[key_col].strip()
            value = float(row[metric_col])
            data[key] = value
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon test between two model CSVs."
    )
    parser.add_argument(
        "--a",
        default="results/llama-3.2-11b-vision-ocr.csv",
        help="CSV for model A (default: results/llama-3.2-11b-vision-ocr.csv)",
    )
    parser.add_argument(
        "--b",
        default="results/qwen2.5-vl-7b-ocr.csv",
        help="CSV for model B (default: results/qwen2.5-vl-7b-ocr.csv)",
    )
    parser.add_argument(
        "--key",
        default="Image Patch",
        help="Join key column (default: Image Patch)",
    )
    parser.add_argument(
        "--metric",
        default="WER_post",
        help="Metric column to compare (default: CER_post)",
    )
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="Alternative hypothesis (default: two-sided)",
    )
    args = parser.parse_args()

    a_vals = read_metric_by_key(args.a, args.key, args.metric)
    b_vals = read_metric_by_key(args.b, args.key, args.metric)

    common_keys = sorted(set(a_vals) & set(b_vals))

    a_list = [a_vals[k] for k in common_keys]
    b_list = [b_vals[k] for k in common_keys]

    diffs = [b - a for a, b in zip(a_list, b_list)]
    stat, p_value = wilcoxon(a_list, b_list, alternative=args.alternative)

    print(f"Model A: {args.a}")
    print(f"Model B: {args.b}")
    print(f"Key: {args.key}")
    print(f"Metric: {args.metric}")
    print(f"Pairs: {len(common_keys)}")
    print(f"Mean diff (B - A): {mean(diffs):.6f}")
    print(f"Median diff (B - A): {median(diffs):.6f}")
    print(f"Wilcoxon statistic: {stat:.6f}")
    print(f"P-value: {p_value:.6g}")


if __name__ == "__main__":
    main()
