import argparse
import json
from collections import defaultdict
from typing import Dict, List


def load_results(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    # Flatten all categories into one list
    items: List[Dict] = []
    for _, vals in data.items():
        items.extend(vals)
    return items


def summarize(items: List[Dict]) -> Dict:
    def avg(key: str, subset: List[Dict]) -> float:
        vals = [float(x.get(key, 0.0)) for x in subset if key in x]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "count": len(items),
        "bleu1_avg": avg("bleu_score", items),
        "f1_avg": avg("f1_score", items),
        "llm_avg": avg("llm_score", items),
    }

    # Per-category averages
    by_cat = defaultdict(list)
    for item in items:
        by_cat[str(item.get("category", "unknown"))].append(item)

    per_cat = {
        cat: {
            "count": len(vals),
            "bleu1_avg": avg("bleu_score", vals),
            "f1_avg": avg("f1_score", vals),
            "llm_avg": avg("llm_score", vals),
        }
        for cat, vals in by_cat.items()
    }
    summary["per_category"] = per_cat
    return summary


def print_summary(summary: Dict) -> None:
    print(f"Total items: {summary['count']}")
    print(f"Overall -> BLEU1: {summary['bleu1_avg']:.4f}, F1: {summary['f1_avg']:.4f}, LLM: {summary['llm_avg']:.4f}")
    print("\nPer category:")
    for cat, vals in sorted(summary["per_category"].items(), key=lambda x: x[0]):
        print(
            f"  Cat {cat:>2}: n={vals['count']:>4}, "
            f"BLEU1={vals['bleu1_avg']:.4f}, F1={vals['f1_avg']:.4f}, LLM={vals['llm_avg']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Print eval metrics summary")
    parser.add_argument(
        "--input_file", type=str, default="results/eval.json", help="Path to eval.json produced by evaluation/evals.py"
    )
    args = parser.parse_args()

    items = load_results(args.input_file)
    summary = summarize(items)
    print_summary(summary)


if __name__ == "__main__":
    main()
