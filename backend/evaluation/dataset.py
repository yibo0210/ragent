"""黄金数据集加载器。"""
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent.parent / "tests" / "golden_dataset.json"


def load_golden_dataset() -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_questions() -> list[str]:
    return [item["question"] for item in load_golden_dataset()]


def get_ground_truths() -> list[str]:
    return [item["ground_truth"] for item in load_golden_dataset()]


def get_expected_agents() -> list[str | None]:
    return [item.get("expected_agent") for item in load_golden_dataset()]
