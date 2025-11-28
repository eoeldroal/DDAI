import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd

random.seed(42)

def load_examples(path: str) -> List[Dict[str, str]]:
    """Load examples from a JSONL file.

    Each line should be a JSON object containing a question under the key
    ``query`` and an expert trajectory under the key ``history``.
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line) #json 형식의 문자열을 딕셔너리 객체로 파싱
            prompt = data.get("query", "") #defalut : ""
            trajectory = json.dumps(data.get("history", []), ensure_ascii=False) #딕셔너리 -> json 문자열
            examples.append({"prompt": prompt, "response": trajectory})
    return examples


def split_dataset(data: List[Dict[str, str]], test_ratio: float = 0.2):
    """Shuffle and split data into train and test portions."""
    random.shuffle(data)
    #split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:8]
    test_data = data[8:]
    return train_data, test_data


def save_parquet(data: List[Dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Convert cot jsonl to train/test parquet files")
    #parser.add_argument("--input", default="example/results/cot_crop.jsonl", help="Input JSONL file") 
    parser.add_argument("--input", default="./lsm_tmp/cot_crop.jsonl", help="Input JSONL file") 
    #parser.add_argument("--output_dir", default="example/results", help="Directory to save parquet files")
    parser.add_argument("--output_dir", default="./lsm_tmp/results", help="Directory to save parquet files")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion of dataset used for test set")
    args = parser.parse_args()

    examples = load_examples(args.input)
    train_data, test_data = split_dataset(examples, test_ratio=args.test_ratio)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "val.parquet")

    save_parquet(train_data, train_path)
    save_parquet(test_data, test_path)


if __name__ == "__main__":
    main()
