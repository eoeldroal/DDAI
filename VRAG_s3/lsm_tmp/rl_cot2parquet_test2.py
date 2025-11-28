import argparse
import json
import os
from typing import List, Dict, Any

from datasets import Dataset

# 시스템 프롬프트와 사용자 프롬프트는 data_construct_pipeline에서 사용한 것과 동일하게 정의
#원래 vrag prompt 사용 
PROMPT_INST = '''Answer the given question. 
You must conduct reasoning inside <think> and </think> every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. 
Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. 
You can search as many times as you want. 
If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. 
Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Question: {question}'''



def _load_json(path: str) -> List[Dict[str, Any]]:
    """json 파일을 로드하여 'examples' 키의 리스트를 반환한다."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 파일 전체를 읽는 json.load() 사용
    return data["examples"] # 실제 데이터 리스트가 담긴 키를 지정


def convert(json_path: str, output_path: str) -> None:
    """JSON 파일을 읽어 Parquet 형식으로 변환하는 함수 (수정 없음)"""
    print(f"\nProcessing '{json_path}'...")
    if not os.path.exists(json_path):
        print(f"!!! Warning: Input file not found at '{json_path}'. Skipping.")
        return

    samples = _load_json(json_path)
    records = []
    for sample in samples:
        messages: List[Dict[str, Any]] = [
            {"role": "assistant", "content": PROMPT_INST},
            {"role": "user", "content": PROMPT_USER_START.format(question=sample["query"])}
        ]

        record = {
            "data_source": "slidevqa",
            "prompt": messages,
            "reward_model": {
                "ground_truth": sample.get("reference_answer", "")
            },
            "reference_page": sample.get("meta_info", {}).get("reference_page"),
            "extra_info": {
                "question": sample.get("query"),
                "answer": sample.get("reference_answer", "")
            },
            "file_name": sample.get("meta_info", {}).get("file_name", "")
        }
        records.append(record)

    ds = Dataset.from_list(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_parquet(output_path)
    print(f"✅ Successfully created '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset JSON to Parquet format.")
    # 기본 훈련 데이터셋 경로
    parser.add_argument("--train_input", type=str, default="./lsm_tmp/rag_dataset.json", help="입력 훈련 json 파일 경로")
    parser.add_argument("--train_output", type=str, default="./lsm_tmp/results/slidevqa_train_crop.parquet", help="저장할 훈련 parquet 파일 경로")
    
    # 추가된 테스트 데이터셋 경로
    parser.add_argument("--test_input", type=str, default="./lsm_tmp/rag_test_dataset.json", help="입력 테스트 json 파일 경로")
    parser.add_argument("--test_output", type=str, default="./lsm_tmp/results/overall_test_crop.parquet", help="저장할 테스트 parquet 파일 경로")
    
    args = parser.parse_args()

    # --- 1. 훈련 데이터셋 변환 (기존 로직) ---
    convert(args.train_input, args.train_output)

    # --- 2. 테스트 데이터셋 변환 (추가된 로직) ---
    convert(args.test_input, args.test_output)


if __name__ == "__main__":
    main()