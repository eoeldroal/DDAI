import json
from pathlib import Path
import re
import pandas as pd

TRAIN_PARQUET = "val.parquet"
DATASET_JSON = "dataset_score1.json"
OUTPUT_PARQUET = "filtered_val.parquet"

def extract_last_corpus_image_path(messages):
    """
    messages 리스트에서 ./search_engine/corpus/img/ 경로를 가진
    마지막 image path만 추출
    """
    last_image = None
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and "image" in item:
                    image_path = item["image"]
                    if "/search_engine/corpus/img/" in image_path:
                        last_image = image_path
    return last_image

def extract_page_from_filename(filename):
    """예: '28_12.jpg' → 12"""
    m = re.search(r"_(\d+)\.jpg$", filename)
    if m:
        return int(m.group(1))
    return None

def main():
    # train.parquet 로드
    df = pd.read_parquet(TRAIN_PARQUET)
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)

    dataset_index = {d["id"]: d for d in dataset_data}
    filtered_rows = []

    for _, row in df.iterrows():
        entry = row.to_dict()
        uid = entry.get("uid")
        if uid not in dataset_index:
            continue

        ds_entry = dataset_index[uid]
        evidence_pages = ds_entry.get("evidence_pages", [])
        if not evidence_pages:
            continue

        messages = entry.get("messages", [])
        # parquet 저장 시 문자열로 변환된 경우 처리
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                continue

        last_image = extract_last_corpus_image_path(messages)
        if not last_image:
            continue

        last_image_name = Path(last_image).name
        page_num = extract_page_from_filename(last_image_name)

        if page_num in evidence_pages:
            filtered_rows.append(entry)

    # 결과를 parquet으로 저장
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"✅ 필터링 완료: {len(filtered_df)}개 데이터가 {OUTPUT_PARQUET}에 저장되었습니다.")
    else:
        print("⚠️ 조건에 맞는 데이터가 없습니다.")

if __name__ == "__main__":
    main()


