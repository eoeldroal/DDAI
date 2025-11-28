import json
import pandas as pd
import re

TRAIN_PARQUET = "filtered_val_fin.parquet"           # 입력 parquet
DATASET_JSON = "dataset_score1.json"                 # 입력 json
OUTPUT_PARQUET = "filtered_val_matched_answer_exact.parquet"  # 출력 parquet


def extract_last_think_text(messages):
    """messages에서 마지막 <think>...</think> 텍스트 추출"""
    if not isinstance(messages, list):
        return ""
    think_texts = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            matches = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
            think_texts.extend(matches)
    return think_texts[-1].strip() if think_texts else ""


def main():
    # 1️⃣ train.parquet 로드
    df = pd.read_parquet(TRAIN_PARQUET)

    # 2️⃣ dataset.json 로드 (id → target 매핑)
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)

    id_to_target = {}
    for entry in dataset_data:
        if "id" in entry and "reward_model" in entry:
            target = entry["reward_model"]["ground_truth"].get("target", [])
            if isinstance(target, list) and len(target) > 0:
                id_to_target[entry["id"]] = target[0]

    matched_entries = []

    # 3️⃣ 각 row 순회
    for _, row in df.iterrows():
        entry = row.to_dict()
        uid = entry.get("uid")
        messages = entry.get("messages")

        # parquet에 문자열로 저장된 경우 JSON 파싱
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                continue

        last_think = extract_last_think_text(messages)
        if not last_think:
            continue

        # 4️⃣ dataset.json의 target과 exact match (대소문자 구분)
        target = id_to_target.get(uid)
        if target and target in last_think:
            matched_entries.append(entry)

    # 5️⃣ 결과 저장
    if matched_entries:
        matched_df = pd.DataFrame(matched_entries)
        matched_df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"✅ 총 {len(matched_df)}개의 데이터가 필터링되어 {OUTPUT_PARQUET}에 저장되었습니다.")
    else:
        print("⚠️ 조건에 맞는 데이터가 없습니다.")


if __name__ == "__main__":
    main()
