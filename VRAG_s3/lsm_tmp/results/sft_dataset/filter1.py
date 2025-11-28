import json
import pandas as pd

TRAIN_PARQUET = "filtered_val.parquet"
OUTPUT_PARQUET = "filtered_val_fin.parquet"

def count_assistant_messages(messages):
    """assistant role 메시지 개수 계산"""
    if not isinstance(messages, list):
        return 0
    return sum(1 for m in messages if m.get("role") == "assistant")

def count_image_crop(messages):
    """'/data/image_crop/' 경로 포함 이미지 개수 계산"""
    if not isinstance(messages, list):
        return 0
    count = 0
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and "image" in item:
                    if "/data/image_crop/" in item["image"]:
                        count += 1
    return count

def main():
    # train.parquet 로드
    df = pd.read_parquet(TRAIN_PARQUET)
    filtered_rows = []

    for _, row in df.iterrows():
        entry = row.to_dict()
        messages = entry.get("messages")

        # parquet에 문자열로 저장된 경우 JSON으로 변환
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                continue

        num_assistant = count_assistant_messages(messages)
        num_crop_images = count_image_crop(messages)

        # 조건: assistant ≤ 5 and crop_image ≤ 2
        if num_assistant <= 5 and num_crop_images <= 2:
            filtered_rows.append(entry)

    # 필터링 결과 저장
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"✅ 필터링 완료: {len(filtered_df)}개 데이터가 {OUTPUT_PARQUET}에 저장되었습니다.")
    else:
        print("⚠️ 조건에 맞는 데이터가 없습니다.")

if __name__ == "__main__":
    main()

