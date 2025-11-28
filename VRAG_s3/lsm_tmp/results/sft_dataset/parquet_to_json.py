import pandas as pd
import json

INPUT_PARQUET = "./filtered_train_matched_answer_exact.parquet"
OUTPUT_JSON = "./filtered_train_matched_answer_exact.json"

def parquet_to_json(input_file, output_file, orient="records", lines=False):
    df = pd.read_parquet(input_file)

    # messages가 문자열이면 dict/list로 변환
    if "messages" in df.columns:
        df["messages"] = df["messages"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    # 보기 좋은 JSON으로 저장
    df.to_json(output_file, orient=orient, lines=lines, force_ascii=False, indent=2)

    print(f"✅ 변환 완료: {input_file} → {output_file}")

if __name__ == "__main__":
    parquet_to_json(INPUT_PARQUET, OUTPUT_JSON)

