import pandas as pd

INPUT_PARQUET = "./slidevqa_train_crop.parquet"
OUTPUT_JSON = "./slidevqa_train_crop.json"

def parquet_to_json(input_file, output_file, orient="records", lines=False):
    df = pd.read_parquet(input_file)

    # 그대로 JSON으로 저장
    df.to_json(output_file, orient=orient, lines=lines, force_ascii=False, indent=2)

    print(f"✅ 변환 완료: {input_file} → {output_file}")

if __name__ == "__main__":
    parquet_to_json(INPUT_PARQUET, OUTPUT_JSON)

