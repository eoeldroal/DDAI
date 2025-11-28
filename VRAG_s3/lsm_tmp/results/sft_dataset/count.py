import pandas as pd

df = pd.read_parquet("filtered_val_fin.parquet")
print(f"✅ 총 데이터 개수: {len(df)}")

# uid 컬럼 기준으로 고유한 데이터 수 확인
if "uid" in df.columns:
    print(f"🆔 고유 UID 개수: {df['uid'].nunique()}")

