import pandas as pd

# 파일 경로
TRAIN_PATH = "filtered_train_matched_answer_exact.parquet"
VAL_PATH = "filtered_val_matched_answer_exact.parquet"

# 파일 로드
train_df = pd.read_parquet(TRAIN_PATH)
val_df = pd.read_parquet(VAL_PATH)

print(f"이전 train 개수: {len(train_df)}, val 개수: {len(val_df)}")

# 1️⃣ train에서 1개 샘플 랜덤 선택
sample_row = train_df.sample(n=1, random_state=42)

# 2️⃣ 선택한 샘플을 train에서 제거
train_df = train_df.drop(sample_row.index)

# 3️⃣ val에 추가
val_df = pd.concat([val_df, sample_row], ignore_index=True)

# 4️⃣ 저장
train_df.to_parquet(TRAIN_PATH, index=False)
val_df.to_parquet(VAL_PATH, index=False)

print(f"✅ 이동 완료! train 개수: {len(train_df)}, val 개수: {len(val_df)}")
print(f"이동된 uid: {sample_row.iloc[0].get('uid')}")
