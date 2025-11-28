import pandas as pd

# --- 파일 경로 ---
a_path = "train.parquet"
b_path = "val.parquet"
c_path = "filtered_train_fin.parquet"
d_path = "filtered_val_fin.parquet"

# --- parquet 로드 ---
a_df = pd.read_parquet(a_path)
b_df = pd.read_parquet(b_path)
c_df = pd.read_parquet(c_path)
d_df = pd.read_parquet(d_path)

# --- uid 집합 구하기 ---
ab_uids = set(a_df["uid"]) | set(b_df["uid"])
cd_uids = set(c_df["uid"]) | set(d_df["uid"])

# --- 차집합 (a+b에만 있는 uid) ---
unique_to_ab = sorted(ab_uids - cd_uids)

print(f"✅ (a+b)에는 있고 (c+d)에는 없는 UID 개수: {len(unique_to_ab)}")
print(unique_to_ab[:50])  # 미리보기

# --- 결과 저장 ---
with open("unique_to_ab_uids.txt", "w") as f:
    for uid in unique_to_ab:
        f.write(uid + "\n")

