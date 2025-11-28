import pandas as pd

# --- 설정 ---
INPUT_FILE = 'filtered_val_fin.parquet'      # 원본 Parquet 파일 이름
OUTPUT_FILE = 'filtered_val_fin_100.parquet' # 저장할 샘플 파일 이름
NUM_ROWS = 10                   # 추출할 데이터 개수

def create_parquet_sample(input_path, output_path, num_samples):
    """
    Parquet 파일에서 지정된 개수만큼의 샘플을 추출하여 새 Parquet 파일로 저장합니다.
    """
    try:
        # 1. 원본 Parquet 파일을 DataFrame으로 읽어옵니다.
        print(f"'{input_path}' 파일을 읽는 중입니다...")
        df = pd.read_parquet(input_path)
        
        print(f"성공! 전체 데이터 개수: {len(df)}개")

        # 2. 원본 데이터가 추출할 개수보다 적은지 확인합니다.
        if len(df) < num_samples:
            print(f"경고: 원본 데이터({len(df)}개)가 요청한 샘플 개수({num_samples}개)보다 적습니다.")
            num_samples_to_take = len(df)
        else:
            num_samples_to_take = num_samples

        # 3. 맨 위에서부터 200개의 데이터를 선택합니다.
        print(f"상위 {num_samples_to_take}개의 데이터를 추출합니다...")
        sample_df = df.head(num_samples_to_take)

        # 4. 선택된 데이터를 새로운 Parquet 파일로 저장합니다.
        # index=False 옵션은 불필요한 인덱스 컬럼이 저장되는 것을 방지합니다.
        sample_df.to_parquet(output_path, index=False)
        
        print(f"성공! '{output_path}' 파일에 {len(sample_df)}개의 데이터가 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    create_parquet_sample(INPUT_FILE, OUTPUT_FILE, NUM_ROWS)