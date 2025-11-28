import pandas as pd
import os

# =====================================================================
# 1단계: 기본 정보 및 파일 경로 설정
# =====================================================================
IMAGE_DIR = "./data/rag/source_images/"
PLACEHOLDER_IMG_PATH = os.path.join(IMAGE_DIR, "placeholder.png")
SALES_CHART_IMG_PATH = os.path.join(IMAGE_DIR, "sales_report_q2.png") # 실제 이미지 경로
ANIMAL_IMG_PATH = os.path.join(IMAGE_DIR, "animal_01.png")       # 실제 이미지 경로

# 원본 데이터 정보: 사람이 관리하기 쉬운 형태
dataset_info = [
    # 1. 멀티모달 (차트 질문)
    {
        "data_source": "vqa_chart", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "이 차트에서 2024년 매출액은 얼마인가요?"}],
        "image_path": SALES_CHART_IMG_PATH,
        "extra_info": {"answer": "$15,000", "question_type": "extractive_qa", "source_file": "sales_report_q2.png"}
    },
    # 2. 멀티모달 (추론)
    {
        "data_source": "vqa_chart", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "2024년도와 비교했을 때 2033년도 매출 성장률은 몇 퍼센트인가요?"}],
        "image_path": SALES_CHART_IMG_PATH,
        "extra_info": {"answer": "25%", "question_type": "reasoning", "source_file": "sales_report_q2.png"}
    },
    # 3. RAG (문서 기반 QA)
    {
        "data_source": "rag_doc", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "다음 문서를 참고하여 'Quantum Processor'의 주요 기능 3가지를 알려줘."}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "AI 최적화, 실시간 노이즈 제거, 전력 효율성 향상", "source_doc_id": "doc-a123-v2", "page_number": 3}
    },
    # 4. 일반 상식
    {
        "data_source": "general", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "대한민국의 수도는 어디인가요?"}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "서울", "category": "geography"}
    },
    # 5. 코딩 (함수 생성)
    {
        "data_source": "coding", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "리스트를 입력받아 평균을 반환하는 파이썬 함수를 작성해줘."}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "def calculate_average(numbers):\n  if not numbers:\n    return 0\n  return sum(numbers) / len(numbers)", "language": "python"}
    },
    # 6. 수학
    {
        "data_source": "math", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "사과 5개와 오렌지 3개가 있습니다. 과일은 총 몇 개인가요?"}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "8", "type": "arithmetic"}
    },
    # 7. 번역
    {
        "data_source": "translation", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "Translate 'Large Language Model' to Korean."}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "거대 언어 모델", "source_lang": "en", "target_lang": "ko"}
    },
    # 8. 요약
    {
        "data_source": "summarization", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "다음 기사를 한 문장으로 요약해줘."}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "최근 발표된 AI 기술은 기존 모델보다 학습 효율을 50% 향상시켰다.", "source_article_id": "news-456"}
    },
    # 9. 창의적 글쓰기
    {
        "data_source": "creative", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "우주를 여행하는 고양이에 대한 짧은 이야기를 써줘."}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": None, "genre": "sci-fi"}
    },
    # 10. 대화
    {
        "data_source": "dialogue", # <-- 'data_source' 키 추가
        "prompt": [
            {"role": "user", "content": "오늘 인천 날씨 어때?"},
            {"role": "assistant", "content": "오늘 인천은 맑고 기온은 24도입니다."},
            {"role": "user", "content": "내일은 어때?"}
        ],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "내일은 대체로 맑겠고, 예상 기온은 25도입니다.", "topic": "weather"}
    },
    # 11. 코딩 (디버깅)
    {
        "data_source": "coding", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "다음 코드의 오류를 찾아줘: print('Hello, World'"}],
        "image_path": PLACEHOLDER_IMG_PATH,
        "extra_info": {"answer": "SyntaxError: Missing closing parenthesis ')'", "language": "python"}
    },
    # 12. 멀티모달 (이미지 분류)
    {
        "data_source": "vqa_classification", # <-- 'data_source' 키 추가
        "prompt": [{"role": "user", "content": "이 이미지에 있는 동물은 무엇인가요?"}],
        "image_path": ANIMAL_IMG_PATH,
        "extra_info": {"answer": "강아지", "source_file": "animal_01.jpg"}
    }
]

# =====================================================================
# 2단계: 데이터 가공 및 Parquet 파일 생성
# =====================================================================

def load_image_bytes(filepath):
    """파일 경로를 받아 바이트 데이터를 안전하게 읽어오는 함수"""
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except FileNotFoundError:
        print(f"!!! 경고: '{filepath}' 파일을 찾을 수 없어 placeholder로 대체합니다.")
        if not os.path.exists(PLACEHOLDER_IMG_PATH):
            raise FileNotFoundError(f"치명적 에러: Placeholder 이미지 '{PLACEHOLDER_IMG_PATH}'도 찾을 수 없습니다.")
        with open(PLACEHOLDER_IMG_PATH, "rb") as f:
            return f.read()

# 2-1. 파일 경로를 실제 바이트 데이터로 변환하면서 필요한 필드 추가
final_dataset = []
for info in dataset_info:
    # 'question' 키 추가
    if 'question' not in info['extra_info']:
        user_prompts = [p['content'] for p in info['prompt'] if p['role'] == 'user']
        info['extra_info']['question'] = user_prompts[0] if user_prompts else ""
    
    # 'data_source' 키를 최상위 레벨로 이동
    item = {
        "data_source": info.get("data_source", "unknown"),
        "prompt": info["prompt"],
        "images": [{"bytes": load_image_bytes(info["image_path"])}],
        "extra_info": info["extra_info"]
    }
    final_dataset.append(item)

# 2-2. 모든 에러 해결을 위한 최종 데이터 가공
corrected_dataset = []
for item in final_dataset:
    item['file_name'] = item['extra_info'].get('source_file', 'placeholder.png')
    item['reference_page'] = item['extra_info'].get('page_number', 0)
    answer = item['extra_info'].get('answer')
    item['reward_model'] = {'ground_truth': answer}
    last_turn = item["prompt"][-1]
    last_turn["content"] = "<image>\n" + last_turn["content"]
    
    corrected_dataset.append(item)

# 2-3. DataFrame으로 변환 및 파일 저장
df = pd.DataFrame(corrected_dataset)
train_df = df.iloc[:8]
test_df = df.iloc[8:]

train_output_path = "./data/rag/slidevqa_train_crop.parquet"
test_output_path = "./data/rag/overall_test_crop.parquet"

os.makedirs(os.path.dirname(train_output_path), exist_ok=True)

train_df.to_parquet(train_output_path, index=False)
print(f"✅ 훈련용 파일 저장 완료: {train_output_path} ({len(train_df)}개 행)")

test_df.to_parquet(test_output_path, index=False)
print(f"✅ 테스트용 파일 저장 완료: {test_output_path} ({len(test_df)}개 행)")