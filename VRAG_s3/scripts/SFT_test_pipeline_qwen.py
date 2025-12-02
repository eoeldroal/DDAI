import os
import json
from tqdm import tqdm
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.schema import NodeWithScore, ImageNode
import sys
import time
import requests
from typing import List, Dict, Any, Optional, Tuple
import zlib  # uid 해시 fallback용

# --- ✨ 1단계 수정: 필요한 라이브러리 추가 ✨ ---
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image # 이미지를 다루기 위해 PIL도 import 합니다.

# (기존 전역 변수들은 그대로 유지)
image_output_dir = './data/image_crop'
raw_image_dir = './search_engine/corpus/img'

# Dashscope 관련 설정은 이제 사용되지 않지만, 다른 기능에 영향을 주지 않도록 남겨둘 수 있습니다.
# import dashscope
# from http import HTTPStatus
# dashscope.base_http_api_url = ...
# API_KEY = ...
# dashscope.api_key = API_KEY

search_engine_url = "http://163.239.28.21:5002/search"
TOPK = 10

# prompt_inst 와 prompt_user_start 는 삭제 또는 주석 처리

# USER_PROMPT = '''You are a search agent.
# You must conduct reasoning inside <think> and </think> every time you get new information. 
# After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. 
# Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. 
# You can search as many times as you want. 
# If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. 
# Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Question: {question}'''
#USER_PROMPT = '''First, think step-by-step about the user's question inside <think> tags. Then, perform an action like <search> or <bbox> or <search_complete>.
#Question: {question}
#'''
# (기존 유틸리티 함수들은 수정 없이 그대로 사용)

# SYSTEM_RULES = (
#     "You are a search agent.\n"
#     "You must always begin with <think>...</think> showing your reasoning about the question.\n"
#     "After reasoning, output exactly one action tag among <search>...</search> or <bbox>[x1, y1, x2, y2]</bbox> or <search_complete>true</search_complete>.\n"
#     "Do not write anything before <think>. Keep actions on a new line after </think>."
# )

SYSTEM_RULES = (
"You are a search agent.\n"
"You must conduct reasoning inside <think> and </think> every time you get new information.\n"
"After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. \n"
"Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. /n"
"You can search as many times as you want. /n"
"If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>.\n"
"Otherwise, continue with <search> or <bbox> actions until you are ready to finish.\n"
)

USER_QUESTION_FMT = (
    "Question: {question}"
)


import re
import ast 
def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math

    if isinstance(image, str):
        image = Image.open(image)
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def _sanitize_for_json(obj):
    # dict, list 재귀 처리
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    # PIL Image → 문자열로 치환
    try:
        from PIL.Image import Image as PILImage
        if isinstance(obj, PILImage):
            return "<PIL.Image omitted>"
    except Exception:
        pass
    return obj




def extract_json(response: str) -> dict:
    """
    <think>, <search>, <bbox>, <search_complete> 태그를 포함한 텍스트에서
    내용을 파싱하여 파이썬 딕셔너리를 생성합니다.
    """
    think_content = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    search_content = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
    bbox_content = re.search(r"<bbox>(.*?)</bbox>", response, re.DOTALL)
    search_complete_found = "<search_complete>true</search_complete>" in response

    result_json = {}

    if think_content:
        result_json["think"] = think_content.group(1).strip()
    
    if search_content:
        result_json["search"] = search_content.group(1).strip()
    
    if bbox_content:
        bbox_str = bbox_content.group(1).strip()
        try:
            result_json["bbox"] = ast.literal_eval(bbox_str)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse bbox string: {bbox_str}")
            result_json["bbox"] = bbox_str

    if search_complete_found:
        result_json["search_complete"] = True

    if not result_json or "think" not in result_json:
        raise ValueError("A valid action with a <think> tag was not found in the response.")

    return result_json
def crop_and_dump(image_path, bbox, output_folder=image_output_dir):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"cannot open {image_path}: {e}")
        return None
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_image = image.crop((x1, y1, x2, y2))
    timestamp = int(time.time() * 1000)
    file_extension = os.path.splitext(image_path)[1]
    output_filename = f"{timestamp}{file_extension}"
    output_path = os.path.join(output_folder, output_filename)
    print(image_path)
    print(output_path)
    try:
        cropped_image.save(output_path)
        return output_path
    except Exception as e:
        print(f"fail: {e}")
        return None

def _uid_to_query_id(uid: str | int | None) -> int:
    if uid is None: return 0
    if isinstance(uid, int): return uid
    s = str(uid)
    m = re.search(r'(\d+)$', s)
    if m:
        try: return int(m.group(1))
        except: pass
    return zlib.crc32(s.encode("utf-8")) & 0xffffffff

class HTTPImageSearcher:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def query(self, query: str, query_id: str | int, topk: int = TOPK) -> List[str]:
        try: query_id = int(query_id)
        except Exception: query_id = _uid_to_query_id(query_id)

        batch_reqs = [{"query": query, "id": str(query_id)}]

        try:
            # POST 방식으로 json을 사용해 데이터 전송
            resp = self.session.post(
                self.base_url,
                json=batch_reqs, # 'params' 대신 'json' 사용
                timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
            batch0 = data[0] if isinstance(data, list) and data else []
            paths = [item["image_file"] for item in batch0 if "image_file" in item]
            return paths[:topk]
        except Exception as e:
            print(f"[SEARCH ERROR] {e}")
            return []


# --- ✨ 1단계 수정: SFT 모델을 위한 새로운 '엔진' 클래스 구현 ✨ ---
class SFT_VLM_Role:
    """로컬 SFT 모델을 로드하고 추론을 수행하는 클래스"""
    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("Model loaded successfully.")

# 텍스트(messages)와 함께 이미지 경로 리스트(image_paths)도 입력으로 받습니다.
    def generate(self, messages, image_paths=None):
        try:
            # (A) 프롬프트 문자열 생성: 토크나이저의 chat_template 사용
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # (B) 이미지 로딩 (있으면)
            raw_images = [Image.open(p).convert("RGB") for p in image_paths] if image_paths else None

            # (C) 텍스트+이미지를 processor로 한번에 텐서화
            inputs = self.processor(
                text=prompt,
                images=raw_images,
                return_tensors="pt"
            ).to(self.model.device)

            # (D) 생성
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            prompt_len = inputs['input_ids'].shape[-1] if 'input_ids' in inputs else 0
            gen_ids = outputs[0, prompt_len:] if prompt_len > 0 else outputs[0]

            # 빈 생성(모델이 아무 것도 안 냈을 때) 대비
            if gen_ids.numel() == 0:
                return ""

            # tokenizer로 디코드 (processor.batch_decode 말고)
            generated = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            return generated

        except Exception as e:
            print(f"[SFT VLM Generate Error] {e}")
            return None



class MMRAG:
    def __init__(self,
                model_path: str,
                dataset='search_engine/corpus',
                query_file='rag_dataset.json',
                experiment_type = 'cot',
                workers_num = 1,
                topk=TOPK):
        
        # --- __init__ 함수의 내용은 대부분 동일 ---
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.query_file = query_file
        self.dataset_dir = os.path.join('./', dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        if experiment_type == 'cot':
            self.eval_func = self.cot_collect
            self.output_file_name = 'sft_test_results.jsonl' 

        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/","-"))
        self.conversation_log_path = os.path.join(self.results_dir, "conversation_history.jsonl")
        
        self.sft_model = SFT_VLM_Role(model_path=model_path)
        self.searcher = HTTPImageSearcher(search_engine_url)

    # --- ✨ 1. postprocess_predictions 함수 추가 ✨ ---
    def postprocess_predictions(self, prediction: str) -> Tuple[str, str]:
        """모델의 원본 응답에서 액션과 내용을 추출합니다."""
        pattern = r'<(search|bbox|answer|search_complete)>(.*?)</\1>'
        match = re.search(pattern, prediction, re.DOTALL)
        if match:
            action = match.group(1)
            content = match.group(2).strip()
            # search_complete는 내용이 없으므로 content를 True로 설정
            if action == 'search_complete':
                content = True
            return action, content
        return None, prediction # 유효한 태그가 없으면 action은 None, content는 원본 응답

    # --- ✨ 2. execute_predictions 함수 추가 ✨ ---
    def execute_predictions(self, action: str, content: Any, uid: str, all_images: List[str]) -> Tuple[Any, bool]:
        """추출된 액션을 실행하고, 다음 observation과 종료 여부를 반환합니다."""
        next_obs = None
        done = False

        if action == 'search':
            query_id = _uid_to_query_id(uid)
            # HTTPImageSearcher를 직접 호출합니다.
            search_results = self.searcher.query(content, query_id, self.top_k)
            next_obs = search_results
        
        elif action == 'bbox':
            if len(all_images) > 0:
                try:
                    bbox_value = ast.literal_eval(content) if isinstance(content, str) else content
                    if len(bbox_value) == 4 and all(isinstance(x, (int, float)) for x in bbox_value):
                         # crop_and_dump 함수는 이미지 경로를 반환합니다.
                        next_obs = crop_and_dump(all_images[-1], bbox_value)
                    else:
                        raise ValueError("Invalid bbox value")
                except:
                    next_obs = "Your previous bbox action was invalid. Please try again."
            else:
                next_obs = "You tried to crop, but there are no images retrieved yet."

        elif action == 'answer' or action == 'search_complete':
            done = True
        
        else: # 유효한 액션 태그가 없는 경우
            next_obs = "Your response did not contain a valid action tag. Please use <search>, <bbox>, or <search_complete>."

        return next_obs, done

    # --- ✨ 3. cot_collect 함수 단순화 ✨ ---
    def cot_collect(self, sample):
        query = sample['query']
        messages = [
            {"role": "system", "content": SYSTEM_RULES},
            {"role": "user",   "content": USER_QUESTION_FMT.format(question=query)}
        ]
            
        all_images = []
        crop_images = []
        status = "failed_unknown"
        final_result = None
        raw_response = ""

        try:
            for _ in range(10): # 최대 10턴
                # response = self.sft_model.generate(
                #     messages=messages, 
                #     image_paths=[all_images[-1]] if all_images else None
                # )
                response = self.sft_model.generate(messages=messages)

                raw_response = response
                if response is None:
                    status = "failed_model_none_response"; break
                
                print("------------대답🚀------------ :\n",response)

                # 1. 모델 응답에서 액션과 내용 추출
                action, content = self.postprocess_predictions(response)
                
                # 2. 추출된 액션 실행
                next_obs, done = self.execute_predictions(action, content, sample.get('uid'), all_images)

                # 3. 대화 기록 업데이트
                messages.append({"role": "assistant", "content": response}) # 모델의 원본 응답을 기록
                '''
                if done:
                    status = "success"
                    sample['collected'] = {"images": all_images, "crops": crop_images}
                    final_result = sample
                    break
                '''
                if done:
                    status = "success"
                    final_result = {
                        "uid": sample.get("uid"),
                        "query": query,
                        "collected": {
                            "images": all_images,          # 문자열 경로 리스트
                            "crops": crop_images           # 문자열 경로 리스트
                        },
                        # 혹시 모를 비직렬화 잔재 대비 습관적으로 sanitize
                        "history": _sanitize_for_json(messages)
                    }
                    break


                # 4. 다음 입력을 위한 observation 처리
                if isinstance(next_obs, str): # 에러 메시지나 텍스트 정보
                    messages.append({"role": "user", "content": next_obs})
                elif isinstance(next_obs, list) and len(next_obs) > 0:  # 검색 결과 (이미지 경로 리스트)
                    selected_image = next_obs[0]
                    all_images.append(selected_image)

                    # ⛔ 이미지 열지 않음, 텐서 생성하지 않음, vision token 만들지 않음
                    # ✅ 로그용으로 "경로 문자열"만 메시지에 남김
                    messages.append({
                        "role": "user",
                        "content": f"[retrieved_image_path]: {selected_image}"
                    })
# '''
#                     selected_image = next_obs[0]
#                     all_images.append(selected_image)

#                     # --- ① 이미지 로드 + 전처리 ---
#                     raw_img = process_image(selected_image)

#                     # --- ② tensor 변환 ---
#                     image_inputs = self.sft_model.processor.image_processor([raw_img], return_tensors="pt")
#                     image_grid_thw = image_inputs['image_grid_thw']

#                     # --- ③ vision 토큰 문자열 생성 ---
#                     merge_length = self.sft_model.processor.image_processor.merge_size**2
#                     vision_tokens = ''.join([
#                         f"<|vision_start|>{self.sft_model.processor.image_token * (grid.prod() // merge_length)}<|vision_end|>"
#                         for grid in image_grid_thw
#                     ])

#                     # --- ④ user role 메시지로 삽입 ---
#                     messages.append({"role": "user", "content": vision_tokens})
#                     messages.append({
#                         "role": "user",
#                         "content": [
#                             {"type": "image", "image": Image.open(selected_image).convert("RGB")},
#                             {"type": "text", "text": "Here is the retrieved image."}
#                         ]
#                     })
#                     # --- ⑤ tensor는 별도로 저장 (모델 forward 시 같이 전달하기 위함) ---
#                     if "multi_modal_inputs" not in sample:
#                         sample["multi_modal_inputs"] = []
#                     sample["multi_modal_inputs"].append(image_inputs)
# '''
# '''
#                 elif isinstance(next_obs, str) and next_obs.endswith(('.png', '.jpg', '.jpeg')):  # 크롭 결과
#                     crop_img = process_image(next_obs)
#                     image_inputs = self.sft_model.processor.image_processor([crop_img], return_tensors="pt")

#                     merge_length = self.sft_model.processor.image_processor.merge_size**2
#                     vision_tokens = ''.join([
#                         f"<|vision_start|>{self.sft_model.processor.image_token * (grid.prod() // merge_length)}<|vision_end|>"
#                         for grid in image_inputs['image_grid_thw']
#                     ])

#                     # user role 메시지 추가
#                     messages.append({"role": "user", "content": vision_tokens})

#                     # tensor 보관
#                     if "multi_modal_inputs" not in sample:
#                         sample["multi_modal_inputs"] = []
#                     sample["multi_modal_inputs"].append(image_inputs)

#                     # crop 이미지 경로도 기록
#                     crop_images.append(next_obs)
# '''

                elif isinstance(next_obs, str) and next_obs.endswith(('.png', '.jpg', '.jpeg')):  # 크롭 결과
                    # ⛔ 이미지 열지 않음, 텐서 생성하지 않음, vision token 만들지 않음
                    # ✅ 로그에 "경로 문자열"만 기록
                    messages.append({"role": "user", "content": f"[crop_image_path]: {next_obs}"})
                    crop_images.append(next_obs)
                

            if not final_result:
                status = "failed_timeout"

        except Exception as e:
            print(f"Processing failed for UID {sample.get('uid')}: {e}\n---")
            status = "failed_exception"
        
        finally:
    
            # --- 로깅은 여기서 딱 한 번만 수행 ---
            safe_messages = _sanitize_for_json(messages)

            with open(self.conversation_log_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "uid": sample.get('uid'), "status": status,
                    "raw_response_on_fail": raw_response if "fail" in status else None,
                    "conversation": safe_messages
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            
            if final_result:
                final_result['history'] = messages
                final_result['recall_results'] = dict(
                    source_nodes=[NodeWithScore(node=ImageNode(image_path=img), score=None).to_dict() for img in all_images],
                    response=None, metadata=None)
            
            return final_result


    # (eval_dataset 함수는 수정 없이 그대로 사용)
    def eval_dataset(self):
        eval_func = self.eval_func
        rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        with open(rag_dataset_path, "r") as f: data = json.load(f)
        data = data['examples']
        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r") as f:
                for line in f:
                    try: results.append(json.loads(line.strip()))
                    except: continue
            uid_already = set()
            for item in results:
                if isinstance(item, dict) and 'uid' in item: uid_already.add(item['uid'])
            data = [item for item in data if item.get('uid') not in uid_already]
        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None: continue
                with open(self.output_file_path, "a", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
        else:
            from threading import Lock
            write_lock = Lock()
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    try: result = future.result()
                    except Exception as e:
                        print(f"[WORKER ERROR] {e}")
                        continue
                    if result is None: continue
                    with write_lock:
                        with open(self.output_file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return self.output_file_path

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,default ='./model/global_step_444/global_step_444', required=True, help="Path to the local SFT model") #추가
    parser.add_argument("--dataset", type=str, default='example', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='./data/test_dataset/test_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='cot', help="The type of experiment")
    parser.add_argument("--workers_num", type=int, default=10, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    mmrag = MMRAG(
        model_path=args.model_path, #추가 필요
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        workers_num=args.workers_num,
        topk=args.topk
    )
    mmrag.eval_dataset()