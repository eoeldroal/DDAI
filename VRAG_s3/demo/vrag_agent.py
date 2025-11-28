# import base64
# import json
# import re
# import requests
# import math
# from io import BytesIO
# import os
# import uuid
# import shutil

# from PIL import Image, ImageDraw

# # ▼▼▼ [수정] AutoModelForVision2Seq 추가 ▼▼▼
# import torch
# from transformers import AutoModelForVision2Seq, AutoTokenizer 
# from dotenv import load_dotenv
# from http import HTTPStatus
# try:
#     import dashscope
#     from dashscope import MultiModalConversation
#     _HAS_DASHSCOPE = True
# except ImportError:
#     _HAS_DASHSCOPE = False
# # ▲▲▲ [수정] ▲▲▲

# # ... (prompt_ins 등 나머지 부분은 동일) ...
# prompt_ins = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
# '''

# class VRAG:
#     def __init__(self,
#                 planner_model_path='/root/workspace/VRAG_test/VRAG_lsm/grpo_model/30_step_checkpoint',
#                 search_url='http://0.0.0.0:8002/search',
#                 generator=True):
        
#         # ... (API 설정 부분은 동일) ...
#         if not _HAS_DASHSCOPE:
#             raise ImportError("DashScope 라이브러리가 필요합니다. 'pip install \"dashscope[vl]\"'를 실행해주세요.")
        
#         dotenv_path = '/root/workspace/VRAG_test/.env'
#         load_dotenv(dotenv_path=dotenv_path)
        
#         dashscope.base_http_api_url = os.getenv("EVAL_BASE_URL")
#         api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("EVAL_API_KEY")
        
#         if not api_key:
#             raise ValueError(f"'{dotenv_path}' 파일에 DASHSCOPE_API_KEY 또는 EVAL_API_KEY를 설정해야 합니다.")
#         dashscope.api_key = api_key
        
#         self.answerer_model_name = os.getenv("EVAL_MODEL", "qwen-vl-max")
#         print(f"✅ '답변 모델'로 외부 API ({self.answerer_model_name})를 사용합니다.")

#         # 2. 검색 계획(Planner) 모델 로드 (지정한 로컬 모델)
#         print("🔍 로컬 '검색 계획 모델'을 로딩합니다...")
        
#         absolute_planner_path = os.path.abspath(planner_model_path)
#         print(f"모델의 절대 경로: {absolute_planner_path}")

#         self.planner_tokenizer = AutoTokenizer.from_pretrained(
#             absolute_planner_path,
#             trust_remote_code=True,
#             local_files_only=True
#         )

#         # ▼▼▼ [수정] AutoModelForCausalLM -> AutoModelForVision2Seq 로 변경 ▼▼▼
#         self.planner_model = AutoModelForVision2Seq.from_pretrained(
#             absolute_planner_path,
#             torch_dtype=torch.bfloat16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#             device_map='auto',
#             local_files_only=True
#         )
#         # ▲▲▲ [수정] ▲▲▲
        
#         print("✅ '검색 계획 모델' 로딩 완료.")

#         self.search_url = search_url
#         self.max_pixels = 512 * 28 * 28
#         self.min_pixels = 256 * 28 * 28
#         self.repeated_nums = 1
#         self.max_steps = 10
#         self.generator = generator
#     # ... (process_image, search, _generate_plan 함수 등 나머지 코드는 이전 답변과 동일합니다) ...
#     def process_image(self, image):
#         if isinstance(image, dict):
#             image = Image.open(BytesIO(image['bytes']))
#         elif isinstance(image, str):
#             image = Image.open(image)

#         if (image.width * image.height) > self.max_pixels:
#             resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
#             width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#             image = image.resize((width, height))

#         if (image.width * image.height) < self.min_pixels:
#             resize_factor = math.sqrt(min_pixels / (image.width * image.height))
#             width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#             image = image.resize((width, height))

#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         byte_stream = BytesIO()
#         image.save(byte_stream, format="JPEG")
#         byte_array = byte_stream.getvalue()
#         base64_encoded_image = base64.b64encode(byte_array)
#         base64_string = base64_encoded_image.decode("utf-8")
#         base64_qwen = f"data:image;base64,{base64_string}"

#         return image, base64_qwen

#     def search(self,query):
#         if isinstance(query,str):
#             query = [query]
#         search_response = requests.get(self.search_url, params={"queries": query})
#         search_results = search_response.json()
#         image_path_list = [result['image_file'] for result in search_results[0]]
#         return image_path_list

#     def _generate_plan(self, messages):
#         query = self.planner_tokenizer.from_list_format(messages)
#         inputs = self.planner_tokenizer([query], return_tensors='pt').to(self.planner_model.device)
#         gen_kwargs = {"max_length": 2048, "do_sample": False} 
#         with torch.no_grad():
#             outputs = self.planner_model.generate(**inputs, **gen_kwargs)
#             response_text = self.planner_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         last_response = response_text.split('<|im_start|>assistant\n')[-1]
#         return last_response.replace('<|im_end|>', '').strip()

#     def _generate_final_answer(self, original_question: str, collected_images: list):
#         print(f"✍️ 최종 답변 생성을 위해 외부 API '{self.answerer_model_name}'를 호출합니다...")

#         temp_dir = "temp_images_for_api"
#         os.makedirs(temp_dir, exist_ok=True)
#         image_paths = []
        
#         try:
#             for i, img in enumerate(collected_images):
#                 path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
#                 img.save(path)
#                 image_paths.append(path)
            
#             user_content = []
#             for path in image_paths:
#                 user_content.append({"image": "file://" + os.path.abspath(path)})
#             user_content.append({"text": original_question})

#             messages = [{
#                 "role": "user",
#                 "content": user_content
#             }]

#             response = MultiModalConversation.call(model=self.answerer_model_name, messages=messages)

#             if response.status_code == HTTPStatus.OK:
#                 content = response.output.choices[0].message.content[0]['text']
#                 raw_response = str(response)
#                 return 'answer', content.strip(), raw_response
#             else:
#                 error_msg = f"API Error: {response.code} - {response.message}"
#                 return 'answer', error_msg, str(response)

#         except Exception as e:
#             return 'answer', f"An exception occurred: {e}", ""
#         finally:
#             if os.path.exists(temp_dir):
#                 shutil.rmtree(temp_dir)
    
#     def run(self, question):
#         self.image_raw = []
#         self.image_input = []
#         self.image_path = []
#         prompt = prompt_ins.format(question=question)
#         messages = [dict(
#             role="user",
#             content=[
#                 {
#                     "type": "text",
#                     "text": prompt,
#                 }
#             ]
#         )]
        
#         max_steps = self.max_steps
#         while max_steps > 0:
#             response_content = self._generate_plan(messages)
#             messages.append(dict(
#                 role="assistant",
#                 content=[{ "type": "text", "text": response_content }]
#             ))
#             pattern = r'<think>(.*?)</think>'
#             match = re.search(pattern, response_content, re.DOTALL)
#             if not match:
#                 print("⚠️ <think> 태그를 찾을 수 없어 계획 단계를 종료합니다.")
#                 break
#             thought = match.group(1)
#             if self.generator:
#                 yield 'think', thought, match.group(0)
#             pattern = r'<(search|bbox)>(.*?)</\1>'
#             match = re.search(pattern, response_content, re.DOTALL)
#             if not match:
#                 print("✅ 검색/BBox 행동이 없어 정보 수집을 완료합니다.")
#                 break
#             raw_content = match.group(0)
#             content = match.group(2).strip()
#             action = match.group(1)
#             if self.generator:
#                 yield action, content, raw_content
            
#             user_content = []
#             if action == 'search':
#                 search_results = self.search(content)
#                 image_path = ""
#                 while len(search_results) > 0:
#                     temp_path = search_results.pop(0)
#                     if self.image_path.count(temp_path) < self.repeated_nums:
#                         self.image_path.append(temp_path)
#                         image_path = temp_path
#                         break
#                 if not image_path:
#                     user_content.append({"type": "text", "text": "Search returned no new images."})
#                 else:
#                     image_raw = Image.open(image_path)
#                     image_input, img_base64 = self.process_image(image_raw)
#                     user_content.append({ 'type': 'image_url', 'image_url': { 'url': img_base64 }})
#                     self.image_raw.append(image_raw)
#                     self.image_input.append(image_input)
#                     if self.generator:
#                         yield 'search_image', self.image_input[-1], raw_content
#             elif action == 'bbox':
#                 bbox = json.loads(content)
#                 input_w, input_h = self.image_input[-1].size
#                 raw_w, raw_h = self.image_raw[-1].size
#                 crop_region_bbox = bbox[0] * raw_w / input_w, bbox[1] * raw_h / input_h, bbox[2] * raw_w / input_w, bbox[3] * raw_h / input_h
#                 pad_size = 56
#                 crop_region_bbox = [max(crop_region_bbox[0]-pad_size,0), max(crop_region_bbox[1]-pad_size,0), min(crop_region_bbox[2]+pad_size,raw_h), min(crop_region_bbox[3]+pad_size,raw_h)]
#                 crop_region = self.image_raw[-1].crop(crop_region_bbox)
#                 image_input, img_base64 = self.process_image(crop_region)
#                 user_content.append({'type': 'image_url', 'image_url': { 'url': img_base64 }})
#                 self.image_raw.append(crop_region)
#                 self.image_input.append(image_input)
#                 if self.generator:
#                     image_to_draw = self.image_input[-2].copy()
#                     draw = ImageDraw.Draw(image_to_draw)
#                     draw.rectangle(bbox, outline=(160, 32, 240), width=7)
#                     yield 'crop_image', self.image_input[-1], image_to_draw
#             max_steps -= 1
#             messages.append(dict( role='user', content=user_content ))

#         action, content, raw_content = self._generate_final_answer(
#             original_question=question, 
#             collected_images=self.image_input
#         )
#         return action, content, raw_content

import base64
import json
import re
import requests
import math
from io import BytesIO
import os
import uuid
import shutil

from PIL import Image, ImageDraw

import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer 
from dotenv import load_dotenv
from http import HTTPStatus
try:
    import dashscope
    from dashscope import MultiModalConversation
    _HAS_DASHOPE = True
except ImportError:
    _HAS_DASHOPE = False

prompt_ins = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
'''

class VRAG:
    def __init__(self,
                planner_model_path='/root/workspace/VRAG_test/VRAG_lsm/grpo_model/30_step_checkpoint',
                search_url='http://0.0.0.0:8002/search',
                generator=True):
        
        # ⬅️ [변경] __init__에서 session_id 자동 생성을 제거하고, 인스턴스 변수만 선언
        self.session_id = None
        self.request_idx = 0
        print(f"✅ VRAG Agent Initialized. Session ID will be set at runtime.")
        
        if not _HAS_DASHOPE:
            raise ImportError("DashScope 라이브러리가 필요합니다. 'pip install \"dashscope[vl]\"'를 실행해주세요.")
        
        dotenv_path = '/root/workspace/VRAG_test/.env'
        load_dotenv(dotenv_path=dotenv_path)
        
        dashscope.base_http_api_url = os.getenv("EVAL_BASE_URL")
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("EVAL_API_KEY")
        
        if not api_key:
            raise ValueError(f"'{dotenv_path}' 파일에 DASHSCOPE_API_KEY 또는 EVAL_API_KEY를 설정해야 합니다.")
        dashscope.api_key = api_key
        
        self.answerer_model_name = os.getenv("EVAL_MODEL", "qwen-vl-max")
        print(f"✅ '답변 모델'로 외부 API ({self.answerer_model_name})를 사용합니다.")

        print("🔍 로컬 '검색 계획 모델'을 로딩합니다...")
        
        absolute_planner_path = os.path.abspath(planner_model_path)
        print(f"모델의 절대 경로: {absolute_planner_path}")

        self.planner_tokenizer = AutoTokenizer.from_pretrained(
            absolute_planner_path,
            trust_remote_code=True,
            local_files_only=True
        )

        self.planner_model = AutoModelForVision2Seq.from_pretrained(
            absolute_planner_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            local_files_only=True
        )
        
        print("✅ '검색 계획 모델' 로딩 완료.")

        self.search_url = search_url
        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.max_steps = 10
        self.generator = generator

    def process_image(self, image):
        # ... process_image 함수는 이전과 동일 ...
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{base64_string}"

        return image, base64_qwen

    def search(self, query: str):
        # ⬅️ [변경] run에서 설정된 self.session_id와 self.request_idx를 사용
        request_body = [{
            "query": query,
            "id": self.session_id,
            "request_idx": self.request_idx
        }]
        
        print(f"➡️ Search Request Body: {json.dumps(request_body, indent=2)}")

        try:
            response = requests.post(self.search_url, json=request_body)
            response.raise_for_status()
            
            search_results_list = response.json()
            print(f"⬅️ Search Response Body: {json.dumps(search_results_list, indent=2)}")
            
            result_for_this_request = next((item for item in search_results_list if item.get("request_idx") == self.request_idx), None)
            
            if result_for_this_request:
                results = result_for_this_request.get("results", [])
                image_path_list = [res.get("image_file") for res in results if "image_file" in res]
                return image_path_list
            else:
                print(f"⚠️ Warning: 응답에서 request_idx {self.request_idx}에 해당하는 결과를 찾을 수 없습니다.")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ Error during search request: {e}")
            return []
        except json.JSONDecodeError:
            print("❌ Error: 서버 응답이 유효한 JSON 형식이 아닙니다.")
            return []

    def _generate_plan(self, messages):
        # ... _generate_plan 함수는 이전과 동일 ...
        query = self.planner_tokenizer.from_list_format(messages)
        inputs = self.planner_tokenizer([query], return_tensors='pt').to(self.planner_model.device)
        gen_kwargs = {"max_length": 2048, "do_sample": False} 
        with torch.no_grad():
            outputs = self.planner_model.generate(**inputs, **gen_kwargs)
            response_text = self.planner_tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_response = response_text.split('<|im_start|>assistant\n')[-1]
        return last_response.replace('<|im_end|>', '').strip()
    
    def _generate_final_answer(self, original_question: str, collected_images: list):
        # ... _generate_final_answer 함수는 이전과 동일 ...
        print(f"✍️ 최종 답변 생성을 위해 외부 API '{self.answerer_model_name}'를 호출합니다...")

        temp_dir = "temp_images_for_api"
        os.makedirs(temp_dir, exist_ok=True)
        image_paths = []
        
        try:
            for i, img in enumerate(collected_images):
                path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
                img.save(path)
                image_paths.append(path)
            
            user_content = []
            for path in image_paths:
                user_content.append({"image": "file://" + os.path.abspath(path)})
            user_content.append({"text": original_question})

            messages = [{
                "role": "user",
                "content": user_content
            }]

            response = MultiModalConversation.call(model=self.answerer_model_name, messages=messages)

            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content[0]['text']
                raw_response = str(response)
                return 'answer', content.strip(), raw_response
            else:
                error_msg = f"API Error: {response.code} - {response.message}"
                return 'answer', error_msg, str(response)

        except Exception as e:
            return 'answer', f"An exception occurred: {e}", ""
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # ⬅️ [변경] run 메소드 시그니처 변경
    def run(self, question, session_id: str, request_idx: int):
        # run 메소드가 시작될 때 UI에서 받은 값으로 인스턴스 변수 설정
        self.session_id = session_id
        self.request_idx = request_idx

        self.image_raw = []
        self.image_input = []
        self.image_path = []
        prompt = prompt_ins.format(question=question)
        messages = [dict(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        )]
        
        max_steps = self.max_steps
        while max_steps > 0:
            response_content = self._generate_plan(messages)
            # ... 이하 run 메소드 로직은 이전과 동일 ...
            messages.append(dict(
                role="assistant",
                content=[{ "type": "text", "text": response_content }]
            ))
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, response_content, re.DOTALL)
            if not match:
                print("⚠️ <think> 태그를 찾을 수 없어 계획 단계를 종료합니다.")
                break
            thought = match.group(1)
            if self.generator:
                yield 'think', thought, match.group(0)
            pattern = r'<(search|bbox)>(.*?)</\1>'
            match = re.search(pattern, response_content, re.DOTALL)
            if not match:
                print("✅ 검색/BBox 행동이 없어 정보 수집을 완료합니다.")
                break
            raw_content = match.group(0)
            content = match.group(2).strip()
            action = match.group(1)
            if self.generator:
                yield action, content, raw_content
            
            user_content = []
            if action == 'search':
                search_results = self.search(content)
                image_path = ""
                while len(search_results) > 0:
                    temp_path = search_results.pop(0)
                    if self.image_path.count(temp_path) < self.repeated_nums:
                        self.image_path.append(temp_path)
                        image_path = temp_path
                        break
                if not image_path:
                    user_content.append({"type": "text", "text": "Search returned no new images."})
                else:
                    image_raw = Image.open(image_path)
                    image_input, img_base64 = self.process_image(image_raw)
                    user_content.append({ 'type': 'image_url', 'image_url': { 'url': img_base64 }})
                    self.image_raw.append(image_raw)
                    self.image_input.append(image_input)
                    if self.generator:
                        yield 'search_image', self.image_input[-1], raw_content
            elif action == 'bbox':
                bbox = json.loads(content)
                input_w, input_h = self.image_input[-1].size
                raw_w, raw_h = self.image_raw[-1].size
                crop_region_bbox = bbox[0] * raw_w / input_w, bbox[1] * raw_h / input_h, bbox[2] * raw_w / input_w, bbox[3] * raw_h / input_h
                pad_size = 56
                crop_region_bbox = [max(crop_region_bbox[0]-pad_size,0), max(crop_region_bbox[1]-pad_size,0), min(crop_region_bbox[2]+pad_size,raw_h), min(crop_region_bbox[3]+pad_size,raw_h)]
                crop_region = self.image_raw[-1].crop(crop_region_bbox)
                image_input, img_base64 = self.process_image(crop_region)
                user_content.append({'type': 'image_url', 'image_url': { 'url': img_base64 }})
                self.image_raw.append(crop_region)
                self.image_input.append(image_input)
                if self.generator:
                    image_to_draw = self.image_input[-2].copy()
                    draw = ImageDraw.Draw(image_to_draw)
                    draw.rectangle(bbox, outline=(160, 32, 240), width=7)
                    yield 'crop_image', self.image_input[-1], image_to_draw
            max_steps -= 1
            messages.append(dict( role='user', content=user_content ))

        action, content, raw_content = self._generate_final_answer(
            original_question=question, 
            collected_images=self.image_input
        )
        return action, content, raw_content