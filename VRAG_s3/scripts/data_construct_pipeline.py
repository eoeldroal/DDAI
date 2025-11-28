import os
import json
from tqdm import tqdm
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.schema import NodeWithScore, ImageNode
import sys
import dashscope
from http import HTTPStatus
import time
from PIL import Image
import requests
from typing import List, Dict, Any, Optional
import zlib  # uid 해시 fallback용

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

image_output_dir = './data/image_crop'
raw_image_dir = './search_engine/corpus/img'
dashscope.base_http_api_url = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/api/v1"
)
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_KEY")
if not API_KEY:
    raise RuntimeError("Set DASHSCOPE_API_KEY (or DASH_SCOPE_KEY).")
dashscope.api_key = API_KEY


search_engine_url = "http://localhost:5002/search"

TOPK = 10

USER_PROMPT = '''You are a search agent.
You must conduct reasoning inside <think> and </think> every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. 
Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. 
You can search as many times as you want. 
If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. 
Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Question: {question}'''

def extract_json(response):
    response = response.replace("```json","").replace("```","")
    response = response.replace("```\n","").replace("\n```","")
    response_json = json.loads(response)
    return response_json

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

    if uid is None:
        return 0
    if isinstance(uid, int):
        return uid
    s = str(uid)
    m = re.search(r'(\d+)$', s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    # 숫자가 없으면 해시로 fallback (32-bit 양의 정수)
    return zlib.crc32(s.encode("utf-8")) & 0xffffffff


class Model_Role:
    def __init__(self, model_name: str, **kwargs):
        self.model = model_name
        self.kwargs = kwargs
        self._local = None  # (processor, model) 캐시

    def _ensure_local_loaded(self):
        if self._local is not None:
            return
        model_path = self.kwargs.get("local_model_path")
        assert model_path, "local_model_path가 필요하다"
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._local = (processor, model)

    @torch.inference_mode()
    def _local_generate_vlm(self, messages):
        # messages: [{"role":"system","content":...}, {"role":"user","content":[{"type":"text",...},{"type":"input_image",...}, ...]}]
        self._ensure_local_loaded()
        processor, model = self._local

        # 1) 텍스트 합치기
        sys_txt = ""
        usr_txt = ""
        images = []
        for m in messages:
            if m.get("role") == "system":
                sys_txt = m.get("content", "")
            elif m.get("role") == "user":
                parts = m.get("content", [])
                for p in parts:
                    if p.get("type") == "text":
                        usr_txt += p.get("text", "")
                    elif p.get("type") == "input_image":
                        # p["image"]가 경로(str)라고 가정
                        images.append(p.get("image"))

        prompt = (sys_txt + "\n" + usr_txt).strip()

        inputs = processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        out = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # 모델들에 따라 프롬프트가 함께 디코딩될 수 있으므로, 마지막 줄을 반환
        return out


class HTTPImageSearcher:
    """
    FastAPI 검색 서버(GET /search)를 호출해 이미지 리스트를 가져옵니다.
    - params: {"queries": <string>, "id": <int>}
    - 응답: [[{"idx": int, "image_file": "<path>"}, ...]]  # 배치 1개면 길이 1의 리스트
    """
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def query(self, query: str, query_id: str | int, topk: int = TOPK) -> List[str]:
        
        try:
            query_id = int(query_id)
        except Exception:
            query_id = _uid_to_query_id(query_id)

        try:
            resp = self.session.get(
                self.base_url,
                params={"queries": query, "id": query_id},
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

class Model_Role:
    def __init__(self,model_name):
        self.model = model_name
    def generate(self,messages):
        if self.model == 'qwen-max-latest':
            return self.generate_llm(messages)
        elif 'vl' in self.model:
            return self.generate_vlm(messages)
    def generate_llm(self,messages):
        dashscope.api_key=API_KEY
        time_left = 5
        while True:
            if time_left == 0:
                return None
            time_left -= 1
            try:
                response = dashscope.Generation.call(
                    model=self.model,
                    messages=messages,
                    result_format='message',
                )
                if response.status_code == HTTPStatus.OK:
                    return response['output']['choices'][0]['message']['content']
                else:
                    raise Exception(f"{response}")
            except Exception as e:
                print(e)
    def generate_vlm(self,messages):
        dashscope.api_key=API_KEY
        headers = {"X-DashScope-DataInspection": "disable"}
        times = 5
        while True:
            if times == 0:
                return None
            times -= 1
            try:
                response = dashscope.MultiModalConversation.call(model=self.model,
                                                                messages=messages,
                                                                headers=headers)
                if response.status_code == HTTPStatus.OK:
                    return response.output.choices[0].message.content[0]['text']
                else:
                    raise Exception(f"{response}")
            except Exception as e:
                print(e)

class MMRAG:
    def __init__(self,
                dataset='search_engine/corpus',
                query_file='rag_dataset.json',
                experiment_type = 'cot',
                workers_num = 1,
                topk=TOPK):
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
            self.output_file_name = f'cot_crop.jsonl'

        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/","-"))

        self.llm = Model_Role(model_name='qwen-max-latest')
        self.vlm = Model_Role(model_name='qwen-vl-max-latest')
        self.vlm_grounding = Model_Role(model_name='qwen2.5-vl-72b-instruct')

        self.searcher = HTTPImageSearcher(search_engine_url)

    def _search(self, search_query: str, query_id: str | int) -> List[str]:
        return self.searcher.query(search_query, query_id=query_id, topk=self.top_k)

    def cot_collect(self,sample):
        query = sample['query']
        reference_images = [f'{raw_image_dir}/'+sample['meta_info']['file_name'].replace('.pdf',f"_{i}.jpg") for i in sample['meta_info']['reference_page']]

        # reference_answer = sample.get('reference_answer')  # (미사용)
        all_images=[]
        crop_images=[]
        history=[{
            "query": query
        }]
        messages = []
        messages.append({
            "role": "system",
            "content": [
                {"text": prompt_inst}
            ]
        })
        messages.append({
            "role": "user",
            "content": [
                {"text": prompt_user_start.replace('{question}', query)}
            ]
        })
        try_times = 10
        grounding = False
        while True:
            try_times -=1
            if try_times < 0:
                return None
            while True:
                try:
                    if grounding:
                        response = self.vlm_grounding.generate(messages)
                    else:
                        response = self.vlm.generate(messages)
                    if response is None:
                        return None
                    print(response)
                    response_json = extract_json(response)
                    break
                except Exception as e:
                    time.sleep(1)
                    continue

            # 종료 조건: search_complete
            if 'search_complete' in response_json and isinstance(response_json['search_complete'], bool):
                history.append(response_json)
                if response_json['search_complete']:
                
                    sample['history'] = history
                    sample['collected'] = {
                        "images": all_images,
                        "crops": crop_images
                    }
                    # recall_results에는 "원본 이미지들만" 저장 (크롭 제외)
                    sample['recall_results'] = dict(
                        source_nodes=[NodeWithScore(
                            node=ImageNode(image_path=image,metadata=dict(file_name=image)), score=None
                        ).to_dict() for image in all_images],
                        response=None,
                        metadata=None)
                    return sample
                else:
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {"text": json.dumps(response_json)}
                        ]
                    })
                    continue

            # search 액션
            if 'think' in response_json and 'search' in response_json:
                search_query = response_json['search']

                
                query_id = _uid_to_query_id(sample.get('uid'))

                image_path_list = self._search(search_query, query_id=query_id)

                def select_element(A, B, C):
                    # A에서 B에 있고 C에는 없는 첫 항목, 없으면 C에 없는 첫 항목
                    for a in A:
                        if a in B and a not in C:
                            return a
                    for a in A:
                        if a not in C:
                            return a
                    return None

                image_input = select_element(image_path_list,reference_images,all_images)
                print(f"[SELECTED IMAGE] {image_input}")
                if image_input is None:
                    # 새 이미지가 없으면 다음 스텝
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {"text": json.dumps(response_json)}
                        ]
                    })
                    continue
                image_path_list = [image_input]

                # assistant
                history.append(response_json)
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"text": json.dumps(response_json)}
                    ]
                })
                # user
                images_content = [{'image': image_path} for image_path in image_path_list[:1]]
                images_content +=[{'text': "You should call crop tool to crop this image. The selected area must be complete and can be larger than the area that needs attention."}]
                messages.append({
                    "role": "user",
                    "content": images_content
                })
                history.append(images_content)
                all_images += image_path_list
                grounding = True
                continue

            # bbox 액션
            if 'think' in response_json and 'bbox' in response_json:
                bbox = response_json['bbox']
                if len(all_images) == 0:
                    continue
                croped_image_path = crop_and_dump(all_images[-1], bbox)
                if croped_image_path is None:
                    continue
                # assistant
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"text": json.dumps(response_json)}
                    ]
                })
                # user
                images_content = [{'image': croped_image_path}]
                messages.append({
                    "role": "user",
                    "content": images_content
                })

                history.append(response_json)
                history.append(images_content)
                crop_images.append(croped_image_path)
                grounding = False
                continue

            # 기타 포맷은 기록만
            messages.append({
                "role": "assistant",
                "content": [
                    {"text": json.dumps(response_json)}
                ]
            })

    def eval_dataset(self):
        eval_func = self.eval_func

        rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        with open(rag_dataset_path, "r") as f:
            data = json.load(f)
        data = data['examples']

        # 이미 저장된 uid는 건너뛰기(resume)
        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r") as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except:
                        continue
            uid_already = set()
            for item in results:
                if isinstance(item, dict) and 'uid' in item:
                    uid_already.add(item['uid'])
            data = [item for item in data if item.get('uid') not in uid_already]

        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None:
                    continue
                with open(self.output_file_path, "a", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
        else:
            # 병렬 처리 시에도 즉시 append하여 중간 저장 보장
            from threading import Lock
            write_lock = Lock()
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"[WORKER ERROR] {e}")
                        continue
                    if result is None:
                        continue
                    with write_lock:
                        with open(self.output_file_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return self.output_file_path

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='example', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='rag_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='cot', help="The type of experiment")
    parser.add_argument("--workers_num", type=int, default=10, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    mmrag = MMRAG(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        workers_num=args.workers_num,
        topk=args.topk
    )
    mmrag.eval_dataset()







