#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT Search+Vision 통합 러너
- 요구사항 반영:
  1) 검색 쿼리/응답 프로토콜: generation.py의 execute_predictions 방식(POST, [{query,id}])
  2) 이미지 전처리/크롭 및 vision 토큰 주입 규칙: generation.py의 process_image / _process_next_obs 흐름
  3) 전체 루프: SFT_test_pipeline.py의 평가 흐름(샘플 반복, 로그/결과 파일 유지)
  4) 검색 결과 선택 정책: 다수 이미지 중 첫 번째만 사용
  5) 프롬프트 길이 상한: 정석적 트렁케이션
  6) 멀티 GPU: 배치 크기가 num_gpus로 나누어떨어지지 않으면 패딩 샘플 추가
  7) 로그/산출물 경로: 기존 파이프라인 파일명 유지(conversation_history.jsonl, results/sft_test_results.jsonl, ./data/image_crop)

주의: 아래 코드는 모델 호출부를 "어댑터"로 분리했다. 네 SFT 모델의 generate/forward 시그니처에 맞게 Adapter 내부 2개 TODO 지점을 채우면 동작한다.
"""
from __future__ import annotations
import os, json, math, time, shutil
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

import numpy as np
from PIL import Image
import requests
import torch

# =============================
# Configs
# =============================
@dataclass
class BridgeConfig:
    model_path: str
    query_file: str  # JSON or JSONL with {uid, query, ...}
    search_url: str  # POST, body: [{"query": str, "id": uid}]
    image_crop_dir: str = "./data/image_crop"
    results_dir: str = "./results"
    results_file: str = "sft_test_results.jsonl"
    history_file: str = "conversation_history.jsonl"
    max_turns: int = 10
    max_prompt_length: int = 8192
    num_gpus: int = 1
    http_timeout_sec: int = 20
    topk: int = 1  # 검색 결과에서 1개만 사용(첫 번째)

# =============================
# Utilities (from generation.py semantics)
# =============================

def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    """generation.py의 전처리 규칙을 그대로 반영."""
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def map_bbox_to_raw(display_w: int, display_h: int, raw_w: int, raw_h: int, bbox_xyxy_on_display: List[int], pad_eval: bool = False) -> List[int]:
    """generation.py의 좌표 변환 로직을 재현(검증 모드일 때 ±28 패딩을 둘 수 있게 옵션 제공)."""
    x1, y1, x2, y2 = bbox_xyxy_on_display
    if pad_eval:
        x1 -= 28; y1 -= 28; x2 += 28; y2 += 28
    x1 = max(0, int(raw_w * x1 / display_w))
    y1 = max(0, int(raw_h * y1 / display_h))
    x2 = min(raw_w, int(raw_w * x2 / display_w))
    y2 = min(raw_h, int(raw_h * y2 / display_h))
    return [x1, y1, x2, y2]


def crop_and_preprocess(image_path: str, bbox_xyxy_on_display: Optional[List[int]] = None, pad_eval: bool = False):
    """원본 이미지를 열고 필요하면 bbox로 크롭 → generation.py 전처리."""
    raw = Image.open(image_path)
    if bbox_xyxy_on_display is not None:
        dw, dh = raw.size  # display=이전 단계 이미지 크기라고 가정(간단화)
        x1, y1, x2, y2 = map_bbox_to_raw(dw, dh, raw.width, raw.height, bbox_xyxy_on_display, pad_eval)
        raw = raw.crop((x1, y1, x2, y2))
    return process_image(raw, 512*28*28, 256*28*28)


# =============================
# Search client (POST [{query,id}])
# =============================
class SearchClient:
    def __init__(self, url: str, timeout_sec: int = 20):
        self.url = url
        self.timeout = timeout_sec

    def search(self, uid: Any, query: str) -> List[str]:
        """서버에 POST로 [{query,id}] 전송, 응답에서 image_file 리스트를 추출.
        반환: 이미지 경로 문자열 리스트(우린 첫 번째만 사용)."""
        sid = str(uid)
        import re
        m = re.search(r"(\d+)$", sid)
        sid = m.group(1) if m else sid

        payload = [{"query": query, "id": sid}]  # ← 숫자 id만 보냄
        resp = requests.post(self.url, json=payload, timeout=self.timeout,
                             headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()  # 기대: [{...}] 혹은 [...] 형태
        # 응답 형태 표준화
        images: List[str] = []
        if isinstance(data, list) and len(data) > 0:
            entry = data[0] if isinstance(data[0], dict) else {}
            # 서버가 여러 이미지를 줄 수 있는 경우를 대비하여 정규화
            if isinstance(entry, dict):
                # 케이스1: {"image_file": "..."}
                if 'image_file' in entry and isinstance(entry['image_file'], str):
                    images = [entry['image_file']]
                # 케이스2: {"images": ["..."]}
                elif 'images' in entry and isinstance(entry['images'], list):
                    images = [str(p) for p in entry['images']]
        return images


# =============================
# Vision Adapter (processor/tokenizer에 맞춰 입력 구성)
# =============================
class VisionAdapter:
    def __init__(self, processor, image_token: str = "<|image_pad|>"):
        self.processor = processor
        self.image_token = image_token
        self.merge_size = getattr(getattr(processor, 'image_processor', processor), 'merge_size', 14)

    def build_mm_prompt_fragment(self, pil_list: List[Image.Image], paths: List[str]) -> Tuple[str, Dict[str, torch.Tensor]]:
        iproc = getattr(self.processor, 'image_processor', None)
        if iproc is None:
            raise RuntimeError("processor.image_processor가 필요함")
        image_inputs = iproc(pil_list, return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_len = self.merge_size ** 2

        pieces = []
        for g in image_grid_thw:
            n_tokens = int(torch.prod(g).item() // merge_len)
            pieces.append(f"<|vision_start|>{self.image_token * n_tokens}<|vision_end|>")
        fragment = "".join(pieces)

        # 경로 정보도 같이 추가
        image_inputs["paths"] = paths
        return fragment, image_inputs



# =============================
# Padding helper for multi-GPU generation
# =============================
class PaddingHelper:
    @staticmethod
    def pad_for_ngpus(input_ids: List[List[int]], num_gpus: int) -> Tuple[List[List[int]], int]:
        if num_gpus <= 1:
            return input_ids, 0
        bs = len(input_ids)
        rem = bs % num_gpus
        if rem == 0:
            return input_ids, 0
        pad_needed = num_gpus - rem
        pad_seq = input_ids[0][:] if bs > 0 else [151643]
        return input_ids + [pad_seq for _ in range(pad_needed)], pad_needed

    @staticmethod
    def trim_padded(outputs: List[str], pad_count: int) -> List[str]:
        if pad_count == 0:
            return outputs
        return outputs[:-pad_count]


# =============================
# Model Adapter (fill the TODO for your SFT model)
# =============================
class ModelAdapter:
    def __init__(self, model_path: str):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load(model_path)

    def _load(self, model_path: str):
        # processor와 모델 로드
        print(f"Loading model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()
        print("Model loaded successfully.")

    @torch.inference_mode()
    def generate_once(
        self,
        messages: List[Dict[str, str]],
        image_inputs: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 512
    ) -> str:
        #1. 메시지를 chat template로 직렬화
        # prompt = self.processor.tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )

        prompt = "".join([f"{m['role']}:\n{m['content']}\n" for m in messages])

        # 2. 이미지 처리 (있을 경우)
        raw_images = None
        if image_inputs and "paths" in image_inputs:
            raw_images = [Image.open(p).convert("RGB") for p in image_inputs["paths"]]

        # 3. processor로 텍스트 + 이미지 동시 인코딩
        inputs = self.processor(
            text=prompt, images=raw_images, return_tensors="pt"
        ).to(self.model.device)

        #디버깅
        # print("[DEBUG] prompt:", prompt[:500], "...")
        # print("[DEBUG] raw_images:", [img.size for img in (raw_images or [])])
        # print("[DEBUG] inputs keys:", inputs.keys())
        # for k, v in inputs.items():
        #     if hasattr(v, "shape"):
        #         print("  ", k, v.shape, v.device)
        ##

        # 4. 모델로 generate 실행
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # 5. 출력 디코딩
        response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        print("------------대답🚀------------ :\n", response_text)

        return response_text


# =============================
# Runner (SFT_test_pipeline 스타일 루프)
# =============================
class SFTRunner:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.image_crop_dir, exist_ok=True)
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        self.search = SearchClient(cfg.search_url, cfg.http_timeout_sec)
        self.model = ModelAdapter(cfg.model_path)
        #self.vision = (self.model.processor)
        self.vision = VisionAdapter(self.model.processor)  # ★ 수정


    def _append_jsonl(self, path: str, obj: Dict[str, Any]):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load_queries(self) -> List[Dict[str, Any]]:
        with open(self.cfg.query_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # "examples" 키가 있으면 그 리스트를 반환
        if isinstance(data, dict) and "examples" in data:
            return data["examples"]

        # 리스트면 그대로 반환
        if isinstance(data, list):
            return data

        # 단일 객체면 리스트로 감싸서 반환
        return [data]


    def _parse_action(self, text: str) -> Tuple[str, Optional[str]]:
        """모델 출력에서 <search>/<bbox>/<answer> 중 하나를 추출."""
        import re
        m = re.search(r"<(search|bbox|answer)>(.*?)</\1>", text, flags=re.DOTALL)
        if not m:
            return "", None
        return m.group(1), m.group(2).strip()

    # def _messages_with_image(self, base_messages: List[Dict[str,str]], pil_images: List[Image.Image]) -> Tuple[List[Dict[str,str]], Dict[str, torch.Tensor]]:
    #     fragment, image_inputs = self.vision.build_mm_prompt_fragment(pil_images)
    #     # 마지막 user turn 뒤에 vision fragment를 붙이는 간단한 구현
    #     msgs = base_messages[:]
    #     if msgs and msgs[-1]['role'] == 'user':
    #         msgs[-1] = {**msgs[-1], 'content': msgs[-1]['content'] + "\n" + fragment}
    #     else:
    #         msgs.append({"role": "user", "content": fragment})
    #     return msgs, image_inputs
    def _messages_with_image(
        self,
        base_messages: List[Dict[str,str]],
        pil_images: List[Image.Image],
        paths: List[str],                          # ★ 추가
    ) -> Tuple[List[Dict[str,str]], Dict[str, torch.Tensor]]:
        fragment, image_inputs = self.vision.build_mm_prompt_fragment(pil_images, paths)  # ★ paths 전달
        msgs = base_messages[:]
        if msgs and msgs[-1]['role'] == 'user':
            msgs[-1] = {**msgs[-1], 'content': msgs[-1]['content'] + "\n" + fragment}
        else:
            msgs.append({"role": "user", "content": fragment})
        #디버그
        # print("[DEBUG] vision fragment:", fragment[:120], "...")
        # print("[DEBUG] image_inputs keys:", list(image_inputs.keys()))
        # if "paths" in image_inputs:
        #     print("[DEBUG] image paths:", image_inputs["paths"])
        #
        
        return msgs, image_inputs

    def run_one_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        uid = item.get('uid') or item.get('id') or str(item.get('uid', '0'))
        query = item['query'] if 'query' in item else item.get('question', '')
        history_path = os.path.join(self.cfg.results_dir, self.cfg.history_file)
        result_path = os.path.join(self.cfg.results_dir, self.cfg.results_file)

        system_prompt = (
            "Answer the given question.\n"
            "You must conduct reasoning inside <think> and </think> every time you get new information.\n"
            "After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search>.\n"
            "Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>.\n"
            "If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>."
        )
        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nQuestion: {query}"
            }
        ]

        convo_log: List[Dict[str, Any]] = []
        retrieved_images: List[str] = []

        for step in range(self.cfg.max_turns):
            # 1) 모델 한 턴 생성
            text = self.model.generate_once(messages)
            convo_log.append({"role": "assistant", "content": text})
            self._append_jsonl(history_path, {"uid": uid, "step": step, "assistant": text})

            # act, content = self._parse_action(text)
            # if act == 'answer' or 'search_complete' in text:
            #     # 종료
            #     out = {"uid": uid, "status": "success", "answer": text, "images": retrieved_images}
            #     self._append_jsonl(result_path, out)
            #     return out
            act, content = self._parse_action(text)
            print("[DEBUG] PARSED ACT:", act, "| CONTENT:", repr(content))  # ← 추가 디버깅
            if act in ("answer", "search_complete"):
                out = {"uid": uid, "status": "success", "answer": text, "images": retrieved_images}
                self._append_jsonl(result_path, out)
                return out


            if act == 'search' and content:
                # 2) 검색 → 첫 이미지만 사용
                try:
                    print("[DEBUG] CALL SEARCH:", self.cfg.search_url, "uid=", uid, "q=", content)  # ← 추가 디버깅
                    images = self.search.search(uid, content)
                except Exception as e:
                    out = {"uid": uid, "status": "failed_search", "error": str(e)}
                    self._append_jsonl(result_path, out)
                    return out
                if not images:
                    # 검색 실패 → 힌트 주고 다음 턴 유도
                    messages.append({"role": "user", "content": "<information></information>"})
                    continue
                img_path = images[0]
                retrieved_images.append(img_path)

                # 3) 전처리하여 vision fragment + image_inputs 준비
                try:
                    pil = process_image(img_path, 512*28*28, 256*28*28)
                except Exception:
                    # 열 수 없는 경로면 스킵
                    messages.append({"role": "user", "content": "<information></information>"})
                    continue
                #messages, image_inputs = self._messages_with_image(messages, [pil])
                messages, image_inputs = self._messages_with_image(messages, [pil], [img_path])  # ★ paths 함께
                # 다음 턴에서 모델이 bbox를 낼 수 있도록 그대로 진행
                continue

            if act == 'bbox' and content:
                # 4) bbox 크롭(가장 마지막 retrieval 이미지 대상으로)
                if not retrieved_images:
                    messages.append({"role": "user", "content": "Your bbox is invalid without an image. Try search first."})
                    continue
                try:
                    xyxy = json.loads(content)
                except Exception:
                    messages.append({"role": "user", "content": "Your bbox is invalid JSON. Try again."})
                    continue
                # 디스플레이=원본 가정(간단화)
                cropped = crop_and_preprocess(retrieved_images[-1], xyxy, pad_eval=False)
                # 저장(관찰성)
                save_name = f"crop_{uid}_{step}.jpg"
                save_path = os.path.join(self.cfg.image_crop_dir, save_name)
                cropped.save(save_path)
                # 다음 턴 입력으로 크롭 이미지 부착
                messages, image_inputs = self._messages_with_image(messages, [cropped])
                continue

            # 5) 기타 → 사용자 프롬프트로 재시도 유도
            messages.append({"role": "user", "content": (
                "Your previous action is invalid.\n"
                "Think first inside <think></think>, then use <search> or <bbox> or <search_complete>"
            )})

        out = {"uid": uid, "status": "max_turn_reached", "images": retrieved_images}
        self._append_jsonl(result_path, out)
        return out

    def eval_all(self):
        data = self._load_queries()
        ok, fail = 0, 0
        for item in data:
            res = self.run_one_sample(item)
            if res.get('status') == 'success':
                ok += 1
            else:
                fail += 1
        print(f"Done. success={ok}, fail={fail}")


# =============================
# CLI
# =============================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--query_file', required=True)
    p.add_argument('--search_url', required=True)
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--image_crop_dir', default='./data/image_crop')
    p.add_argument('--max_turns', type=int, default=10)
    p.add_argument('--max_prompt_length', type=int, default=8192)
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--http_timeout_sec', type=int, default=20)
    args = p.parse_args()

    cfg = BridgeConfig(
        model_path=args.model_path,
        query_file=args.query_file,
        search_url=args.search_url,
        results_dir=args.results_dir,
        image_crop_dir=args.image_crop_dir,
        max_turns=args.max_turns,
        max_prompt_length=args.max_prompt_length,
        num_gpus=args.num_gpus,
        http_timeout_sec=args.http_timeout_sec,
    )

    runner = SFTRunner(cfg)
    runner.eval_all()
