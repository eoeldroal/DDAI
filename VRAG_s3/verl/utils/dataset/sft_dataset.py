# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

"""
VRAG/VRAG에서
python -m verl.utils.dataset.sft_dataset   --input ./lsm_tmp/results/sft_dataset/train_10.parquet   --tokenizer Qwen/Qwen2.5-7B-Instruct
위 명령어 실행
"""

from typing import List, Union

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
import json

class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=2048,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys
        self.response_dict_keys = [] if not response_dict_keys else response_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    """def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.prompts={self.prompts}')
                raise
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.responses={self.responses}')
                raise
        self.responses = self.responses.tolist()"""
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # parquet 파일 읽기
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)

        # 여러 개 parquet이면 concat
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        # messages 컬럼 확인
        if "messages" not in self.dataframe.columns:
            raise ValueError(
                f"parquet 파일에 'messages' 컬럼이 없습니다. "
                f"현재 컬럼들: {list(self.dataframe.columns)}"
            )


    def __len__(self):
        #return len(self.prompts)
        return len(self.dataframe)

    """def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{'role': 'user', 'content': prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }"""
    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.dataframe.iloc[item]["messages"]

        # 문자열(JSON)일 경우 dict 리스트로 변환
        if isinstance(messages, str):
            messages = json.loads(messages)

        # content가 list인 경우 문자열 변환
        for m in messages:
            if isinstance(m.get("content"), list):
                m["content"] = json.dumps(m["content"], ensure_ascii=False)

        # === 전체 대화 텍스트 ===
        chat_str = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # padding / truncation
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif len(input_ids) > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:
                raise NotImplementedError

        position_ids = compute_position_id_with_mask(attention_mask)

        # === loss_mask (assistant만 1) ===
        loss_mask = torch.zeros_like(attention_mask)

        for msg in messages:
            if msg.get("role") == "assistant":
                # assistant content만 토큰화
                resp_ids = tokenizer(
                    msg["content"] + tokenizer.eos_token,
                    return_tensors="pt",
                    add_special_tokens=False
                )["input_ids"][0]

                # input_ids 안에서 resp_ids 위치 찾기 (substring search)
                for i in range(len(input_ids) - len(resp_ids) + 1):
                    if torch.equal(input_ids[i:i+len(resp_ids)], resp_ids):
                        loss_mask[i:i+len(resp_ids)-1] = 1  # 마지막 토큰 제외
                        break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask
        }    

    """def __getitem__(self, item):
        tok = self.tokenizer
        row = self.dataframe.iloc[item]
        messages = row["messages"]

        # parquet에 문자열(JSON)로 저장된 경우 파싱
        if isinstance(messages, str):
            messages = json.loads(messages)

        # content가 list면 JSON 문자열로 변환 (chat template는 문자열을 기대)
        for m in messages:
            if isinstance(m.get("content"), list):
                m["content"] = json.dumps(m["content"], ensure_ascii=False)

        # 전체 대화 문자열 (Qwen ChatML 가정)
        chat_str = tok.apply_chat_template(messages, tokenize=False)

        enc = tok(chat_str, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].squeeze(0)        # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ---- pad / truncation ----
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        L = input_ids.numel()
        if L < self.max_length:
            pad_len = self.max_length - L
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif L > self.max_length:
            if self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:
                raise RuntimeError(f"sequence_length={L} > max_length={self.max_length}")

        position_ids = compute_position_id_with_mask(attention_mask)

        # ---- loss_mask: <|im_start|>assistant ... <|im_end|> 본문만 1
        loss_mask = torch.zeros_like(attention_mask)

        # 토큰 헬퍼
        def toks(s: str) -> torch.Tensor:
            return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

        # 경계 토큰(개행 유무 모두 대응)
        start_assistant_a = toks("<|im_start|>assistant")
        start_assistant_b = toks("<|im_start|>assistant\n")
        end_tok = toks("<|im_end|>")

        # search_complete 태그 (우선 true 포함, 없으면 닫힘태그만이라도)
        sc_true = toks("<search_complete>true</search_complete>")
        sc_close = toks("</search_complete>")

        # 부분열 검색 (간단 선형)
        def find_subseq(hay: torch.Tensor, nee: torch.Tensor, st: int = 0) -> int:
            Lh, Ln = hay.numel(), nee.numel()
            if Ln == 0 or Lh < Ln:
                return -1
            for i in range(st, Lh - Ln + 1):
                if torch.equal(hay[i:i+Ln], nee):
                    return i
            return -1

        # 블록 찾기
        def find_blocks(ids: torch.Tensor, start_tok: torch.Tensor) -> list[tuple[int,int]]:
            blocks, cur = [], 0
            while True:
                s = find_subseq(ids, start_tok, cur)
                if s < 0:
                    break
                cs = s + start_tok.numel()           # 본문 시작
                e = find_subseq(ids, end_tok, cs)
                if e < 0:
                    break
                blocks.append((cs, e))               # [본문시작, end_tok 시작)
                cur = e + end_tok.numel()
            return blocks

        blocks = find_blocks(input_ids, start_assistant_b) + find_blocks(input_ids, start_assistant_a)
        blocks = sorted(blocks, key=lambda x: x[0])

        # 중복/겹침 제거(개행 유무로 두 번 잡힌 경우)
        merged = []
        for b in blocks:
            if not merged or b[0] >= merged[-1][1]:
                merged.append(b)
            # 겹치면 건너뜀

        # 각 블록에서 search_complete까지 마스킹
        for (cs, e) in merged:
            ce = min(e, input_ids.numel())
            rel = input_ids[cs:ce]

            pos = find_subseq(rel, sc_true, 0)
            if pos >= 0:
                cut_end = cs + pos + sc_true.numel()
            else:
                pos2 = find_subseq(rel, sc_close, 0)
                if pos2 >= 0:
                    cut_end = cs + pos2 + sc_close.numel()
                else:
                    cut_end = ce

            cut_end = min(cut_end, input_ids.numel())
            # 마지막 토큰(보통 EOS 성격)은 예측 제외
            if cut_end - cs > 1:
                loss_mask[cs:cut_end] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }"""





        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=2048)
    #parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()

    # === 데이터셋 로드 ===
    dataset = SFTDataset(
        parquet_files=args.input,
        tokenizer=args.tokenizer,
        max_length=args.max_len,
        truncation="right"
    )
    print(f"✅ parquet 로드 완료: {len(dataset)} samples")

    results = []

    # === 샘플 점검 & JSON 생성 ===
    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        tokenizer = dataset.tokenizer

        # loss_mask=1 인 위치
        assistant_ids = input_ids[loss_mask == 1]

        # 디코딩된 텍스트
        assistant_text = tokenizer.decode(assistant_ids)

        # 원래 parquet 안의 id 필드 사용
        # 만약 parquet에 'id'가 없다면 'uid' 등 다른 키 이름 확인 필요
        sample_id = item.get("uid", i)  # 'id'가 없으면 인덱스 사용

        results.append({
            "id": sample_id,
            "assistant_text": assistant_text
        })


    # === 원본 parquet 옆에 저장 경로 만들기 ===
    base, ext = os.path.splitext(args.input)
    output_file = base + "_assistant_only.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {output_file}")