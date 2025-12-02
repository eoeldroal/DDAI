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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
import requests
import numpy as np
import os
from time import sleep

def recall_ret(sorted_docs, golden_answer_list):
    """
    计算召回率（Recall）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: Recall 值
    """
    sorted_docs_set = set(sorted_docs)
    golden_answer_set = set(golden_answer_list)
    relevant_retrieved = len(sorted_docs_set.intersection(golden_answer_set))
    if len(golden_answer_set) == 0:
        return 0.0
    recall_value = relevant_retrieved / len(golden_answer_set)
    return recall_value

def dcg(relevance_scores):
    """
    计算折扣累积增益（DCG）
    :param relevance_scores: 一个列表，表示每个文档的相关性分数
    :return: DCG 值
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value

def ndcg(sorted_docs, golden_answer_list):
    """
    计算归一化折扣累积增益（NDCG）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: NDCG 值
    """
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    dcg_value = dcg(relevance_scores)
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    if idcg_value == 0:
        return 0.0
    ndcg_value = dcg_value / idcg_value
    return ndcg_value

def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'
    end_pos = text.rfind(end_tag)
    if end_pos == -1: return None
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1: return None
    start_pos += len(start_tag)
    return text[start_pos:end_pos]


class RMEvalManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, rm_url="http://0.0.0.0:8003/eval") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.rm_url = rm_url

    def verify(self, data):
        # (이 함수는 현재 학습 흐름에서 직접 사용되지 않으므로 수정하지 않아도 무방합니다.)
        scores = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        recall_list = []
        ndcg_list = []

        already_print_data_sources = {}

        data_eval = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                generated_answer = 'Please Judge False'
            data_eval.append(dict(
                query = extra_info['question'],
                generated_answer = generated_answer,
                reference_answer = data_item.non_tensor_batch['reward_model']['ground_truth']
            ))

        data_to_be_eval = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if score >0.0:
                data_to_be_eval.append(data_eval[i])

        if len(data_to_be_eval) > 0:
            bs = 200
            while True:
                try:
                    request_data_to_be_eval = dict(
                        bs=bs,
                        prompts=data_to_be_eval
                    )
                    #prompts_json = json.dumps(request_data_to_be_eval)
                    print("=====================eval model start=====================")
                    #response = requests.post(self.rm_url, json=prompts_json) #외부 api 출력 수정 제거
                    response = requests.post(self.rm_url, json=request_data_to_be_eval) #외부 api 출력 추가

                    response.raise_for_status() # [수정] 응답 상태 코드가 200이 아니면 에러를 발생시킵니다.
                    eval_results = response.json()
                    print("=====================eval model end=====================")
                    eval_results_iter = iter(eval_results) # ✨ API 결과를 이터레이터로 변환 수정 index error 해결
                    break
                except requests.exceptions.RequestException as e: # [수정] 더 구체적인 예외를 잡습니다.
                    print(f"=====================eval model error: {e}=====================")
                bs = max(int(bs/2),5)
                sleep(5)
        
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            #############수정 (주석 처리) ########
            #이유: NotImplementedError를 유발하는 내부 점수 계산을 비활성화합니다.
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            #############수정 완료 (주석 처리) #########

            # ###############수정 (삽입) ###########
            # # 이유: 내부 점수 대신 API 결과와 NDCG, Recall 점수만으로 최종 점수를 계산합니다.
            # score = eval_results[i] if i < len(eval_results) else 0.0
            
            if score > 0.0: # [주석 처리] 내부 점수 필터링을 제거합니다.
            #try:
                #score = eval_results.pop(0) #format_score
                score = next(eval_results_iter) # ✨ 필요할 때마다 순서대로 안전하게 하나씩 꺼내 사용 수정            
                retrievaled_images_basename_list = [os.path.basename(item.rstrip('/')).split(".jpg")[0] for item in data_item.non_tensor_batch['retrievaled_images']]
                reference_images_basename_list = [f'{extra_info["file_name"].split(".pdf")[0]}_{page}' for page in extra_info["reference_page"].tolist()]
                recall_list.append(recall_ret(retrievaled_images_basename_list, reference_images_basename_list))
                ndcg_list.append(ndcg(retrievaled_images_basename_list, reference_images_basename_list))
            # except Exception as e:
            #     # RAG 관련 데이터가 아닐 경우 에러가 날 수 있으므로, 기본값을 추가하고 넘어갑니다.
            #     recall_list.append(0.0)
            #     ndcg_list.append(0.0)
            # #################수정 완료 (삽입) ###############

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor, ndcg_list, recall_list

