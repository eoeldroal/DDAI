import os
import re
import requests
#import time qwen 72b 사용경우

DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""


class RemoteLLMEvaluator:
    """Client for calling a remote evaluation model via HTTP."""

    def __init__(self, model_name: str, api_key: str, base_url: str) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    # def batch_generate(self, prompts):
    #     outputs = []
    #     for prompt in prompts:
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": DEFAULT_SYSTEM_TEMPLATE.replace("{query}", prompt["query"]).replace("{reference_answer}", prompt["reference_answer"]).replace("{generated_answer}", prompt["generated_answer"]),
    #             }
    #         ]
    #         payload = {"model": self.model_name, "messages": messages}
    #         headers = {"Authorization": f"{self.api_key}"}
    #         url = f"{self.base_url}/api/v1/services/aigc/text-generation/generation"
    #         response = requests.post(url, json=payload, headers=headers, timeout=60)
    #         response.raise_for_status()
    #         content = response.json()["choices"][0]["message"]["content"]
    #         outputs.append(content)
    #     return outputs

    def batch_generate(self, prompts):
        outputs = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": DEFAULT_SYSTEM_TEMPLATE.replace("{query}", prompt["query"]).replace("{reference_answer}", prompt["reference_answer"]).replace("{generated_answer}", prompt["generated_answer"]),
                }
            ]
            
            # 1. Payload를 Dashscope의 공식 양식으로 수정
            payload = {
                "model": self.model_name,
                "input": {"messages": messages},
                "parameters": {"result_format": "message"}
            }

            headers = {"Authorization": f"{self.api_key}"}
            url = f"{self.base_url}/api/v1/services/aigc/text-generation/generation"
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            # 2. 응답 데이터 추출 방식을 Dashscope의 공식 양식에 맞게 수정
            content = response.json()["output"]["choices"][0]["message"]["content"]
            outputs.append(content)
            #time.sleep(1)  필요할 경우 qwen 72b 사용경우
        return outputs

    def eval_func(self, prompts):
        responses = self.batch_generate(prompts)
        eval_results = []
        for judge in responses:
            match = re.search(r"<judge>(.*?)</judge>", judge, re.DOTALL)
            if match:
                judge_str = match.group(1).strip().lower()
                eval_results.append(1.0 if "true" in judge_str else 0.0)
            else:
                eval_results.append(0.0)
        return eval_results