import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
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

class LLMGenerator:
    def __init__(self, model_name):
        """
        初始化模型和分词器
        :param model_name: 模型名称或路径
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="balanced"
        )

    def batch_generate(self, prompts):
        messages_batch = []
        for prompt in prompts:
            messages = [
                {"role": "user", "content": DEFAULT_SYSTEM_TEMPLATE.replace("{query}", prompt["query"]).replace("{reference_answer}", prompt["reference_answer"]).replace("{generated_answer}", prompt["generated_answer"])}
            ]
            messages_batch.append(messages)

        text = self.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer(text, return_tensors="pt",padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )

        generated_ids_batch = generated_ids[:, model_inputs.input_ids.shape[1]:].tolist()
        
        if 'Qwen3' in self.model_name:
            generated_ids_batch_trimmed = []
            for output_ids in generated_ids_batch:
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                generated_ids_batch_trimmed.append(output_ids[index:])
            output_response = self.tokenizer.batch_decode(generated_ids_batch_trimmed, skip_special_tokens=True)
        else:
            output_response = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
        return output_response
    
    def eval_func(self, prompts):
        response = self.batch_generate(prompts)
        print(response)
        eval_results = []
        for judge in response:
            # judge = judge.replace('```', '').replace('```\n', '').replace('```json', '')
            # pattern = r'\{.*?\}'
            pattern = r'<judge>(.*?)</judge>'
            match = re.search(pattern, judge, re.DOTALL)
            if match:
                try:
                    judge_str = match.group(0)
                    if 'true' in judge_str or 'True' in judge_str:
                        eval_results.append(1.0)
                    else:
                        eval_results.append(0.0)
                except Exception as e:
                    eval_results.append(0.0)
            else:
                eval_results.append(0.0)
        return eval_results