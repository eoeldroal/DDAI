from fastapi import FastAPI, Request
import uvicorn

#외부 api 수정(제거)
# from typing import List, Dict, Any  
# from model_eval import LLMGenerator
###################
from dotenv import load_dotenv

from tqdm import tqdm
import os

#import json #외부 api 수정(제거)
#외부 api 수정(추가)
from .external_client import RemoteLLMEvaluator

app = FastAPI()
model_eval = None
load_dotenv() #외부 api 수정 추가

@app.on_event("startup")
async def startup_event():
    global model_eval
    #외부 api 수정(제거)
    # #model_name = "Qwen/Qwen2.5-72B-Instruct"
    # model_name = "Qwen/Qwen2.5-7B-Instruct" #수정
    # model_eval = LLMGenerator(model_name)
    #외부 api 수정(추가)
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get("EVAL_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("EVAL_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    model_eval = RemoteLLMEvaluator(model_name=model_name, api_key=api_key, base_url=base_url)    

@app.post("/eval")
async def eval(request: Request):
    prompts = await request.json()
    # prompts = json.loads(prompts) #외부 apu 수정 제거
    bs = int(prompts["bs"])
    data_eval = prompts["prompts"]

    eval_results = []
    for i in tqdm(range(0, len(data_eval), bs)):
        eval_results.extend(model_eval.eval_func(data_eval[i : i + bs]))
    return eval_results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)