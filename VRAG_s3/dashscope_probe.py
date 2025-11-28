# dashscope_probe.py
import os
import time
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from http import HTTPStatus



# ===== 환경변수 =====
# 필수: DASHSCOPE_API_KEY
# 선택: DASHSCOPE_BASE_URL, D_MODEL, TOTAL_REQS_PER_LEVEL, CONCURRENCY_LEVELS, MAX_TOKENS
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASH_SCOPE_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/api/v1")
MODEL = os.getenv("D_MODEL", "qwen2.5-vl-72b-instruct")  # 네가 쓰는 모델로 맞춰
TOTAL_REQS_PER_LEVEL = int(os.getenv("TOTAL_REQS_PER_LEVEL", "30"))  # 각 동시성 레벨에서 보낼 총 요청 수
CONCURRENCY_LEVELS = [int(x) for x in os.getenv("CONCURRENCY_LEVELS", "1,2,4,8,16,32").split(",")]
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "64"))
WARMUP = int(os.getenv("WARMUP", "3"))  # 각 레벨 시작 전에 워밍업 요청 수(소량)
RETRIES = int(os.getenv("RETRIES", "2"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.6"))
QPS_LIMIT = os.getenv("QPS_LIMIT")  # 예: "6" 지정 시 아주 느슨한 토큰버킷 (없으면 미사용)

assert API_KEY, "DASHSCOPE_API_KEY를 환경변수로 지정하세요."

# ===== SDK 로드 =====
import dashscope
dashscope.api_key = API_KEY
dashscope.base_http_api_url = BASE_URL
from dashscope import MultiModalConversation

def extract_text(resp):
    try:
        ot = getattr(resp, "output_text", None)
        if ot:
            return str(ot).strip()
    except Exception:
        pass

    out = getattr(resp, "output", None)
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or []
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("text") is not None:
                        texts.append(str(part["text"]))
                    elif part.get("type") == "text" and part.get("text") is not None:
                        texts.append(str(part["text"]))
            if texts:
                return "".join(texts).strip()
            if msg.get("text") is not None:
                return str(msg["text"]).strip()
        if out.get("text") is not None:
            return str(out["text"]).strip()
    return None

def call_once(qtext: str):
    """단건 호출 (텍스트만). 상태코드/지연/텍스트길이 반환"""
    sys_prompt = (
        "You are a visual QA generator. If no images are provided, "
        "answer concisely based on the question."
    )
    messages = [
        {"role": "system", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": f"Question: {qtext}"}]},
    ]
    t0 = time.perf_counter()
    try:
        try:
            resp = MultiModalConversation.call(
                model=MODEL, messages=messages, max_output_tokens=MAX_TOKENS
            )
        except TypeError:
            resp = MultiModalConversation.call(
                model=MODEL, messages=messages, max_tokens=MAX_TOKENS
            )
        dt = time.perf_counter() - t0
        code = getattr(resp, "status_code", None)
        txt = extract_text(resp) if code == HTTPStatus.OK else None
        return code, dt, (len(txt) if txt else 0)
    except Exception:
        return None, time.perf_counter() - t0, 0

# 느슨한 QPS 토큰버킷 (옵션)
_sema = None
_sleep_quantum = 0.0
if QPS_LIMIT:
    _qps = max(1, int(QPS_LIMIT))
    _sema = threading.BoundedSemaphore(_qps)
    _sleep_quantum = 1.0 / _qps

def respect_rate_limit():
    if not _sema:
        return
    _sema.acquire()
    # 초간단 pacing
    try:
        time.sleep(_sleep_quantum)
    finally:
        _sema.release()

def call_once_with_retry(q: str):
    delay = 0.0
    for a in range(RETRIES + 1):
        if delay > 0:
            time.sleep(delay)
        respect_rate_limit()
        code, dt, outlen = call_once(q)
        if code == HTTPStatus.OK and outlen >= 0:
            return code, dt, outlen, a  # a=재시도 횟수
        # 429/5xx에만 백오프
        if code in (429, 500, 502, 503, 504, None):
            delay = (BACKOFF_BASE ** a) + random.uniform(0, 0.2)
            continue
        else:
            # 4xx 등은 재시도 의미 적음
            return code, dt, outlen, a
    return code, dt, outlen, RETRIES

def run_level(concurrency: int, total_reqs: int):
    # 워밍업
    for _ in range(WARMUP):
        call_once_with_retry("warmup")

    results = []
    started = time.perf_counter()
    # 요청 개수를 concurrency 덩어리로 분할
    to_send = total_reqs
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = []
        for i in range(to_send):
            q = f"probe #{i} give me a one-word answer."
            futs.append(ex.submit(call_once_with_retry, q))
        for f in as_completed(futs):
            try:
                code, dt, outlen, retries = f.result()
            except Exception:
                code, dt, outlen, retries = None, 0.0, 0, RETRIES
            results.append((code, dt, outlen, retries))
    elapsed = time.perf_counter() - started

    # 집계
    codes = [r[0] for r in results]
    lat = [r[1] for r in results]
    outlens = [r[2] for r in results]
    retr = [r[3] for r in results]
    code_cnt = Counter(codes)
    succ = code_cnt.get(HTTPStatus.OK, 0)
    err = len(results) - succ

    report = {
        "concurrency": concurrency,
        "total": len(results),
        "success": succ,
        "error": err,
        "codes": {str(k): v for k, v in code_cnt.items()},
        "avg_latency_s": round(sum(lat) / max(1, len(lat)), 3),
        "p95_latency_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3) if lat else 0.0,
        "throughput_req_per_s": round(len(results) / elapsed, 2) if elapsed > 0 else 0.0,
        "avg_retries": round(sum(retr) / max(1, len(retr)), 3),
        "nonempty_text_ratio": round(sum(1 for L in outlens if L > 0) / max(1, len(outlens)), 3),
        "elapsed_s": round(elapsed, 2),
    }
    return report

def main():
    print(f"[probe] model={MODEL} base={BASE_URL} total_per_level={TOTAL_REQS_PER_LEVEL} max_tokens={MAX_TOKENS}")
    if QPS_LIMIT:
        print(f"[probe] soft QPS limit enabled: {QPS_LIMIT}")

    all_reports = []
    for c in CONCURRENCY_LEVELS:
        rep = run_level(c, TOTAL_REQS_PER_LEVEL)
        all_reports.append(rep)
        print("\n=== CONCURRENCY:", c, "===")
        for k, v in rep.items():
            if k == "codes":
                print("codes:", v)
            else:
                print(f"{k}: {v}")

    # 간단 추천치: 성공률 0.98+ & 429 거의 없음인 가장 큰 동시성
    best = None
    for r in all_reports:
        ok_ratio = r["success"] / max(1, r["total"])
        codes = r["codes"]
        too_many = sum(v for k, v in codes.items() if k in ("429",))
        if ok_ratio >= 0.98 and too_many <= 1:
            best = r
    if best:
        print("\n[recommendation] 안정 동시성 추정:", best["concurrency"])
    else:
        print("\n[recommendation] 조건을 만족하는 안정 구간을 찾지 못했어요. 더 낮은 동시성/요청수로 다시 측정해 보세요.")

if __name__ == "__main__":
    main()
















