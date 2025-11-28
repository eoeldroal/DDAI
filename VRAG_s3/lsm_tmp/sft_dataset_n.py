#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9/15(월)
SFT 데이터셋 변환기 (v2)
- 입력: cot_crop.jsonl (각 줄: {"query": str, "history": list[dict], (옵션) images, uid, ...})
- 출력: results/train.parquet, results/val.parquet
- 목적: 모델이 태그 문법(<think>, <search>, <bbox>, <search_complete>)을 그대로 따르도록 response를 태그 텍스트로 학습.


#이거를 터미널에 붙여넣기 하면 됩니다. 
python sft_dataset_n.py \
  --input ./cot_crop.jsonl \
  --outdir ./results/sft_dataset \
  --val_ratio 0.1 \
  --seed 42 \
  --save parquet <- 다운 파일 형식 지정(par / json)


"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

PROMPT_TEMPLATE = (
    "You are a search agent.\n"
    "You must always begin with <think>...</think> showing your reasoning about the question.\n"
    "After reasoning, output exactly one action tag among <search>...</search> or <bbox>[x1, y1, x2, y2]</bbox> or <search_complete>true</search_complete>.\n"
    "Do not write anything before <think>. Keep actions on a new line after </think>.\n"
    "When using <search>, vary or refine the query using evidence from previous steps, and do not repeat the same query twice.\n"
)

def build_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """history를 대화 시퀀스로 변환. 크롭 힌트 text/description은 제거한다."""
    query = obj.get("query", "").strip()
    history = obj.get("history", [])

    # 크롭 툴 힌트 문장 패턴 (약간 느슨한 포함 매칭)
    CROP_HINT_SUBSTR = "You should call crop tool to crop this image"
    # (필요시 추가: 다른 변형 문구가 있다면 여기에 서브스트링을 더 append)

    messages: List[OrderedDict] = []
    # system
    messages.append(OrderedDict([("role", "system"), ("content", PROMPT_TEMPLATE)]))
    # first user question
    messages.append(OrderedDict([("role", "user"), ("content", query)]))

    for step in history:
        # assistant side (think/search/bbox/search_complete 등)
        if isinstance(step, dict):
            parts = []

            if "think" in step:
                parts.append(f"<think>{step['think']}</think>")

            if "search" in step:
                sval = step["search"]
                if not isinstance(sval, str):
                    sval = json.dumps(sval, ensure_ascii=False)
                parts.append(f"<search>{sval}</search>")

            if "bbox" in step:
                b = step["bbox"]
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in b]
                    parts.append(f"<bbox>[{x1}, {y1}, {x2}, {y2}]</bbox>")
                # ✅ 기존의 description 전파 로직은 제거한다 (무시)

            if "search_complete" in step:
                val = step["search_complete"]
                if isinstance(val, str):
                    val = val.strip().lower() in ("true", "1", "yes")
                parts.append(f"<search_complete>{str(bool(val)).lower()}</search_complete>")

            if parts:
                messages.append(OrderedDict([("role", "assistant"), ("content", "\n".join(parts))]))

            # history 내 query 키는 중복 질문이므로 무시
            # (description도 여기에 왔다면 더 이상 붙이지 않는다)
            continue

        # user image / search result 응답 (list payload)
        elif isinstance(step, list):
            filtered_list = []
            for item in step:
                # dict가 아니면 통과
                if not isinstance(item, dict):
                    filtered_list.append(item)
                    continue

                # 1) description 키가 있으면 제거
                if "description" in item:
                    continue

                # 2) text가 크롭 힌트 문구면 제거 (느슨 매칭)
                if "text" in item and isinstance(item["text"], str):
                    if CROP_HINT_SUBSTR.lower() in item["text"].lower():
                        continue

                # 그 외는 유지
                filtered_list.append(item)

            messages.append({"role": "user", "content": filtered_list})

    # OrderedDict -> dict로 변환(키 순서 유지)
    return [dict(m) for m in messages]




'''def build_messages(question: str, assistant_trajectory: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant_trajectory}
    ]

def render_history_to_text(history: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
    """history -> 태그 텍스트로 렌더. 순서 고정: think → search → bbox → search_complete → answer"""
    lines: List[str] = []
    stats = {k: 0 for k in ["think", "search", "bbox", "search_complete", "answer", "unknown"]}

    for step in history:
        if not isinstance(step, dict):
            stats["unknown"] += 1
            continue
        emitted = False

        # think
        if "think" in step and isinstance(step["think"], str):
            lines.append(f"<think>{step['think']}</think>")
            stats["think"] += 1
            emitted = True

        # search
        if "search" in step:
            sval = step["search"]
            if not isinstance(sval, str):
                try:
                    sval = json.dumps(sval, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    sval = str(sval)
            lines.append(f"<search>{sval}</search>")
            stats["search"] += 1
            emitted = True

        # bbox (여러 포맷 허용)
        if "bbox" in step:
            b = step["bbox"]
            coords = None
            try:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    coords = [int(round(float(v))) for v in b]
                elif isinstance(b, dict):
                    if "box_2d" in b and isinstance(b["box_2d"], (list, tuple)) and len(b["box_2d"]) == 4:
                        coords = [int(round(float(v))) for v in b["box_2d"]]
                    elif all(k in b for k in ("x1", "y1", "x2", "y2")):
                        coords = [int(round(float(b[k]))) for k in ("x1", "y1", "x2", "y2")]
                    elif all(k in b for k in ("x", "y", "w", "h")):
                        x1 = float(b["x"]); y1 = float(b["y"])
                        x2 = x1 + float(b["w"]); y2 = y1 + float(b["h"])
                        coords = [int(round(v)) for v in (x1, y1, x2, y2)]
                if coords is not None:
                    x1, y1, x2, y2 = coords
                    lines.append(f"<bbox>[{x1}, {y1}, {x2}, {y2}]</bbox>")
                    stats["bbox"] += 1
                    emitted = True
                else:
                    stats["unknown"] += 1
            except Exception:
                stats["unknown"] += 1

        # search_complete
        if "search_complete" in step:
            val = step["search_complete"]
            if isinstance(val, str):
                val = val.strip().lower() in ("true", "1", "yes")
            lines.append(f"<search_complete>{str(bool(val)).lower()}</search_complete>")
            stats["search_complete"] += 1
            emitted = True

        # answer
        if "answer" in step and isinstance(step["answer"], str):
            lines.append(f"Answer: {step['answer']}")
            stats["answer"] += 1
            emitted = True

        if not emitted:
            stats["unknown"] += 1

    return "\n".join(lines), stats
'''

@dataclass
class Row:
    uid: str
    format_version: str
    messages: List[Dict[str, str]]
    #response_json: str
    #images: str

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {i}: {e}", file=sys.stderr)
    return data

def build_rows(objs: List[Dict[str, Any]], escape_slashes: bool=False) -> Tuple[List[Row], Dict[str, Any]]:
    rows: List[Row] = []
    report = {"total": len(objs), "kept": 0, "skipped": 0,
              "stats_sum": {k: 0 for k in ["think", "search", "bbox", "search_complete", "answer", "unknown"]}}

    for idx, obj in enumerate(objs):
        query = obj.get("query")
        history = obj.get("history")
        if not isinstance(query, str) or not isinstance(history, list):
            report["skipped"] += 1
            continue

        # assistant content 만들기
        '''response_text, stats = render_history_to_text(history)
        for k, v in stats.items():
            report["stats_sum"][k] += int(v)'''

        #messages = build_messages(question=query, assistant_trajectory=response_text)
        messages = build_messages(obj)
        
        
        uid = obj.get("uid") or f"ex_{idx+1:06d}"
        response_json = json.dumps(history, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        if escape_slashes:
            response_json = response_json.replace("/", r"\/")

        images = obj.get("images", [])
        if not isinstance(images, list):
            images = []
        images_str = json.dumps(images, ensure_ascii=False, separators=(",", ":"))
        if escape_slashes:
            images_str = images_str.replace("/", r"\/")

        rows.append(Row(
            uid=uid,
            format_version="v1",
            messages=messages#,
            #response_json=response_json,
            #images=images_str
        ))
        report["kept"] += 1

    return rows, report

def split_rows(rows: List[Row], val_ratio: float, seed: int):
    n = len(rows)
    random.seed(seed)
    idxs = list(range(n))
    random.shuffle(idxs)
    v = max(1, int(round(n * float(val_ratio)))) if n > 1 else 0
    val_idx = set(idxs[:v])
    train, val = [], []
    for i, r in enumerate(rows):
        (val if i in val_idx else train).append(r)
    return train, val

def save_jsonl(rows: List[Row], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "uid": r.uid,
                "format_version": r.format_version,
                "messages": r.messages#,          # 배열 그대로
                #"response_json": r.response_json, # 문자열
                #"images": r.images               # 문자열
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return path

def save_parquet_or_csv(rows: List[Row], path: str, fmt: str) -> str:
    # Parquet/CSV는 리스트 컬럼 호환성 문제로 messages를 문자열에 담아 저장
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([{
        "uid": r.uid,
        "format_version": r.format_version,
        #"messages": r.messages
        "messages": json.dumps(r.messages, ensure_ascii=False, separators=(",", ":"))#,
        #"response_json": r.response_json,
        #"images": r.images,
    } for r in rows])
    if fmt == "parquet":
        df.to_parquet(path, engine="pyarrow", index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("fmt must be parquet or csv")
    return path

def save_dataset(rows: List[Row], path: str, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "jsonl":
        return save_jsonl(rows, path)
    elif fmt in ("parquet", "csv"):
        return save_parquet_or_csv(rows, path, fmt)
    elif fmt == "auto":
        try:
            return save_parquet_or_csv(rows, path, "parquet")
        except Exception as e:
            alt = os.path.splitext(path)[0] + ".jsonl"
            print(f"[WARN] parquet save failed -> fallback to JSONL: {e}")
            return save_jsonl(rows, alt)
    else:
        raise ValueError(f"Unknown save format: {fmt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default="jsonl", choices=["auto", "parquet", "jsonl", "csv"])
    ap.add_argument("--escape_slashes", action="store_true",
                    help="response_json/images 내 슬래시를 \\/ 로 이스케이프(호환 필요시 사용).")
    args = ap.parse_args()

    objs = load_jsonl(args.input)
    rows, report = build_rows(objs, escape_slashes=args.escape_slashes)
    
    
    
    
    print(f"[INFO] total={report['total']} kept={report['kept']} skipped={report['skipped']}")
    print(f"[INFO] stats={report['stats_sum']}")

    train, val = split_rows(rows, args.val_ratio, args.seed)
    print(f"[INFO] split -> train={len(train)}, val={len(val)}")

    os.makedirs(args.outdir, exist_ok=True)
    train_path = os.path.join(args.outdir, f"train.{args.save}")
    val_path   = os.path.join(args.outdir, f"val.{args.save}")
    if args.save == "jsonl":
        save_dataset(train, train_path, fmt="jsonl")
        save_dataset(val,   val_path,   fmt="jsonl")
    elif args.save in ("parquet", "csv", "auto"):
        save_dataset(train, train_path, fmt=args.save)
        save_dataset(val,   val_path,   fmt=args.save)
    print(f"[OK] saved -> {train_path}, {val_path}")
    
    
    
    # === DEBUG: parquet 저장 내용 확인 ===
    if args.save == "parquet":
        import pandas as pd
        print("\n[DEBUG] ==== Train parquet preview ====")
        df_train = pd.read_parquet(train_path)
        print(df_train.head(1).to_dict())
        print("\n[DEBUG] Train messages raw:")
        print(df_train.iloc[0]["messages"])

        print("\n[DEBUG] ==== Val parquet preview ====")
        df_val = pd.read_parquet(val_path)
        print(df_val.head(1).to_dict())
        print("\n[DEBUG] Val messages raw:")
        print(df_val.iloc[0]["messages"])

if __name__ == "__main__":
    main()
