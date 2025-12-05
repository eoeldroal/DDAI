import re
import json

def simple_format_checker(data_source, solution_str, ground_truth, extra_info):
    """
    Assistant의 전체 대화 기록(solution_str)을 검사하여,
    모든 턴이 정해진 문법 규칙을 따랐는지 확인하고, 실패 원인을 출력 및 반환합니다.
    Returns:
        score (float): 1.0 (Pass) or 0.0 (Fail)
        reason (str): 실패 원인 설명 (성공 시 None)
    """

    # 1단계: Assistant의 턴(Turn)만 분리하기
    assistant_turns = re.findall(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", solution_str, re.DOTALL)

    if not assistant_turns:
        msg = f"DEBUG: No assistant turns found. Input: {solution_str}"
        print(msg)
        return 0.0, "No assistant turns found"

    # 2단계: 모든 턴에 대한 '공통 문법' 검사
    for i, turn in enumerate(assistant_turns):
        cleaned_turn = turn.strip()

        # 2-A: <think> 태그 검사
        if not cleaned_turn.startswith('<think>'):
            msg = f"DEBUG: Turn {i} does not start with <think>. Content: {cleaned_turn[:50]}..."
            #print(msg)
            return 0.0, f"Turn {i} missing <think> start tag.{cleaned_turn[:50]}..."
            
        if cleaned_turn.count('<think>') != 1 or cleaned_turn.count('</think>') != 1:
            msg = f"DEBUG: Turn {i} has incorrect <think> tag count."
            #print(msg)
            return 0.0, f"Turn {i} incorrect <think> tag count. {cleaned_turn[:50]}..."

        # 2-B: 행동(Action) 태그 개수 검사
        action_count = cleaned_turn.count('<search>') + cleaned_turn.count('<bbox>') + cleaned_turn.count('<search_complete>')
        if action_count != 1:
            msg = f"DEBUG: Turn {i} has invalid action count: {action_count}"
            #print(msg)
            return 0.0, f"Turn {i} invalid action count ({action_count})"

        # 2-C: 행동(Action) 태그 내용 검사 (세부 규칙)
        if '<search>' in cleaned_turn:
            match = re.search(r"<search>(.*?)</search>", cleaned_turn, re.DOTALL)
            if not match or not match.group(1).strip():
                msg = f"DEBUG: Turn {i} has empty or malformed <search>."
                print(msg)
                return 0.0, f"Turn {i} empty/malformed <search>"

        elif '<bbox>' in cleaned_turn:
            match = re.search(r"<bbox>(.*?)</bbox>", cleaned_turn, re.DOTALL)
            if not match:
                msg = f"DEBUG: Turn {i} has malformed <bbox>."
                #print(msg)
                return 0.0, f"Turn {i} malformed <bbox>"
            try:
                bbox_content = json.loads(match.group(1).strip())
                if not isinstance(bbox_content, list) or len(bbox_content) != 4:
                    msg = f"DEBUG: Turn {i} bbox is not a list of 4. Got: {bbox_content}"
                    #print(msg)
                    return 0.0, f"Turn {i} bbox format error (not length 4)"
                if not all(isinstance(coord, (int, float)) for coord in bbox_content):
                    msg = f"DEBUG: Turn {i} bbox contains non-numbers. Got: {bbox_content}"
                    #print(msg)
                    return 0.0, f"Turn {i} bbox non-number values"
            except json.JSONDecodeError:
                msg = f"DEBUG: Turn {i} bbox JSON decode error."
                #print(msg)
                return 0.0, f"Turn {i} bbox JSON decode error"

        elif '<search_complete>' in cleaned_turn:
            if '<search_complete>true</search_complete>' not in cleaned_turn.replace(" ", ""):
                msg = f"DEBUG: Turn {i} <search_complete> is not 'true'."
                #print(msg)
                return 0.0, f"Turn {i} <search_complete> value error"

    # 3단계: '마지막 턴' 특별 규칙 검사
    last_turn = assistant_turns[-1].strip()
    if '<search_complete>' not in last_turn:
        msg = f"DEBUG: Last turn does not have <search_complete>. Last turn content: {last_turn[-50:]}"
        print(msg)
        return 0.0, "Last turn missing <search_complete>"

    # 4단계: 최종 합격 판정
    return 1.0, None