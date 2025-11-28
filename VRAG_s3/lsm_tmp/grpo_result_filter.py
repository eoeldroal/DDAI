import json
import collections

def analyze_grpo_logs(input_file_path, output_file_path):
    """
    GRPO 로그 JSON 파일을 분석하여 step별로 결과를 집계하고,
    그 결과를 'queries' 리스트만 한 줄로 압축하여 JSON 파일로 저장합니다.
    """
    
    print(f"'{input_file_path}' 파일 분석을 시작합니다...")
    
    all_results = {} 
    placeholders = {} # "자리표시자": "압축된 리스트 문자열" 을 저장할 딕셔너리

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("오류: JSON 파일의 최상위 구조가 딕셔너리(객체)가 아닙니다.")
            return

        step_counter = 1
        
        def get_step_number(step_name):
            try:
                num_str = step_name.split('[')[1].split(',')[0]
                return int(num_str)
            except:
                return step_name

        sorted_step_names = sorted(data.keys(), key=get_step_number)

        for step_name in sorted_step_names:
            step_key = f"step_{step_counter}"
            step_results = {}
            step_data = data[step_name]
            
            print(f"... step {step_counter} 분석 중 ...")
            
            raw_score_1_count = 0
            raw_score_0_count = 0
            
            group_1_queries = set()
            group_2_queries = set()
            group_3_queries = set()

            if not isinstance(step_data, dict):
                print(f"  [경고] '{step_name}'의 데이터 형식이 딕셔너리가 아닙니다. 건너뜁니다.")
                continue

            for query_uuid, query_data in step_data.items():
                
                if not isinstance(query_data, dict):
                    print(f"  [경고] UUID {query_uuid}의 데이터 형식이 딕셔너리가 아닙니다. 건너뜁니다.")
                    continue
                    
                agents = query_data.get("agents", [])
                
                if not isinstance(agents, list) or len(agents) != 5:
                    continue 

                all_raw_score_1 = True
                final_score_1_count = 0
                query_identifier = None
                has_valid_identifier = False

                try:
                    first_agent_scores = agents[0].get("scores", {})
                    if not isinstance(first_agent_scores, dict): continue
                    ndcg_details = first_agent_scores.get("ndcg_details", {})
                    if not isinstance(ndcg_details, dict): continue
                    ref_docs = ndcg_details.get("reference_documents", [])
                    
                    if ref_docs and isinstance(ref_docs, list) and len(ref_docs) > 0:
                        ref_doc_str = ref_docs[0]
                        if isinstance(ref_doc_str, str) and '_' in ref_doc_str:
                            query_identifier = ref_doc_str.split('_')[0]
                            if query_identifier.isdigit():
                                 has_valid_identifier = True
                except Exception:
                    pass

                for agent in agents:
                    scores = agent.get("scores", {})
                    if not isinstance(scores, dict): continue
                        
                    raw_score = scores.get("raw_score", 0)
                    final_score = scores.get("⭐️final_score⭐️", 0.0)

                    if raw_score == 1:
                        raw_score_1_count += 1
                    else:
                        raw_score_0_count += 1
                    
                    if raw_score != 1:
                        all_raw_score_1 = False
                    
                    if final_score == 1.0:
                        final_score_1_count += 1

                if all_raw_score_1 and has_valid_identifier:
                    if final_score_1_count == 5:
                        group_3_queries.add(query_identifier)
                    elif 2 <= final_score_1_count <= 4:
                        group_2_queries.add(query_identifier)
                    else:
                        group_1_queries.add(query_identifier)

            # --- 이 Step의 결과를 딕셔너리에 저장 ---
            
            step_results["raw_score_results"] = {
                "raw_score_1_count": raw_score_1_count,
                "raw_score_0_count": raw_score_0_count
            }
            
            # --- [수정된 부분] ---
            # 1. 쿼리 ID 리스트를 정렬
            sorted_group_3 = sorted(list(group_3_queries), key=int)
            sorted_group_2 = sorted(list(group_2_queries), key=int)
            sorted_group_1 = sorted(list(group_1_queries), key=int)

            # 2. 고유한 자리표시자(placeholder) 문자열 생성
            g3_placeholder = f"__STEP{step_counter}_G3_QUERIES__"
            g2_placeholder = f"__STEP{step_counter}_G2_QUERIES__"
            g1_placeholder = f"__STEP{step_counter}_G1_QUERIES__"

            # 3. 압축된(한 줄짜리) JSON 리스트 문자열을 placeholders 딕셔너리에 저장
            # (indent=None이 기본값이며, 한 줄로 만듭니다)
            placeholders[g3_placeholder] = json.dumps(sorted_group_3, ensure_ascii=False)
            placeholders[g2_placeholder] = json.dumps(sorted_group_2, ensure_ascii=False)
            placeholders[g1_placeholder] = json.dumps(sorted_group_1, ensure_ascii=False)

            # 4. 쿼리 분류 결과에 리스트 대신 자리표시자 문자열을 넣음
            step_results["query_classification"] = {
                "group_3": {
                    "description": "5개 에이전트 모두 final_score 1.0 달성",
                    "count": len(sorted_group_3),
                    "queries": g3_placeholder # 실제 리스트 대신 임시 문자열
                },
                "group_2": {
                    "description": "2개, 3개, 또는 4개 에이전트가 final_score 1.0 달성",
                    "count": len(sorted_group_2),
                    "queries": g2_placeholder
                },
                "group_1": {
                    "description": "0개 또는 1개 에이전트만 final_score 1.0 달성",
                    "count": len(sorted_group_1),
                    "queries": g1_placeholder
                }
            }
            
            all_results[step_key] = step_results
            step_counter += 1

        # --- [수정된 부분] ---
        # 1. 먼저 indent=4 를 적용하여 전체 JSON 문자열을 생성
        output_string = json.dumps(all_results, ensure_ascii=False, indent=4)
        
        # 2. 저장해둔 placeholders를 순회하며 문자열 교체
        for placeholder, compact_list_string in placeholders.items():
            # JSON 문자열 안에 있는 "자리표시자" (따옴표 포함)를
            # 압축된 리스트 문자열 (따옴표 없음)로 교체합니다.
            output_string = output_string.replace(f'"{placeholder}"', compact_list_string)

        # 3. 최종 수정된 문자열을 파일에 씁니다.
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(output_string)
            
        print(f"\n분석 완료! 결과가 '{output_file_path}' 파일에 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: '{input_file_path}' 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        print(f"오류: '{input_file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"분석 중 예기치 않은 오류가 발생했습니다: {e}")

# --- 스크립트 실행 ---
input_file = './lsm_tmp/grpo_log.json'
output_file = './lsm_tmp/analysis_results.json'

analyze_grpo_logs(input_file, output_file)