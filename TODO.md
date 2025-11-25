# HPT 코드베이스 이해 계획

> **목표**: HPT (Hybrid Post-Training) 코드베이스를 단계적으로 이해하고 수정/확장 가능한 수준에 도달하기

**마지막 업데이트**: 2025-11-25

---

## 📚 Phase 1: 기초 개념 이해

### 1.1 논문 및 배경 이론 학습
- [ ] HPT 논문 정독 ([arXiv:2509.04419](https://arxiv.org/abs/2509.04419))
  - [ ] Introduction 및 Motivation 이해
  - [ ] Unified Policy Gradient Estimator 수식 이해
  - [ ] Switch vs Soft 전략 차이점 파악
- [ ] 관련 개념 복습
  - [ ] PPO (Proximal Policy Optimization) 알고리즘
  - [ ] GRPO (Group Relative Policy Optimization)
  - [ ] Advantage Estimation (GAE, REINFORCE)
  - [ ] On-policy vs Off-policy 강화학습
- [ ] 참고 자료
  - [ ] 원본 HPT GitHub 저장소 확인: [TsinghuaC3I/Unify-Post-Training](https://github.com/TsinghuaC3I/Unify-Post-Training)

### 1.2 프로젝트 구조 파악
- [ ] README.md 읽기 (프로젝트 개요)
- [ ] Agent.md 정독 (상세 기술 문서)
  - [ ] Section 1-10: 전체 아키텍처
  - [ ] Section 11-15: 설정 및 실행
  - [ ] Section 16-24: 수정 가이드 및 참조
- [ ] 디렉토리 구조 탐색
  - [ ] `exp_scripts/`: 실행 스크립트
  - [ ] `hpt/verl/verl/mix_src/`: 핵심 구현
  - [ ] 누락 파일 목록 확인 (VRAG agent, dataset 등)

---

## 🔍 Phase 2: 코드 세부 분석

### 2.1 메인 트레이너 분석 (`mix_trainer.py`)

#### 2.1.1 클래스 구조 이해
- [ ] `MIXRayPPOTrainer` 클래스 전체 구조 파악 (205-1577줄)
  - [ ] 상속 관계: `RayPPOTrainer` 확인
  - [ ] 주요 멤버 변수 목록 작성
  - [ ] 주요 메서드 목록 작성

#### 2.1.2 초기화 과정 분석
- [ ] `__init__` 메서드 (212-256줄)
  - [ ] KL 컨트롤러 설정 로직
  - [ ] Hybrid engine 설정
  - [ ] 데이터로더 생성 과정
- [ ] `init_workers` 메서드 (258-336줄)
  - [ ] Ray 워커 그룹 생성 과정
  - [ ] Actor, Critic, RefPolicy 초기화
  - [ ] Resource pool 관리 방식
- [ ] `_create_dataloader` 메서드 (338-390줄)
  - [ ] `RLHFDatasetWithTarget` 사용법
  - [ ] Batch size 설정
  - [ ] Shuffle 및 Sampler 설정

#### 2.1.3 훈련 루프 상세 분석
- [ ] **Phase 1: 데이터 준비** (510-530줄)
  - [ ] UID 매핑 로직 이해
  - [ ] `index` → `id` / `uid` 변환 과정
  - [ ] DataProto 구조 파악

- [ ] **Phase 2: 생성** (547-590줄)
  - [ ] Switch 모드: VRAG 멀티턴 생성 (573-588줄)
  - [ ] 일반 모드: 표준 생성
  - [ ] `generation_manager.run_llm_loop()` 동작 방식
  - [ ] `actor_rollout_wg.generate_sequences()` 동작 방식

- [ ] **Phase 3: 보상 계산** (612-708줄)
  - [ ] `reward_fn()` 호출 및 결과 처리
  - [ ] `reward_impl_version=7` (RAG) vs 기본 버전 차이
  - [ ] UID별 성공 집계 로직 (668-692줄)
  - [ ] `solve_none`, `solve_all` 플래그 의미

- [ ] **Phase 4: 데이터 균형 조정** ⭐ **HPT 핵심** (711-1110줄)
  - [ ] `select_on_off_ada_balance()` 메서드 상세 분석 (405-426줄)
    - [ ] Switch 전략 로직 (406-418줄)
    - [ ] Soft 전략 로직 (420-426줄)
  - [ ] On-policy 데이터 제거 로직 (754-776줄)
  - [ ] On-policy 데이터 추가 로직 (778-845줄)
  - [ ] Off-policy 데이터 추가 로직 (882-1032줄)
  - [ ] `prefix_mask` 생성 및 활용 방법
  - [ ] UID별 데이터 처리 루프 이해

- [ ] **Phase 5: Advantage 계산** (1265-1296줄)
  - [ ] `compute_advantage()` 함수 (117-203줄)
  - [ ] GRPO 구현 상세
  - [ ] prefix_mask 가중치 적용 (1274-1281줄)
  - [ ] `alpha_weight`, `beta_weight` 계산

- [ ] **Phase 6: 모델 업데이트** (1345-1358줄)
  - [ ] `actor_rollout_wg.update_actor()` 호출
  - [ ] Batch 메타데이터 전달 내용 확인
  - [ ] Critic 업데이트 (GAE 사용 시)

#### 2.1.4 보조 함수 분석
- [ ] `apply_kl_penalty()` (86-115줄)
- [ ] `compute_advantage()` (117-203줄)
  - [ ] GAE, GRPO, REINFORCE 각 구현
- [ ] `select_on_off_ada_balance()` (405-426줄)
- [ ] Memory management 함수들 (70-82줄)

### 2.2 설정 파일 분석 (`debug.sh`)

#### 2.2.1 주요 파라미터 그룹별 이해
- [ ] **HPT 전략 파라미터**
  - [ ] `trainer.unify_strategy` 옵션들
  - [ ] `trainer.switch_gate` 의미 및 조정 방법
  - [ ] `trainer.switch_gate_off` 역할
  - [ ] `trainer.remove_sfted_data` 효과

- [ ] **손실 함수 파라미터**
  - [ ] `actor_rollout_ref.actor.offline_loss_type`
  - [ ] `actor_rollout_ref.actor.sft_loss_coef`
  - [ ] `actor_rollout_ref.actor.off_policy_loss_impl`
  - [ ] KL 관련 설정들 (모두 0으로 비활성화)

- [ ] **알고리즘 파라미터**
  - [ ] `algorithm.adv_estimator=grpo`
  - [ ] `algorithm.grpo_use_std=False`
  - [ ] `algorithm.kl_ctrl.*` 설정

- [ ] **데이터 파라미터**
  - [ ] `data.train_files`, `data.val_files`
  - [ ] `data.reward_impl_version=7`
  - [ ] Batch size 관련 설정들
  - [ ] Sequence length 제한들

- [ ] **생성 파라미터**
  - [ ] `actor_rollout_ref.rollout.n`, `n_verify`, `n_val`
  - [ ] Temperature, top_p 설정
  - [ ] VRAG 관련: `max_turns`, `search_url`

- [ ] **메모리 최적화 파라미터**
  - [ ] FSDP offloading 옵션들
  - [ ] `gpu_memory_utilization`
  - [ ] Gradient checkpointing

- [ ] **분산 훈련 파라미터**
  - [ ] `trainer.n_gpus_per_node`, `trainer.nnodes`
  - [ ] Parallelism 설정들

### 2.3 누락 파일 이해 (코드 요청 필요)

- [ ] `rl_dataset_with_target.py`
  - [ ] `RLHFDatasetWithTarget` 클래스 구조
  - [ ] `random_get()` 메서드 구현
  - [ ] `remove_data()` 메서드 구현

- [ ] `vrag_agent/generation_phase1.py`
  - [ ] `LLMGenerationManager` 클래스
  - [ ] `GenerationConfig` 구조
  - [ ] `run_llm_loop()` 멀티턴 로직

- [ ] `vrag_agent/gpu_monitor.py`
  - [ ] GPU 모니터링 기능

- [ ] `mix_core_alg.py`
  - [ ] `compute_grpo_outcome_advantage_split()` 구현

---

## 🛠️ Phase 3: 실습 및 실험

### 3.1 환경 설정 및 실행
- [ ] 필요 라이브러리 설치
  - [ ] Ray, PyTorch, FSDP
  - [ ] vLLM
  - [ ] HuggingFace Transformers
  - [ ] W&B (선택)
- [ ] 모델 다운로드
  - [ ] Qwen2.5-VL-7B-Instruct 또는 다른 모델
- [ ] 데이터셋 준비
  - [ ] Parquet 파일 형식 확인
  - [ ] 필수 컬럼: `index`, `prompt`, `target`
- [ ] 테스트 실행
  - [ ] `bash exp_scripts/debug.sh` 실행
  - [ ] 에러 확인 및 해결
  - [ ] 로그 확인

### 3.2 파라미터 실험

#### 3.2.1 HPT 전략 비교
- [ ] **Baseline (no HPT)** 실행
  ```bash
  trainer.unify_strategy="no"
  ```
  - [ ] 결과 기록: 성능, 수렴 속도

- [ ] **Switch 전략** 실행
  ```bash
  trainer.unify_strategy="switch"
  trainer.switch_gate=0
  ```
  - [ ] 결과 기록 및 baseline과 비교

- [ ] **Soft 전략** 실행
  ```bash
  trainer.unify_strategy="soft"
  ```
  - [ ] 결과 기록 및 다른 전략들과 비교

#### 3.2.2 Switch Gate 조정 실험
- [ ] `switch_gate=0` (즉시 RL)
- [ ] `switch_gate=3` (3회 성공 후 RL)
- [ ] `switch_gate=5` (5회 성공 후 RL)
- [ ] 각 설정의 수렴 곡선 비교

#### 3.2.3 SFT Loss 계수 실험
- [ ] `sft_loss_coef=0.5` (SFT 약화)
- [ ] `sft_loss_coef=1.0` (기본값)
- [ ] `sft_loss_coef=2.0` (SFT 강화)
- [ ] 각 설정의 영향 분석

#### 3.2.4 Advantage Estimator 비교
- [ ] `algorithm.adv_estimator=grpo`
- [ ] `algorithm.adv_estimator=gae` (Critic 필요)
- [ ] `algorithm.adv_estimator=reinforce`
- [ ] 성능 및 안정성 비교

### 3.3 디버깅 및 모니터링
- [ ] 메모리 사용량 모니터링
  - [ ] `check_memory_usage()` 호출 추가
  - [ ] GPU 메모리 프로파일링
- [ ] 데이터 균형 로깅
  - [ ] UID별 on/off-policy 비율 출력
  - [ ] Switch 결정 로그 추가
- [ ] Prefix mask 검증
  - [ ] 각 배치의 on/off-policy 샘플 수 확인
  - [ ] Mask 정확성 검사
- [ ] W&B 대시보드 활용
  - [ ] 주요 메트릭 추적
  - [ ] 비교 실험 시각화

---

## 🔧 Phase 4: 코드 수정 및 확장

### 4.1 간단한 수정 실습

#### 4.1.1 로깅 개선
- [ ] 추가 메트릭 로깅
  - [ ] UID별 상세 통계
  - [ ] 단계별 시간 측정
  - [ ] 메모리 사용량 추적
- [ ] 로그 포맷 개선
  - [ ] 더 읽기 쉬운 출력
  - [ ] 색상 코딩 (선택)

#### 4.1.2 하이퍼파라미터 하드코딩 제거
- [ ] `on_remove_num = 8` 설정 가능하게 변경 (409줄)
- [ ] Soft 전략 계수 매핑 설정화 (1124-1178줄)
- [ ] RAG 성공 임계값 `0.1` 설정 가능하게 (633줄)

#### 4.1.3 검증 기능 강화
- [ ] Assertion 추가
  - [ ] prefix_mask 형태 검증
  - [ ] UID 매핑 검증
  - [ ] Batch size 일관성 검사
- [ ] 예외 처리 개선
  - [ ] 파일 누락 시 명확한 에러 메시지
  - [ ] GPU OOM 시 복구 전략

### 4.2 중급 수정 과제

#### 4.2.1 새로운 Unify 전략 구현
- [ ] "adaptive" 전략 설계
  - [ ] Switch와 Soft의 장점 결합
  - [ ] 성능 기반 동적 계수 조정
- [ ] 구현
  - [ ] `select_on_off_ada_balance()` 확장
  - [ ] 새로운 로직 추가
- [ ] 테스트 및 비교

#### 4.2.2 보상 함수 커스터마이징
- [ ] 새로운 `reward_impl_version` 추가
  - [ ] 도메인 특화 보상 로직
  - [ ] 다단계 보상 (partial credit)
- [ ] 성공 판단 기준 변경
  - [ ] 임계값 조정 가능하게
  - [ ] 복합 조건 지원

#### 4.2.3 데이터 샘플링 전략 개선
- [ ] Hard example mining
  - [ ] 어려운 샘플 우선 선택
  - [ ] 성능 낮은 UID 집중 학습
- [ ] Curriculum learning
  - [ ] 쉬운 것부터 어려운 순서로
  - [ ] 단계적 난이도 증가

### 4.3 고급 확장 과제

#### 4.3.1 멀티 모달 지원 강화
- [ ] 이미지 처리 파이프라인 최적화
- [ ] 비전-언어 데이터 증강
- [ ] 모달별 보상 가중치

#### 4.3.2 효율성 개선
- [ ] Batch 크기 동적 조정 개선
- [ ] KV Cache 재사용 전략
- [ ] Gradient accumulation 최적화

#### 4.3.3 새로운 알고리즘 통합
- [ ] DPO (Direct Preference Optimization) 지원
- [ ] RLHF with rejection sampling
- [ ] Iterative refinement 전략

---

## 📊 Phase 5: 분석 및 문서화

### 5.1 실험 결과 분석
- [ ] 각 전략별 성능 비교표 작성
- [ ] 수렴 곡선 그래프 생성
- [ ] 메모리/시간 효율성 분석
- [ ] Best practice 도출

### 5.2 코드 문서화
- [ ] Docstring 추가
  - [ ] 주요 클래스
  - [ ] 주요 메서드
  - [ ] 복잡한 로직 부분
- [ ] 인라인 주석 개선
  - [ ] 한글 주석 영어로 통일 (선택)
  - [ ] 불분명한 부분 명확화
- [ ] 예제 코드 작성
  - [ ] 간단한 사용 예제
  - [ ] 커스터마이징 예제

### 5.3 기술 보고서 작성
- [ ] HPT 구현 세부사항 정리
- [ ] 실험 결과 및 인사이트
- [ ] 개선 제안 및 향후 과제
- [ ] 블로그 포스트 또는 논문 작성 (선택)

---

## 🎯 Phase 6: 응용 및 프로젝트

### 6.1 실제 문제 적용
- [ ] 특정 도메인 선택 (예: 코딩, 수학, 대화)
- [ ] 데이터셋 준비
- [ ] HPT로 모델 훈련
- [ ] 성능 평가 및 분석

### 6.2 오픈소스 기여
- [ ] 버그 수정 및 개선 사항 PR
- [ ] 문서 개선 기여
- [ ] 예제 추가

### 6.3 연구 확장
- [ ] 새로운 연구 아이디어 도출
- [ ] 실험 설계 및 수행
- [ ] 결과 발표 (컨퍼런스/저널)

---

## 📝 학습 노트

### 핵심 개념 요약

#### HPT의 핵심 아이디어
```
성능 낮음 → SFT (모방 학습) → 기초 능력 확보
성능 높음 → RL (강화 학습) → 자체 개선 및 탐색
```

#### 통합 손실 함수
```
L_total = L_RL(on-policy) + λ_SFT × L_SFT(off-policy)
```
- On-policy: 현재 모델이 생성한 데이터
- Off-policy: 데이터셋의 타겟 시퀀스
- prefix_mask로 구분

#### 3가지 전략
1. **Switch**: 임계값 기반 하드 스위칭
2. **Soft**: 점진적 계수 블렌딩
3. **No**: 베이스라인 (HPT 없음)

### 주요 코드 위치 빠른 참조

| 기능 | 파일 | 라인 |
|------|------|------|
| HPT 전략 선택 | mix_trainer.py | 405-426 |
| 훈련 메인 루프 | mix_trainer.py | 428-1395 |
| 데이터 균형 조정 | mix_trainer.py | 711-1110 |
| Advantage 계산 | mix_trainer.py | 117-203, 1265-1296 |
| 보상 계산 | mix_trainer.py | 612-708 |
| VRAG 생성 | mix_trainer.py | 573-588 |
| 설정 파일 | debug.sh | 전체 |

### 질문 및 메모
- [ ] Q: VRAG와 표준 생성의 성능 차이는?
- [ ] Q: Switch gate를 동적으로 조정할 수 있는가?
- [ ] Q: Off-policy 데이터가 너무 많으면 과적합 위험은?
- [ ] TODO: UID별 학습 곡선 시각화 도구 만들기
- [ ] TODO: Hyperparameter sweep 자동화 스크립트

---

## ✅ 진행 상황 추적

### 완료된 항목
- [x] Agent.md 작성 (상세 기술 문서)
- [x] README.md 작성 (한글 개요)
- [x] 코드베이스 초기 탐색

### 현재 진행 중
- [ ] Phase 1: 기초 개념 이해
- [ ] Phase 2: 코드 세부 분석

### 다음 단계
- [ ] Phase 3: 실습 및 실험

---

## 📌 참고 자료

### 논문
- [HPT 논문](https://arxiv.org/abs/2509.04419)
- [PPO 원본 논문](https://arxiv.org/abs/1707.06347)
- [GRPO 관련 자료 찾기]

### 코드 저장소
- [공식 HPT 구현](https://github.com/TsinghuaC3I/Unify-Post-Training)
- [VERL 프레임워크](https://github.com/volcengine/verl)

### 블로그/튜토리얼
- [Ray 분산 훈련 가이드](https://docs.ray.io/en/latest/)
- [PyTorch FSDP 문서](https://pytorch.org/docs/stable/fsdp.html)
- [vLLM 문서](https://docs.vllm.ai/)

---

**마지막 업데이트**: 2025-11-25
**작성자**: Claude & User
**버전**: 1.0
