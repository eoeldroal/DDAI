# HPT: 대규모 언어 모델을 위한 하이브리드 사후 훈련

[*"Towards a Unified View of Large Language Model Post-Training"*](https://arxiv.org/abs/2509.04419) 논문(Tsinghua University, Shanghai AI Lab, WeChat AI, 2024)의 **Hybrid Post-Training (HPT)** 구현체입니다.

## HPT란?

HPT는 훈련 중에 **지도 학습 미세조정(SFT)**과 **강화학습(RL)**을 동적으로 결합합니다:

- **낮은 성능** → 시연 데이터로부터 학습 (SFT 모드)
- **높은 성능** → 탐색하며 자체 개선 (RL 모드)

이러한 적응형 접근법은 두 방법의 장점을 활용하면서 각각의 한계를 피합니다.

## 주요 특징

- **동적 전략 전환**: 성공률에 따라 SFT와 RL 사이를 자동으로 전환
- **세 가지 훈련 모드**:
  - `switch`: 성능 임계값 기반 하드 스위칭
  - `soft`: 적응형 계수를 통한 점진적 블렌딩
  - `no`: HPT 없는 베이스라인 PPO
- **VRAG 통합**: 복잡한 추론을 위한 멀티턴 비전 검색 증강 생성
- **분산 훈련**: Ray 기반 아키텍처, FSDP 및 vLLM 사용

## 빠른 시작

### 훈련 실행

```bash
cd /home/user/DDAI
bash exp_scripts/debug.sh
```

### 설정

주요 설정 파일: `exp_scripts/debug.sh`

핵심 파라미터:
```bash
trainer.unify_strategy="switch"     # HPT 모드: switch|soft|no
trainer.switch_gate=0               # RL 모드 진입 성공 임계값
algorithm.adv_estimator=grpo        # Advantage 추정기
data.reward_impl_version=7          # 보상 버전
```

### 훈련 모니터링

훈련 메트릭은 다음에 기록됩니다:
- 콘솔 출력
- W&B 대시보드 (설정된 경우)

## 프로젝트 구조

```
DDAI/
├── exp_scripts/
│   └── debug.sh              # 훈련 실행 스크립트
├── hpt/
│   └── verl/verl/mix_src/
│       └── mix_trainer.py    # 주요 HPT 구현 (1,577줄)
├── Agent.md                  # AI 에이전트용 상세 기술 가이드
└── README.md                 # 이 파일
```

## 핵심 개념

HPT는 단일 손실 함수를 통해 SFT와 RL을 통합합니다:

```
L_total = L_RL(on-policy) + λ_SFT × L_SFT(off-policy)
```

- **On-policy 데이터**: 새로 생성된 모델 응답 → RL 손실 (탐색)
- **Off-policy 데이터**: 데이터셋의 타겟 시퀀스 → SFT 손실 (모방)
- **prefix_mask**: 어떤 토큰이 어떤 손실을 사용할지 표시

On-policy와 off-policy 데이터의 비율은 프롬프트별 성공률에 따라 적응적으로 조정됩니다.

## 작동 방식

1. **응답 생성**: 현재 모델을 사용하여 응답 생성
2. **보상 계산**: 각 응답에 대한 보상 계산
3. **성공 카운트**: 프롬프트 그룹별 성공 횟수 집계
4. **데이터 균형 조정**:
   - 성공 횟수 적음 → on-policy 제거, off-policy 추가 (SFT 모드)
   - 성공 횟수 많음 → on-policy 유지, off-policy 제거 (RL 모드)
5. **모델 업데이트**: RL + SFT 통합 손실로 업데이트

## 훈련 모드

### Switch 모드 (기본값)

```bash
trainer.unify_strategy="switch"
trainer.switch_gate=0
```

성공 횟수 기반 하드 스위칭:
- `≤ switch_gate` 성공 → SFT 모드
- `> switch_gate` 성공 → RL 모드

### Soft 모드

```bash
trainer.unify_strategy="soft"
```

점진적 계수 블렌딩:
- 1회 성공 → 100% SFT
- 2-4회 성공 → SFT + RL 혼합
- 5회 이상 성공 → 100% RL

### Baseline 모드

```bash
trainer.unify_strategy="no"
```

HPT 없는 표준 PPO.

## 모델 지원

현재 다음 모델로 설정되어 있습니다:
- **Qwen2.5-VL-7B-Instruct** (비전-언어 모델)
- 수정을 통해 모든 HuggingFace 호환 모델 지원 가능

## 하드웨어 요구사항

**최소 사양** (현재 설정 기준):
- GPU 2개 (debug.sh에서 설정됨)
- GPU당 ~40GB GPU 메모리 (FSDP 오프로딩 사용 시)
- 64GB 이상 시스템 RAM (CPU 오프로딩용)

**적용된 최적화**:
- FSDP 파라미터/그래디언트/옵티마이저 오프로딩
- 그래디언트 체크포인팅
- 동적 배치 크기 조정
- vLLM 메모리 제어 (GPU 활용률 40%)

## 문서

### 개발자/연구자용

이 README는 높은 수준의 개요를 제공합니다.

### AI 코딩 에이전트용

**상세 기술 문서는 [Agent.md](./Agent.md)를 참조하세요:**
- 상세한 아키텍처 문서
- 코드 위치 참조 (파일 경로 + 라인 번호)
- 구현 세부사항 및 알고리즘
- 설정 파라미터 레퍼런스
- 수정 가이드 및 디버깅 팁

`Agent.md` 파일은 AI 에이전트가 광범위한 탐색 없이 이 코드베이스를 빠르게 이해하고 작업할 수 있도록 합니다.

## 주요 논문

- **HPT 논문**: [Towards a Unified View of Large Language Model Post-Training](https://arxiv.org/abs/2509.04419)
- **공식 코드**: [TsinghuaC3I/Unify-Post-Training](https://github.com/TsinghuaC3I/Unify-Post-Training)

## 인용

```bibtex
@article{lv2024unified,
  title={Towards a Unified View of Large Language Model Post-Training},
  author={Lv, Xingtai and Zuo, Yuxin and Sun, Youbang and Liu, Hongyi and Wei, Yuntian and Chen, Zhekai and He, Lixuan and Xuekai, Zhu and Zhang, Kaiyan and Wang, Bingning and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2509.04419},
  year={2024}
}
```

## 라이선스

라이선스 정보는 원본 HPT 저장소를 참조하세요.

## 참고사항

이 코드베이스는 연구 구현체입니다. 일부 참조되는 파일들(예: `vrag_agent/`, `rl_dataset_with_target.py`)은 이 저장소에 포함되어 있지 않습니다.

---

**빠른 링크**:
- [Agent.md](./Agent.md) - AI 에이전트용 기술 문서
- [exp_scripts/debug.sh](./exp_scripts/debug.sh) - 훈련 설정
- [hpt/verl/verl/mix_src/mix_trainer.py](./hpt/verl/verl/mix_src/mix_trainer.py) - 핵심 구현
