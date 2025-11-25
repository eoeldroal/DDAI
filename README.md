# HPT: Hybrid Post-Training for LLMs

Implementation of **Hybrid Post-Training (HPT)** from the paper [*"Towards a Unified View of Large Language Model Post-Training"*](https://arxiv.org/abs/2509.04419) (Tsinghua University, Shanghai AI Lab, WeChat AI, 2024).

## What is HPT?

HPT dynamically combines **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning (RL)** during training:

- **Low performance** → Learn from demonstration data (SFT mode)
- **High performance** → Explore and self-improve (RL mode)

This adaptive approach leverages the strengths of both methods while avoiding their individual limitations.

## Key Features

- **Dynamic Strategy Switching**: Automatically transitions between SFT and RL based on success rate
- **Three Training Modes**:
  - `switch`: Hard switching based on performance thresholds
  - `soft`: Gradual blending with adaptive coefficients
  - `no`: Baseline PPO without HPT
- **VRAG Integration**: Multi-turn Visual Retrieval-Augmented Generation for complex reasoning
- **Distributed Training**: Ray-based architecture with FSDP and vLLM

## Quick Start

### Run Training

```bash
cd /home/user/DDAI
bash exp_scripts/debug.sh
```

### Configuration

Main configuration file: `exp_scripts/debug.sh`

Key parameters:
```bash
trainer.unify_strategy="switch"     # HPT mode: switch|soft|no
trainer.switch_gate=0               # Success threshold for RL mode
algorithm.adv_estimator=grpo        # Advantage estimator
data.reward_impl_version=7          # Reward version
```

### Monitor Training

Training metrics are logged to:
- Console output
- W&B dashboard (if configured)

## Project Structure

```
DDAI/
├── exp_scripts/
│   └── debug.sh              # Training launch script
├── hpt/
│   └── verl/verl/mix_src/
│       └── mix_trainer.py    # Main HPT implementation (1,577 lines)
├── Agent.md                  # Detailed technical guide for AI agents
└── README.md                 # This file
```

## Core Concept

HPT unifies SFT and RL through a single loss function:

```
L_total = L_RL(on-policy) + λ_SFT × L_SFT(off-policy)
```

- **On-policy data**: Fresh model generations → RL loss (exploration)
- **Off-policy data**: Dataset target sequences → SFT loss (imitation)
- **prefix_mask**: Marks which tokens use which loss

The ratio of on-policy to off-policy data adapts per-prompt based on success rate.

## How It Works

1. **Generate responses** using current model
2. **Compute rewards** for each response
3. **Count successes** per prompt group
4. **Adjust data balance**:
   - Few successes → Remove on-policy, add off-policy (SFT mode)
   - Many successes → Keep on-policy, remove off-policy (RL mode)
5. **Update model** with combined RL + SFT loss

## Training Modes

### Switch Mode (Default)

```bash
trainer.unify_strategy="switch"
trainer.switch_gate=0
```

Hard switching based on success count:
- `≤ switch_gate` successes → SFT mode
- `> switch_gate` successes → RL mode

### Soft Mode

```bash
trainer.unify_strategy="soft"
```

Gradual coefficient blending:
- 1 success → 100% SFT
- 2-4 successes → Mixed SFT + RL
- 5+ successes → 100% RL

### Baseline Mode

```bash
trainer.unify_strategy="no"
```

Standard PPO without HPT.

## Model Support

Currently configured for:
- **Qwen2.5-VL-7B-Instruct** (vision-language model)
- Supports any HuggingFace-compatible model with modification

## Hardware Requirements

**Minimum** (current config):
- 2× GPUs (configured in debug.sh)
- ~40GB GPU memory per GPU (with FSDP offloading)
- 64GB+ system RAM (for CPU offloading)

**Optimizations enabled**:
- FSDP parameter/gradient/optimizer offloading
- Gradient checkpointing
- Dynamic batch sizing
- vLLM memory control (40% GPU utilization)

## Documentation

### For Developers/Researchers

This README provides a high-level overview.

### For AI Coding Agents

**See [Agent.md](./Agent.md)** for:
- Detailed architecture documentation
- Code location references (file paths + line numbers)
- Implementation details and algorithms
- Configuration parameter reference
- Modification guides and debugging tips

The `Agent.md` file enables AI agents to quickly understand and work with this codebase without extensive exploration.

## Key Papers

- **HPT Paper**: [Towards a Unified View of Large Language Model Post-Training](https://arxiv.org/abs/2509.04419)
- **Official Code**: [TsinghuaC3I/Unify-Post-Training](https://github.com/TsinghuaC3I/Unify-Post-Training)

## Citation

```bibtex
@article{lv2024unified,
  title={Towards a Unified View of Large Language Model Post-Training},
  author={Lv, Xingtai and Zuo, Yuxin and Sun, Youbang and Liu, Hongyi and Wei, Yuntian and Chen, Zhekai and He, Lixuan and Xuekai, Zhu and Zhang, Kaiyan and Wang, Bingning and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2509.04419},
  year={2024}
}
```

## License

See original HPT repository for licensing information.

## Notes

This codebase is a research implementation. Some referenced files (e.g., `vrag_agent/`, `rl_dataset_with_target.py`) are not included in this repository.

---

**Quick Links**:
- [Agent.md](./Agent.md) - Technical documentation for AI agents
- [exp_scripts/debug.sh](./exp_scripts/debug.sh) - Training configuration
- [hpt/verl/verl/mix_src/mix_trainer.py](./hpt/verl/verl/mix_src/mix_trainer.py) - Core implementation
