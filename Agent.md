# HPT Codebase Technical Guide for AI Agents

**Last Updated**: 2025-11-25
**Purpose**: Fast onboarding guide for AI coding agents working on this HPT (Hybrid Post-Training) implementation.

---

## 1. Project Overview

This codebase implements **Hybrid Post-Training (HPT)** from the paper ["Towards a Unified View of Large Language Model Post-Training"](https://arxiv.org/abs/2509.04419).

**Core Innovation**: Dynamically switches between SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning) based on model performance:
- Low success rate → SFT mode (learn from demonstration data)
- High success rate → RL mode (explore and self-improve)

**Technology Stack**: Ray distributed training, PyTorch FSDP, vLLM for inference, VRAG (Visual Retrieval-Augmented Generation)

---

## 2. Critical File Locations

### Main Implementation
- **`hpt/verl/verl/mix_src/mix_trainer.py`** (1,577 lines)
  - **Line 205**: `class MIXRayPPOTrainer` - Main trainer class
  - **Line 405-426**: `select_on_off_ada_balance()` - HPT core logic
  - **Line 428-1395**: `fit()` - Main training loop
  - **Line 711-1110**: Data balancing logic (on/off-policy mixing)
  - **Line 1265-1296**: Advantage computation with prefix masking

### Configuration
- **`exp_scripts/debug.sh`** - Launch script with all hyperparameters
  - Line 118-120: HPT strategy parameters (`unify_strategy`, `switch_gate`)
  - Line 122-139: Loss function configuration
  - Line 69-147: Full training command with Hydra overrides

### Missing Files (Referenced but Not in Repo)
These are imported but not committed:
- `rl_dataset_with_target.py` - `RLHFDatasetWithTarget` class
- `vrag_agent/generation_phase1.py` - `LLMGenerationManager`, `GenerationConfig`
- `vrag_agent/gpu_monitor.py` - `GPUMonitor` class
- `mix_core_alg.py` - `compute_grpo_outcome_advantage_split`

**Action Required**: When modifying training logic, you may need to request these files from the user.

---

## 3. Architecture Deep Dive

### 3.1 Training Loop Structure

**Entry Point**: `MIXRayPPOTrainer.fit()` (mix_trainer.py:428)

```
Initialization (465-480):
  ├─ Create LLMGenerationManager (VRAG agent)
  └─ Set up GenerationConfig

Main Loop (482-1395):
  for each epoch:
    for each batch:
      ├─ Phase 1: Data Preparation (510-530)
      │   └─ Map dataset 'index' → 'uid' for grouping
      │
      ├─ Phase 2: Generation (547-590)
      │   ├─ switch mode → VRAG multi-turn (generation_manager.run_llm_loop)
      │   └─ other modes → Standard generation (actor_rollout_wg.generate_sequences)
      │
      ├─ Phase 3: Reward Computation (612-708)
      │   ├─ Call reward_fn() → token_level_scores
      │   └─ Aggregate per-UID success counts
      │
      ├─ Phase 4: Data Balancing (711-1110) ⭐ HPT CORE
      │   for each UID:
      │     ├─ Count successes: on_solve_num
      │     ├─ Call select_on_off_ada_balance(on_solve_num)
      │     ├─ Remove/add on-policy samples
      │     └─ Remove/add off-policy samples (with targets)
      │
      ├─ Phase 5: Advantage Computation (1265-1296)
      │   ├─ compute_advantage() → advantages
      │   └─ Apply prefix_mask weighting
      │
      └─ Phase 6: Model Update (1345-1358)
          ├─ actor_rollout_wg.update_actor() → RL + SFT loss
          └─ critic_wg.update_critic() (if using GAE)
```

### 3.2 HPT Strategy Implementation

**Location**: `select_on_off_ada_balance()` (mix_trainer.py:405-426)

#### Strategy A: "switch" (Primary HPT Mode)

**Parameters** (from debug.sh):
```python
trainer.unify_strategy = "switch"
trainer.switch_gate = 0          # Threshold to enter RL mode
trainer.switch_gate_off = 0      # Secondary threshold
```

**Logic** (mix_trainer.py:406-418):
```python
if on_solve_num <= switch_gate:
    # SFT Mode: Few/No successes
    on_remove_num = 8    # Remove all on-policy data
    off_add_num = 1      # Add off-policy (SFT) data

elif on_solve_num <= switch_gate_off:
    # Mixed Mode: Intermediate performance
    on_remove_num = 8
    off_add_num = -1     # Special flag (negative)

else:
    # RL Mode: Many successes
    on_remove_num = 0    # Keep on-policy data
    off_add_num = 0      # No off-policy additions
```

**Success Counting** (mix_trainer.py:721-734):
```python
if reward_impl_version == 7:  # RAG rewards
    is_solved = (uid_rewards > 0.1)  # Format pass + retrieval success
else:  # Exact match rewards
    is_solved = (uid_rewards == 1.0)
```

#### Strategy B: "soft" (Soft Blending)

**Logic** (mix_trainer.py:420-426, 1121-1178):
- Always adds off-policy: `off_add_num = 1`
- Assigns per-sample coefficients based on success count:

```python
coef_mapping = {
    1: (on_coef=0.0,   off_coef=1.0, sft_coef=1.0),   # Pure SFT
    2: (on_coef=0.125, off_coef=1.0, sft_coef=0.5),   # Mostly SFT
    3: (on_coef=0.25,  off_coef=1.0, sft_coef=0.25),  # Mixed
    4: (on_coef=0.5,   off_coef=1.0, sft_coef=0.125), # Mostly RL
    5+: (on_coef=1.0,  off_coef=1.0, sft_coef=0.0),   # Pure RL
}
```

Coefficients attached to batch:
- `batch.batch['on_coef']` - Weight for on-policy loss
- `batch.batch['off_coef']` - Weight for off-policy loss
- `batch.batch['sft_coef']` - Weight for SFT loss

Used in actor update (implementation in actor module).

#### Strategy C: "no" (Baseline)

Standard PPO with fixed on/off-policy ratio. No HPT.

---

## 4. On-Policy vs Off-Policy Data

### 4.1 Core Mechanism: prefix_mask

**Location**: mix_trainer.py:1274

```python
prefix_mask = batch.batch['prefix_mask']  # Shape: [batch_size, seq_len]
# True  = off-policy (SFT) token
# False = on-policy (RL) token

# Apply to advantages
alpha_weight = prefix_mask.float() * prefix_reward_weight_alpha
beta_weight = (~prefix_mask).float() * prefix_reward_weight_beta
batch.batch['advantages'] = (alpha_weight + beta_weight) * raw_advantages
```

### 4.2 Data Generation Types

| Type | Generation Method | prefix_mask | Loss Type | Code Location |
|------|-------------------|-------------|-----------|---------------|
| **On-policy** | Fresh generation from current model | `False` | RL (policy gradient) | 778-845 |
| **Off-policy** | Use dataset target sequences | `True` | SFT (cross-entropy) | 882-1032 |

### 4.3 Data Balancing Flow

**Per-UID Processing** (mix_trainer.py:711-1110):

```python
for uid in unique_uids:
    # 1. Count successes for this UID
    uid_mask = (batch.non_tensor_batch['uid'] == uid)
    uid_rewards = rewards[uid_mask]
    on_solve_num = (uid_rewards > threshold).sum()

    # 2. Determine data operations
    on_remove_num, on_add_num, off_add_num = select_on_off_ada_balance(on_solve_num)

    # 3. Apply operations
    if on_remove_num > 0:
        # Remove on-policy samples (lines 754-776)
        remove_indices = on_policy_indices_for_uid

    if off_add_num > 0:
        # Add off-policy samples with targets (lines 882-1032)
        off_batch = generate_with_prefix(tgt_input_ids)
        off_batch.batch['prefix_mask'] = create_prefix_mask(tgt_length)
```

### 4.4 On-Policy Generation

**Location**: mix_trainer.py:778-845

```python
# Collect UIDs needing additional on-policy data
on_add_uid_list = [...]

# Batch generation
on_gen_batch = merge_batches(on_add_uid_list)
on_gen_output = actor_rollout_wg.generate_on_sequences(on_gen_batch)

# No prefix masking (pure on-policy)
on_gen_output.batch['prefix_mask'] = torch.zeros(...)
```

### 4.5 Off-Policy Generation

**Location**: mix_trainer.py:882-1032

```python
# Use target sequences as prefix
off_gen_batch.batch['input_ids'] = tgt_input_ids  # From dataset
off_gen_batch.batch['attention_mask'] = tgt_attention_mask

# Generate continuation
off_gen_output = actor_rollout_wg.generate_off_sequences(off_gen_batch)

# Mark prefix tokens
prefix_length = tgt_input_ids.size(1)
prefix_mask = torch.zeros(batch_size, total_length)
prefix_mask[:, :prefix_length] = True  # True for target tokens
off_gen_output.batch['prefix_mask'] = prefix_mask
```

---

## 5. Loss Function Architecture

### 5.1 Conceptual Formula

```
L_total = L_RL(on-policy) + λ_SFT × L_SFT(off-policy)

where:
  L_RL  = -Σ advantages[~prefix_mask] × log_probs[~prefix_mask]
  L_SFT = -Σ log_probs[prefix_mask]
  λ_SFT = actor_rollout_ref.actor.sft_loss_coef (default: 1.0)
```

### 5.2 Configuration Parameters

**From debug.sh (lines 122-139)**:

```bash
# Off-policy loss configuration
actor_rollout_ref.actor.offline_loss_type="sft"        # Type: 'sft' or 'dpo'
actor_rollout_ref.actor.sft_loss_coef=1.0              # λ_SFT weight
actor_rollout_ref.actor.off_policy_loss_impl=token     # Token-level implementation

# KL divergence (disabled in this config)
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.00
algorithm.kl_ctrl.kl_coef=0.000

# Prefix reward weighting
actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0  # Weight for off-policy
# prefix_reward_weight_beta (default)                     # Weight for on-policy

# Loss implementation details
actor_rollout_ref.actor.off_policy_normalize=False
actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1"
actor_rollout_ref.actor.loss_remove_token_mean=True
actor_rollout_ref.actor.loss_remove_clip=True
```

### 5.3 Loss Computation Flow

**Note**: Actual loss computation happens in actor module (not in mix_trainer.py). The trainer prepares batch metadata.

**Batch Metadata Passed to Actor** (mix_trainer.py:1356):
```python
batch.batch['advantages']           # Weighted by prefix_mask
batch.batch['prefix_mask']          # Token-level on/off-policy indicator
batch.batch['old_log_probs']        # Log probs from rollout policy
batch.batch['responses']            # Generated token IDs
batch.batch['attention_mask']       # Valid token mask

# For soft strategy only:
batch.batch['on_coef']              # Per-sample on-policy coefficient
batch.batch['off_coef']             # Per-sample off-policy coefficient
batch.batch['sft_coef']             # Per-sample SFT coefficient
```

**Actor Update Call**:
```python
# mix_trainer.py:1356
actor_metrics = self.actor_rollout_wg.update_actor(batch)
# Actor internally:
#   1. Compute RL loss on on-policy tokens
#   2. Compute SFT loss on off-policy tokens
#   3. Combine: total_loss = rl_loss + sft_loss_coef * sft_loss
#   4. Backward and optimizer step
```

---

## 6. VRAG (Visual Retrieval-Augmented Generation)

### 6.1 Integration Points

**Initialization** (mix_trainer.py:465-480):
```python
from .vrag_agent.generation_phase1 import LLMGenerationManager, GenerationConfig

gen_config = GenerationConfig(
    max_turns=self.config.actor_rollout_ref.rollout.get('max_turns', 5),
    max_prompt_length=self.config.actor_rollout_ref.rollout.max_prompt_length,
    num_gpus=self.config.trainer.n_gpus_per_node,
    search_url=self.config.get('retriever', {}).get('url', None),
)

generation_manager = LLMGenerationManager(
    processor=self.processor,  # Qwen2.5-VL processor for vision-language
    actor_rollout_wg=self.actor_rollout_wg,
    config=gen_config,
)
```

**Usage in Generation** (mix_trainer.py:573-588):
```python
if self.config.trainer.unify_strategy == 'switch':
    # Multi-turn VRAG generation
    gen_batch_output = generation_manager.run_llm_loop(
        gen_batch=gen_batch,
        initial_input_ids=first_input_ids
    )
    # Returns:
    #   - gen_batch_output.batch['responses']: Generated sequences
    #   - gen_batch_output.batch['old_log_probs']: Log probabilities
else:
    # Standard single-pass generation
    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

### 6.2 VRAG Parameters

**From debug.sh (lines 145-146)**:
```bash
actor_rollout_ref.rollout.search_url="http://163.239.28.21:5002/search"
actor_rollout_ref.rollout.max_turns=5  # Number of retrieval iterations
```

**Purpose**:
- Multi-turn iterative refinement with external retrieval
- Particularly useful for visual question answering tasks
- Integrates with external search API for knowledge augmentation

---

## 7. Dataset Structure

### 7.1 RLHFDatasetWithTarget

**Initialization** (mix_trainer.py:342-367):
```python
from .rl_dataset_with_target import RLHFDatasetWithTarget

self.train_dataset = RLHFDatasetWithTarget(
    config=self.config,
    parquet_files=self.config.data.train_files,
    tokenizer=self.tokenizer,
    prompt_key=self.config.data.prompt_key,
    max_prompt_length=self.config.data.max_prompt_length,
    filter_prompts=True,
    return_raw_chat=self.config.data.get('return_raw_chat', False),
    processor=self.processor  # For vision-language models
)
```

### 7.2 Dataset Features

**Unique Aspects**:
1. **Dual Data Storage**: Contains both prompts AND target completions
   - `input_ids`: Tokenized prompt
   - `tgt_input_ids`: Tokenized target completion (for SFT)

2. **UID Mapping** (mix_trainer.py:514-530):
   ```python
   if 'index' in batch.non_tensor_batch:
       indices = batch.non_tensor_batch['index']
       batch.non_tensor_batch['id'] = indices    # For VRAG manager
       batch.non_tensor_batch['uid'] = indices   # For HPT logic
   ```

3. **Dynamic Removal** (mix_trainer.py:484-503):
   ```python
   if self.config.trainer.remove_sfted_data:
       self.train_dataset.remove_data(sfted_data_item_list)
       # Reconstruct dataloader
   ```

4. **Random Sampling** (mix_trainer.py:904):
   ```python
   off_data = self.train_dataset.random_get(num_samples)
   # Used to fetch additional off-policy examples
   ```

### 7.3 Data Flow

```
Parquet File (columns: index, prompt, target)
    ↓
RLHFDatasetWithTarget.__getitem__()
    ↓
{
  'input_ids': tensor,           # Tokenized prompt
  'attention_mask': tensor,
  'position_ids': tensor,
  'tgt_input_ids': tensor,       # Tokenized target (for SFT)
  'tgt_attention_mask': tensor,
  'index': int                   # Dataset index
}
    ↓
DataLoader (batch_size=1 in debug.sh)
    ↓
DataProto.from_single_dict()
    ↓
Training Loop
```

---

## 8. Advantage Computation

### 8.1 Supported Estimators

**Configuration**: `algorithm.adv_estimator` (debug.sh:70)

**Implementation** (mix_trainer.py:117-203):
```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    if adv_estimator == 'gae':
        # Generalized Advantage Estimation (requires critic)
        values = data.batch['values']
        advantages = compute_gae(rewards, values, gamma, lam)

    elif adv_estimator == 'grpo':
        # Group Relative Policy Optimization
        advantages = (rewards - rewards.mean(dim=1, keepdim=True))
        if grpo_use_std:
            advantages /= (rewards.std(dim=1, keepdim=True) + 1e-8)

    elif adv_estimator == 'grpo_balance':
        # Balanced GRPO with on/off policy separation
        advantages = compute_grpo_with_balance(...)

    elif adv_estimator == 'grpo_split':
        # Split GRPO (from mix_core_alg.py - not in repo)
        advantages = compute_grpo_outcome_advantage_split(...)

    elif adv_estimator == 'reinforce':
        # Standard REINFORCE (rewards as advantages)
        advantages = rewards

    elif adv_estimator == 'reinforce_plus_plus':
        # REINFORCE++ with baseline
        advantages = rewards - rewards.mean()
```

### 8.2 GRPO Configuration

**From debug.sh (lines 70, 136)**:
```bash
algorithm.adv_estimator=grpo
algorithm.grpo_use_std=False  # Don't divide by std
```

**Actual Computation** (mix_trainer.py:1265-1296):
```python
# 1. Compute raw advantages
advantages = compute_advantage(
    data=batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=1.0,
    lam=1.0,
    grpo_use_std=self.config.algorithm.grpo_use_std
)

# 2. Apply prefix masking weights
prefix_mask = batch.batch['prefix_mask']
alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
beta_weight = (~prefix_mask).float() * 1.0  # Default beta = 1.0

batch.batch['advantages'] = (alpha_weight + beta_weight) * advantages

# 3. Compute returns (for critic training)
if self.use_critic:
    batch.batch['returns'] = advantages + batch.batch['values']
```

---

## 9. Reward Computation

### 9.1 Reward Function Call

**Location**: mix_trainer.py:612-708

```python
# Generation produces responses
gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch)

# Compute rewards
batch.batch['token_level_scores'] = self.reward_fn(batch)
# Shape: [batch_size, response_length]
# Values: 0.0 (failure) to 1.0 (success), or custom for reward_impl_version=7
```

### 9.2 Reward Versions

**Configuration**: `data.reward_impl_version` (debug.sh:139)

**Version 7 (RAG Rewards)** - mix_trainer.py:628-663:
```python
if reward_impl_version == 7:
    fail_value = 0
    success_value = 1  # Symbolic, actual values from reward_fn

    # Success criteria: reward > 0.1
    # Reward components:
    #   - Format correctness: 0.0 or 0.5
    #   - Retrieval success: additional boost
    is_solved = (uid_rewards > 0.1)
```

**Other Versions (Exact Match)** - default:
```python
else:
    fail_value = 0.0
    success_value = 1.0
    is_solved = (uid_rewards == 1.0)
```

### 9.3 UID-Level Aggregation

**Location**: mix_trainer.py:668-692

```python
for uid in unique_uids:
    uid_mask = (batch.non_tensor_batch['uid'] == uid)
    uid_rewards = rewards[uid_mask]  # All responses for this UID

    # Count outcomes
    solve_none = (uid_rewards <= fail_value).all()      # All failed
    solve_all = (uid_rewards >= success_value).all()    # All succeeded
    solve_none_format = (uid_rewards <= 0.1).all()      # All format failures

    # Track metrics
    metrics[f'uid_{uid}/solve_none'] = float(solve_none)
    metrics[f'uid_{uid}/solve_all'] = float(solve_all)
    metrics[f'uid_{uid}/mean_reward'] = uid_rewards.mean().item()
```

---

## 10. Memory Management

### 10.1 Memory Optimization Utilities

**Location**: mix_trainer.py:70-82

```python
def check_memory_usage(stage=""):
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    print(f"[{stage}] Memory usage: {memory_gb:.2f} GB")

def memory_cleanup():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 10.2 Cleanup Points

**Strategic Cleanup** (mix_trainer.py):
- Line 505: After removing SFTed data
- Line 774: After on-policy data removal
- Line 876: After on-policy generation
- Line 1032: After off-policy generation

**Usage Pattern**:
```python
# Heavy operation
gen_output = actor_rollout_wg.generate_sequences(batch)

# Immediate cleanup
memory_cleanup()
```

### 10.3 FSDP Offloading

**From debug.sh (lines 88-90)**:
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True      # Offload params to CPU
actor_rollout_ref.actor.fsdp_config.grad_offload=True       # Offload grads to CPU
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True  # Offload optimizer to CPU
```

**Purpose**: Reduce GPU memory by offloading to CPU RAM.

### 10.4 VLLM Memory Control

**From debug.sh (line 96)**:
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # VLLM uses max 40% GPU memory
```

**Tradeoff**: Lower utilization → smaller KV cache → more memory for model training.

---

## 11. Distributed Training Architecture

### 11.1 Ray Worker Groups

**Initialization** (mix_trainer.py:258-336):

```python
# Define roles and resource pools
role_worker_mapping = {
    Role.ActorRollout: ActorRolloutWorker,  # Combined actor + rollout
    Role.Critic: CriticWorker,               # Value function (if GAE)
    Role.RefPolicy: RefPolicyWorker,         # Reference policy (if KL)
    Role.RewardModel: RewardModelWorker,     # Reward model (if not using reward_fn)
}

# Create worker groups
self.actor_rollout_wg = RayWorkerGroup(resource_pool, actor_rollout_cls)
self.critic_wg = RayWorkerGroup(resource_pool, critic_cls)
self.ref_policy_wg = RayWorkerGroup(resource_pool, ref_policy_cls)

# Initialize models
self.actor_rollout_wg.init_model()
```

### 11.2 Worker Group Methods

**Key Methods**:
```python
# Generation
gen_output = self.actor_rollout_wg.generate_sequences(batch)
on_output = self.actor_rollout_wg.generate_on_sequences(batch)
off_output = self.actor_rollout_wg.generate_off_sequences(batch)

# Training
actor_metrics = self.actor_rollout_wg.update_actor(batch)
critic_metrics = self.critic_wg.update_critic(batch)

# Reference policy (for KL)
ref_log_probs = self.ref_policy_wg.compute_ref_log_prob(batch)

# Reward model
rewards = self.rm_wg.compute_rm_score(batch)
```

### 11.3 Parallelism Configuration

**From debug.sh (lines 93, 114-115)**:
```bash
# Tensor parallelism (model sharding)
actor_rollout_ref.rollout.tensor_model_parallel_size=1

# Data parallelism (FSDP)
trainer.n_gpus_per_node=2
trainer.nnodes=1

# Sequence parallelism (Ulysses)
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
```

**Total GPUs**: `n_gpus_per_node × nnodes = 2`

---

## 12. Checkpoint Management

### 12.1 Save/Load Logic

**Save Checkpoint** (mix_trainer.py:1397-1418):
```python
def _save_checkpoint(self):
    ckpt_path = os.path.join(
        self.config.trainer.default_local_dir,
        f'global_step_{self.global_steps}'
    )

    # Save actor
    self.actor_rollout_wg.save_checkpoint(ckpt_path)

    # Save critic (if using GAE)
    if self.use_critic:
        self.critic_wg.save_checkpoint(ckpt_path)

    # Save optimizer states
    self.actor_rollout_wg.save_optimizer(ckpt_path)
```

**Load Checkpoint** (called in fit() line 445):
```python
def _load_checkpoint(self):
    if checkpoint_path_exists:
        self.actor_rollout_wg.load_checkpoint(ckpt_path)
        if self.use_critic:
            self.critic_wg.load_checkpoint(ckpt_path)
```

### 12.2 Configuration

**From debug.sh (lines 116, 140, 144)**:
```bash
trainer.save_freq=50              # Save every 50 steps
trainer.max_optim_to_keep=2       # Keep only 2 latest checkpoints
trainer.default_local_dir=$ROOT/checkpoints/$EXP_NAME
```

---

## 13. Validation Loop

### 13.1 Validation Trigger

**Location**: mix_trainer.py:1420+

**Triggering Conditions**:
1. Before training (if `trainer.val_before_train=True`)
2. Every `trainer.test_freq` steps (if > 0)
3. After training completes

**Configuration** (debug.sh):
```bash
trainer.val_before_train=False  # Skip initial validation
trainer.test_freq=0             # No periodic validation
```

### 13.2 Validation Process

```python
def _validate(self):
    val_metrics = {}

    for val_batch in self.val_dataloader:
        # Generate with validation settings
        val_output = self.actor_rollout_wg.generate_sequences(
            val_batch,
            temperature=self.config.actor_rollout_ref.rollout.val_temperature,
            top_p=self.config.actor_rollout_ref.rollout.val_top_p,
            n=self.config.actor_rollout_ref.rollout.n_val
        )

        # Compute validation rewards
        val_rewards = self.val_reward_fn(val_batch)
        val_metrics.update(compute_metrics(val_rewards))

    return val_metrics
```

**Validation-Specific Parameters** (debug.sh:99-102):
```bash
actor_rollout_ref.rollout.val_temperature=0.6  # Lower temp = less random
actor_rollout_ref.rollout.val_top_p=0.95
actor_rollout_ref.rollout.n_val=1              # Generate 1 response
```

---

## 14. Metrics and Logging

### 14.1 Tracking System

**Initialization** (mix_trainer.py:434-440):
```python
from verl.utils.tracking import Tracking

logger = Tracking(
    project_name=self.config.trainer.project_name,
    experiment_name=self.config.trainer.experiment_name,
    default_backend=self.config.trainer.logger,  # ['console', 'wandb']
    config=OmegaConf.to_container(self.config, resolve=True)
)
```

**Configuration** (debug.sh:51-53, 110-112):
```bash
export WANDB_API_KEY="..."
export WANDB_PROJECT="unified-ft-debug"

trainer.logger=['console','wandb']
trainer.project_name="$WANDB_PROJECT"
trainer.experiment_name="$EXP_NAME"
```

### 14.2 Metric Categories

**Reward Metrics** (mix_trainer.py:668-692):
```python
metrics[f'uid_{uid}/solve_none'] = ...       # All failed
metrics[f'uid_{uid}/solve_all'] = ...        # All succeeded
metrics[f'uid_{uid}/mean_reward'] = ...      # Average reward
metrics['reward/mean'] = ...                 # Global average
```

**On/Off-Policy Metrics** (mix_trainer.py:1467-1476):
```python
on_policy_mask = ~batch.batch['prefix_mask'].any(-1)
off_policy_mask = batch.batch['prefix_mask'].any(-1)

metrics['on_off_metrics/off_example_ratio'] = off_policy_mask.float().mean()
metrics['on_off_metrics/on_response_length'] = response_length[on_policy_mask].mean()
metrics['on_off_metrics/off_response_length'] = response_length[off_policy_mask].mean()
```

**Unify Strategy Metrics** (mix_trainer.py:1570-1576):
```python
metrics['uni/on_data_ratio'] = on_data_count / total_count
metrics['uni/off_data_ratio'] = off_data_count / total_count
metrics['uni/sft_data_ratio'] = sft_data_count / total_count
```

**Training Metrics** (from actor/critic updates):
```python
metrics['actor/loss'] = ...
metrics['actor/entropy'] = ...
metrics['actor/kl'] = ...
metrics['critic/loss'] = ...
metrics['critic/value_mean'] = ...
```

### 14.3 Logging Call

**Location**: mix_trainer.py (end of each training step):
```python
logger.log(data=metrics, step=self.global_steps)
```

---

## 15. Key Configuration Parameters Reference

### 15.1 HPT Core

```bash
# Strategy selection
trainer.unify_strategy="switch"         # "switch" | "soft" | "no"
trainer.switch_gate=0                   # Success threshold for RL mode
trainer.switch_gate_off=0               # Secondary threshold
trainer.remove_sfted_data=False         # Remove data after SFT

# Off-policy loss
actor_rollout_ref.actor.offline_loss_type="sft"
actor_rollout_ref.actor.sft_loss_coef=1.0
actor_rollout_ref.actor.off_policy_loss_impl=token
actor_rollout_ref.actor.off_policy_normalize=False
actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1"
```

### 15.2 Algorithm

```bash
algorithm.adv_estimator=grpo            # grpo | gae | reinforce | ...
algorithm.grpo_use_std=False
algorithm.kl_ctrl.kl_coef=0.0           # KL penalty (0 = disabled)

actor_rollout_ref.actor.entropy_coeff=0.001
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.use_kl_loss=False
```

### 15.3 Data

```bash
data.train_files=$TRAIN_FILE
data.val_files=$TEST_FILE
data.train_batch_size=1
data.val_batch_size=1
data.max_prompt_length=256
data.max_response_length=256
data.reward_impl_version=7
data.shuffle=True
```

### 15.4 Generation

```bash
# On-policy generation
actor_rollout_ref.rollout.n=1           # Samples per prompt
actor_rollout_ref.rollout.n_verify=1    # Verification samples
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=0.95

# Validation generation
actor_rollout_ref.rollout.n_val=1
actor_rollout_ref.rollout.val_temperature=0.6
actor_rollout_ref.rollout.val_top_p=0.95

# VRAG
actor_rollout_ref.rollout.max_turns=5
actor_rollout_ref.rollout.search_url="http://..."
```

### 15.5 Prefix Mechanism

```bash
actor_rollout_ref.actor.use_sft_prefix_reward=False
actor_rollout_ref.rollout.prefix_share_across_samples=False
actor_rollout_ref.rollout.prefix_strategy=random
actor_rollout_ref.rollout.n_prefix=1
actor_rollout_ref.rollout.min_prefix_ratio=1.0
actor_rollout_ref.rollout.max_prefix_ratio=1.0
actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0
```

### 15.6 Model & Training

```bash
actor_rollout_ref.model.path=$MODEL_PATH
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.model.torch_dtype=bfloat16

actor_rollout_ref.actor.optim.lr=5e-6
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.actor.ppo_micro_batch_size=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
actor_rollout_ref.actor.max_grad_norm=80.0
```

### 15.7 Memory Optimization

```bash
# FSDP offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.grad_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# VLLM
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.enforce_eager=True
```

### 15.8 Trainer

```bash
trainer.total_training_steps=1          # Number of training steps
trainer.total_epochs=1                  # Number of epochs
trainer.n_gpus_per_node=2
trainer.nnodes=1
trainer.save_freq=50
trainer.test_freq=0
trainer.val_before_train=False
trainer.max_optim_to_keep=2
trainer.default_local_dir=$ROOT/checkpoints/$EXP_NAME
```

---

## 16. Common Modification Scenarios

### 16.1 Changing HPT Strategy

**To switch from "switch" to "soft" strategy**:

1. **Edit debug.sh**:
   ```bash
   trainer.unify_strategy="soft"
   ```

2. **Understand behavior change**:
   - "switch": Hard switching based on `switch_gate`
   - "soft": Gradual blending based on success count coefficients

3. **Relevant code**: mix_trainer.py:420-426, 1121-1178

### 16.2 Adjusting Success Threshold

**To change when RL mode activates**:

1. **Edit debug.sh**:
   ```bash
   trainer.switch_gate=5      # Require 5+ successes for RL
   trainer.switch_gate_off=3  # Intermediate threshold
   ```

2. **Relevant code**: mix_trainer.py:406-418

### 16.3 Modifying SFT Loss Weight

**To increase/decrease SFT influence**:

1. **Edit debug.sh**:
   ```bash
   actor_rollout_ref.actor.sft_loss_coef=2.0  # Double SFT weight
   ```

2. **Effect**: `L_total = L_RL + 2.0 × L_SFT`

3. **Relevant code**: Loss computation in actor module (not in mix_trainer.py)

### 16.4 Adding New Reward Version

**To implement custom reward logic**:

1. **Locate reward function**: Passed as `reward_fn` parameter to trainer

2. **Modify success counting** (mix_trainer.py:721-734):
   ```python
   if reward_impl_version == 8:  # Your new version
       success_threshold = 0.8
       is_solved = (uid_rewards >= success_threshold)
   ```

3. **Update debug.sh**:
   ```bash
   data.reward_impl_version=8
   ```

### 16.5 Disabling VRAG

**To use standard generation instead of VRAG**:

1. **Change strategy** in debug.sh:
   ```bash
   trainer.unify_strategy="soft"  # or "no"
   # VRAG only used when unify_strategy="switch"
   ```

2. **Relevant code**: mix_trainer.py:573-590

### 16.6 Batch Size Adjustment

**To increase batch size**:

1. **Edit debug.sh**:
   ```bash
   data.train_batch_size=8
   actor_rollout_ref.actor.ppo_mini_batch_size=16
   actor_rollout_ref.actor.ppo_micro_batch_size=2
   ```

2. **Ensure divisibility**: `mini_batch_size % (micro_batch_size × n_gpus) = 0`

3. **Memory impact**: May require reducing `gpu_memory_utilization` or enabling more offloading

---

## 17. Debugging Tips

### 17.1 Enable Memory Monitoring

**Uncomment existing calls** (mix_trainer.py:508, etc.):
```python
check_memory_usage("batch_start")
check_memory_usage("after_generation")
check_memory_usage("after_update")
```

### 17.2 Print Data Balancing Decisions

**Add logging** in mix_trainer.py:718:
```python
on_remove_num, on_add_num, off_add_num = select_on_off_ada_balance(on_solve_num)
print(f"[UID {uid}] Successes: {on_solve_num}, "
      f"Remove: {on_remove_num}, Add On: {on_add_num}, Add Off: {off_add_num}")
```

### 17.3 Verify prefix_mask Correctness

**Add assertion** after data balancing (mix_trainer.py:~1200):
```python
prefix_mask = batch.batch['prefix_mask']
on_policy_count = (~prefix_mask.any(-1)).sum()
off_policy_count = prefix_mask.any(-1).sum()
print(f"Batch composition: {on_policy_count} on-policy, {off_policy_count} off-policy")
```

### 17.4 Monitor Coefficient Assignment (Soft Strategy)

**Add logging** in mix_trainer.py:~1150:
```python
if self.config.trainer.unify_strategy == 'soft':
    print(f"Assigned coefficients - on: {batch.batch['on_coef']}, "
          f"off: {batch.batch['off_coef']}, sft: {batch.batch['sft_coef']}")
```

### 17.5 Check Reward Distribution

**Add histogram** in mix_trainer.py:~650:
```python
import numpy as np
print(f"Reward distribution: {np.histogram(rewards.cpu().numpy(), bins=10)}")
```

---

## 18. Important Caveats

### 18.1 Missing Files

**Action Required**: If you need to modify:
- Dataset loading logic → Request `rl_dataset_with_target.py`
- VRAG generation → Request `vrag_agent/generation_phase1.py`
- Split GRPO algorithm → Request `mix_core_alg.py`

### 18.2 Hardcoded Values

**Locations to check**:
- mix_trainer.py:409: `on_remove_num = 8` (hardcoded removal count)
- mix_trainer.py:1124-1178: Coefficient mapping for soft strategy

**Recommendation**: Consider making these configurable.

### 18.3 Ray Version Compatibility

**Note** (mix_trainer.py:319):
```python
# keep the referece of WorkerDict to support ray >= 2.31
# Ref: https://github.com/ray-project/ray/pull/45699
self.wg_dicts.append(wg_dict)
```

**Implication**: Code may break on older Ray versions.

### 18.4 VLLM Backend

**VLLM attention backend** (debug.sh:37):
```bash
#export VLLM_ATTENTION_BACKEND=XFORMERS  # Commented out
```

**Note**: Default attention backend may vary by VLLM version.

### 18.5 Data Shuffle

**Important** (debug.sh:141):
```bash
data.shuffle=True
```

**Implication**: Training order is non-deterministic. Set to `False` for reproducibility.

---

## 19. Code Quality Notes

### 19.1 TODOs and Comments

**Korean comments**: Several comments in Korean (lines 2-25, 39-46, 512-530)
- Indicate GPU memory optimization notes
- VRAG integration notes
- UID mapping explanation

**Commented code**:
- Line 536-545: Alternative data popping logic (commented out)
- Line 37: VLLM_ATTENTION_BACKEND export (commented)

### 19.2 Code Duplication

**On-policy generation** (lines 778-845) and **off-policy generation** (lines 882-1032) share similar structure:
- Batch merging
- Worker group call
- Output processing

**Refactoring opportunity**: Extract common generation logic.

### 19.3 Magic Numbers

**Examples**:
- Line 409: `on_remove_num = 8`
- Line 1134-1168: Hardcoded coefficient values for soft strategy
- Line 633: `0.1` threshold for RAG success

**Recommendation**: Move to configuration.

---

## 20. Quick Start Commands

### 20.1 Run Training

```bash
cd /home/user/DDAI
bash exp_scripts/debug.sh
```

### 20.2 Monitor Training

**Watch logs**:
```bash
tail -f /path/to/log/file
```

**Check GPU usage**:
```bash
watch -n 1 nvidia-smi
```

**Monitor W&B** (if configured):
```
https://wandb.ai/your-entity/unified-ft-debug
```

### 20.3 Resume from Checkpoint

**Edit debug.sh** to add:
```bash
trainer.load_checkpoint=/path/to/checkpoint/global_step_N
```

### 20.4 Run with Different Strategy

**Switch mode**:
```bash
bash exp_scripts/debug.sh  # Default: switch strategy
```

**Soft mode**:
```bash
# Edit debug.sh: trainer.unify_strategy="soft"
bash exp_scripts/debug.sh
```

**Baseline (no HPT)**:
```bash
# Edit debug.sh: trainer.unify_strategy="no"
bash exp_scripts/debug.sh
```

---

## 21. Performance Optimization Checklist

### 21.1 Memory

- [ ] Enable FSDP offloading (already enabled in debug.sh)
- [ ] Reduce `gpu_memory_utilization` if OOM
- [ ] Decrease batch sizes
- [ ] Enable gradient checkpointing (already enabled)
- [ ] Use `torch_dtype=bfloat16` (already enabled)
- [ ] Call `memory_cleanup()` after heavy operations

### 21.2 Speed

- [ ] Increase batch sizes (if memory allows)
- [ ] Increase `ppo_micro_batch_size`
- [ ] Disable validation (`test_freq=0`)
- [ ] Use tensor parallelism if multi-GPU
- [ ] Disable unnecessary logging
- [ ] Use `enforce_eager=False` for vLLM (if stable)

### 21.3 Convergence

- [ ] Tune learning rate (`actor.optim.lr`)
- [ ] Adjust `switch_gate` thresholds
- [ ] Tune `sft_loss_coef`
- [ ] Enable KL penalty if needed
- [ ] Increase `entropy_coeff` for exploration
- [ ] Adjust advantage estimator (`grpo_use_std`)

---

## 22. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     MIXRayPPOTrainer (Driver)                   │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──> RLHFDatasetWithTarget (Parquet → Batch)
             │
             ├──> Ray Worker Groups:
             │    ├─ actor_rollout_wg (Generation + Actor Update)
             │    ├─ critic_wg (Value Function)
             │    ├─ ref_policy_wg (Reference Policy for KL)
             │    └─ rm_wg (Reward Model, optional)
             │
             └──> Training Loop:
                  │
                  ├─ [1] Data Preparation
                  │   └─ Map index → uid
                  │
                  ├─ [2] Generation
                  │   ├─ switch → VRAG (LLMGenerationManager)
                  │   └─ other → Standard (actor_rollout_wg)
                  │
                  ├─ [3] Reward Computation
                  │   └─ reward_fn() → token_level_scores
                  │
                  ├─ [4] Data Balancing ⭐ HPT CORE
                  │   │
                  │   └─ For each UID:
                  │       ├─ Count successes
                  │       ├─ select_on_off_ada_balance()
                  │       │   ├─ switch strategy → Hard switch
                  │       │   └─ soft strategy → Coefficient blend
                  │       │
                  │       ├─ Remove/add on-policy samples
                  │       └─ Remove/add off-policy samples
                  │
                  ├─ [5] Advantage Computation
                  │   ├─ compute_advantage() (GRPO/GAE/...)
                  │   └─ Apply prefix_mask weights
                  │
                  └─ [6] Model Update
                      ├─ actor_rollout_wg.update_actor()
                      │   └─ L_total = L_RL + λ_SFT × L_SFT
                      └─ critic_wg.update_critic()
```

---

## 23. File-to-Concept Mapping

| Concept | Primary File | Key Lines |
|---------|-------------|-----------|
| HPT Strategy Selection | mix_trainer.py | 405-426 |
| Switch Strategy Logic | mix_trainer.py | 406-418 |
| Soft Strategy Logic | mix_trainer.py | 420-426, 1121-1178 |
| Training Loop | mix_trainer.py | 428-1395 |
| Data Balancing | mix_trainer.py | 711-1110 |
| On-Policy Generation | mix_trainer.py | 778-845 |
| Off-Policy Generation | mix_trainer.py | 882-1032 |
| prefix_mask Application | mix_trainer.py | 1274-1281 |
| Advantage Computation | mix_trainer.py | 117-203, 1265-1296 |
| Reward Computation | mix_trainer.py | 612-708 |
| Success Counting | mix_trainer.py | 721-734 |
| VRAG Integration | mix_trainer.py | 41-46, 465-480, 573-588 |
| Dataset Initialization | mix_trainer.py | 342-367 |
| UID Mapping | mix_trainer.py | 514-530 |
| Memory Management | mix_trainer.py | 70-82 |
| Worker Group Setup | mix_trainer.py | 258-336 |
| Checkpoint Management | mix_trainer.py | 1397-1418 |
| Metrics Logging | mix_trainer.py | 434-440, 668-692, 1467-1476 |
| Configuration | debug.sh | All |

---

## 24. Glossary

| Term | Definition |
|------|------------|
| **HPT** | Hybrid Post-Training - Unified framework for SFT + RL |
| **On-policy** | Data generated by current model (for RL loss) |
| **Off-policy** | Data using dataset targets (for SFT loss) |
| **prefix_mask** | Boolean tensor marking off-policy (SFT) tokens |
| **UID** | Unique identifier grouping responses per prompt |
| **VRAG** | Visual Retrieval-Augmented Generation (multi-turn) |
| **GRPO** | Group Relative Policy Optimization (advantage estimator) |
| **switch_gate** | Success threshold for switching to RL mode |
| **sft_loss_coef** | Weight λ_SFT for SFT loss in total loss |
| **Ray Worker Group** | Distributed workers for actor/critic/rollout |
| **FSDP** | Fully Sharded Data Parallel (PyTorch distributed training) |
| **vLLM** | Fast LLM inference engine (used for rollout) |
| **DataProto** | Data structure for passing batches between components |
| **Advantage** | Estimate of how good an action is (for policy gradient) |

---

**End of Technical Guide**

For human-readable overview, see README.md.
