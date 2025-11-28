
#SFT 데이터셋 개수 약 3000개
#batch size: 64
#epoch:44
#-> 총 188 step


torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
  data.train_files=lsm_tmp/results/sft_dataset/train1.parquet \
  data.val_files=lsm_tmp/results/sft_dataset/val1.parquet \
  model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct \
  trainer.default_local_dir=outputs/sft-test-filtered \
  model.trust_remote_code=true \
  data.micro_batch_size_per_gpu=4 \
  data.train_batch_size=32 \
  data.max_length=1024 \
  data.use_remove_padding=false \
  model.enable_gradient_checkpointing=false \
  trainer.total_epochs=12 \
  optim.warmup_steps_ratio=0.12 \
  optim.lr=1e-5 \
  optim.weight_decay=0.05 | tee logs/sft_test.log