#!/bin/bash

export NNODES=8
export NPROC_PER_NODE=16
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export GLOO_SOCKET_IFNAME=xxx
export HCCL_SOCKET_IFNAME=xxx
export HCCL_IF_BASE_PORT=61000
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'

export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
export PYTHONPATH=/path/to/MindSpeed:$PYTHONPATH

export QWEN_CHECKPOINT_TIMEOUT_SECOND=3600
export USE_MCORE_GDN=1
export IMAGE_MAX_TOKEN_NUM=256
export PLE_CPU_OFFLOAD=0

export MODEL=/path/to/Qwen3.8-Flash-Next
export DATASET=/path/to/dataset


megatron sft \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --save_safetensors true \
    --save_missing_weights true \
    --add_non_thinking_prefix true \
    --enable_thinking false \
    --tuner_type full \
    --bf16 true \
    --nccl_comm_warmup true \
    --overlap_grad_reduce false \
    --overlap_param_gather false \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 14 \
    --decoder_last_pipeline_num_layers 34 \
    --expert_model_parallel_size 64 \
    --expert_tensor_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --vit_attn_impl sdpa \
    --freeze_parameters_regex '.*\.self_attention\.indexer\..*' \
    --micro_batch_size 1 \
    --global_batch_size 80 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_aux_loss_coeff 1e-6 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --packing true \
    --padding_free true \
    --num_iters 100 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --use_precision_aware_optimizer false \
    --optimizer_cpu_offload false \
    --use_megatron_fsdp false \
    --use_distributed_optimizer true \
    --optimizer adam \
    --accumulate_allreduce_grads_in_fp32 true \
    --gradient_accumulation_fusion false \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --lr_decay_style cosine \
    --add_version false \
    --load_args false \
    --dataset_shuffle true \
    --max_length 32000 \
    --truncation_strategy delete \
    --split_dataset_ratio 0 \
    --eval_iters 0 \
    --manual_gc false \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend auto \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 1