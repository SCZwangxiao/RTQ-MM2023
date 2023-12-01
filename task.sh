################################## Set your own env variables ##################################
export TRANSFORMERS_CACHE=./modelzoo/huggingface

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# For debug only
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO


################################## training commands for 8 gpus ##################################
numGPU=8

# MSR-VTT
bash run_scripts/dist_train.sh configs/rtq/msrvtt_ret/rtq_vitb16_f12to6_top128_local_neg_lr2e-6_8*b8_e6_amp.yaml $numGPU
bash run_scripts/dist_train.sh configs/rtq/msrvtt_cap/rtq_vitb16_f12to6_lr2e-6_8*b16_e10_amp.yaml $numGPU
bash run_scripts/dist_train.sh configs/rtq/msrvtt_qa/rtq_vitb16_f12to6_lr2e-5_8*b16_e6_amp.yaml $numGPU

# NeXt-QA
bash run_scripts/dist_train.sh configs/rtq/nextqa/rtq_vitb16_f16to4_lr1e-5_8*b16_e6_amp.yaml $numGPU


################################## some useful command templates ##################################
# training using fewer gpus
bash run_scripts/dist_train.sh <path_to_your_config> $numGPU --options run.batch_size_train=<global_batch_size>/$numGPU
# For example
bash run_scripts/dist_train.sh configs/rtq/nextqa/rtq_vitb16_f16to4_lr1e-5_8*b16_e6_amp.yaml 2 --options run.batch_size_train=64


# Resume training
bash run_scripts/dist_train.sh <path_to_your_config> $numGPU --options run.resume_ckpt_path=<path_to_latest_checkpoint>

# Evaluation
bash run_scripts/dist_test.sh <path_to_your_config> $numGPU --options model.load_finetuned=<path_to_the_checkpoint>
