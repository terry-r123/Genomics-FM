#!/bin/bash

export KMER=6
export TRAIN_FILE=
export TEST_FILE=
export TRAIN_FILE_KMER=
export TEST_FILE_KEMR=
export SOURCE=../
export OUTPUT_PATH=
export RUNS_PATH=
export S_OUTPUT_PATH=
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1



srun python srun_pretrain.py \
    --output_dir=$OUTPUT_PATH \
    --s_output_dir=$S_OUTPUT_PATH \
    --model_type=genomics-fm \
    --tokenizer_name=./genomics-fm/tokenizer_bpe \
    --tokenizer_name_kmer=./genomics-fm/tokenizer_kmer \
    --config_name=./genomics-fm/config/config_prompt.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --train_data_file_kmer=$TRAIN_FILE_KMER \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --eval_data_file_kmer=$TEST_FILE_KEMR \
    --mlm \
    --is_shuffle    \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 192 \
    --per_gpu_eval_batch_size 192 \
    --save_steps 25000 \
    --save_total_limit 2 \
    --max_steps 1000000 \
    --evaluate_during_training \
    --logging_steps 5000 \
    --line_by_line \
    --learning_rate 1e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.15 \
    --kmer_sample_probability 0.28 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 1   \
    --num_workers 1 \
    --tensorboard_dir=$RUNS_PATH \
    --num_train_sample 298018034 \
    --num_eval_sample 420748 \
    --fp16  \
    --fp16_opt_level O1
