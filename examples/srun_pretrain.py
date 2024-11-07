# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
from copy import deepcopy
from multiprocessing import Pool
import pdb
import subprocess
import signal
from utils_ceph import PetrelBackend

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import math
import sys 
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

import subprocess
import torch.distributed as dist

import re

# import wandb
sys.path.append("..")

from genomics_fm.configuration_bert import BertConfig
from genomics_fm.bert_layers import BertForMaskedLM

from src.transformers import DNATokenizer


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def remove_non_acgt_chars(sequence):
    pattern = '[^ACGT]'
    cleaned_sequence = re.sub(pattern, 'N', sequence)
    return cleaned_sequence

def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join([MAP[c] for c in reversed(sequence)])

MODEL_CLASSES = {
    "genomics-fm": (BertConfig, BertForMaskedLM),
}

MASK_LIST = {
    "1": [-1],
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3],
    "14": [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]
}
port = None
#if num_gpus != 1 and num_gpus != 0:

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "39500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        print(f"rank {rank}")
        print(f"world_size{world_size}")
        print(f"node_list{node_list}")
        print(f"addr {addr}")    
        print(os.environ['MASTER_ADDR'])
        print(os.environ['MASTER_PORT'])
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

class CSVIterableDataset(IterableDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args, file_path: str, train_file_lines, pertrel_oss: PetrelBackend, block_size=512):      
        self.folder_path = file_path.split(",")
        self.pertrel_oss = pertrel_oss
        self.file_list = []
        for subfile_path in self.folder_path:
            for file in self.pertrel_oss.list_dir_or_file(dir_path=subfile_path):
                filename = os.path.join(subfile_path, file)
                self.file_list.append(filename)
        self.start = 0
        self.args = args
        self.end = 0
        self.num_tokenizer = 2
        self.word_size = int(os.environ.get('WORLD_SIZE'))
        self.rank = int(os.environ.get('RANK'))
        self.tokenizer = tokenizer
        self.max_seq_length = block_size
        if file_path == args.train_data_file:
            #print(args.n_gpu)
            self.len_data = args.num_train_sample // args.n_gpu
        else:
            self.len_data = args.num_eval_sample // args.n_gpu
    def __iter__(self):
        random.shuffle(self.file_list)
        print(f"rank编号:{self.rank} len(self.file_list) = {len(self.file_list)}")
        for file_name in self.file_list:
            # file_path = os.path.join(self.folder_path, file_name)
            # data = pd.read_csv(file_path, header=None)
            info = self.pertrel_oss.get_text(file_name).strip()
            data_list = info.split('\n')
            self.end = len(data_list)
            
            worker_info = torch.utils.data.get_worker_info()
            total_data_loader_count = worker_info.num_workers * self.word_size // self.num_tokenizer # 2 tokenizer: bpe and kmer
            per_worker_lines = int(math.ceil((self.end - self.start) / total_data_loader_count))
            if self.args.use_bpe:
                iter_start = self.start + (self.rank * worker_info.num_workers + worker_info.id) * per_worker_lines
            else:
                rank_start = self.rank - self.word_size // self.num_tokenizer 
                iter_start = self.start + (rank_start * worker_info.num_workers + worker_info.id) * per_worker_lines
            
            iter_end = min(self.end, iter_start + per_worker_lines)

            print(f"------------------------------------------------------------------------{file_name}")
            print(f"系统进程号:{os.getpid()} rank编号:{self.rank} dataloader子进程编号:{worker_info.id}, \
                  iter_start = {iter_start}, self.end = {iter_end}, prompt_token = {self.args.prompt_tokenizer}, {self.args.prompt_position}")
            data_list = random.sample(data_list, len(data_list))
            rows = data_list[iter_start:iter_end]
            # rows = random.sample(rows, len(rows))
            for row in rows:
                text = row.strip().upper()
                text = remove_non_acgt_chars(text)
                # 0.5 use reverse
                if random.random() > 0.5:
                    text = get_alter_of_dna_sequence(text)
                if len(text) < 10:
                    print("so short < 10: ",text)
                    continue
                # if not self.args.use_bpe:
                #     if random.random() >= self.args.kmer_sample_probability:
                #         continue
                # print(text, len(text))
                if not self.args.use_bpe:
                    # print(text)
                    text = generate_kmer_str(text, k=6)
                    # print(text)
                text = self.args.prompt_tokenizer + self.args.prompt_position + text 
                inputs = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_attention_mask=True, 
                    return_tensors='pt'
                )
                
                input_ids = inputs["input_ids"].squeeze(0)
                attention_mask = inputs["attention_mask"].squeeze(0)

                input_ids_padded = torch.nn.functional.pad(input_ids, (0, self.max_seq_length - len(input_ids)), 
                                                           value=self.tokenizer.pad_token_id)
                attention_mask_padded = torch.nn.functional.pad(attention_mask, (0, self.max_seq_length - len(attention_mask)), value=0)
                line_list = {
                    'input_ids': input_ids_padded,
                    'attention_mask': attention_mask_padded
                }
                yield line_list
                
    def shuffle_file(self):
        random.shuffle(self.file_list)
    def __len__(self):       
        return self.len_data



def load_and_cache_examples(args, tokenizer, pertrel_oss, evaluate=False):
    word_size = int(os.environ.get('WORLD_SIZE'))
    rank = int(os.environ.get('RANK'))
    if args.use_bpe:
        file_path = args.eval_data_file if evaluate else args.train_data_file     
    else:
        file_path = args.eval_data_file_kmer if evaluate else args.train_data_file_kmer

    
    print(f"rank = {rank}, use_bpe = {args.use_bpe}, file_path = {file_path}")

    train_file_lines = args.num_eval_sample if evaluate else args.num_train_sample
    if args.line_by_line:
        TDataset = CSVIterableDataset(tokenizer, args, pertrel_oss=pertrel_oss, file_path=file_path, train_file_lines=train_file_lines, block_size=args.block_size)
    else:
        print("error")
        return

    return TDataset
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)




def mask_tokens(inputs: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if args.use_bpe:
        mask_list = MASK_LIST["1"]
        mask_probability = args.mlm_probability
    else:
        mask_list = MASK_LIST["14"]
        mask_probability = args.mlm_probability / 6.0
    # print(f"args.use_bpe = {args.use_bpe}, mask_list = {mask_list}, mask_probability = {mask_probability}")

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mask_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    # print(f" this is mask_token special_tokens_mask = {special_tokens_mask[0]}")
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        try:
            end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
        except:
            print(f" i = {i}, probability_matrix.shape = {probability_matrix.shape}, {torch.where(probability_matrix[i])}")
        mask_centers = set(torch.where(masked_index==1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        if mask_list is not None:
            for center in mask_centers:
                
                for mask_number in mask_list:
                    current_index = center + mask_number
                    if current_index <= end and current_index >= 1:
                        new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True


    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, pertrel_oss) -> Tuple[int, float]:
    """ Train the model """
    if args.rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.per_gpu_train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn
    )
    len_train_dataloader = args.num_train_sample // args.train_batch_size
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len_train_dataloader// args.gradient_accumulation_steps) + 1
    else:
        t_total = args.len_train_dataloader // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
        # and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        # and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    if (
        args.model_name_or_path
    ):

        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location=torch.device('cpu')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"), map_location=torch.device('cpu')))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        num_params = count_parameters(model)
        print(f"local_rank = {args.local_rank}, 模型的参数数量为：{num_params}")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", args.num_train_sample)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_gpu_train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    print(f"!!!!!!!!!!!!!!!!!!!!!!!torch.distributed.get_world_size() =  {torch.distributed.get_world_size()}")
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    iteration_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len_train_dataloader // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len_train_dataloader // args.gradient_accumulation_steps)
            iteration_trained_in_current_epoch = steps_trained_in_current_epoch * args.gradient_accumulation_steps
            # print(global_step)
            # batch_pos = torch.load(os.path.join(args.model_name_or_path, "dataloader_pos.pt"))[:, :1].reshape(-1)
            
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            logger.info("  Will skip the first %d iteration in the first epoch", iteration_trained_in_current_epoch)
            # logger.info("  dataloader pos = %d", batch_pos)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    #print(len(tokenizer))
    # model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    # model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    # ids_set = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0}
    print("!!!!!!!!!!!!!!!")
    print(f"model = {model}")
    step_flag = True
    word_size = int(os.environ.get('WORLD_SIZE'))
    for _ in train_iterator:
        train_dataset.shuffle_file()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            inputs_ids = batch["input_ids"]
            
            inputs, labels = mask_tokens(inputs_ids, tokenizer, args) if args.mlm else (inputs_ids, inputs_ids)
            if step_flag:
                print(f"args.rank = {args.local_rank} mask_inputs = {inputs[0]}, {inputs_ids.shape}")
                step_flag = False
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            # print(inputs[0])
            # print(attention_mask[0])
            model.train()
            outputs = model(inputs, attention_mask=attention_mask, labels=labels, use_alibi=args.use_alibi) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc) 

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                
                global_step += 1

                if args.rank in [-1, 0, int(word_size/2)] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        (args.rank == 0 or args.rank == int(word_size/2)) and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, pertrel_oss)
                        for key, value in results.items():
                            tb_writer.add_scalar(f"eval_{args.rank}_{key}", value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                # 
                if args.rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}/".format(checkpoint_prefix, global_step))
                    s_output_dir = os.path.join(args.s_output_dir, "{}-{}/".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)
                    optimizer_filename = "optimizer.pt"
                    scheduler_filename = "scheduler.pt"
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, optimizer_filename))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, scheduler_filename))
                    rclone_command = "rclone copy --progress --transfers 200 --checkers 200 {} {}".format(output_dir, s_output_dir)
                    print(rclone_command)
                    try:
                        output = subprocess.check_output(rclone_command, shell=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Command execution failed: {e}")

                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, pertrel_oss, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    #pdb.set_trace()
    eval_dataset = load_and_cache_examples(args, tokenizer, pertrel_oss, evaluate=True)

    if args.rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn
    )


    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        
        inputs_ids = batch["input_ids"]
            
        inputs, labels = mask_tokens(inputs_ids, tokenizer, args) if args.mlm else (inputs_ids, inputs_ids)

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, labels=labels, use_alibi=args.use_alibi) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "eval_loss": eval_loss}


    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write(str(float(perplexity)) + "\n")
            # writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--train_data_file_kmer", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--s_output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## prompt
    parser.add_argument(
        "--use_alibi",
        type=bool,
        default=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--user_bpe",
        type=bool,
        default=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prompt_tokenizer",
        type=str,
        default="[BPE]",
        help="The prompt tokenizer : [BPE] or [KMER].",
    )
    parser.add_argument(
        "--prompt_position",
        type=str,
        default="[ALIBI]",
        help="The prompt tokenizer : [ALIBI] or [APE].",
    )


    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        default=None,
        type=str,
        help="The output directory of tensorboard runs.",
    )
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--eval_data_file_kmer",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--is_shuffle", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--kmer_sample_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--wandb_name",
        default=None,
        type=str,
        help="wandb name.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_name_kmer",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--n_process", type=int, default=1, help="")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--rank", type=int, default=-1, help="For distributed training: rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--num_train_sample", type=int, default=5000000, help="num_train_sample")
    parser.add_argument("--num_eval_sample", type=int, default=7557629, help="num_eval_sample")
    parser.add_argument("--num_workers", type=int, default=1, help="num_workers")
    #parser.add_argument("--len_train_dataloader", type=int, default=0, help="len_train_dataloader")
    args = parser.parse_args()
    pertrel_oss = PetrelBackend()

    #print(args)
    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    setup_distributed()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.n_gpu = torch.distributed.get_world_size()
    rank = int(os.environ["RANK"])
    args.rank = rank
    device = torch.device("cuda", args.local_rank)
    args.device = device
    print(f"[init] == local rank: {args.local_rank}, global rank: {rank} ==")
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


    if args.rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    config_class, model_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()
    

    if args.tokenizer_name:

        tokenizer_bpe = AutoTokenizer.from_pretrained(args.tokenizer_name)

        tokenizer_kmer = DNATokenizer.from_pretrained(args.tokenizer_name_kmer)
        new_special_token = "[BPE]"
        new_special_token_kmer = "[KMER]"
        new_special_token_alibi = "[ALIBI]"
        new_special_token_ape = "[APE]"
        special_tokens_dict = {'additional_special_tokens': [new_special_token, new_special_token_kmer, new_special_token_alibi, new_special_token_ape]}
        tokenizer_bpe.add_special_tokens(special_tokens_dict)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
        
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        #pdb.set_trace()
        if args.rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        word_size = int(os.environ.get('WORLD_SIZE'))
        rank = int(os.environ.get('RANK'))
        if rank < word_size // 2:
            args.use_bpe = True
            file_path = args.eval_data_file if evaluate else args.train_data_file
            args.prompt_tokenizer = "[BPE]"
            tokenizer=tokenizer_bpe
        else:
            args.use_bpe = False
            file_path = args.eval_data_file_kmer if evaluate else args.train_data_file_kmer
            args.prompt_tokenizer = "[KMER] "
            tokenizer=tokenizer_kmer

        # use alibi
        if rank % 2 == 0:
            args.use_alibi = True
            args.prompt_position = "[ALIBI]"
        else:
            args.use_alibi = False
            args.prompt_position = "[APE]"
        print("load examples")
        train_dataset = load_and_cache_examples(args, tokenizer, pertrel_oss, evaluate=False)


        if args.rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, pertrel_oss)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = DNATokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, pertrel_oss=pertrel_oss, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
