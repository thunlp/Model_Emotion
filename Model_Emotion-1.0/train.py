import os
os.environ['HF_HOME'] = './huggingface_cache'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch,gc
gc.collect()
torch.cuda.empty_cache()


import sys
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AdamW

from framework import RobertaForMaskedLMPrompt, Trainer, get_loader


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentiment', default='admiration', type=str,
            help='training mode.')
    parser.add_argument('--random_seed', default=1, type=int,
            help='training mode.')

    parser.add_argument('--mode', default='hard_verbalizer',
            help='training mode.')
    parser.add_argument('--prompt_len', default=100, type=int,
            help='prompt length')

    parser.add_argument('--epochs', type=int, default=50, 
            help='epochs to train')
    parser.add_argument('--train_batch_size', default=8, type=int,
            help='train batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int,
            help='eval batch size')
    parser.add_argument('--max_length', default=256, type=int,
            help='max length')
    parser.add_argument('--lr', default=2e-3, type=float,
            help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
            help='weight decay')
    parser.add_argument('--warmup_step', default=300, type=int,
            help='number of warmup steps')
    parser.add_argument('--dropout', default=0.0, type=float,
            help='dropout rate')
    parser.add_argument('--fp16', action='store_true',
            help='use nvidia apex fp16')
    
    # Dataset
    parser.add_argument('--benchmark', default='glue',
            help='benchmark. glue or superglue')
    parser.add_argument('--dataset', default='chli',
            help='Fine tune dataset')
    parser.add_argument('--shot', type=str, default='all',
            help='suffix of dataset. "all" or "K shot"')

    parser.add_argument('--pretrained_model', type=str, default='roberta-base',
            help='Pretrained LM')

    args = parser.parse_args()

    # if args.mode == 'warp':
    #     args.max_length = 512 - args.prompt_len

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        args.train_batch_size = args.train_batch_size * n_gpu
        
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
        
    # Name for log file and checkpoint
    config = [args.sentiment, args.random_seed]
    config = '-'.join([str(i) for i in config])
    pl.seed_everything(args.random_seed)

    os.makedirs('./log', exist_ok=True)
    logfile = os.path.join('./log', config+'.log')
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(format='%(asctime)s - %(name)s -   %(message)s',
                        filename=logfile, level=logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().addHandler(TqdmLoggingHandler())
    logger = logging.getLogger(__name__)
    logger.info(config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=False)

    # Dataset
    train_loader, valid_loader, _, num_labels = get_loader(args)

    # Model
    init_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['<mask>'] * args.prompt_len))
    model = RobertaForMaskedLMPrompt.from_pretrained(
            args.pretrained_model, prompt_len=args.prompt_len, init_ids=init_ids, num_labels=num_labels)

    print('Parameters that requires grad:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() 
            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters_to_optimize, lr=args.lr, correct_bias=False)
    scheduler = StepLR(optimizer, step_size=1, gamma=1)

    # Run
    framework = Trainer(args, logger, tokenizer, model, optimizer, scheduler, num_labels, train_loader, valid_loader)

    best_acc = framework.train()

    logger.info(config)
    logger.info(f'Final best valid accuracy {best_acc:.2%}')

if __name__ == "__main__":
    main()
