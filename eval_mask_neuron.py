import os
os.environ['HF_HOME'] = './huggingface_cache'

import json
import pickle
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from framework import RobertaForMaskedLMPrompt, get_loader


nlayers = 12
state = 'best'

def get_pred(logits):
    _, pred = torch.max(logits.view(-1, 2), 1)
    return pred

def accuracy(pred, label):
    return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

def checkIfPathExist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Make dirs:", folder_path)
    else:
        print("Folder path already exis.")


@torch.no_grad()
def evaluate(args, model, loader, tokenizer):
    model.eval()

    it = 0
    total_acc = 0
    with torch.no_grad():
        for sentence, label in tqdm(loader, desc='Eval'):
            inputs = tokenizer(
                sentence, max_length=args.max_length, padding='max_length', truncation=True, return_tensors='pt')

            inputs = inputs.to('cuda')
            label = label.to('cuda')
            outputs = model(**inputs, labels=label)

            pred = get_pred(outputs.logits)
            acc = accuracy(pred, label)
            total_acc += acc
            it += 1
    return total_acc / it

@torch.no_grad()
def run(args, mask):
    valid_loader = get_loader(args, eval_only=True)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=False, local_files_only=True)
    init_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['<mask>'] * args.prompt_len))
    model = RobertaForMaskedLMPrompt.from_pretrained(
            args.pretrained_model, prompt_len=args.prompt_len, init_ids=init_ids, num_labels=2)

    ckpt = torch.load(f'./checkpoint/{args.sentiment}-{args.random_seed}.pt')
    prompt_emb = torch.nn.Parameter(ckpt[f'{state}_prompt'])
    model.roberta.embeddings.prompt_embedding.weight = prompt_emb

    if torch.cuda.is_available():
        model = model.to('cuda')
    
    def save_ppt_outputs1_hook(n):
        def fn(_,__,output):
            output = output * mask[n].to('cuda')
            return output
        return fn

    for n in range(nlayers):
        model.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))

    acc = evaluate(args, model, valid_loader, tokenizer)

    del model, save_ppt_outputs1_hook

    return ckpt['best_acc'].item(), acc.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentiment', default='anger', type=str,
            help='sentiment')
    parser.add_argument('--random_seed', default=1, type=int,
            help='random seed')

    parser.add_argument('--mode', default='warp',
            help='training mode.')
    parser.add_argument('--prompt_len', default=100, type=int,
            help='prompt length')

    parser.add_argument('--train_batch_size', default=8, type=int,
            help='train batch size')
    parser.add_argument('--eval_batch_size', default=100, type=int,
            help='eval batch size')
    parser.add_argument('--max_length', default=256, type=int,
            help='max length')
    
    # Dataset
    parser.add_argument('--dataset', default='goemotions',
            help='Fine-tune dataset')
    parser.add_argument('--shot', type=str, default='all',
            help='suffix of dataset. "all" or "K shot"')

    parser.add_argument('--pretrained_model', type=str, default='roberta-base',
            help='Pretrained LM')
    
    # Path
    parser.add_argument('--maskPath', type=str, default='./masks/RSA_14property_top_500_6000.pkl',
            help='mask path')
    parser.add_argument('--resultPath', type=str, default='./eval_mask_neuron_result/RSA_14property_top_500_6000',
            help='result path')

    args = parser.parse_args()
    
    
    checkIfPathExist(args.resultPath)
    
    # if the result already exists, skip it
    if os.path.exists(f'{args.resultPath}/{args.sentiment}-{args.random_seed}.json'):
        return

    with open(f'{args.maskPath}', 'rb') as f:
        all_mask = pickle.load(f)

    result = {'sentiment': args.sentiment, 'seed': args.random_seed}
    for name, mask in all_mask.items():
        best_acc, acc = run(args, mask)
        result[name] = acc
        result['best'] = best_acc
        print(result)
        # break
    
    checkIfPathExist(f'{args.resultPath}')
    with open(f'{args.resultPath}/{args.sentiment}-{args.random_seed}.json', 'w') as f:
        json.dump(result, f)    # save result

if __name__ == "__main__":
    main()
