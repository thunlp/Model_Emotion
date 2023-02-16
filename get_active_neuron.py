import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from framework.roberta import RobertaWarp
import argparse

# Define 12 Random Seeds
random_seeds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 42, 100]

# Define 27 Emotion Tasks 
sentiments = ['amusement','excitement','joy','love','desire','optimism','caring',
            'pride','admiration','gratitude','relief', 'approval', 
            'realization', 'surprise', 'curiosity', 'confusion', 
            'fear', 'nervousness', 'remorse', 'embarrassment', 'disappointment', 
            'sadness', 'grief', 'disgust', 'anger', 'annoyance', 'disapproval']


class ActiveNeuronDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return '<s>'

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_mode', type=str, default='before_relu',
            help='Activate neurons before or after ReLU')
    args = parser.parse_args()
    
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=False)
    init_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['<mask>'] * 100))
    model = RobertaWarp.from_pretrained('roberta-base', prompt_len=100, init_ids=init_ids, num_labels=2)
    model = model.to('cpu')

    loader = DataLoader(ActiveNeuronDataset(), batch_size=1, shuffle=False)

    nlayers = 12    # RoBERTa has 12 layers
    result = []
    acc = []
    state = 'best'
    activation_mode = args.activation_mode.lower()
    neurons_path = f'./active_neuron_{activation_mode}_csv'
    if not os.path.exists(neurons_path):
        os.makedirs(neurons_path)
        
    for sentiment in tqdm(sentiments, 'Sentiment'):

        per_sentiment_out = []
        for seed in tqdm(random_seeds, 'Random seed', leave=False):
            ckpt = torch.load(f'./checkpoint/{sentiment}-{seed}.pt')
            prompt_emb = torch.nn.Parameter(ckpt[f'{state}_prompt']).to('cpu')
            model.roberta.embeddings.prompt_embedding.weight = prompt_emb

            # Accuracy
            acc.append({
                'sentiment': sentiment,
                'random_seed': seed,
                'accuracy': ckpt['best_acc'].item()
            })

            # Forward pass to get active neuron
            outputs = [[] for _ in range(nlayers)]
            def save_ppt_outputs1_hook(n):
                def fn(_,__,output):
                    outputs[n] = output.detach().to('cpu')
                return fn

            for n in range(nlayers):
                model.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))

            for sentence in loader:
                inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=False).to('cpu')
                _ = model(**inputs)

            outputs = torch.stack(outputs)
            # print(outputs.shape)          # [12, 1, 102, 3072]
            outputs = outputs[:,:1,:1,:]    # [12 layers, 1 output list for each layer, '<s>', 3072 neurons]
            outputs = outputs.flatten()
            
            # Get neurons before or after activation function.
            if activation_mode == "before_relu":
                activeted_neurons = outputs.numpy().reshape((1, -1))
            
            elif activation_mode == "after_relu":
                activeted_neurons = (outputs > 0).int().numpy().reshape((1, -1))

            else:
                raise NotImplementedError("Please check your activation mode. Use 'before_relu' or 'after_relu'")
            
            activeted_neurons = pd.DataFrame(activeted_neurons)
            activeted_neurons.to_csv(f'{neurons_path}/{sentiment}-{seed}.csv', index=None)

            per_sentiment_out.append(activeted_neurons)
        
        # Add up the neurons of all random seeds
        neuron_sum = (np.stack(per_sentiment_out, axis=0)).sum(0)
        neuron_sum = pd.DataFrame(neuron_sum)
        neuron_sum.to_csv(f'{neurons_path}/{sentiment}-all.csv', index=None)


    acc = pd.DataFrame(acc)
    acc = acc.sort_values(['random_seed', 'sentiment'])
    acc.to_csv('./goemotions_prompt_tuning_accuracy.csv', index=None)

if __name__ == "__main__":
    main()