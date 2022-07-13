import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from framework.roberta import RobertaWarp

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

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=False)
init_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['<mask>'] * 100))
model = RobertaWarp.from_pretrained('roberta-base', prompt_len=100, init_ids=init_ids, num_labels=2)
model = model.to('cpu')

loader = DataLoader(ActiveNeuronDataset(), batch_size=1, shuffle=False)

nlayers = 12
result = []
acc = []
state = 'best'
for sentiment in tqdm(sentiments, 'Sentiment'):

    per_sentiment_out_before_relu = []
    per_sentiment_out_after_relu = []
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

        # Prompt embedding
        # emb = state_dict['roberta.embeddings.prompt_embedding.weight'].detach().cpu().numpy().reshape((1, -1))
        # emb = pd.DataFrame(emb)
        # emb.to_csv(f'./prompt_embedding_csv/{sentiment}-{seed}.csv', index=None)

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
        
    # ---- Active neuron before ReLU ------------------------------------
        neuron_before_relu = outputs.numpy().reshape((1, -1))
        neuron_before_relu = pd.DataFrame(neuron_before_relu)
        if not os.path.exists('./active_neuron_before_relu_csv'):
            os.makedirs('./active_neuron_before_relu_csv')
        neuron_before_relu.to_csv(f'./active_neuron_before_relu_csv/{sentiment}-{seed}.csv', index=None)

        per_sentiment_out_before_relu.append(neuron_before_relu)
    
    neuron_sum_before_relu = (np.stack(per_sentiment_out_before_relu, axis=0)).sum(0)
    neuron_sum_before_relu = pd.DataFrame(neuron_sum_before_relu)
    neuron_sum_before_relu.to_csv(f'./active_neuron_before_relu_csv/{sentiment}-all.csv', index=None)
    # -------------------------------------------------------------------

    # ---- Active neuron after ReLU -------------------------------------
    #     neuron_after_relu = (outputs > 0).int().numpy().reshape((1, -1))
    #     neuron_after_relu = pd.DataFrame(neuron_after_relu)
    #     if not os.path.exists('./active_neuron_after_relu_csv'):
    #         os.makedirs('./active_neuron_after_relu_csv')
    #     neuron_after_relu.to_csv(f'./active_neuron_after_relu_csv/{sentiment}-{seed}.csv', index=None)

    #     per_sentiment_out_after_relu.append(neuron_after_relu)
    
    # neuron_sum_after_relu = (np.stack(per_sentiment_out_after_relu, axis=0)).sum(0)
    # neuron_sum_after_relu = pd.DataFrame(neuron_sum_after_relu)
    # neuron_sum_after_relu.to_csv(f'./active_neuron_after_relu_csv/{sentiment}-all.csv', index=None)
    # -------------------------------------------------------------------

# os.system('tar -zcvf prompt_embedding_csv.tar.gz prompt_embedding_csv')
# os.system('tar -zcvf active_neuron_before_relu_csv.tar.gz active_neuron_before_relu_csv')
# os.system('tar -zcvf active_neuron_after_relu_csv.tar.gz active_neuron_after_relu_csv')

acc = pd.DataFrame(acc)
acc = acc.sort_values(['random_seed', 'sentiment'])
acc.to_csv('./goemotions_prompt_tuning_accuracy.csv', index=None)
