import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed


class EmotionDataset(Dataset):

    def __init__(self, sentiment, tokenizer, max_length, seed, split, root='../data'):
        self.data = []
        self.sentiment = sentiment

        path = os.path.join(root, f"goemotions/{split}.csv")
        data = pd.read_csv(path, usecols=['text', 'classes', sentiment])
        set_seed(seed)
        pos_index = data[data[sentiment] == 1].index
        neg_index = np.random.choice(data.index, len(pos_index))
        data = data.iloc[list(pos_index) + list(neg_index)]

        self.inputs = tokenizer(data['text'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        self.labels = data[sentiment].tolist()
        self.num_labels = 2

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        inputs = {}
        for k, v in self.inputs.items():
            inputs[k] = v[idx]
        
        inputs['labels'] = self.labels[idx]

        return inputs
