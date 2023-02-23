import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def get_loader(args, num_workers=8, root='./data', eval_only=False):
    train_dataset = RandomSequenceClassificationDataset(args.sentiment, args.random_seed, 'train')
    valid_dataset = RandomSequenceClassificationDataset(args.sentiment, args.random_seed, 'dev')
    test_dataset = RandomSequenceClassificationDataset(args.sentiment, args.random_seed, 'test')

    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    if eval_only:
        return valid_loader

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader, train_dataset.num_labels

class RandomSequenceClassificationDataset(Dataset):

    def __init__(self, sentiment, seed, split, root='./data'):
        self.data = []
        self.sentiment = sentiment

        path = os.path.join(root, f"goemotions/{split}.csv")
        self.data = pd.read_csv(path, usecols=['text', 'classes', sentiment])
        np.random.seed(seed)
        pos_index = self.data[self.data[sentiment] == 1].index
        neg_index = np.random.choice(self.data.index, len(pos_index))
        self.data = self.data.iloc[list(pos_index) + list(neg_index)]

        self.num_labels = 2

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        
        sentence = data['text']

        return sentence, data[self.sentiment]
