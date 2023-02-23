import torch
import pickle
import numpy as np
import pandas as pd
import argparse
import os


def inverse_mask(mask):
    '''
    Convert a mask that masks neuron by ranking their importance into a mask that randomly masks neurons.
        All 0s get converted into 1s.
        Randomly sample the same amount of 1s to convert to 0s. The number of 0s is the same as the original mask.

        E.G.
        >>> mask = [[1, 1, 1, 1], 
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0]]
        >>> inv_mask = inverse_mask(mask)
        >>> print(inv_mask)

        0s -> 1s. Randomly sample same amount of 0s at where used to be 1s.
        [[0, 1, 1, 1], 
         [0, 0, 1, 1],
         [1, 0, 1, 1],
         [1, 0, 1, 1]]

    '''

    new_mask = torch.ones_like(mask)            # 用于创建一个与已知 tensor 形状相同的 tensor; mask:12*3072
    num_zeros = mask.numel() - mask.sum()       # 返回数组中元素的个数 -  可迭代对象的个数
    # print(num_zeros)
    idx_zero = (mask == 0).nonzero()            # nonzero()返回输入数组中非零元素的索引
    idx_one = (mask == 1).nonzero()
    
    # Replace original 1s with 0s, same count
    i = torch.randperm(idx_one.size(0))[:num_zeros]     # torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列
    # print(idx_one.size(0))
    new_idx_zero = idx_one[i]
    new_mask[new_idx_zero[:, 0], new_idx_zero[:, 1]] = 0
    
    assert (mask[idx_zero[:, 0], idx_zero[:, 1]] * new_mask[idx_zero[:, 0], idx_zero[:, 1]]).sum() == 0
    assert (mask[new_idx_zero[:, 0], new_idx_zero[:, 1]] * new_mask[new_idx_zero[:, 0], new_idx_zero[:, 1]]).sum() == 0    
    
    return new_mask

def gen_random_mask(mask):
    '''
    Randomly sample the same amount of 1s to convert to 0s. The number of 0s is the same as the original mask.
    '''
    num_zeros = mask.numel() - mask.sum()       # 0元素的个数 = 返回数组中元素的个数(12*3072) -  可迭代对象的个数(1的个数)
    
    # Randomly assign same amount of 1s and 0s
    i = torch.randperm(mask.numel())[:int(num_zeros)] # torch.randperm(n)：将0~n-1（包括0和n-1）随机打乱后获得的数字序列
    # print(i)
    random_mask  = np.ones(mask.numel())
    for j in i:
        random_mask[j] = 0 
    random_mask = torch.Tensor(random_mask)
    random_mask = random_mask.reshape(mask.shape)
    # print(random_mask)
     
    return random_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CSV_Path', type=str, default='./neuron_rank/neuron_rank_by_searchlight_RSA_14property.csv',
            help='Path that stores neuron rank')
    parser.add_argument('--mode', type=str, default='14Properties',      # 3Spaces
            help='Choose whether to use 3 Spaces or 14 Properties')
    parser.add_argument('--mask_method', type=str, default='BINARY_MASK',
            help='Choose mask neuron method:BINARY_MASK or FLOAT_MASK')
    parser.add_argument('--PKL_File', type=str, default='./masks/RSA_14property_top_500_6000.pkl',
            help='Path that stores masks')
    
    parser.add_argument('--thresholds', type=list, default=[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000])
    
    if not os.path.exists('./masks'):
        os.makedirs('./masks')
    
    args = parser.parse_args()
    
    # Number of neurons to be masked
    thresholds = args.thresholds
    
    if args.mode == '3Spaces':
        # 3 spaces
        cols = ['Affective.Space', 'Basic.Emotions.Space', 'Appraisal.Space']
    elif args.mode == '14Properties':    
        # 14 properties
        cols = ['arousal', 'valence', 'happy', 'anger', 'sad', 'fear', 'surprise',
            'disgust', 'control', 'fairness', 'self-related', 'other-related',
            'expectedness', 'non-novelty', 'all_1'][:-1]
    
    # The ranking of neurons
    score = pd.read_csv(f'{args.CSV_Path}')
    print('score shape: ',score.shape)
    
    masks = {}
    for feature in range(0,len(cols)):
        for k in thresholds:
            all_idx = (pd.DataFrame(score, columns=[cols[feature]])).values      # Get a column (1 property) of data
            all_idx = (all_idx.reshape(1,12*3072))[0]
            # print("all_idx: \t",all_idx)
            all_idx = np.array(np.argsort(np.abs(all_idx)))     # Neuron sorted by importance (1 is the most important)
            remove_idx = all_idx[:k]                            # Get the top k most important neuron index
            # print("remove_idx:\t", remove_idx)
            
            # Method 1: 0&1 mask (default)
            if args.mask_method == 'BINARY_MASK':
                abs_mask = torch.ones(12*3072, dtype=torch.int8)
                abs_mask[remove_idx] = 0
                abs_mask = abs_mask.reshape(12, 3072)
                
            # Method 2: Float mask
            elif args.mask_method == 'FLOAT_MASK':
                abs_mask = np.ones(12*3072)
                abs_mask = np.float32(abs_mask)
                # print(abs_mask)
                randomMask = np.random.random(size=(k))     # Return k random floats in the half-open interval [0.0, 1.0)
                # print(randomMask)
                abs_mask[remove_idx] = randomMask
                abs_mask = torch.from_numpy(abs_mask)       # Turn numpy to tensor
                abs_mask = abs_mask.reshape(12, 3072)       # Reshape tensor
            
            masks[f'{cols[feature]}_top{k}'] = abs_mask
        print(f"{cols[feature]}")
         
    # Generate random mask
    inverse_masks = {}
    for n, m in masks.items():
        inverse_masks[n+'_random'] = gen_random_mask(m)

    masks.update(inverse_masks)
    # print(masks)
    
    print('Number of masks:', len(masks.keys()))
    
    with open(f'{args.PKL_File}', 'wb') as f:
        pickle.dump(masks, f)       # save masks
    print("------------------------------------------")
                
if __name__ == "__main__":
    main()
