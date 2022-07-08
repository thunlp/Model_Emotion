import copy
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import os

PKL_Path = './masks_linearModel/beta_3space_500_6000_1.pkl'
CSV_Path = './LiMing_Mask/neuron_rank_by_linear_regression_3space.csv'

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
    print(random_mask)
     
    return random_mask

def main():
    # Number of neurons to be masked
    thresholds = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
    
    # 3 spaces
    cols = ['Affective.Space', 'Basic.Emotions.Space', 'Appraisal.Space']
    
    # # 14 properties
    # cols = ['arousal', 'valence', 'happy', 'anger', 'sad', 'fear', 'surprise',
    #     'disgust', 'control', 'fairness', 'self-related', 'other-related',
    #     'expectedness', 'non-novelty', 'all_1'][:-1]
    
    # The ranking of neurons
    score = pd.read_csv(f'{CSV_Path}')
    print('score: ',score.shape)
    
    masks = {}
    for feature in range(0,len(cols)):
        for k in thresholds:
            print(f"{cols[feature]}\t{k}")
            all_idx = (pd.DataFrame(score, columns=[cols[feature]])).values      # 获取一列（一个property排序）数据
            all_idx = (all_idx.reshape(1,12*3072))[0]
            # print("all_idx: \t",all_idx)
            all_idx = np.array(np.argsort(np.abs(all_idx)))     # Neuron sorted by importance (1 is the most important)
            remove_idx = all_idx[:k]                            # Get the top k most important neuron index
            # print("remove_idx:\t", remove_idx)
            
            # Method 1: 0&1 mask
            abs_mask = torch.ones(12*3072, dtype=torch.int8)
            abs_mask[remove_idx] = 0
            abs_mask = abs_mask.reshape(12, 3072)
            
            # # Method 2: Float mask
            # abs_mask = np.ones(12*3072)
            # abs_mask = np.float32(abs_mask)
            # # print(abs_mask)
            # randomMask = np.random.random(size=(k))     # Return k random floats in the half-open interval [0.0, 1.0)
            # print(randomMask)
            # abs_mask[remove_idx] = randomMask
            # abs_mask = torch.from_numpy(abs_mask)       # Turn numpy to tensor
            # abs_mask = abs_mask.reshape(12, 3072)       # Reshape tensor
            
            # print(abs_mask)
            # if feature>=14:
            #     continue
            masks[f'{cols[feature]}_top{k}'] = abs_mask
            
    # print(masks)
    inverse_masks = {}
    for n, m in masks.items():
        print(n)
        # generate random mask
        inverse_masks[n+'_random'] = gen_random_mask(m)

    masks.update(inverse_masks)
    # print(masks)

    print('Number of masks:', len(masks.keys()))
    with open(f'{PKL_Path}', 'wb') as f:
        pickle.dump(masks, f)       # save masks
    print("------------------------------------------")
                
if __name__ == "__main__":
    main()
