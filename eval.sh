#! /bin/bash 

for sentiment in 'realization' 'surprise' 'admiration' 'gratitude' 'optimism' 'approval' \
                 'pride' 'excitement' 'joy' 'love' 'amusement' 'caring' \
                 'relief' 'curiosity' 'desire' 'disgust' 'disapproval' 'anger' \
                 'annoyance' 'confusion' 'fear' 'disappointment' 'grief' 'sadness' \
                 'nervousness' 'embarrassment' 'remorse'
do 
  for seed in {1,3,5,7,9,11,13,15,17,19,42,100}
  do
    echo "sentiment=$sentiment,random_seed=$seed"
    CUDA_VISIBLE_DEVICES=0 \
    python eval_mask_neuron.py \
    --sentiment=$sentiment \
    --random_seed=$seed \
    --maskPath='./masks_linearModel/RSA_14property_500_6000_1.pkl' \
    --resultPath="./neuron/new_61/RSA_14property_500_6000_1"
  done
done
