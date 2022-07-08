## GoEmotions

To train prompt tuning model
```
    python train.py --sentiment $SENTIMENT --random_seed $SEED
```

To get the activated neurons by each prompt
``` 
    python get_active_neuron.py
```

To train a linear model to predict activated neurons
```
    python train_linear_model.py
```

To generage masks
```
    python gen_mask.py
```

To run experiments with the masks
```
    python eval_mask_neuron.py
```
