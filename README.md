<h1><img src="docs/logo.png" height="28px" /> OpenNeuron</h1>

**Activate neurons in RoBERTa**

<p align="center">
	<a href="#overview">Overview</a> 
    • <a href="#documentation">Documentation</a> 
    • <a href="#installation">Installation</a> 
    • <a href="#usage">Usage</a> 
    • <a href="#result">Result</a> 
    • <a href="./README-ZH.md" target="_blank">简体中文</a>
</p>

<p align="center">
<a href='https://bmtrain.readthedocs.io/en/latest/?badge=latest'>
<img src='https://readthedocs.org/projects/bmtrain/badge/?version=latest' alt='Documentation Status'>
</a>
<a href="https://github.com/OpenBMB/BMTrain/releases">
<img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/OpenBMB/BMTrain?include_prereleases">
</a>
<a href="https://github.com/OpenBMB/BMTrain/blob/main/LICENSE">
<img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMTrain">
</a>
</p>

<div id="overview"></div>

## Overview

OpenNeuron is a method that uses emotional prompts to obtain the activated neurons in the pre-train language model (Roberta) to figure out whether the model can pick up human emotion by studying large amounts of texts.

<div id="documentation"></div>

## Documentation
Our [documentation](https://bmtrain.readthedocs.io/en/latest/index.html) provides more information about the package.

<div id="installation"></div>

## Installation

- From pip （recommend） : ``pip install -r requirements.txt``


<div id="usage"></div>

## Usage

### Step 1: Train Emotional Prompt

Before activating neurons in the pretrain language model, we need emotional prompts to change the attention distribution in the model. 

We use **[GoEmotions](https://doi.org/10.48550/arXiv.2005.00547)** as the prompt training dataset. It is the largest manually annotated dataset of 58k English Reddit comments, labeled for 27 emotion categories or Neutral. It has been proved that it can generalize well to other domains and different emotion taxonomies.

In ``train.py``, we define **27 Emotional tasks** according to the labels of **GoEmotions**. Each one of the tasks was trained by **12 Random Seeds** (Data used for training will slightly differ when using different random seeds).

You can train all 27 prompts by using `./train.sh` or `python train.py`.  

After training, you will get **27×12** checkpoints.




### Step 2: Modify RoBERTa

To load prompts in RoBERTa, we need to make some changes in the original code of ``transformers.models.roberta.modeling_roberta``. 

* `RobertaForMaskedLM` -> `framework.roberta.RobertaForMaskedLMPrompt`
* `RobertaLMHead` -> `framework.roberta.RobertaLMHeadWarp` 
* `RobertaEmbeddings` -> `framework.roberta.RobertaEmbeddingsWarp`
* `RobertaModel` -> `framework.roberta.RobertaModelWarp`



### Step 3: Activate Neurons in RoBERTa

We use the special token '**\<s>**' to activate the neurons in RoBERTa.

RoBERTa has 12 following layers.  ( You can view the complete model by using `print(model)` )

**The Activated Neurons** are the output of the intermediate layer between two Feed-Forward Networks.

```
RobertaLayer(
       (attention): RobertaAttention(...)
       (intermediate): RobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
       )
       (output): RobertaOutput(...)
     )
```

You can get the activated neurons by running `python get_active_neuron.py`



### Step 4: Sort Activated Neurons by RSA Seachlight







### Step 5: Generate Masks Based on Importance of Neurons

After getting the neurons sorted out, we will generate masks for the top k important neurons, and we will also generate random masks for setting the baseline.

In the `gen_mask.py`, we provide several methods for masking the neurons and generate random masks.

**Methods of masking neurons**

* 0&1 mask (Use 0 to represent the masked neurons, and 1 to represent the unmasked neurons)
* Float mask (Use random floats in the half-open interval [0.0, 1.0) to represent the mask neurons)

You can get the masks by running `python gen_mask.py`

```shell
python gen_mask.py \
--CSV_Path='./neuron_rank/neuron_rank_by_searchlight_RSA_14property.csv' \
--mode='14Properties' \
--mask_method=1 \
--PKL_File='./masks/RSA_14property_top_500_6000.pkl'
```




### Step 6: Evaluate Masked Neurons

We evaluate the masked neurons using the modified RoBERTa model loaded with the prompt to do the binary classification task. Each time the forward layer of the intermediate module is called, we multiply the corresponding mask by the layer's output.

You can evaluate  the masks by running `python eval_mask_neuron.py` or `./eval.sh`



<div id="result"></div>

## Result

### Neurons Correspond Selectively to Emotional Attributes

![Neurons Correspond Selectively to Emotional Attributes](pic/Neurons_Correspond_Selectively_to_Emotional_Attributes.png)

\* For each attribute, show the top 4000 neurons of correspondence.



### Accuracy after Masking Corresponding Neurons

 ![Acc after masking correspond neuron](pic/Acc_after_masking_correspond_neuron.png)



## Community
We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/BMTrain/blob/master/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 
- Website: 
- Weibo: 
- Twitter: 

## License
The package is released under the [Apache 2.0](https://github.com/OpenBMB/BMTrain/blob/master/LICENSE) License.

## Other Notes

`BMTrain` makes underlying changes to PyTorch, so if your program outputs unexpected results, you can submit information about it in an issue.

