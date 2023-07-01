<!--<h1><img src="pic/emotions_fig.png" height="28px" /> Model Emotion</h1>-->

<h1><img src="Prompt-Transferability-1.0/pic/human_emotion.png" height="42px" /> Model Emotion</h1>


<p align="center">
	<a href="#overview">Overview</a> 
    • <a href="#news">News</a> 
    • <a href="#installation">Installation</a> 
    • <a href="#usage">Usage</a> 
    • <a href="#result">Result</a> 
    • <a href="./README-ZH.md" target="_blank">简体中文</a>
</p>


<p align="center">
	
<!--	
<a href='https://bmtrain.readthedocs.io/en/latest/?badge=latest'>
<img src='https://readthedocs.org/projects/bmtrain/badge/?version=latest' alt='Documentation Status'>
</a>
<a href="https://github.com/OpenBMB/BMTrain/releases">
<img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/OpenBMB/BMTrain?include_prereleases">
</a>
<a href="https://github.com/OpenBMB/BMTrain/blob/main/LICENSE">
<img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMTrain">
</a>
-->
	
<a href="https://colab.research.google.com/drive/1VCSIDaX_pgkrSjzouaNH14D8Fo7G9GBz?usp=sharing">
<img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>
</p>


<p align="center">
<img src="pic/cover.jpg" height="256px" />
<!--![Human Emotion Knowledge Representation Emerges in Large Language Models and Supports Discrete Emotion Inference![image](pic/cover.png)-->
</p>

<div id="overview"></div>

## Overview
This is the toolkit to explore the emerging emotion concept in Large-Scale Language Models. Specifically, we can utilize this toolkit to observe the human emotion knowledge representation emerges in large language models and supports discrete emotion inference.

<div id="news"></div>

## News
- 2023.02.15 Support RoBERTa.

- 2023.03.20 More supported models coming soon...

<div id="installation"></div>

## Installation
To set up the environment, run the following code in bash:
```bash
conda create -n openNeuron python=3.8
conda activate openNeuron
pip install -r requirements.txt
```


<div id="usage"></div>

## Usage

### **Step 1: Train Emotional Prompt**

Before activating neurons in the PLMs, we need emotional prompts to change the attention distribution in the model. 

We use **[GoEmotions](https://doi.org/10.48550/arXiv.2005.00547)** as the prompt training dataset. It is the largest manually annotated dataset of 58k English Reddit comments, labeled for 27 emotion categories or Neutral. It has been proved that it can generalize well to other domains and different emotion taxonomies.

In ``train.py``, we define **27 Emotional tasks** according to the labels of **GoEmotions**. Each one of the tasks was trained by **12 Random Seeds** (Data used for training will slightly differ when using different random seeds).

You can train all 27 prompts by running the default script `./train.sh` in bash or the following script: 
```bash
python train.py \
    --sentiment='joy' \
    --random_seed=1 \
    --epochs=10 \
    --train_batch_size=16 \
    --eval_batch_size=32
```


### **Step 2: Activate Neurons in Model**

We use the special token '**\<s>**' to activate the neurons in RoBERTa.

The **activated neurons** of the model are the output of the `intermediate` layer between two Feed-Forward Networks. You can get the activated neurons `before` or `after` the activation function `ReLU` by running:
```bash
python get_active_neuron.py --activation_mode='before_relu'
python get_active_neuron.py --activation_mode='after_relu'
```

### **Step 4: Sort Activated Neurons by RSA Seachlight**

**Methods for Sorting Neurons**

* `14Properties` [ DEFAULT ]

  TODO
* `3Spaces`: Affective Space, Basic Emotions Space, Appraisal Space

  TODO


### **Step 5: Generate Masks Based on the Importance of Neurons**

After getting the neurons sorted out, we will generate masks for the top k important neurons. We will also generate random masks for setting the baseline.

In the `gen_mask.py`, we provide several methods for masking the neurons and generating random masks.

**Methods for Masking Neurons**

* `BINARY_MASK` [ DEFAULT ]

  Use 0 to represent the masked neurons, and 1 to represent the unmasked neurons.
* `FLOAT_MASK`

  Use random floats in the half-open interval [0.0, 1.0) to represent the mask neurons

You can get the masks by running the following script:

```bash
python gen_mask.py \
--CSV_Path='./neuron_rank/neuron_rank_by_searchlight_RSA_14property.csv' \
--mode='14Properties' \
--mask_method='BINARY_MASK' \
--PKL_File='./masks/RSA_14property_top_500_6000.pkl' \
--thresholds=[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
```



### **Step 6: Evaluate Masked Neurons**

We evaluate the masked neurons using the modified RoBERTa model loaded with the prompt to do the binary classification task. Each time the forward layer of the intermediate module is called, we multiply the corresponding mask by the layer's output.

You can evaluate  the masks by running the default script `./eval.sh` in bash or :
```bash
python eval_mask_neuron.py \
    --sentiment='joy' \
    --random_seed=1 \
    --maskPath='./masks/RSA_14property_top_500_6000.pkl' \
    --resultPath='./eval_mask_neuron_result/RSA_14property_top_500_6000'
```


<div id="result"></div>

## Result

### Neurons Correspond Selectively to Emotional Attributes

![Neurons Correspond Selectively to Emotional Attributes](pic/Neurons_Correspond_Selectively_to_Emotional_Attributes.png)

\* For each attribute, show the top 4000 neurons of correspondence.



### Accuracy after Masking Corresponding Neurons

 ![Acc after masking correspond neuron](pic/Acc_after_masking_correspond_neuron.png)


<!--
## Community
We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/BMTrain/blob/master/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 
- Website: 
- Weibo: 
- Twitter: 
-->

<!--
## License
The package is released under the [Apache 2.0](https://github.com/OpenBMB/BMTrain/blob/master/LICENSE) License.
-->

<!--
## Other Notes

`BMTrain` makes underlying changes to PyTorch, so if your program outputs unexpected results, you can submit information about it in an issue.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VCSIDaX_pgkrSjzouaNH14D8Fo7G9GBz?usp=sharing)
-->

## Contact
- [Yusheng Su](https://yushengsu-thu.github.io/): yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn

- Ming Li: liming16@tsinghua.org.cn

- Xiuyuan Huang: 
