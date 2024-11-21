<!--<h1><img src="pic/emotions_fig.png" height="28px" /> Model Emotion</h1>-->

<h1><img src="pic/human_emotion.png" height="42px" /> Model Emotion</h1>

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

Humans no doubt use language to communicate about their emotional experiences, but does language in turn help humans understand emotions, or is language just a vehicle of communication? This study used a form of artificial intelligence (AI) known as large language models (LLMs) to assess whether language-based representations of emotion causally contribute to the AI’s ability to generate inferences about the emotional meaning of novel situations. Fourteen attributes of human emotion concept representation were found to be represented by the LLM's distinct artificial neuron populations. By manipulating these attribute-related neurons, we in turn demonstrated the role of emotion concept knowledge in generative emotion inference. The attribute-specific performance deterioration was related to the importance of different attributes in human mental space. Our findings provide a proof-in-concept that even a LLM can learn about emotions in the absence of sensory-motor representations and highlight the contribution of language-derived emotion-concept knowledge for emotion inference.

This toolkit can help explore the emerging emotion concept in Large-Scale Language Models. Specifically, we can utilize this toolkit to observe the human emotion knowledge representation that emerges in large language models and supports discrete emotion inference.

<div id="news"></div>

## News

- (Future):  We will provide more user-friendly API to enable non-CS background research to play our codes.
- 2023.02.20: The version2.0 supports you to access any existing pre-trained language models o explore the humna-like emotion concepts.
- 2023.02.15: These codes support you to access RoBERTa to explore the humna-like emotion concepts.

## Note

- [Model_Emotion-1.0](https://github.com/thunlp/Model_Emotion/tree/main/Model_Emotion-1.0) provides the original codes and details to reproduce the results in the paper.
- [Model_Emotion-2.0-latest](https://github.com/thunlp/Model_Emotion/tree/main) refactors the Model_Emotion-1.0 and provides more user-friendly codes for users. In this `README.md`, we mainly demostrate the usage of the `Model_Emotion-2.0-latest`.

<!--
## Setups
- torch==1.12.0+cu118
- python>=3.7

You could refer environment.yml for more details.
-->

## Setups

- python==3.8.0

We recommend to create a new Anaconda environment to manage the required packages.

```bash
conda create -n openNeuron python=3.8
conda activate openNeuron
bash requirements.sh
```

If the system shows the error about torch. Please find the proper torch that can match your CPUs or GPUs.

<!--`ERROR: Invalid requirement: 'torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0' (from line 10 of requirements.txt)`. Please manually run `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`.-->

User can also directly create the environment via `Model_Emotion-2.0-latest/environment.yml`.

```bash
conda env create -f environment.yml
```

<div id="usage"></div>

## Usage

### Quick Example

Go to `Model_Emotion-2.0-latest` and run `train.sh`

```bash
BACKBONE=roberta-base
PROMPTLEN=100


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
    python train.py \
    --output_dir outputs \
    --backbone $BACKBONE \
    --prompt_len $PROMPTLEN \
    --sentiment=$sentiment \
    --seed $seed
  done
done
```

- `output_dir`: Required. Output directory for checkpoint, results, etc. E.g. 'outputs'.
- `backbone`: The PLM to use. E.g. 'roberta-base'.
- `prompt_len`: The number of soft prompt tokens. E.g. 100.
- `sentiment`: The emotion to use. See the below section for more details. E.g. 'surprise'.
- `max_source_length`: The maximum length of input. E.g. 128.

This script train 27 emotion prompts with 12 different seeds. The `train.py` includes prompt tuning , evaluation, and activated neuron analysis.

```python
from transformers import tokenizer
from framework import RobertaForMaskedLMPrompt
from framework.training_args import ModelEmotionArguments, RemainArgHfArgumentParser


# Training config
#args = ModelEmotionArguments(
#  output_dir='outputs',
#  backbone='roberta-base',
#  prompt_len=100,
#  sentiment='surprise'
#)
parser = RemainArgHfArgumentParser(ModelEmotionArguments)
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    json_file=os.path.abspath(sys.argv[1])
    args = parser.parse_json_file(json_file, return_remaining_args=True)[0] #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    args = parser.parse_args_into_dataclasses()[0]


tokenizer = AutoTokenizer.from_pretrained(args.backbone, max_length=args.max_source_length, use_fast=False)
model = RobertaForMaskedLMPrompt.from_pretrained(args.backbone, prompt_len=args.prompt_len, num_labels=2)

# Initialize trainer
trainer = ModelEmotionTrainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

# Train prompt
train_result = trainer.train_prompt()
eval_result = trainer.eval_prompt()

# Find activated neuron
neuron_before_relu, neuron_after_relu = trainer.activated_neuron()

# Mask neurons for evaluation
eval_results, mask = trainer.mask_activated_neuron()

```

### Detailed Usage

#### **Step 1: Arguments**

Before running any experiments, we need to set the correct arguments. We extend the original `transformers.TrainingArguments`. Please refer to `transformers` documentation and `Model_Emotion-2.0-latest/framework/training_args.py` for the full list of arguments.
For most arguments, the default values work fine. Here are some important parameters you may want to change. The following parameters are passed from the above `train.sh`.

```python
# Main Training config
#args = ModelEmotionArguments(
#  output_dir='outputs',
#  backbone='roberta-base',
#  prompt_len=100,
#  sentiment='surprise'
#)
parser = RemainArgHfArgumentParser(ModelEmotionArguments)
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    json_file=os.path.abspath(sys.argv[1])
    args = parser.parse_json_file(json_file, return_remaining_args=True)[0] #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    args = parser.parse_args_into_dataclasses()[0]


```

#### **Step 2: Train Emotional Prompt**

Before activating neurons in the PLMs, we need emotional prompts to change the attention distribution in the model. We use **[GoEmotions](https://doi.org/10.48550/arXiv.2005.00547)** as the prompt training dataset. It is the largest manually annotated dataset of 58k English Reddit comments, labeled for 27 emotion categories or Neutral. It has been proved that it can generalize well to other domains and different emotion taxonomies. We define **27 Emotional tasks** according to the labels of **GoEmotions**. Each one of the tasks was trained by **12 Random Seeds** (Data used for training will slightly differ when using different random seeds).

You can train prompt for a given emotion by running the following function:

```python
from transformers import tokenizer
from framework import RobertaForMaskedLMPrompt

tokenizer = AutoTokenizer.from_pretrained(args.backbone, max_length=args.max_source_length, use_fast=False)
model = RobertaForMaskedLMPrompt.from_pretrained(args.backbone, prompt_len=args.prompt_len, num_labels=2)

# Initialize trainer
trainer = ModelEmotionTrainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

# Train
train_result = trainer.train_prompt()
eval_result = trainer.eval_prompt()
```

#### **Step 3: Activate Neurons in Model**

We use the special token '**\<s>**' to activate the neurons in RoBERTa.

The **activated neurons** of the model are the output of the `intermediate` layer between two Feed-Forward Networks. You can get the activated neurons `before` or `after` the activation function `ReLU` by running:

```python
neuron_before_relu, neuron_after_relu = trainer.activated_neuron()
```

#### **Step 4: Sort Activated Neurons by RSA Seachlight**

**Methods for Sorting Neurons**

* `14Properties` [ DEFAULT ]

```
Excute `Code/analysis_RSA_by_neuron.R`
```

#### **Step 5: Generate Masks Based on the Importance of Neurons**

After getting the neurons sorted out, we will generate masks for the top k important neurons. We will also generate random masks for setting the baseline.

In the `ModelEmotionTrainer.mask_activated_neuron`, we provide several methods for generating random masks and evaluating model's performance with the generated mask.

**Methods for Masking Neurons**

* `BINARY_MASK` [ DEFAULT ]

  Use 0 to represent the masked neurons, and 1 to represent the unmasked neurons.
* `FLOAT_MASK`

  Use random floats in the half-open interval [0.0, 1.0) to represent the mask neurons

You can get the masks by running the following script:

```python
eval_results, mask = trainer.mask_activated_neuron()
```

<!-- ### **Step 6: Evaluate Masked Neurons**

We evaluate the masked neurons using the modified RoBERTa model loaded with the prompt to do the binary classification task. Each time the forward layer of the intermediate module is called, we multiply the corresponding mask by the layer's output.

You can evaluate  the masks by running the default script `./eval.sh` in bash or :
```bash
python eval_mask_neuron.py \
    --sentiment='joy' \
    --random_seed=1 \
    --maskPath='./masks/RSA_14property_top_500_6000.pkl' \
    --resultPath='./eval_mask_neuron_result/RSA_14property_top_500_6000'
``` -->

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

## Citation

Please cite our paper if it is helpful to your work!

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.09582,
  doi = {10.48550/ARXIV.2302.09582},
  url = {https://arxiv.org/abs/2302.09582},
  author = {Li, Ming and Su, Yusheng and Huang, Hsiu-Yuan and Cheng, Jiali and Hu, Xin and Zhang, Xinmiao and Wang, Huadong and Qin, Yujia and Wang, Xiaozhi and Liu, Zhiyuan and Zhang, Dan},
  title = {Human Emotion Knowledge Representation Emerges in Large Language Model and Supports Discrete Emotion Inference},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Contact

- [Yusheng Su](https://yushengsu-thu.github.io/): yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn
- Ming Li: liming16@tsinghua.org.cn
- Hsiu-Yuan Huang: hsiuyuanhuang.pku@gmail.com
