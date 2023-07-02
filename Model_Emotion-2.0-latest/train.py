import os
import sys
import logging
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AdamW,
    set_seed,
    TrainingArguments,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction
)
from datasets import load_metric

from framework import RobertaForMaskedLMPrompt, ModelEmotionTrainer
from framework.training_args import ModelEmotionArguments, RemainArgHfArgumentParser


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    parser = RemainArgHfArgumentParser(ModelEmotionArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        json_file=os.path.abspath(sys.argv[1])
        args = parser.parse_json_file(json_file, return_remaining_args=True)[0] #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]

    #set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, max_length=args.max_source_length, use_fast=False)

    # Model
    num_labels = 2
    init_ids = torch.tensor([tokenizer.mask_token_id] * args.prompt_len)
    model = RobertaForMaskedLMPrompt.from_pretrained(
            args.backbone, prompt_len=args.prompt_len, init_ids=init_ids, num_labels=num_labels)

    print('Parameters that requires grad:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)


    metric = load_metric("framework/glue_metrics.py")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        result["combined_score"] = np.mean(list(result.values())).item()

        return result

    # Train
    trainer = ModelEmotionTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    # Train
    train_result = trainer.train_prompt()
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")
    metrics = trainer.eval_prompt()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    results['eval'] = metrics

    print(results)

    trainer.activated_neuron()

    trainer.mask_activated_neuron()

    trainer.plot_neuron()


if __name__ == "__main__":
    main()
