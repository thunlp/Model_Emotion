import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from .dataset import EmotionDataset
from .utils import decorate


class ModelEmotionTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = EmotionDataset(self.args.sentiment, self.tokenizer, self.args.max_source_length, self.args.seed, 'train')

        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=self._train_batch_size,
            shuffle=True)

        return train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = EmotionDataset(self.args.sentiment, self.tokenizer, self.args.max_source_length, self.args.seed, 'dev')

        validation_dataloader = DataLoader(
            dataset=eval_dataset, 
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False) 

        return validation_dataloader

    @torch.no_grad()
    def get_prompt_emb(self):
        prompt_emb = self.model.roberta.embeddings.prompt_embeddings.weight

        return prompt_emb.detach().cpu()

    def train_prompt(self, **kwargs):
        self.args.output_dir = os.path.join(self.args.out_dir_root, 'prompt_emb')
        os.makedirs(self.args.output_dir, exist_ok=True)

        return super().train(**kwargs)
        
    def eval_prompt(self):
        r"""Evaluate a prompt"""

        self.args.output_dir = os.path.join(self.args.out_dir_root, 'prompt_emb')
        os.makedirs(self.args.output_dir, exist_ok=True)

        metrics = self.evaluate()
        self.save_metrics("eval_prompt", metrics)

        return metrics

    def activated_neuron(self, layers=None):
        r"""Get activated neuron."""

        model = self.model
        model.eval()
        self._move_model_to_device(model, self.args.device)
        num_layers = model.config.num_hidden_layers
        
        outputs = [[] for _ in range(num_layers)]
        def save_ppt_outputs1_hook(n):
            def fn(_,__,output):
                outputs[n].append(output.detach().to('cpu'))
            return fn

        hook_handles = []
        for n in range(num_layers):
            hd = model.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))
            hook_handles.append(hd)

        inputs = self.tokenizer([self.tokenizer.mask_token], return_tensors='pt', add_special_tokens=False)#.to('cpu')
        inputs = self._prepare_inputs(inputs)
        _ = model(**inputs)

        for k in range(num_layers):
            outputs[k] = torch.cat(outputs[k])

        outputs = torch.stack(outputs)
        outputs = outputs[:,:1,:1,:]    # Get the output of the <mask> position

        if layers is not None:
            outputs = outputs[torch.tensor(layers)]
            num_layers = len(layers)
        
        outputs = outputs.view(num_layers, -1)

        # Active neuron before ReLU
        torch.save(outputs, os.path.join(self.args.out_dir_root, 'activated_neuron_before_relu.pt'))

        # Active neuron after ReLU
        neuron_after_relu = (outputs > 0).int()
        torch.save(neuron_after_relu, os.path.join(self.args.out_dir_root, 'activated_neuron_after_relu.pt'))

        # Remove hook of masking neurons
        _ = [hd.remove() for hd in hook_handles]

        return outputs, neuron_after_relu

    def mask_activated_neuron(self, layers=None, ratio=0.2):

        model = self.model
        self._move_model_to_device(model, self.args.device)
        num_layers = model.config.num_hidden_layers

        neuron = torch.load(os.path.join(self.args.out_dir_root, 'activated_neuron_before_relu.pt'))
        original_shape = neuron.shape
        neuron = neuron.reshape(-1)
        mask = torch.ones_like(neuron)
        idx = torch.argsort(neuron, descending=True)
        idx = idx[:int(ratio * len(idx))]
        mask[idx] = 0
        mask = mask.view(original_shape)

        def save_ppt_outputs1_hook(n):
            def fn(_,__,output):
                output = output * mask[n].to('cuda')
                return output
            return fn

        if layers is None:
            layers = range(num_layers)

        hook_handles = []
        for n in layers:
            hd = model.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))
            hook_handles.append(hd)

        self.model = model
        eval_results = self.eval_prompt()

        # Remove hook of masking neurons
        _ = [hd.remove() for hd in hook_handles]

        return eval_results, mask
        
    def plot_neuron(self, **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(40, 20))
        neuron = torch.load(os.path.join(self.args.out_dir_root, 'activated_neuron_before_relu.pt'))
        sns.heatmap(neuron.numpy(), cmap='Reds', **kwargs)
        plt.xlabel('Neuron')
        plt.ylabel('Layer')

    def set_active_state_dict(self, module: nn.Module, includes=['prompt_model.template.soft_embeds']):
        r"""modify the state_dict function of the model (by default, the backbone model) to return only the tunable part.
        Args:
            module (:obj:`nn.Module`): The module modified. The modification is in-place.
        """
        
        def _caller(_org_func, includes,  *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n  in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict
        includes = includes          # use excludes will have trouble when the model have shared weights
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning("The forward function might have been wrapped by a decorator, is it intended? Do you freeze the parameters twice?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(includes,), kwsyntax=True)


class Trainer:
    def __init__(self, args=None, logger=None, tokenizer=None, model=None, optimizer=None, scheduler=None, 
                 num_labels=None, train_loader=None, val_loader=None, test_loader=None):
        self.logger = logger
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_labels = num_labels

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_pred(self, logits):
        _, pred = torch.max(logits.view(-1, self.num_labels), 1)
        return pred

    def multilabel_accuracy(self, logits, label):
        return accuracy_score(label.numpy(), logits.round().numpy())

    def accuracy(self, pred, label):
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def train(self):
        self.model.train()
        valid_acc = []
        best_acc = 0
        best_epoch = 0
        for epoch in trange(self.args.epochs, desc='Epoch'):
            it = 0
            iter_loss = 0.0
            iter_acc = 0.0
            for sentence, label in tqdm(self.train_loader, desc='Iteration'):#, disable=True):
                inputs = self.tokenizer(
                    sentence, max_length=self.args.max_length, padding='max_length', truncation=True, return_tensors='pt')
                inputs = inputs.to('cuda')
                label = label.to('cuda')
                outputs = self.model(**inputs, labels=label)

                if hasattr(self.model, 'module'):
                    loss = outputs.loss.mean()
                else:
                    loss = outputs.loss
                
                pred = self.get_pred(outputs.logits)
                acc = self.accuracy(pred, label)
                # Multilabel
                # self.multilabel_accuracy(outputs.logits.detach().cpu(), label.cpu())

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                it += 1
                iter_loss += loss.item()
                iter_acc += acc.item()

            self.logger.info(f'Epoch: {epoch:2} | loss: {iter_loss / it:2.6f} | acc: {iter_acc / it:.2%}')

            acc_valid = self.eval(use_valid='valid')
            self.model.train()
            
            valid_acc.append(acc_valid)

            # Save best checkpoint
            if acc_valid > best_acc:
                best_acc = acc_valid
                best_ckpt = {
                    'best_prompt': copy.deepcopy(self.model.roberta.embeddings.prompt_embedding.weight).detach().cpu(),
                    'best_acc': best_acc,
                    'epoch': epoch,
                }
                torch.save(best_ckpt, f'checkpoint/{self.args.sentiment}-{self.args.seed}.pt')

        # Save final checkpoint
        best_ckpt['final_prompt'] = self.model.roberta.embeddings.prompt_embedding.weight.detach().cpu()
        best_ckpt['final_acc'] = acc_valid
        # torch.save(best_ckpt, f'checkpoint/{self.sentiment}-{self.args.seed}.json')
        torch.save(best_ckpt, f'checkpoint/{self.args.sentiment}-{self.args.seed}.pt')

        return best_acc
        
    @torch.no_grad()
    def eval(self, use_valid=True):
        self.model.eval()

        if use_valid:
            eval_loader = self.val_loader
            log_prefix = 'Valid'
        else:
            eval_loader = self.test_loader
            log_prefix = 'Test'

        it = 0
        total_loss = 0
        total_acc = 0
        all_pred = []
        all_label = []
        for sentence, label in tqdm(eval_loader, desc='Eval'):
            inputs = self.tokenizer(
                sentence, max_length=self.args.max_length, padding='max_length', truncation=True, return_tensors='pt')
            
            inputs = inputs.to('cuda')
            label = label.to('cuda')
            outputs = self.model(**inputs, labels=label)

            if hasattr(self.model, 'module'):
                loss = outputs.loss.mean()
            else:
                loss = outputs.loss

            pred = self.get_pred(outputs.logits)
            all_pred.append(pred.cpu())
            all_label.append(label.cpu())

            total_loss += loss.item()
            it += 1

        all_pred = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        acc = self.accuracy(all_pred, all_label)
        
        self.logger.info(f'[EVAL] {log_prefix} | loss: {total_loss / it:.2f} | acc: {acc:.2%}')
        
        return acc.item()
