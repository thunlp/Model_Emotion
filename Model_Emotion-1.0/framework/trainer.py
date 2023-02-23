import os
import copy
import torch
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score


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
                torch.save(best_ckpt, f'checkpoint/{self.args.sentiment}-{self.args.random_seed}.pt')

        # Save final checkpoint
        best_ckpt['final_prompt'] = self.model.roberta.embeddings.prompt_embedding.weight.detach().cpu()
        best_ckpt['final_acc'] = acc_valid
        # torch.save(best_ckpt, f'checkpoint/{self.sentiment}-{self.args.random_seed}.json')
        torch.save(best_ckpt, f'checkpoint/{self.args.sentiment}-{self.args.random_seed}.pt')

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
