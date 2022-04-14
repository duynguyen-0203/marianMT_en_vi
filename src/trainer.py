import csv
from datetime import datetime
import logging
import math
import os
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers
from torch.optim import AdamW
from transformers import MarianTokenizer, MarianMTModel

from . import logger_utils
from . import utils
from .reader import Reader
from .entities import *
from .loss import ModelLoss
from .evaluation import Evaluator


class Trainer:
    def __init__(self, args):
        self.args = args
        self._init_logger()
        self._device = torch.device(utils.get_device())
        if utils.get_device() == 'cuda':
            self._logger.info(f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        self._tokenizer = MarianTokenizer.from_pretrained(args.model_path)
        utils.set_seed(args.seed)

    def train(self):
        args = self.args
        self._logger.info(f'Model: {args.model_name}')
        self._logger.info(f'Machine translation translates from {args.src_lang} to {args.tgt_lang}')

        # read data
        reader = Reader(self._tokenizer)
        train_dataset = reader.read(args.src_train_data, args.tgt_train_data, args.data_name, args.max_length)
        valid_dataset = reader.read(args.src_valid_data, args.tgt_valid_data, args.data_name, args.max_length)
        self._log_dataset(train_dataset, valid_dataset)
        n_train_samples = len(train_dataset)
        updates_epoch = n_train_samples // args.train_batch_size
        if args.steps_per_epoch is not None and updates_epoch > args.steps_per_epoch:
            updates_epoch = args.steps_per_epoch
        n_updates = updates_epoch * args.epochs

        self._logger.info('--------- Running training ---------')
        self._logger.info(f'Updates per epoch: {updates_epoch}')
        self._logger.info(f'Updates total: {n_updates}')

        # create model
        model = MarianMTModel.from_pretrained(args.model_path)
        model.to(self._device)
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.lr_warmup * n_updates,
                                                                 num_training_steps=n_updates)

        # create loss function
        criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self._tokenizer.pad_token_id)
        loss_calculator = ModelLoss(criterion, model, optimizer, scheduler, args.max_grad_norm)

        best_bleu = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            self._train_epoch(model, loss_calculator, scheduler, train_dataset, epoch)
            score = self._eval(model, valid_dataset, epoch)
            if score > best_bleu:
                self._logger.info(f'Best BLEU score update, from {best_bleu} to {score}, at epoch {epoch}')
                best_bleu = score
                best_epoch = epoch
                self._save_model(model, optimizer, scheduler, epoch, flag='bestModel')

        # save final model
        self._save_model(model, optimizer, scheduler, args.epochs - 1, flag='finalModel')
        self._logger.info('Finish training!!!')
        self._logger.info(f'Best model at epoch {best_epoch}')

    def _save_model(self, model, optimizer, scheduler, epoch, flag: str):
        save_path = os.path.join(self._path, flag + '.pt')
        saved_point = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(saved_point, save_path)

    def _load_model(self, model_path):
        saved_point = torch.load(model_path, map_location=self._device)

        return saved_point['model']

    def _eval(self, model, dataset, epoch=None):
        if epoch is not None:
            self._logger.info(f'Evaluate epoch {epoch}')
            desc = f'Evaluate epoch {epoch}'
        else:
            desc = 'Evaluate epoch'
        evaluator = Evaluator(dataset, self._tokenizer)
        dataset.set_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 collate_fn=self._collate_fn)

        with torch.no_grad():
            model.eval()
            total = math.ceil(len(dataset) / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc=desc):
                batch = utils.to_device(batch, self._device)
                generated_tokens = model.generate(inputs=batch['encoding'], pad_token_id=self._tokenizer.pad_token_id)
                evaluator.eval_batch(generated_tokens)

        score = evaluator.compute_scores()
        if self.args.mode == 'train':
            self._log_eval(score, epoch)

        return score['bleu']

    def eval(self):
        args = self.args
        model = self._load_model(args.saved_model_path)
        model.to(self._device)
        reader = Reader(self._tokenizer)
        dataset = reader.read(args.src_test_data, args.tgt_test_data, args.test_data_name, args.max_length)
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Test dataset: {len(dataset)} samples')

        self._logger.info('--------- Evaluation phrase ---------')
        score = self._eval(model, dataset, epoch=None)
        self._logger.info('BLEU score: {}'.format(score))

    def _train_epoch(self, model, loss_calculator, scheduler, dataset, epoch):
        self._logger.info(f'--------- EPOCH {epoch} ---------')
        dataset.set_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 collate_fn=self._collate_fn)
        model.zero_grad()
        iteration = 0
        total = len(dataset) // self.args.train_batch_size
        if self.args.steps_per_epoch is not None and total > self.args.steps_per_epoch:
            total = self.args.steps_per_epoch

        for batch in tqdm(data_loader, total=total - 1, desc=f'Train epoch {epoch}'):
            model.train()
            batch = utils.to_device(batch, self._device)
            logits = model(input_ids=batch['encoding'], attention_mask=batch['attention_mask'],
                           labels=batch['label'])['logits']
            batch_loss = loss_calculator.compute(logits, batch['label'])

            global_iteration = total * epoch + iteration
            if global_iteration % self.args.log_iter == 0:
                self._log_train(scheduler, batch_loss, epoch, iteration, global_iteration)

            iteration += 1
            if self.args.steps_per_epoch is not None and iteration == self.args.steps_per_epoch:
                break

        return iteration

    def _init_logger(self):
        time = str(datetime.now()).replace(' ', '_').replace(':', '-')
        log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s [%(levelname)-5.5s] %(message)s')
        self._logger = logging.getLogger()
        logger_utils.reset_logger(self._logger)

        if self.args.mode == 'train':
            self._path = os.path.join(self.args.save_path, time)
            self._log_path = os.path.join(self._path, 'log')
            os.makedirs(self._path, exist_ok=True)
            os.makedirs(self._log_path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self._log_path, 'all.log'))
            self._init_csv_logger()
            # init tensorboard logger
            os.makedirs(self.args.tensorboard_path, exist_ok=True)
            self._writer = SummaryWriter(os.path.join(self.args.tensorboard_path, time))
        else:
            self._eval_path = os.path.join(self.args.eval_path, time)
            os.makedirs(self._eval_path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self._eval_path, 'all.log'))

        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)
        self._logger.setLevel(logging.INFO)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        return optimizer_params

    def _collate_fn(self, batch):
        padded_batch = dict()
        keys = batch[0].keys()

        for key in keys:
            samples = [s[key] for s in batch]
            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                if key == 'encoding' or key == 'label':
                    padding = self._tokenizer.pad_token_id
                else:
                    padding = 0

                padded_batch[key] = utils.padded_stack(samples, padding=padding)

        return padded_batch

    def _log_dataset(self, train_dataset: Dataset, valid_dataset: Dataset):
        self._logger.info(f'Dataset: {self.args.data_name}')
        self._logger.info(f'Train dataset: {len(train_dataset)} samples')
        self._logger.info(f'Validation dataset: {len(valid_dataset)} samples')
        self._logger.info(f'Vocabulary size of tokenizer: {self._tokenizer.vocab_size}')

    def _log_train(self, scheduler, batch_loss, epoch, iteration, global_iteration):
        lr = scheduler.get_last_lr()
        loss = batch_loss / self.args.train_batch_size
        data = [global_iteration, epoch, iteration, loss, lr]
        logger_utils.log_csv(self._loss_csv, data)
        self._logger.info(f'Training loss at epoch {epoch}, iteration {iteration}, global iteration {global_iteration}:'
                          f' {loss}')
        self._writer.add_scalar('Training loss per iteration', loss, global_step=global_iteration)
        self._writer.add_scalar('Learning rate', lr[0], global_step=global_iteration)

    def _log_eval(self, score, epoch):
        self._logger.info(f"BLEU score at epoch {epoch}: {score['bleu']}")
        logger_utils.log_csv(self._eval_csv, [epoch, score['bleu'], score['length_ratio']])
        self._writer.add_scalar('Validation BLEU score per epoch', score['bleu'], global_step=epoch)
        self._writer.add_scalar('Validation length ratio per epoch', score['length_ratio'], global_step=epoch)

    def _init_csv_logger(self):
        self._loss_csv = os.path.join(self._log_path, 'loss.csv')
        header = ['global_iteration', 'epoch', 'epoch_iteration', 'train_loss', 'current_lr']
        with open(self._loss_csv, mode='w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

        self._eval_csv = os.path.join(self._log_path, 'eval.csv')
        header = ['epoch', 'bleu_score', 'length_ratio']
        with open(self._eval_csv, mode='w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
