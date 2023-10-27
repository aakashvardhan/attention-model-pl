from data.dataset import casual_mask
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from models.model import *
from config import *
from tokenizers import Tokenizer
from models.model_utils import *
from torch.utils.tensorboard import SummaryWriter
import os
import random

config=get_config()

class LitModel(pl.LightningModule):
    def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                eps=1e-9, 
                label_smoothing=0.1,
                num_examples=5):
        super().__init__()
        self.eps = eps
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.label_smoothing = label_smoothing
        self.num_examples = num_examples

        self.model : Transformer = build_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            config['seq_len'],
            config['seq_len'],
            config['d_model'])
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], eps=self.eps)
        self.save_hyperparameters(ignore=['model']) 
    
    def initialize_attributes(self):
        self.tk_src = self.trainer.datamodule.tokenizer_src
        self.tk_tgt = self.trainer.datamodule.tokenizer_tgt
        # self.src_vocab_size = self.tk_src.get_vocab_size()
        # self.tgt_vocab_size = self.tk_tgt.get_vocab_size()

    def on_train_start(self):
        self.initialize_attributes()
        self.tgt_pad_token = self.tk_tgt.token_to_id("[PAD]")
        self.tgt_sos_token = self.tk_tgt.token_to_id("[SOS]")
        self.tgt_eos_token = self.tk_tgt.token_to_id("[EOS]")

        
    def configure_optimizers(self):
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], eps=self.eps)
        return self.optimizer
    
    def calculate_loss(self, proj_output: torch.Tensor, label):
        return nn.CrossEntropyLoss(ignore_index=self.tgt_pad_token, 
                                   label_smoothing=self.label_smoothing)(proj_output, label)
    
    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input'] # (batch_size, seq_len)
        decoder_input = batch['decoder_input'] # (batch_size, seq_len)

        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']

        # Run the tensors through the encoder, decoder and the projection layer
        encoder_output = self.model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
        decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
        proj_output = self.model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size)
        label = batch['label'] # (batch_size, seq_len)

        # Calculate the loss
        loss = self.calculate_loss(proj_output.view(-1, self.tgt_vocab_size), label.view(-1))
        self.log('train_loss', loss, 
                 on_step=True, on_epoch=True, 
                 prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.source_texts = []
        self.expected = []
        self.predicted = []

    def validation_step(self, batch, batch_idx):
        self.initialize_attributes()
        cer_metric = torchmetrics.text.CharErrorRate()
        wer_metric = torchmetrics.text.WordErrorRate()
        bleu_metric = torchmetrics.text.BLEUScore()
        if batch_idx < self.num_examples:
            encoder_input = batch['encoder_input']
            encoder_mask = batch['encoder_mask']

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(self.model, encoder_input, encoder_mask, self.tk_src, self.tk_tgt, config['seq_len'])

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = self.tk_tgt.decode(model_out.detach().cpu().numpy())

            self.source_texts.append(source_text)
            self.expected.append(target_text)
            self.predicted.append(model_out_text)

            print("-" * 80)
            print(f"{f'SOURCE: ':>12}{self.source_texts}")
            print(f"{f'TARGET: ':>12}{self.expected}")
            print(f"{f'PREDICTED: ':>12}{self.predicted}")

        if batch_idx == self.num_examples-1:      
            cer = cer_metric(self.predicted, self.expected)
            self.log("Validation CER", cer)

            # Word Error Rate
            wer = wer_metric(self.predicted, self.expected)
            self.log("Validation WER", wer)

            # BLEU Score
            
            bleu = bleu_metric(self.predicted, self.expected)
            self.log("Validation BLEU", bleu)




        
        

            

                                      
