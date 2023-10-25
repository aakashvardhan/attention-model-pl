from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data.dataset import *
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

class BilingualDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item['translation'][lang]

    def get_or_build_tokenizer(self, ds, lang):
        tokenizer_path = Path(self.config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"],
                min_frequency=2,
            )
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer
    
    def prepare_data(self):
        # Download the dataset
        load_dataset('opus_books', f"{self.config['lang_src']}-{self.config['lang_tgt']}", split="train")

    def setup(self, stage=None):
        # it only has the train split, so we divide it ourselves
        ds_raw = load_dataset('opus_books', f"{self.config['lang_src']}-{self.config['lang_tgt']}", split="train")

        # Build tokenizers
        self.tokenizer_src = self.get_or_build_tokenizer(ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = self.get_or_build_tokenizer(ds_raw, self.config['lang_tgt'])

        # Keep 90% of the data for training, 10% for validation
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BillingualDataset(
            train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config['lang_src'],
            self.config['lang_tgt'],
            self.config['seq_len'],
        )
        
        self.val_ds = BillingualDataset(
            val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config['lang_src'],
            self.config['lang_tgt'],
            self.config['seq_len'],
        )

        # find the max length of each sentence in the source and target language
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
            tgt_ids = self.tokenizer_tgt.encode(item['translation'][self.config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source language: {max_len_src}")
        print(f"Max length of target language: {max_len_tgt}")

    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=True)