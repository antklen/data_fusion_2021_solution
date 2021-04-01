"""
Train different tokenizers on all item names.
"""

import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer


TRAIN_DATA_PATH = 'data/data_fusion_train.parquet'
OUTPUT_PATH = 'data/tokenizers/'

# Prepare data
train = pd.read_parquet(TRAIN_DATA_PATH, columns=['item_name'])
item_names = train.item_name.drop_duplicates().tolist()


# WordPiece tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits()])
tokenizer.normalizer = Lowercase()

trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                           vocab_size=70000)
tokenizer.train_from_iterator(item_names, trainer)
tokenizer.save(os.path.join(OUTPUT_PATH, 'wordpiece_70k.json'))


# BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits()])
tokenizer.normalizer = Lowercase()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                     vocab_size=60000)
tokenizer.train_from_iterator(item_names, trainer)
tokenizer.save(os.path.join(OUTPUT_PATH, 'bpe_60k.json'))


# Unigram tokenizer
tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Sequence([Whitespace(), Digits()])
tokenizer.normalizer = Lowercase()

trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                         vocab_size=50000)
tokenizer.train_from_iterator(item_names, trainer)
tokenizer.save(os.path.join(OUTPUT_PATH, 'unigram_50k.json'))
