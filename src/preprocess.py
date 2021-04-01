"""
Preprocessing.
"""

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def preprocess(texts, tokenizer_path, max_len=32):

    input_ids, input_masks = [], []

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.mask_token = '[MASK]'
    tokenizer.pad_token = "[PAD]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.unk_token = "[UNK]"

    for text in tqdm(texts):
        encoded = tokenizer.encode_plus(text, max_length=max_len,
                                        pad_to_max_length=True, truncation=True)
        input_ids.append(encoded['input_ids'])
        input_masks.append(encoded['attention_mask'])

    return [np.array(input_ids), np.array(input_masks)]
