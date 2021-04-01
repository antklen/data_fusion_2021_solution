"""
Prerain Distilbert as masked language model on all item names.
"""

import argparse
import os

from omegaconf import OmegaConf
from transformers import (DataCollatorForLanguageModeling, DistilBertConfig,
                          DistilBertForMaskedLM, LineByLineTextDataset,
                          PreTrainedTokenizerFast, Trainer, TrainingArguments)


DATA_PATH = 'data/item_name.txt'

parser = argparse.ArgumentParser(description='Training language model')
parser.add_argument('--config_path', type=str, default='src/configs/train_lm1.yaml',
                    help='path to config file')
args = parser.parse_args()

config = OmegaConf.load(args.config_path)
print(OmegaConf.to_yaml(config))

os.environ['WANDB_DISABLED'] = 'true'

tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.tokenizer_path)
tokenizer.mask_token = '[MASK]'
tokenizer.pad_token = "[PAD]"
tokenizer.sep_token = "[SEP]"
tokenizer.cls_token = "[CLS]"
tokenizer.unk_token = "[UNK]"

distilbert_config = DistilBertConfig(vocab_size=config.vocab_size,
                                     n_heads=8, dim=512, hidden_dim=2048)
model = DistilBertForMaskedLM(distilbert_config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=DATA_PATH,
    block_size=64)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=config.mlm_probability)

training_args = TrainingArguments(
    output_dir=config.output_path,
    overwrite_output_dir=True,
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    save_steps=300000,
    save_total_limit=1)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=None)

trainer.train()
trainer.save_model(os.path.join(config.output_path, 'final'))
