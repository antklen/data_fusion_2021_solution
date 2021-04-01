"""
Training model on all data.
"""

import argparse

import pandas as pd
import tensorflow as tf
from omegaconf import OmegaConf

from models import distilbert_model
from preprocess import preprocess


DATA_PATH = 'data/train_data.csv'

tf.config.experimental.set_memory_growth(
    device=tf.config.experimental.get_visible_devices('GPU')[0],
    enable=True)

parser = argparse.ArgumentParser(description='Training model on all data')
parser.add_argument('--config_path', type=str, default='configs/train1.yaml',
                    help='path to config file')
args = parser.parse_args()

config = OmegaConf.load(args.config_path)
print(OmegaConf.to_yaml(config))

train = pd.read_csv(DATA_PATH)
X = preprocess(train.item_name, **config.preprocess)
y = pd.get_dummies(train.category_id)

model = distilbert_model(**config.model)
print(model.summary())

model.fit(X, y, verbose=1, **config.train)
model.save(config.output_path)
