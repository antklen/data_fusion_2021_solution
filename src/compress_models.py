"""
Save models without optimizer state.
"""

import os

from tensorflow.keras.models import load_model


MODELS_PATH = 'data/models/'
MODEL_NAMES = ['distilbert_wordpiece_70k',
               'distilbert_bpe_60k',
               'distilbert_unigram_50k']

for model_name in MODEL_NAMES:
    path = os.path.join(MODELS_PATH, model_name)
    model = load_model(path)
    model.save(path, include_optimizer=False)
