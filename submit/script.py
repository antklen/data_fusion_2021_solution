import pandas as pd
from tensorflow.keras.models import load_model

from preprocess import preprocess


DATA_PATH = 'data/task1_test_for_user.parquet'
TOKENIZER_PATH1 = 'tokenizers/unigram_50k.json'
TOKENIZER_PATH2 = 'tokenizers/wordpiece_70k.json'
TOKENIZER_PATH3 = 'tokenizers/bpe_60k.json'
MODEL_PATH1 = 'models/distilbert_unigram_50k'
MODEL_PATH2 = 'models/distilbert_wordpiece_70k'
MODEL_PATH3 = 'models/distilbert_bpe_60k'
CATEGORIES_PATH = 'categories.csv'

test = pd.read_parquet(DATA_PATH)
test_unique = test[['item_name']].drop_duplicates('item_name')
categories = pd.read_csv(CATEGORIES_PATH)['category'].tolist()

inputs1 = preprocess(test_unique.item_name, tokenizer_path=TOKENIZER_PATH1)
inputs2 = preprocess(test_unique.item_name, tokenizer_path=TOKENIZER_PATH2)
inputs3 = preprocess(test_unique.item_name, tokenizer_path=TOKENIZER_PATH3)

model1 = load_model(MODEL_PATH1)
proba1 = model1.predict(inputs1, batch_size=256, verbose=True)
model2 = load_model(MODEL_PATH2)
proba2 = model2.predict(inputs2, batch_size=256, verbose=True)
model3 = load_model(MODEL_PATH3)
proba3 = model3.predict(inputs3, batch_size=256, verbose=True)

proba = proba1 + proba2 + proba3
proba = pd.DataFrame(proba, columns=categories)
pred = proba.idxmax(axis=1).astype(int)

test_unique['pred'] = pred.values
test = test.merge(test_unique, on='item_name', how='left')
test[['id', 'pred']].to_csv('answers.csv', index=None)
