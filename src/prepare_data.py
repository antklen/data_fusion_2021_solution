"""
Prepare all required data.
"""

import pandas as pd


DATA_PATH = 'data/data_fusion_train.parquet'

train = pd.read_parquet(DATA_PATH)

# prepare data for training language model
item_names = train.item_name.drop_duplicates()
item_names = item_names.map(lambda x: x + '\n')

with open('data/item_name_100k.txt', 'w') as f:
    f.writelines(item_names[:100000].tolist())
with open('data/item_name.txt', 'w') as f:
    f.writelines(item_names.tolist())

# prepare training data
train2 = train[train.category_id != -1].drop_duplicates('item_name')
train2 = train2[train2.item_name != '']
train2.to_csv('data/train_data.csv', index=False)

# prepare list of categories
categories = sorted(train2.category_id.unique())
categories = pd.Series(categories, name='category')
categories.to_csv('data/categories.csv', index=False)
