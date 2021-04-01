# Data Fusion Contest

4th place solution for [Goodsification task of Data Fusion Contest](https://boosters.pro/championship/data_fusion/overview).

## Task

Multiclass classification. Predict category of item from receipt based on its name and some additional data. There is a lot of unlabeled data (~3m items) and small part of labeled data (~48k items). Item names are very dirty.

## Approach

- Use only text data (names of items).
- Train tokenizer from scratch on all data.
- Pretrain small custom distilbert from scratch on all data as masked language model.
- Train this distilbert on labeled data.
- Make ensemble (simple average) of 3 such models with different tokenizers (wordpiece, BPE and unigram).

There was 500 mb solution size limit. So training small custom models helps.

## Details

File with data `data_fusion_train.parquet` should be added to `data` folder.

`run_all.sh ` contains all steps to fully reproduce solution:
- `python src/prepare_data.py` - prepare data for training language model and training on labeled data.
- `python src/train_tokenizers.py` - train 3 different tokenizers.
- `python src/train_lm.py --config_path=src/configs/train_lm{1,2,3}.yaml` - pretrain 3 language models with this tokenizers.
- `python src/train.py --config_path=src/configs/train{1,2,3}.yaml` - train this models on labeled data.
- `python src/compress_models.py` - save models without optimizer state, makes them much smaller. Saving without optimizer state during training didn't work as expected.

`submit` folder contains final submission. `copy_to_submit.sh` copies all required generated files to it.

Pretraining language models takes a lot of time. Smaller data file `item_name_100k.txt` can be used for testing purposes.

