python src/prepare_data.py

python src/train_tokenizers.py

python src/train_lm.py --config_path=src/configs/train_lm1.yaml
python src/train_lm.py --config_path=src/configs/train_lm2.yaml
python src/train_lm.py --config_path=src/configs/train_lm3.yaml

python src/train.py --config_path=src/configs/train1.yaml
python src/train.py --config_path=src/configs/train2.yaml
python src/train.py --config_path=src/configs/train3.yaml

python src/compress_models.py
