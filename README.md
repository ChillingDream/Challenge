# Recsys Challenge

## Usage
### Prepare
```bash
python data_count.py
```
### Train
``` bash
python train.py \
--device 0 \
--batch 2000 \
--model autoint \
--model_name exp1 \
--label retweet \
--data_name all \
--epoch 10 \
--lr 1e-3 \
--weight_decay 1e-6 \
--dropout 0.5 \
--n_workers 0 \
--use_user_info \
--save_latest
```
### Prediction
This is used for submission. Eliminate --make_prediction for evaluation on the validation dataset.
```bash
python test.py --device 0 model_name exp1 --label retweet --data_name all --make_prediction

```
