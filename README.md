# Recsys Challenge
## Requirements
* python3
* torch>=1.1.0
* pytorch-pretrained-bert==0.6.2
* tensorboardx==2.0
* tqdm==4.44

## Usage
### Prepare
```bash
python data_count.py
```
This program will count the language, hashtag and user information. The path of the source data is assigned in the code.
### Train
``` bash
python train.py \
--device 0 \
--batch 2000 \
--model autoint \
--model_name exp1 \
--label retweet \
--data_name all \
--epoch 6 \
--lr 1e-3 \
--weight_decay 1e-6 \
--dropout 0.5 \
--n_workers 0 \
--use_user_info \
--save_latest
```
If using \[--use_user_info\], the model will use user's information collected from training and validation set, which 
will consume huge RAM memory, and only when the test set user set coincides a lot with training set can this option improve the model.
n_workers can be set higher to speed up data processing by multiprocessing. Note that the memory occupation is proportional to the number of processes.

The path of the training set and validation set is assigned in `config.py`.
### Prediction
```bash
python test.py --device 0 model_name exp1 --label retweet --data_name all --use_user_info --make_prediction

```
This is used for submission. Eliminate --make_prediction for evaluation on the validation dataset. \[--use_user_info\]
should keep consistent with training. In practice, we split the test data into two parts, one where all user appear
at least once in training data and vice versa, and apply corresponding models.

The path of the test set is assigned in `config.py`.
