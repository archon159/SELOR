# SELOR

This repository is the official implementation of **S**elf-**E**xplaining deep models with **LO**gic rule **R**easoning.

## Requirements
numpy==1.21.6

scikit-learn==0.24.2

torch==1.9.0+cu111

tqdm==4.62.3

pandas==1.1.4

## Datasets
Yelp Review Polarity (Kaggle)
>https://www.kaggle.com/datasets/irustandi/yelp-review-polarity

Clickbait News Detection (Kaggle)
>https://www.kaggle.com/c/clickbait-news-detection

Adult (UCI Machine Learning Repository)
>https://archive.ics.uci.edu/ml/datasets/adult

## Training
Train the base model with the following command.
```
python3 base.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```
Extract the embedding of base model for train instances with the following command.
```
python3 extract_base_embedding.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```
Build the atom pool with the following command.
```
python3 build_atom_pool.py --dataset <DATASET> --base_model <BASE_MODEL>
```
Sample atoms to pretrain the consequent estimator with the following command.
```
python3 pretrain_consequent_estimator.py --dataset <DATASET> --base_model <BASE_MODEL> --gpu <GPU>
```
Train the SELOR model with the following command.
```
python3 selor.py --dataset <DATASET> --base_model <BASE_MODEL> --gpu <GPU>
```

## Evaluation
Evaluation results are automatically provided after training.
If you want to produce only evaluation result of a trained model, please use the following command.
```
python3 selor.py --dataset <DATASET> --base_model <BASE_MODEL> --gpu <GPU> --only_eval
```

## Result
This will be updated soon.
