# SELOR

This repository is the official implementation of **S**elf-**E**xplaining deep models with **LO**gic rule **R**easoning.

## Environment Setup
Hardware
>Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz with 376GB RAM

>NVIDIA A100 GPU with 40GB memory

Operating System
>CentOS Linux release 7.9.2009

Python Version
>3.7

CUDA Version
>11.0

PyTorch Version
>1.9.0

## Datasets
Yelp Review Polarity (Kaggle)
>https://www.kaggle.com/datasets/irustandi/yelp-review-polarity

Clickbait News Detection (Kaggle)
>https://www.kaggle.com/c/clickbait-news-detection

Adult (UCI Machine Learning Repository)
>https://archive.ics.uci.edu/ml/datasets/adult

## Quick Start
Run following code to train the base model and the SELOR model.
```
python3 run_all.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

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
python3 build_atom_pool.py --dataset <DATASET> --base <BASE_MODEL>
```

Sample atoms to pretrain the consequent estimator with the following command.
```
python3 sample_antecedents.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

Pretrain a consequent estimator with the following command.
```
python3 pretrain_consequent_estimator.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

Train the SELOR model with the following command.
```
python3 selor.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU>
```

## Evaluation
Evaluation results are automatically provided after training.
If you want to produce only evaluation result of a trained model, please use the following command.
```
python3 selor.py --dataset <DATASET> --base <BASE_MODEL> --gpu <GPU> --only_eval
```

## Result
This will be updated soon.

## Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
