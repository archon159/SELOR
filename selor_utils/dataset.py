"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The module that contains utility functions and classes related to datasets
"""
from typing import Dict, List, Tuple
import math
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

NLP_DATASET = ['yelp', 'clickbait']
NLP_BASE = ['bert', 'roberta']
TAB_DATASET = ['adult']
TAB_BASE = ['dnn']

class YelpDataset(Dataset):
    """
    Dataset structure for Yelp data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        atom_pool: object=None,
        atom_tokenizer: object=None,
        tf_tokenizer: object=None,
        max_len: int=512,
    ):
        self.text = data_df['text']
        self.y = data_df['label'].astype(dtype='int64')
        self.atom_pool = atom_pool
        self.atom_tokenizer = atom_tokenizer
        self.tf_tokenizer = tf_tokenizer
        self.max_len = max_len

    def __len__(
        self
    ) -> int:
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        encoding = self.tf_tokenizer.encode_plus(
            text=self.text[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = input_ids.squeeze(dim=0).long()
        attention_mask = attention_mask.squeeze(dim=0).long()

        if self.atom_pool is not None:
            text_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count = Counter(self.atom_tokenizer.tokenize(self.text[i]))

            for word, count in dict(x_count).items():
                text_[word] = count

            # x_ indicates if the satisfaction of atoms for current sample
            x_ = self.atom_pool.check_atoms(text_)

            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y[i]

        return (input_ids, attention_mask, x_), y

class ClickbaitDataset(Dataset):
    """
    Dataset structure for clickbait data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        atom_pool: object,
        atom_tokenizer: object,
        tf_tokenizer: object,
        max_len: int=512
    ):
        self.title = data_df['title']
        self.text = data_df['text']
        self.y = data_df['label'].astype(dtype='int64')
        self.atom_pool = atom_pool
        self.tf_tokenizer = tf_tokenizer
        self.atom_tokenizer = atom_tokenizer
        self.max_len = max_len

        print(f"Data Num: {len(self.y)}")

    def __len__(
        self
    ) -> int:
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        encoding = self.tf_tokenizer.encode_plus(
            text=self.title[i],
            text_pair=self.text[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = input_ids.squeeze(dim=0).long()
        attention_mask = attention_mask.squeeze(dim=0).long()

        if self.atom_pool is not None:
            title_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count_title = Counter(self.atom_tokenizer.tokenize(self.title[i]))

            for word, count in dict(x_count_title).items():
                title_[word] = count

            text_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count_text = Counter(self.atom_tokenizer.tokenize(self.text[i]))

            for word, count in dict(x_count_text).items():
                text_[word] = count

            article_ = np.concatenate((title_, text_))

            x_ = self.atom_pool.check_atoms(article_)
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y[i]

        return (input_ids, attention_mask, x_), y

class AdultDataset(Dataset):
    """
    Dataset structure for adult data
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        atom_pool: object,
    ):
        self.x = data_df.drop(columns=['income'])
        self.y = data_df['income'].astype(dtype='int64')
        self.atom_pool = atom_pool

        print(f"Data Num: {len(self.y)}")

    def __len__(
        self
    ):
        return len(self.y)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[Tuple[torch.Tensor, ...], int]:
        x_dummy = self.x.loc[i]
        x = torch.Tensor(x_dummy).float()

        if self.atom_pool is not None:
            x_ = self.atom_pool.check_atoms(np.array(x_dummy))
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        y = self.y[i]

        return (x, x_), y


class ClassSamples:
    """
    Class to count the number of samples in each class for initialization of pretraining datasets
    """
    def __init__(
        self,
        class_idx: int,
        need: int,
    ):
        self.class_idx = class_idx
        self.num_samples = 0
        self.need = need
        self.done = False

        self.x = []
        self.mu = []
        self.sigma = []
        self.n = []

    def add(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        n: torch.Tensor,
    ):
        """
        Add multiple samples.
        """
        self.x.append(x)
        self.mu.append(mu)
        self.sigma.append(sigma)
        self.n.append(n)

        self.num_samples += len(x)
        if self.num_samples >= self.need:
            self.x = torch.cat(self.x)
            self.mu = torch.cat(self.mu)
            self.sigma = torch.cat(self.sigma)
            self.n = torch.cat(self.n)

            self.x = self.x[:self.need]
            self.mu = self.mu[:self.need]
            self.sigma = self.sigma[:self.need]
            self.n = self.n[:self.need]

            self.num_samples = self.need

        if self.num_samples == self.need:
            self.done=True


class PretrainDataset(Dataset):
    """
    Dataset structure for pretraining consequent estimator
    """
    def __init__(
        self,
        candidate: torch.Tensor,
        true_matrix: torch.Tensor,
        train_y: torch.Tensor,
        n_sample: int,
        **kwargs
    ):
        default_kwargs = {
            'num_classes': 2,
            'min_df': 200,
            'max_df': 0.95,
        }
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        self.n_atom, self.n_data = true_matrix.shape
        self.n_candidate, self.antecedent_len = candidate.shape
        num_classes = kwargs['num_classes']
        self.num_classes = num_classes

        self.x = []
        self.mu = []
        self.sigma = []
        self.n = []


        if self.antecedent_len == 1:
            self.x = candidate
            self.mu, self.sigma, self.n = self.__get_answer__(
                candidate,
                true_matrix,
                train_y
            )

        else:
            assert n_sample % num_classes == 0
            bsz = n_sample
            n_batch = math.ceil(self.n_candidate / bsz)

            sample_dict = {}
            for i in range(num_classes):
                sample_dict[i] = ClassSamples(
                    class_idx=i,
                    need=(n_sample // num_classes),
                )

            pbar = tqdm(range(n_sample))

            b_cnt = 0
            while not all(sample_dict[i].done for i in range(num_classes)):
                start = (b_cnt % n_batch) * bsz
                end = min(start + bsz, self.n_candidate)
                b_cnt += 1

                cand = candidate[start:end]
                mu, sigma, n = self.__get_answer__(
                    cand,
                    true_matrix,
                    train_y
                )

                max_value, max_index = torch.max(mu, dim=1)

                min_df_mask = (n >= kwargs['min_df'])
                max_df_mask = (n <= (kwargs['max_df'] * self.n_data))
                uniform_prob_mask = (max_value != (1 / num_classes))
                mask = min_df_mask & max_df_mask & uniform_prob_mask

                for i in range(num_classes):
                    if not sample_dict[i].done:
                        bef = sample_dict[i].num_samples
                        class_mask = mask & (max_index==i)
                        sample_dict[i].add(
                            cand[class_mask],
                            mu[class_mask],
                            sigma[class_mask],
                            n[class_mask]
                        )
                        aft = sample_dict[i].num_samples
                        pbar.update(aft-bef)

            pbar.close()

            self.x = torch.cat([sample_dict[i].x for i in range(num_classes)])
            self.mu = torch.cat([sample_dict[i].mu for i in range(num_classes)])
            self.sigma = torch.cat([sample_dict[i].sigma for i in range(num_classes)])
            self.n = torch.cat([sample_dict[i].n for i in range(num_classes)])

            assert len(self.x) == len(self.mu) == len(self.sigma) == len(self.n) == n_sample

            rand_idx = torch.randperm(n_sample, device=candidate.device)

            self.x = self.x[rand_idx]
            self.mu = self.mu[rand_idx]
            self.sigma = self.sigma[rand_idx]
            self.n = self.n[rand_idx]

        self.x = self.x.cpu()
        self.mu = self.mu.cpu()
        self.sigma = self.sigma.cpu()
        self.n = self.n.cpu()

    def __get_answer__(
        self,
        candidate: torch.Tensor,
        true_matrix: torch.Tensor,
        train_y: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        bsz, _ = candidate.shape
        target = torch.index_select(true_matrix, 0, candidate.flatten())
        target = target.reshape(bsz, self.antecedent_len, self.n_data)

        satis_num = torch.sum(target, dim=1)
        satis_mask = (satis_num == self.antecedent_len)

        n = torch.sum(satis_mask, dim=1)

        mu = []
        sigma = []
        for mask in satis_mask:
            satis_ans = train_y[mask]
            if len(satis_ans) == 0:
                mu.append(
                    torch.ones(
                        self.num_classes,
                        device=satis_ans.device
                    ) * (1 / self.num_classes)
                )
                sigma.append(
                    torch.zeros(
                        self.num_classes,
                        device=satis_ans.device
                    )
                )
            else:
                satis_ans = F.one_hot(satis_ans.long(), num_classes=self.num_classes).float()
                mu.append(torch.mean(satis_ans, dim=0))
                sigma.append(torch.std(satis_ans, dim=0, unbiased=False))

        mu = torch.stack(mu)
        sigma = torch.stack(sigma)

        return mu, sigma, n

    def __len__(
        self
    ) -> int:
        return len(self.x)

    def __getitem__(
        self,
        i=int
    ) -> Tuple[torch.Tensor, ...]:
        return self.x[i], self.mu[i], self.sigma[i], self.n[i]

def load_data(
    dataset: str='yelp',
    data_dir: str='./data/',
    seed: int=7,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load data and split into train, valid, test dataset.
    """
    if dataset=='yelp':
        # Negative = 0, Positive = 1
        data_path = f'{data_dir}/yelp_review_polarity_csv'

        train_df = pd.read_csv(f'{data_path}/train.csv', header=None)
        train_df = train_df.rename(columns={0: 'label', 1: 'text'})
        train_df['label'] = train_df['label'] - 1
        _, train_df = train_test_split(
            train_df,
            test_size=0.1,
            random_state=seed,
            stratify=train_df['label']
        )

        test_df = pd.read_csv(f'{data_path}/test.csv', header=None)
        test_df = test_df.rename(columns={0: 'label', 1: 'text'})
        test_df['label'] = test_df['label'] - 1

        test_df = test_df.reset_index(drop=True)
        test_df, valid_df = train_test_split(
            test_df,
            test_size=0.5,
            random_state=seed,
            stratify=test_df['label']
        )

    elif dataset=='clickbait':
        # news = 0, clickbait = 1
        data_path = f'{data_dir}/clickbait_news_detection'

        train_df = pd.read_csv(f'{data_path}/train.csv')
        train_df = train_df.loc[
            train_df['label'].isin(['news', 'clickbait']),
            ['title', 'text', 'label']
        ]
        train_df = train_df.dropna()
        train_df.at[train_df['label']=='news', 'label'] = 0
        train_df.at[train_df['label']=='clickbait', 'label'] = 1

        valid_df = pd.read_csv(f'{data_path}/valid.csv')
        valid_df = valid_df.loc[
            valid_df['label'].isin(['news', 'clickbait']),
            ['title', 'text', 'label']
        ]
        valid_df = valid_df.dropna()
        valid_df.at[valid_df['label']=='news', 'label'] = 0
        valid_df.at[valid_df['label']=='clickbait', 'label'] = 1

        test_df, valid_df = train_test_split(
            valid_df,
            test_size=0.5,
            random_state=seed,
            stratify=valid_df['label']
        )

    elif dataset=='adult':
        # <=50K = 0, >50K = 1
        data_path = f'{data_dir}/adult'

        data_df = pd.read_csv(f'{data_path}/adult.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
        cat_map = get_tabular_category_map(data_df, dataset)

        number_data_df = numerize_tabular_data(data_df, cat_map, dataset)
        number_data_df = number_data_df[numerical_x_col + categorical_x_col + y_col]
        dummy_data_df = pd.get_dummies(number_data_df, columns=categorical_x_col)

        train_df, test_df = train_test_split(
            dummy_data_df,
            test_size=0.2,
            random_state=seed,
            stratify=number_data_df[y_col[0]]
        )

        valid_df, test_df = train_test_split(
            test_df,
            test_size=0.5,
            random_state=seed,
            stratify=test_df[y_col[0]]
        )

    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    train_df, valid_df, test_df = [
        df.reset_index(
            drop=True
        ) for df in [train_df, valid_df, test_df]]

    return train_df, valid_df, test_df

def get_dataset_type(
    dataset: str='yelp'
) -> str:
    """
    Return the type of the dataset.
    """
    if dataset in NLP_DATASET:
        ret = 'nlp'
    elif dataset in TAB_DATASET:
        ret = 'tab'
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return ret

def get_base_type(
    base: str='bert'
) -> str:
    """
    Return the type of the base model.
    """
    if base in NLP_BASE:
        ret = 'nlp'
    elif base in TAB_BASE:
        ret = 'tab'
    else:
        raise ValueError(f'Base model {base} is not supported.')

    return ret

def get_label_column(
    dataset: str='yelp'
) -> str:
    """
    Return the label column of the dataset.
    """
    if dataset == 'yelp':
        label = 'label'
    elif dataset == 'clickbait':
        label = 'label'
    elif dataset == 'adult':
        label = 'income'
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return label

def get_context_columns(
    dataset: str='yelp'
) -> List[str]:
    """
    Return the context columns of the dataset.
    This also can be used as position indicator.
    """
    if dataset == 'yelp':
        cols = ['text']
    elif dataset == 'clickbait':
        cols = ['title', 'text']
    elif dataset == 'adult':
        categorical_x_col, numerical_x_col, _ = get_tabular_column_type(dataset)
        cols = categorical_x_col + numerical_x_col
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return cols

def get_class_names(
    dataset: str='yelp'
) -> List[str]:
    """
    Return the class names of the dataset.
    """
    if dataset == 'yelp':
        class_names = ['Negative', 'Positive']
    elif dataset == 'clickbait':
        class_names = ['news', 'clickbait']
    elif dataset == 'adult':
        class_names = ['<=50K', '>50K']
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    # For extensibility
    class_names = [str(c) for c in class_names]

    return class_names

def get_tabular_column_type(
    dataset: str='adult'
) -> List[str]:
    """
    Return the type of columns for the tabular dataset.
    """
    if dataset == 'adult':
        categorical_x_col = [
            'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
            'gender', 'native-country'
        ]

        numerical_x_col = [
            'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'
        ]

        y_col = ['income']
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return categorical_x_col, numerical_x_col, y_col

def get_tabular_category_map(
    data_df: pd.DataFrame,
    dataset: str='adult'
) -> Dict[str, str]:
    """
    Return the category map that maps string-like categories to index
    """
    categorical_x_col, _, y_col = get_tabular_column_type(dataset)

    cat_map = {}

    for cat in categorical_x_col + y_col:
        cat_map[f'{cat}_idx2key'] = {}
        cat_map[f'{cat}_key2idx'] = {}

        count = Counter(data_df[cat])

        cat_keys = sorted(count.keys())
        for i, key in enumerate(cat_keys):
            cat_map[f'{cat}_idx2key'][i] = key
            cat_map[f'{cat}_key2idx'][key] = i

    return cat_map

def numerize_tabular_data(
    data_df: pd.DataFrame,
    cat_map: Dict[str, str],
    dataset: str='adult'
) -> pd.DataFrame:
    """
    Convert the given dataframe.
    Categorical column would become index from string,
    and numerical column would normalized by its maximum value.
    """
    def convert_key2idx(target, map_dict):
        return map_dict[target]

    categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)

    for cat in categorical_x_col + y_col:
        map_dict = cat_map[f'{cat}_key2idx']
        col = data_df[cat]
        new_col = col.apply(convert_key2idx, args=(map_dict, ))
        data_df[cat] = new_col

    data_df = data_df.astype(float)
    for col in numerical_x_col:
        data_df[col] = data_df[col] / max(data_df[col])

    return data_df

def get_tabular_numerical_threshold(
    data_df: pd.DataFrame,
    dataset: str='adult',
    interval: int=4
) -> Dict[str, list]:
    """
    Get thresholds to create atoms for each column of the tabular dataset.
    """
    _, numerical_x_col, _ = get_tabular_column_type(dataset)

    numerical_threshold = {}
    if dataset == 'adult':
        for col in numerical_x_col:
            numerical_threshold[col] = []
            if col in ['capital-gain', 'capital-loss']:
                target = data_df[col][data_df[col] != 0]
            else:
                target = data_df[col]

            target = target.to_numpy()
            for i in range(1, interval):
                percent = i * (100 / interval)
                numerical_threshold[col].append(np.percentile(target, percent))
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return numerical_threshold

def get_tabular_numerical_max(
    data_df: pd.DataFrame,
    dataset: str='adult'
) -> Dict[str, float]:
    """
    Get the maximum value for each column of the tabular dataset.
    """
    _, numerical_x_col, _ = get_tabular_column_type(dataset)

    numerical_max = {}
    if dataset in ['adult']:
        for col in numerical_x_col:
            numerical_max[col] = data_df[col].describe()['max']

    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return numerical_max

def load_tabular_info(
    dataset: str='adult',
    data_dir: str='./data/'
) -> Tuple[Dict[str, str], Dict[str, list], Dict[str, float]]:
    """
    Returns the data structures that contains information of the tabular dataset.
    """
    data_path = f'{data_dir}/{dataset}'

    if dataset=='adult':
        data_df = pd.read_csv(f'{data_path}/adult.csv')
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    cat_map = get_tabular_category_map(data_df, dataset)
    numerical_threshold = get_tabular_numerical_threshold(data_df, dataset=dataset)
    numerical_max = get_tabular_numerical_max(data_df, dataset=dataset)

    return cat_map, numerical_threshold, numerical_max

def create_dataset(
    data_df: pd.DataFrame,
    dataset: str='yelp',
    atom_pool: object=None,
    atom_tokenizer: object=None,
    tf_tokenizer: object=None,
    config: object=None,
) -> object:
    """
    Create a dataset with the given dataframe.
    """
    if dataset == 'yelp':
        ret = YelpDataset(
            data_df,
            atom_pool=atom_pool,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            max_len=config.max_position_embeddings,
        )
    elif dataset == 'clickbait':
        ret = ClickbaitDataset(
            data_df,
            atom_pool=atom_pool,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            max_len=config.max_position_embeddings,
        )
    elif dataset == 'adult':
        ret = AdultDataset(
            data_df,
            atom_pool=atom_pool
        )
    else:
        raise ValueError(f'Dataset {dataset} is not supported.')

    return ret

def create_dataloader(
    dataset: object,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> object:
    """
    Create a dataloader with the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=False,
        pin_memory=False,
    )

def get_single_context(
    target: pd.DataFrame,
    dataset: str,
    tabular_column_type: List[str],
    tabular_info: Tuple[dict, dict, dict],
):
    """
    Get a single context for given instance.
    Used to get an base contents for explanation of a single instance.
    """
    if dataset in TAB_DATASET:
        categorical_x_col, numerical_x_col, y_col = tabular_column_type
        cat_map, _, numerical_max = tabular_info

    target_context = ''

    if dataset == 'yelp':
        target_context += f'text: {target["text"]}\n'
    elif dataset == 'clickbait':
        target_context += f'title: {target["title"]}\n'
        target_context += f'text: {target["text"]}\n'
    elif dataset == 'adult':
        for key in target.index:
            value = target[key]
            if key in numerical_x_col:
                target_context += f'{key}: {round(value * numerical_max[key], 1)}\n'
            elif key in y_col:
                continue
            else:
                if value == 1:
                    context, category = key.split('_')
                    assert context in categorical_x_col
                    cur = cat_map[f'{context}_idx2key'][int(float(category))]
                    target_context += f'{context}: {cur}\n'

    return target_context

def get_single_input(
    target: pd.DataFrame,
    dataset: str,
    atom_pool: object,
    tf_tokenizer: object=None,
    atom_tokenizer: object=None,
    max_len: int=512,
) -> Tuple[torch.Tensor, ...]:
    """
    Get a single input for given instance.
    Used to get an explanation of a single instance.
    """
    assert atom_pool is not None

    if dataset in NLP_DATASET:
        assert tf_tokenizer is not None
        assert atom_tokenizer is not None

        if dataset == 'yelp':
            text = target['text']
            text_pair = None

            text_bow = np.zeros(atom_tokenizer.vocab_size)
            text_count = Counter(atom_tokenizer.tokenize(target['text']))

            for word, count in dict(text_count).items():
                text_bow[word] = count
            bow = text_bow
        elif dataset == 'clickbait':
            text = target['title']
            text_pair = target['text']

            text_bow = np.zeros(atom_tokenizer.vocab_size)
            title_bow = np.zeros(atom_tokenizer.vocab_size)

            text_count = Counter(atom_tokenizer.tokenize(target['text']))
            title_count = Counter(atom_tokenizer.tokenize(target['title']))

            for word, count in dict(text_count).items():
                text_bow[word] = count

            for word, count in dict(title_count).items():
                title_bow[word] = count

            bow = np.concatenate((title_bow, text_bow))

        encoding = tf_tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = input_ids.squeeze(dim=0).long()
        attention_mask = attention_mask.squeeze(dim=0).long()

        x_ = atom_pool.check_atoms(bow)
        x_ = torch.Tensor(x_).long()

        ret = (input_ids, attention_mask, x_)

    if dataset in TAB_DATASET:
        col_label = get_label_column(dataset)
        target = target.drop(index=[col_label])
        x = torch.Tensor(target).float()

        x_ = atom_pool.check_atoms(target)
        x_ = torch.Tensor(x_).long()

        ret = (x, x_)

    return ret

def get_weight(
    pretrain_dataset: object,
    n_data: int,
) -> Tuple[torch.Tensor, ...]:
    """
    Get weight between losses for pretraining consequent estimator.
    """
    noise_mu = torch.std(pretrain_dataset.mu)
    noise_sigma = torch.std(pretrain_dataset.sigma)
    noise_coverage = torch.std(pretrain_dataset.n / n_data)

    w_mu = 1 / (2 * (noise_mu ** 2))
    w_sigma = 1 / (2 * (noise_sigma ** 2))
    w_coverage = 1 / (2 * (noise_coverage ** 2))

    return w_mu, w_sigma, w_coverage

def create_pretrain_dataset(
    candidate: torch.Tensor,
    true_matrix: torch.Tensor,
    train_y: torch.Tensor,
    n_sample: int,
    **kwargs
) -> object:
    """
    Create a dataset for pretraining consequent estimator with given antecedent candidates.
    """
    p_ds = PretrainDataset(
        candidate=candidate,
        true_matrix=true_matrix,
        train_y=train_y,
        n_sample=n_sample,
        num_classes=kwargs['num_classes'],
        min_df=kwargs['min_df'],
        max_df=kwargs['max_df'],
    )

    return p_ds

def create_pretrain_dataloader(
    pretrain_dataset: object,
    batch_size: int,
    num_workers: int,
    test_ratio: float=0.2,
    seed: int=7,
) -> Tuple[object, ...]:
    """
    Create a dataloader for pretraining consequent estimator with given dataset.
    """
    n_test = int(len(pretrain_dataset) * test_ratio)
    n_train = len(pretrain_dataset) - n_test

    test_dataset, train_dataset = random_split(
        pretrain_dataset,
        [n_test, n_train],
        torch.Generator().manual_seed(seed)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=False,
        pin_memory=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
    )

    return train_dataloader, test_dataloader
