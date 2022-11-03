import pandas as pd
import math
import functools
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import pickle
import random
from tqdm import tqdm

NLP_DATASET = ['yelp', 'clickbait']
NLP_BASE = ['bert', 'roberta']
    
TAB_DATASET = ['adult']
TAB_BASE = ['dnn']

def multi_AND(l):
    return functools.reduce(operator.and_, l)

def load_data(
    dataset='yelp',
    data_dir='./data/',
    seed=7,
):
    if dataset=='yelp':
        # Negative = 0, Positive = 1
        data_path = f'{data_dir}/yelp_review_polarity_csv'

        train_df = pd.read_csv(f'{data_path}/train.csv', header=None)
        train_df = train_df.rename(columns={0: 'label', 1: 'text'})
        train_df['label'] = train_df['label'] - 1
        _, train_df = train_test_split(train_df, test_size=0.1, random_state=seed, stratify=train_df['label'])

        test_df = pd.read_csv(f'{data_path}/test.csv', header=None)
        test_df = test_df.rename(columns={0: 'label', 1: 'text'})
        test_df['label'] = test_df['label'] - 1

        test_df = test_df.reset_index(drop=True)
        test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=seed, stratify=test_df['label'])
    
    elif dataset=='clickbait':
        # news = 0, clickbait = 1
        data_path = f'{data_dir}/clickbait_news_detection'

        train_df = pd.read_csv(f'{data_path}/train.csv')
        train_df = train_df.loc[train_df['label'].isin(['news', 'clickbait']), ['title', 'text', 'label']]
        train_df = train_df.dropna()
        train_df.at[train_df['label']=='news', 'label'] = 0
        train_df.at[train_df['label']=='clickbait', 'label'] = 1
        
        valid_df = pd.read_csv(f'{data_path}/valid.csv')
        valid_df = valid_df.loc[valid_df['label'].isin(['news', 'clickbait']), ['title', 'text', 'label']]
        valid_df = valid_df.dropna()
        valid_df.at[valid_df['label']=='news', 'label'] = 0
        valid_df.at[valid_df['label']=='clickbait', 'label'] = 1

        test_df, valid_df = train_test_split(valid_df, test_size=0.5, random_state=seed, stratify=valid_df['label'])
    
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
            dummy_data_df, test_size=0.2, random_state=seed, stratify=number_data_df[y_col[0]]
        )
        valid_df, test_df = train_test_split(
            test_df, test_size=0.5, random_state=seed, stratify=test_df[y_col[0]]
        )

    else:
        assert(0)
        
    train_df, valid_df, test_df = [df.reset_index(drop=True) for df in [train_df, valid_df, test_df]]
    
    return train_df, valid_df, test_df

def get_dataset_type(dataset='yelp'):
    if dataset in NLP_DATASET:
        return 'nlp'
    elif dataset in TAB_DATASET:
        return 'tab'
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
        
def get_base_type(base='bert'):
    if base in NLP_BASE:
        return 'nlp'
    elif base in TAB_BASE:
        return 'tab'
    else:
        raise NotImplementedError(f'Base model {base} is not supported.')

def get_label_column(dataset='yelp'):
    if dataset == 'yelp':
        label = 'label'
    elif dataset == 'clickbait':
        label = 'label'
    elif dataset == 'adult':
        label = 'income'
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
        
    return label
        
def get_pos_type(
    dataset='yelp'
):
    if dataset=='yelp':
        pos_type = ['text']
    elif dataset=='clickbait':
        pos_type = ['title', 'text']
    else:
        pos_type = []
        
    return pos_type
    
def get_class_names(dataset='yelp'):
    if dataset == 'yelp':
        class_names = ['Negative', 'Positive']
    elif dataset == 'clickbait':
        class_names = ['news', 'clickbait']
    elif dataset == 'adult':
        class_names = ['<=50K', '>50K']
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
        
    # For extensibility
    class_names = [str(c) for c in class_names]
        
    return class_names
        
def get_tabular_column_type(dataset='adult'):
    if dataset == 'adult':
        categorical_x_col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
        numerical_x_col = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        y_col = ['income']
    else:
        raise NotImplementedError(f'Base model {base} is not supported.')
    
    return categorical_x_col, numerical_x_col, y_col

def get_tabular_category_map(data_df, dataset='adult'):
    categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
    
    cat_map = {}

    for cat in categorical_x_col + y_col:
        cat_map[f'{cat}_idx2key'] = {}
        cat_map[f'{cat}_key2idx'] = {}

        c = Counter(data_df[cat])

        cat_keys = sorted(c.keys())
        for i, k in enumerate(cat_keys):
            cat_map[f'{cat}_idx2key'][i] = k
            cat_map[f'{cat}_key2idx'][k] = i
                
    return cat_map

def numerize_tabular_data(data_df, cat_map, dataset='adult'):
    def convert_key2idx(target, map_dict):
        return map_dict[target]
    
    categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
    
    for cat in categorical_x_col + y_col:
        map_dict = cat_map[f'{cat}_key2idx']

        col = data_df[cat]
        new_col = col.apply(convert_key2idx, args=(map_dict, ))
        data_df[cat] = new_col
        
    data_df = data_df.astype(float)
    for c in numerical_x_col:
        data_df[c] = data_df[c] / max(data_df[c])
        
    
        
    return data_df
        
def get_tabular_numerical_threshold(data_df, dataset='adult', interval=4):
    categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)

    numerical_threshold = {}
    if dataset == 'adult':
        for c in numerical_x_col:
            numerical_threshold[c] = []
            if c in ['capital-gain', 'capital-loss']:
                target = data_df[c][data_df[c] != 0]
            else:
                target = data_df[c]
            
            target = target.to_numpy()
            for i in range(1, interval):
                p = i * (100 / interval)
                numerical_threshold[c].append(np.percentile(target, p))
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
        
    return numerical_threshold

def get_tabular_numerical_max(data_df, dataset='adult'):
    categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
    
    numerical_max = {}
    if dataset in ['adult']:
        for c in numerical_x_col:
            numerical_max[c] = data_df[c].describe()['max']
            
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
        
    return numerical_max
        
def load_tabular_info(
    dataset='adult',
    data_dir='./data/'
):
    data_path = f'{data_dir}/{dataset}'

    if dataset=='adult':
        data_df = pd.read_csv(f'{data_path}/adult.csv')
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported as a tabular dataset.')
        
    cat_map = get_tabular_category_map(data_df, dataset)
    numerical_threshold = get_tabular_numerical_threshold(data_df, dataset=dataset)
    numerical_max = get_tabular_numerical_max(data_df, dataset=dataset)
    
    return cat_map, numerical_threshold, numerical_max

def create_dataset(
    data_df,
    dataset='yelp',
    atom_pool=None,
    atom_tokenizer=None,
    tf_tokenizer=None,
    config=None,
):
    if dataset == 'yelp':
        ds = YelpDataset(
            data_df,
            atom_pool=atom_pool,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            max_len=config.max_position_embeddings,
        )
        
    elif dataset == 'clickbait':
        ds = ClickbaitDataset(
            data_df,
            atom_pool=atom_pool,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            max_len=config.max_position_embeddings,
        )
        
    elif dataset == 'adult':
        ds = AdultDataset(
            data_df,
            atom_pool=atom_pool
        )
    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported.')
    
    return ds
        
def create_dataloader(
    dataset,
    batch_size,
    num_workers,
    shuffle
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=False,
        pin_memory=False,
    )
    
    
class YelpDataset(Dataset):
    def __init__(
        self,
        df,
        atom_pool=None,
        atom_tokenizer=None,
        tf_tokenizer=None,
        max_len=512,
    ):
        self.text = df['text']
        self.y = df['label'].astype(dtype='int64')
        self.ap = atom_pool
        self.atom_tokenizer = atom_tokenizer
        self.tf_tokenizer = tf_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
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
        
        if self.ap != None:
            text_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count = Counter(self.atom_tokenizer.tokenize(self.text[i]))

            for k, v in dict(x_count).items():
                text_[k] = v

            # x_ indicates if the satisfaction of atoms for current sample
            x_ = self.ap.check_atoms(text_)
                
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
            
        
        y = self.y[i]
#         y = torch.Tensor([y]).long()
        
        return (input_ids, attention_mask, x_), y
        
class ClickbaitDataset(Dataset):
    def __init__(
        self,
        df,
        atom_pool,
        atom_tokenizer,
        tf_tokenizer,
        max_len=512
    ):
        self.title = df['title']
        self.text = df['text']
        self.y = df['label'].astype(dtype='int64')
        self.ap = atom_pool
        self.tf_tokenizer = tf_tokenizer
        self.atom_tokenizer = atom_tokenizer        
        self.max_len = max_len
        
        print(f"Data Num: {len(self.y)}")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        encoding = self.tf_tokenizer.encode_plus(
            text=self.title[i],
            text_pair=self.text[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='np',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = torch.tensor(input_ids).squeeze(dim=0).long()
        attention_mask = torch.tensor(attention_mask).squeeze(dim=0).long()
        
        if self.ap != None:
            title_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count_title = Counter(self.atom_tokenizer.tokenize(self.title[i]))

            for k, v in dict(x_count_title).items():
                title_[k] = v
            
            text_ = np.zeros(self.atom_tokenizer.vocab_size)
            x_count_text = Counter(self.atom_tokenizer.tokenize(self.text[i]))

            for k, v in dict(x_count_text).items():
                text_[k] = v
                
            article_ = np.concatenate((title_, text_))

            x_ = self.ap.check_atoms(article_)
                
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
            
        
        y = self.y[i]
        
        return (input_ids, attention_mask, x_), y
        
class AdultDataset(Dataset):
    def __init__(
        self,
        df,
        atom_pool,
    ):
        self.x = df.drop(columns=['income'])
        self.y = df['income'].astype(dtype='int64')
        self.ap = atom_pool
        
        print(f"Data Num: {len(self.y)}")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        x_dummy = self.x.loc[i]
        x = torch.tensor(x_dummy).float()

        if self.ap != None:
            x_ = self.ap.check_atoms(x_dummy)
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        
        y = self.y[i]
        
        return (x, x_), y
    
def get_weight(
    pretrain_dataset,
    n_data,
):
    noise_mu = torch.std(pretrain_dataset.mu)
    noise_sigma = torch.std(pretrain_dataset.sigma)
    noise_coverage = torch.std(pretrain_dataset.n / n_data)
    
    weight_mu = 1 / (2 * (noise_mu ** 2))
    weight_sigma = 1 / (2 * (noise_sigma ** 2))
    weight_coverage = 1 / (2 * (noise_coverage ** 2))
    
    return weight_mu, weight_sigma, weight_coverage
    
def create_pretrain_dataset(
    candidate,
    true_matrix,
    train_y,
    dataset,
    n_sample,
    num_classes=2,
    min_df=200,
    max_df=0.95,
):
    p_ds = PretrainDataset(
        candidate,
        true_matrix,
        train_y,
        n_sample,
        num_classes,
        min_df,
        max_df,
    )
    
    return p_ds
    
def create_pretrain_dataloader(
    pretrain_dataset,
    batch_size,
    num_workers,
    test_ratio=0.2,
    seed=7,
):
    n_test = int(len(pretrain_dataset) * 0.2)
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
    
class ClassSamples():
    def __init__(
        self,
        class_idx,
        need,
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
        x,
        mu,
        sigma,
        n,
    ):
        self.x.append(x)
        self.mu.append(mu)
        self.sigma.append(sigma)
        self.n.append(n)
        
        self.num_samples += 1
        
        if self.num_samples == self.need:
            self.done=True
        
    def add_multiple(
        self,
        x,
        mu,
        sigma,
        n,
    ):
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
    def __init__(
        self,
        candidate,
        tm,
        train_y,
        n_sample,
        num_classes=2,
        min_df=200,
        max_df=0.95,
    ):
        self.n_atom, self.n_data = tm.shape
        self.n_candidate, self.rule_length = candidate.shape
        self.num_classes = num_classes
        
        self.x = []
        self.mu = []
        self.sigma = []
        self.n = []
        
        with torch.no_grad():
            if self.rule_length == 1:
                self.x = candidate
                self.mu, self.sigma, self.n = self.__get_answer__(
                    candidate,
                    tm,
                    train_y
                )
            else:
                assert(n_sample % num_classes == 0)
                bsz = n_sample
                n_batch = math.ceil(self.n_candidate / bsz)
                
                sample_dict = {}
                for i in range(num_classes):
                    sample_dict[i] = ClassSamples(
                        class_idx=i,
                        need=(n_sample // num_classes),
                    )
                
                pbar = tqdm(range(n_sample))
                
                b = 0
                while not multi_AND([sample_dict[i].done for i in range(num_classes)]):
                    start = (b % n_batch) * bsz
                    end = min(start + bsz, self.n_candidate)
                    b += 1
                    
                    c = candidate[start:end]
                    mu, sigma, n = self.__get_answer__(
                        c,
                        tm,
                        train_y
                    )
                
                    mv, mi = torch.max(mu, dim=1)
                    mask = (n >= min_df) & (n <= (max_df * self.n_data)) & (mv != (1 / num_classes))
                    class_counter = Counter(mi.cpu().numpy())
                    
                    for i in range(num_classes):
                        if not sample_dict[i].done:
                            bef = sample_dict[i].num_samples
                            class_mask = mask & (mi==i)
                            sample_dict[i].add_multiple(c[class_mask], mu[class_mask], sigma[class_mask], n[class_mask])
                            aft = sample_dict[i].num_samples
                            pbar.update(aft-bef)
                            
                pbar.close()
                        
                self.x = torch.cat([sample_dict[i].x for i in range(num_classes)])
                self.mu = torch.cat([sample_dict[i].mu for i in range(num_classes)])
                self.sigma = torch.cat([sample_dict[i].sigma for i in range(num_classes)])
                self.n = torch.cat([sample_dict[i].n for i in range(num_classes)])
                
                assert(len(self.x) == len(self.mu) == len(self.sigma) == len(self.n) == n_sample)
                
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
        c,
        tm,
        train_y
    ):
        bsz, _ = c.shape
        target = torch.index_select(tm, 0, c.flatten()) 
        target = target.reshape(bsz, self.rule_length, self.n_data)

        satis_num = torch.sum(target, dim=1)
        satis_mask = (satis_num == self.rule_length)
        
        n = torch.sum(satis_mask, dim=1)

        mu = []
        sigma = []
        for m in satis_mask:
            satis_ans = train_y[m]
            satis_ans = F.one_hot(satis_ans.long(), num_classes=self.num_classes).float()
            mu.append(torch.mean(satis_ans, dim=0))
            sigma.append(torch.std(satis_ans, dim=0, unbiased=False))

        mu = torch.stack(mu)
        sigma = torch.stack(sigma)
        
        return mu, sigma, n
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.mu[i], self.sigma[i], self.n[i]