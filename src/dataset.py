import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import random
from tqdm import tqdm

def create_dataloader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        persistent_workers=False,
        pin_memory=False,
    )

def get_dataset(
    dataset='yelp',
    seed=7,
):
    if dataset=='yelp':
        # Negative = 0, Positive = 1
        DATASET='yelp'
        DATA_DIR = './data/yelp_review_polarity_csv'

        train_df = pd.read_csv(f'{DATA_DIR}/train.csv', header=None)
        train_df = train_df.rename(columns={0: 'label', 1: 'text'})
        train_df['label'] = train_df['label'] - 1
        _, train_df = train_test_split(train_df, test_size=0.1, random_state=seed, stratify=train_df['label'])

        test_df = pd.read_csv(f'{DATA_DIR}/test.csv', header=None)
        test_df = test_df.rename(columns={0: 'label', 1: 'text'})
        test_df['label'] = test_df['label'] - 1

        test_df = test_df.reset_index(drop=True)
        test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=seed, stratify=test_df['label'])
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, valid_df, test_df
    
    elif dataset=='clickbait':
        # news = 0, clickbait = 1
        DATASET='clickbait'
        DATA_DIR = './data/clickbait_news_detection'

        train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
        train_df = train_df.loc[train_df['label'].isin(['news', 'clickbait']), ['title', 'text', 'label']]
        train_df = train_df.dropna()
        train_df.at[train_df['label']=='news', 'label'] = 0
        train_df.at[train_df['label']=='clickbait', 'label'] = 1
        
        valid_df = pd.read_csv(f'{DATA_DIR}/valid.csv')
        valid_df = valid_df.loc[valid_df['label'].isin(['news', 'clickbait']), ['title', 'text', 'label']]
        valid_df = valid_df.dropna()
        valid_df.at[valid_df['label']=='news', 'label'] = 0
        valid_df.at[valid_df['label']=='clickbait', 'label'] = 1

        test_df, valid_df = train_test_split(valid_df, test_size=0.5, random_state=seed, stratify=valid_df['label'])
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, valid_df, test_df
    
    elif dataset=='adult':
        # <=50K = 0, >50K = 1
        DATASET='adult'
        DATA_DIR = './data/adult'
        
        data_df = pd.read_csv(f'{DATA_DIR}/adult.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
        assert(set(data_df.columns) == set(categorical_x_col + numerical_x_col + y_col))
        
        cat_map = get_tabular_category_map(dataset)
        
        number_data_df = numerize_tabular_data(data_df, cat_map, categorical_x_col + y_col, numerical_x_col, dataset)
        dummy_data_df = pd.get_dummies(number_data_df, columns=categorical_x_col)
        
        number_train_df, number_test_df, dummy_train_df, dummy_test_df = train_test_split(
            number_data_df, dummy_data_df, test_size=0.2, random_state=seed, stratify=number_data_df[y_col[0]]
        )
        number_valid_df, number_test_df, dummy_valid_df, dummy_test_df = train_test_split(
            number_test_df, dummy_test_df, test_size=0.5, random_state=seed, stratify=number_test_df[y_col[0]]
        )

        number_train_df = number_train_df.reset_index(drop=True)
        number_valid_df = number_valid_df.reset_index(drop=True)
        number_test_df = number_test_df.reset_index(drop=True)

        dummy_train_df = dummy_train_df.reset_index(drop=True)
        dummy_valid_df = dummy_valid_df.reset_index(drop=True)
        dummy_test_df = dummy_test_df.reset_index(drop=True)
        
        return number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df
    else:
        assert(0)


def numerize_tabular_data(data_df, cat_map, categorical_column, numerical_column, dataset='adult'):
    def convert_key2idx(target, map_dict):
        return map_dict[target]
    
    for cat in categorical_column:
        map_dict = cat_map[f'{cat}_key2idx']

        col = data_df[cat]
        new_col = col.apply(convert_key2idx, args=(map_dict, ))
        data_df[cat] = new_col
        
    data_df = data_df.astype(float)
    for c in numerical_column:
        data_df[c] = data_df[c] / max(data_df[c])
        
    return data_df
        
def get_tabular_category_map(dataset='adult'):
    if dataset == 'adult':
        DATASET='adult'
        DATA_DIR = './data/adult'
        
        data_df = pd.read_csv(f'{DATA_DIR}/adult.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
        
    else:
        assert(0)
    
    cat_map = {}
    if dataset in ['adult']:
        for cat in categorical_x_col + y_col:
            cat_map[f'{cat}_idx2key'] = {}
            cat_map[f'{cat}_key2idx'] = {}

            c = Counter(data_df[cat])
            cat_keys = sorted(c.keys())
            for i, k in enumerate(cat_keys):
                cat_map[f'{cat}_idx2key'][i] = k
                cat_map[f'{cat}_key2idx'][k] = i
                
    return cat_map
        
def get_tabular_numerical_threshold(dataset='adult', interval=4):
    if dataset == 'adult':
        DATASET='adult'
        DATA_DIR = './data/adult/'
        
        data_df = pd.read_csv(f'{DATA_DIR}/adult.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
    else:
        assert(0)
    
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
        assert(0)
        
    return numerical_threshold

def get_tabular_numerical_max(dataset='adult'):
    if dataset == 'adult':
        DATASET='adult'
        DATA_DIR = './data/adult'
        
        data_df = pd.read_csv(f'{DATA_DIR}/adult.csv')
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset)
    else:
        assert(0)
    
    numerical_max = {}
    if dataset in ['adult']:
        for c in numerical_x_col:
            numerical_max[c] = data_df[c].describe()['max']
            
    else:
        assert(0)
        
    return numerical_max

def get_tabular_column_type(dataset='adult'):
    if dataset == 'adult':
        categorical_x_col = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
        numerical_x_col = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        y_col = ['income']
    else:
        assert(0)
    
    return categorical_x_col, numerical_x_col, y_col
        
def get_class_names(dataset='yelp'):
    if dataset == 'yelp':
        class_names = ['Negative', 'Positive']
    elif dataset == 'clickbait':
        class_names = ['news', 'clickbait']
    elif dataset == 'adult':
        class_names = ['<=50K', '>50K']
    else:
        assert(0)
        
    return class_names

class AdultDataset(Dataset):
    def __init__(self, number_df, dummy_df, atom_pool, args):
        self.x_number = number_df.drop(columns=['income'])
        self.x_dummy = dummy_df.drop(columns=['income'])
        self.y = number_df['income']
        self.ap = atom_pool
        self.base = args.base_model
        
        print(f"Data Num: {len(self.y)}")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        x_number = self.x_number.loc[i]
        x_dummy = self.x_dummy.loc[i]
        
        x = torch.tensor(x_dummy).float()
        if self.ap != None:
            x_ = self.ap.check_atoms(x_number)
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        
        y = self.y[i]
        y = torch.Tensor([y]).long()
        
        return x, x_, y

class ClickbaitDataset(Dataset):
    def __init__(self, df, atom_pool, tf_tokenizer, atom_tokenizer, args, max_len=512):
        self.title = df['title']
        self.text = df['text']
        self.y = df['label']
        self.ap = atom_pool
        self.tf_tokenizer = tf_tokenizer
        self.atom_tokenizer = atom_tokenizer        
        self.max_len = max_len
        self.base = args.base_model
        
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

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        
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

            # x_ indicates if the satisfaction of atoms for current sample
            x_ = self.ap.check_atoms(article_)
                
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
            
        
        y = self.y[i]
        y = torch.Tensor([y]).long()
        
        return input_ids, attention_mask, x_, y

class YelpDataset(Dataset):
    def __init__(self, df, atom_pool, tf_tokenizer, atom_tokenizer, args, max_len=512):
        self.text = df['text']
        self.y = df['label']
        self.ap = atom_pool
        self.tf_tokenizer = tf_tokenizer
        self.atom_tokenizer = atom_tokenizer        
        self.max_len = max_len
        self.base = args.base_model
        
        print(f"Data Num: {len(self.y)}")
        
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
            return_tensors='np',
            truncation=True,
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        
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
        y = torch.Tensor([y]).long()
        
        return input_ids, attention_mask, x_, y
    
class AdultPretrainDataset(Dataset):
    def __init__(
        self,
        tm,
        train_y_,
        n_sample,
        rule_length,
        min_df=200,
        max_df=0.95,
        candidate=None,
        args=None,
    ):
        self.gpu = torch.device(f'cuda:{args.gpu}')
        self.tm = tm.to(self.gpu)
        self.train_y_ = torch.tensor(train_y_).to(self.gpu)
        
        n_atom, n_data = tm.shape
        
        self.x = []
        self.mu = []
        self.sigma = []
        self.n = []
        self.rule_length = rule_length
        
        if rule_length==1:
            assert(n_sample == n_atom)
            for i in tqdm(range(n_sample)):
                rule = set([i])
                mu, sigma, n = self.__get_answer__(rule)

                self.x.append(rule)
                self.mu.append(mu)
                self.sigma.append(sigma)
                self.n.append(n)
                
            return
        
        pbar = tqdm(range(n_sample))
        
        n_poor = 0
        n_rich = 0
        
        while len(self.x) < n_sample:
            rules = random.sample(candidate, n_sample)
            for rule in rules:
                mu, sigma, n = self.__get_answer__(rule)
                if (n < min_df) or (n > (max_df * n_data)):
                    continue
                    
                _, mi = torch.max(mu, dim=0)
                mi = mi.item()

                if mi == 0 and n_poor < (n_sample / 2):
                    self.x.append(rule)
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.n.append(n)
                    n_poor += 1
                    pbar.update(1)
                    
                elif mi == 1 and n_rich < (n_sample / 2):
                    self.x.append(rule)
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.n.append(n)
                    n_rich += 1
                    pbar.update(1)
                else:
                    continue

        pbar.close()
        
    def __get_answer__(self, rule):
        out = torch.index_select(self.tm, 0, torch.tensor(list(rule)).to(self.gpu).long())
        out = torch.sum(out, dim=0)
        out = (out == self.rule_length)
        n = torch.sum(out).item()

        satis_ans = self.train_y_[out]
        if len(satis_ans) == 0:
            mu = torch.tensor([1/2, 1/2]).to(self.gpu)
            sigma = torch.tensor([0, 0]).to(self.gpu)
        else:
            satis_ans = satis_ans.long()
            satis_ans_one_hot = F.one_hot(satis_ans, num_classes=2).float()

            mu = torch.mean(satis_ans_one_hot, dim=0)
            sigma = torch.std(satis_ans_one_hot, unbiased=False, dim=0)
        
        return mu.cpu(), sigma.cpu(), n
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return torch.tensor(list(self.x[i])), self.mu[i], self.sigma[i], self.n[i]
    
class ClickbaitPretrainDataset(Dataset):
    def __init__(
        self,
        tm,
        train_y_,
        n_sample,
        rule_length,
        min_df=200,
        max_df=0.95,
        candidate=None,
        args=None,
    ):
        assert(rule_length == 1 or n_sample % 2 == 0)
        self.gpu = torch.device(f'cuda:{args.gpu}')
        self.tm = tm.to(self.gpu)
        self.train_y_ = torch.tensor(train_y_).to(self.gpu)
        
        n_atom, n_data = tm.shape
        
        self.x = []
        self.mu = []
        self.sigma = []
        self.n = []
        self.rule_length = rule_length
        
        if rule_length==1:
            assert(n_sample == n_atom)
            for i in tqdm(range(n_sample)):
                rule = set([i])
                mu, sigma, n = self.__get_answer__(rule)

                self.x.append(rule)
                self.mu.append(mu)
                self.sigma.append(sigma)
                self.n.append(n)
                
            return
        
        pbar = tqdm(range(n_sample))
        
        n_clickbait = 0
        n_news = 0
        
        while len(self.x) < n_sample:
            rules = random.sample(candidate, n_sample)
            for rule in rules:
                mu, sigma, n = self.__get_answer__(rule)
                if (n < min_df) or (n > (max_df * n_data)):
                    continue
                    
                _, mi = torch.max(mu, dim=0)
                mi = mi.item()

                if mi == 0 and n_news < (n_sample / 2):
                    self.x.append(rule)
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.n.append(n)
                    n_news += 1
                    pbar.update(1)
                    
                elif mi == 1 and n_clickbait < (n_sample / 2):
                    self.x.append(rule)
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.n.append(n)
                    n_clickbait += 1
                    pbar.update(1)
                else:
                    continue

        pbar.close()
        
    def __get_answer__(self, rule):
        out = torch.index_select(self.tm, 0, torch.tensor(list(rule)).to(self.gpu).long())
        out = torch.sum(out, dim=0)
        out = (out == self.rule_length)
        n = torch.sum(out).item()

        satis_ans = self.train_y_[out]
        if len(satis_ans) == 0:
            mu = torch.tensor([1/2, 1/2]).to(self.gpu)
            sigma = torch.tensor([0, 0]).to(self.gpu)
        else:
            satis_ans = satis_ans.long()
            satis_ans_one_hot = F.one_hot(satis_ans, num_classes=2).float()

            mu = torch.mean(satis_ans_one_hot, dim=0)
            sigma = torch.std(satis_ans_one_hot, unbiased=False, dim=0)
        
        return mu.cpu(), sigma.cpu(), n
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return torch.tensor(list(self.x[i])), self.mu[i], self.sigma[i], self.n[i]
    
    
class YelpPretrainDataset(Dataset):
    def __init__(
        self,
        tm,
        train_y_,
        n_sample,
        rule_length,
        min_df=200,
        max_df=0.95,
        candidate=None,
        args=None,
    ):
        assert(rule_length == 1 or n_sample % 2 == 0)
        self.gpu = torch.device(f'cuda:{args.gpu}')
        self.tm = tm.to(self.gpu)
        self.train_y_ = torch.tensor(train_y_).to(self.gpu)
        
        n_atom, n_data = tm.shape
        
        self.x = []
        self.mu = []
        self.sigma = []
        self.n = []
        self.rule_length = rule_length
        
        if rule_length==1:
            assert(n_sample == n_atom)
            for i in tqdm(range(n_sample)):
                rule = set([i])
                mu, sigma, n = self.__get_answer__(rule)

                self.x.append(rule)
                self.mu.append(mu)
                self.sigma.append(sigma)
                self.n.append(n)
                
            return
        
        pbar = tqdm(range(n_sample))
        
        n_pos = 0
        n_neg = 0
        
        while len(self.x) < n_sample:
            rules = random.sample(candidate, n_sample)
            for rule in rules:
                mu, sigma, n = self.__get_answer__(rule)
                if (n < min_df) or (n > (max_df * n_data)):
                    continue
                    
                if mu > 0.5 and n_pos < (n_sample / 2):
                    self.x.append(rule)
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.n.append(n)
                    n_pos += 1
                    pbar.update(1)
                elif mu < 0.5 and n_neg < (n_sample / 2):
                    self.x.append(rule)
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.n.append(n)
                    n_neg += 1
                    pbar.update(1)
                else:
                    continue

        pbar.close()
        
    def __get_answer__(self, rule):
        out = torch.index_select(self.tm, 0, torch.tensor(list(rule)).to(self.gpu).long())
        out = torch.sum(out, dim=0)
        out = (out == self.rule_length)
        n = torch.sum(out).item()

        satis_ans = self.train_y_[out]
        if len(satis_ans) == 0:
            satis_ans = torch.tensor([0])
        satis_ans = satis_ans.float()

        mu = torch.mean(satis_ans)
        sigma = torch.std(satis_ans, unbiased=False)
        return mu.cpu(), sigma.cpu(), n
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return torch.tensor(list(self.x[i])), self.mu[i], self.sigma[i], self.n[i]