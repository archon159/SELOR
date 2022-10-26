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

NLP_DATASET = ['yelp', 'clickbait']
NLP_BASE = ['bert', 'roberta']
    
TAB_DATASET = ['adult']
TAB_BASE = ['dnn']

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

            # x_ indicates if the satisfaction of atoms for current sample
            x_ = self.ap.check_atoms(article_)
                
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
            
        
        y = self.y[i]
#         y = torch.Tensor([y]).long()
        
        return (input_ids, attention_mask, x_), y
        
class AdultDataset(Dataset):
#     def __init__(self, number_df, dummy_df, atom_pool, args):
    def __init__(
        self,
        df,
        atom_pool,
    ):
#         self.x_number = number_df.drop(columns=['income'])
#         self.x_dummy = dummy_df.drop(columns=['income'])
        self.x = df.drop(columns=['income'])
        self.y = df['income'].astype(dtype='int64')
        self.ap = atom_pool
        
        print(f"Data Num: {len(self.y)}")
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
#         x_number = self.x_number.loc[i]
#         x_dummy = self.x_dummy.loc[i]
        x_dummy = self.x.loc[i]
        x = torch.tensor(x_dummy).float()

        if self.ap != None:
            x_ = self.ap.check_atoms(x_dummy)
            x_ = torch.Tensor(x_).long()
        else:
            x_ = torch.Tensor([0]).long()
        
        y = self.y[i]
#         y = torch.Tensor([y]).long()
        
        return (x, x_), y
        
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
#         print(dummy_data_df.columns)
#         assert(0)
        
#         number_train_df, number_test_df, dummy_train_df, dummy_test_df = train_test_split(
#             number_data_df, dummy_data_df, test_size=0.2, random_state=seed, stratify=number_data_df[y_col[0]]
#         )
#         number_valid_df, number_test_df, dummy_valid_df, dummy_test_df = train_test_split(
#             number_test_df, dummy_test_df, test_size=0.5, random_state=seed, stratify=number_test_df[y_col[0]]
#         )

#         number_train_df = number_train_df.reset_index(drop=True)
#         number_valid_df = number_valid_df.reset_index(drop=True)
#         number_test_df = number_test_df.reset_index(drop=True)

#         dummy_train_df = dummy_train_df.reset_index(drop=True)
#         dummy_valid_df = dummy_valid_df.reset_index(drop=True)
#         dummy_test_df = dummy_test_df.reset_index(drop=True)
        
#         return (number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df), cat_map
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
