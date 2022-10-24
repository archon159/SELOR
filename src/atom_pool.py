from collections import OrderedDict, defaultdict, Counter
from sklearn.tree import _tree
import numpy as np
import re
from tqdm import tqdm
import pickle
import torch
from dataset import get_tabular_numerical_max

def get_true_matrix(atom_pool):
    n = atom_pool.num_atoms()
    sd = atom_pool.atom_satis_dict
    targets = atom_pool.atoms
        
    tm = np.zeros((n, len(atom_pool.train_x_)))
    
    for ri, satis in tqdm(sd.items()):
        r = targets[ri]
        for s in satis:
            tm[r.atom_idx, s] = 1

    return torch.tensor(tm).float().detach()

def get_word_count(
    df,
    tokenizer,
    dataset='yelp',
    create=False,
    save_path='',
):
    assert(len(save_path) > 0)

    if create:
        if dataset=='yelp':
            pos_type = ['text']
        elif dataset=='clickbait':
            pos_type = ['title', 'text']
            
        x_ = np.zeros((len(df), len(pos_type) * tokenizer.vocab_size))
        for pos, pt in enumerate(pos_type):
            for i, t in enumerate(tqdm(df[pt])):
                tokens = tokenizer.tokenize(t)
                c = Counter(tokens)

                for k, v in c.items():
                    x_[i, pos * tokenizer.vocab_size + k] = v

        with open(save_path, 'wb') as f:
            pickle.dump(x_, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'rb') as f:
            x_ = pickle.load(f)

    return x_

class AtomTokenizer():
    def __init__(self, data_df, min_freq=10, max_len=512, dataset='yelp'):
        self.dataset = dataset
        if dataset=='yelp':
            data_list = data_df['text'].tolist()
        elif dataset == 'clickbait':
            data_list = data_df['title'].tolist() + data_df['text'].tolist()
        else:
            assert(0)
            
        word_dict = defaultdict(int)
        for t in data_list:
            t = re.sub(r'[^a-zA-Z0-9]', ' ', t)
            t = t.lower()
            t = t.split(' ')
            for w in t:
                word_dict[w] += 1

        word_dict = {k:v for k,v in word_dict.items() if v >= min_freq}
        
        self.max_len = max_len
        self.word2idx = {}
        
        self.word2idx['[PAD]'] = 0
        self.word2idx['[DUM]'] = 1
        self.word2idx['[UNK]'] = 2
        
        i = 3
        for k in word_dict:
            self.word2idx[k] = i
            i += 1
        self.vocab_size = len(self.word2idx)
        self.idx2word = {}
        for k, v in self.word2idx.items():
            self.idx2word[v] = k
            
    def tokenize(self, s):
        s = re.sub(r'[^a-zA-Z0-9]', ' ', s)
        s = s.lower()
        words = s.split(' ')
        tokens = []
        for w in words:
            if w in self.word2idx:
                tokens.append(self.word2idx[w])
            else:
                tokens.append(self.word2idx['[UNK]'])
                
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
            
        return tokens
    
class Atom():
    def __init__(
        self,
        c_type, # dummy, text, categorical, numeric
        context, # For all kinds of dataset
        bigger, # For text, numeric, categorical. In categorical case, bigger==True means context == target
        target,
        n, 
        consequent,
        atom_idx,
        position=0, # For multiple input dataset
        tokenizer=None, # For text
        col_list=None, # For categorical or numeric
        cat_map=None, # For categorical or numeric
        dataset='yelp',
    ):
        self.c_type = c_type
        if c_type == 'dummy':
            self.context = context
        elif c_type == 'text':
            self.context = context
            self.word = tokenizer.idx2word[int(context)]
            self.feature_id = tokenizer.vocab_size * position + context
            self.bigger = bigger
        elif c_type == 'categorical':
            self.feature_id = col_list.index(context)
            self.bigger = bigger
        elif c_type == 'numerical':
            self.feature_id = col_list.index(context)
            self.bigger = bigger
            numerical_max = get_tabular_numerical_max(dataset=dataset)
            self.m = numerical_max[context]
        else:
            assert(0)
            
        self.target = target
        self.consequent = consequent
        self.n = n
        self.atom_idx = atom_idx
        self.position = position
        
        if dataset == 'yelp':
            pos_type = ['text']
        elif dataset == 'clickbait':
            pos_type = ['title', 'text']
        else:
            pos_type = []
        
        if c_type == 'text':
            if bigger:
                self.display_str = f'({pos_type[position]}) {self.word} >= {target}'
            else:
                self.display_str = f'({pos_type[position]}) {self.word} < {target}'
                
        elif c_type == 'categorical':
            target_word = cat_map[f'{context}_idx2key'][target]
            if bigger:
                self.display_str = f'{context} == {target_word}'
            else:
                self.display_str = f'{context} != {target_word}'
            
        elif c_type == 'numerical':
            if bigger:
                self.display_str = f'{context} >= {round(target * self.m, 1)}'
            else:
                self.display_str = f'{context} < {round(target * self.m, 1)}'
                
        elif c_type == 'dummy':
            self.display_str = f'[DUMMY]'
            
        else:
            assert(0)
    
    # Check if given x_ satisfies this atom
    def check(
        self,
        x_
    ):
        if len(x_.shape) == 1:
            x_ = np.array([x_])

        if self.c_type in ['text', 'numerical']:
            if self.bigger:
                ret = (x_[:, self.feature_id] >= self.target)
            else:
                ret = (x_[:, self.feature_id] < self.target)
                
        elif self.c_type == 'categorical':
            if self.bigger:
                ret = (x_[:, self.feature_id] == self.target)
            else:
                ret = (x_[:, self.feature_id] != self.target)

        elif self.c_type == 'dummy':
            ret = np.zeros(x_.shape[0])
        else:
            assert(0)
            
        return ret.astype(int)
    
    def display(
        self,
    ):
        dummy_consequent = [round(p, 4) for p in self.consequent]
        print(f'Atom {self.atom_idx}: {self.display_str}, Type: {self.c_type}, Basis: {self.n}, Consequent: {dummy_consequent}')
        
        
class AtomPool():
    def __init__(
        self,
        tokenizer,
        col_list,
        cat_map,
        train_x_,
        train_y_,
        dataset='yelp',
        alpha=1,
    ):
        self.tokenizer = tokenizer
        self.col_list = col_list
        self.cat_map = cat_map
        self.train_x_ = train_x_
        self.train_y_ = train_y_

        self.dataset = dataset
        self.alpha = alpha
        self.n_class = len(set(train_y_))

        self.atom_idx = 0

        self.atoms = OrderedDict()
        self.atom_id2key = []
        self.atom_satis_dict = {}
    
    def add_atom(
        self,
        c_type,
        context,
        bigger,
        target,
        position=0,
    ):
        atom_key = (c_type, context, bigger, target, position)
        
        if atom_key in self.atoms:
            return
        
        consequent, n, ids = self.check_atom_consequent(c_type, context, bigger, target, position)
        
        if c_type == 'dummy':
            r = Atom(c_type, None, None, None, n, consequent, self.atom_idx, dataset=self.dataset)
        elif c_type == 'text':
            r = Atom(c_type, context, bigger, target, n, consequent, self.atom_idx, position=position, tokenizer=self.tokenizer, dataset=self.dataset)
        elif c_type == 'categorical':
            r = Atom(c_type, context, bigger, target, n, consequent, self.atom_idx, col_list=self.col_list, cat_map=self.cat_map, dataset=self.dataset)
        elif c_type == 'numerical':
            r = Atom(c_type, context, bigger, target, n, consequent, self.atom_idx, col_list=self.col_list, cat_map=self.cat_map, dataset=self.dataset)
        else:
            assert(0)
        
        self.atoms[atom_key] = r
        self.atom_id2key.append(atom_key)
        self.atom_idx += 1
        self.atom_satis_dict[atom_key] = ids
        
    def check_atom_consequent(
        self,
        c_type,
        context,
        bigger,
        target,
        position,
    ):
        if c_type == 'dummy':
            ids = np.array(range(len(self.train_x_)))
            
        elif c_type == 'text':
            feature_id = self.tokenizer.vocab_size * position + context
            if bigger:
                ids = np.where(self.train_x_[:, feature_id] >= target)[0]
            else:
                ids = np.where(self.train_x_[:, feature_id] < target)[0]
                
        elif c_type == 'categorical':
            feature_id = self.col_list.index(context)
            if bigger:
                ids = np.where(self.train_x_[:, feature_id] == target)[0]
            else:
                ids = np.where(self.train_x_[:, feature_id] != target)[0]
            
        elif c_type == 'numerical':
            feature_id = self.col_list.index(context)
            if bigger:
                ids = np.where(self.train_x_[:, feature_id] >= target)[0]
            else:
                ids = np.where(self.train_x_[:, feature_id] < target)[0]
        else:
            assert(0)
            
        n = len(ids)
        if c_type == 'dummy':
            consequent = []
            for i in range(self.n_class):
                prob = 1 / self.n_class
                consequent.append(prob)
                
        else:
            c = Counter(self.train_y_[ids])
            consequent = []

            for i in range(self.n_class):
                prob = (c[i] + self.alpha) / (n + self.alpha * self.n_class)
                consequent.append(prob)
        
        return consequent, n, ids
        
    def check_atoms(
        self,
        x_,
    ):
        result = [v.check(x_)[0] for k, v in self.atoms.items()]

        return result
        
    def get_atom_consequent(self):
        ret = []
        ret += [v.consequent for k, v in self.atoms.items()]
        
        return np.array(ret)
    
    def get_coverage_atom(self):
        ret = []
        ret += [v.n / len(self.train_x_) for k, v in self.atoms.items()]
        
        return np.array(ret)
        
    def num_atoms(
        self,
    ):
        return len(self.atoms)
    
    def display_atoms(
        self,
        n=0,
    ):
        for i, (k, v) in enumerate(self.atoms.items()):
            v.display()
            if n > 0 and i==n:
                break