"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The module that contains utility functions and classes related to atoms
"""
from collections import OrderedDict, defaultdict, Counter
from typing import List, Tuple
import re
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd

from .utils import check_kwargs

class AtomTokenizer:
    """
    The tokenizer for atoms
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        dataset: str='yelp',
        min_freq: int=10,
        max_len: int=512
    ):
        self.dataset = dataset
        if dataset=='yelp':
            data_list = data_df['text'].tolist()
        elif dataset == 'clickbait':
            data_list = data_df['title'].tolist() + data_df['text'].tolist()
        else:
            raise ValueError(f'Dataset {dataset} is not supported.')

        word_dict = defaultdict(int)
        for instance in data_list:
            words = self.preprocess(instance)
            for word in words:
                word_dict[word] += 1

        word_dict = {word:count for word,count in word_dict.items() if count >= min_freq}

        self.max_len = max_len
        self.word2idx = {}

        self.word2idx['[PAD]'] = 0
        self.word2idx['[DUM]'] = 1
        self.word2idx['[UNK]'] = 2

        i = 3
        for word in word_dict:
            self.word2idx[word] = i
            i += 1
        self.vocab_size = len(self.word2idx)
        self.idx2word = {}
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word

    def preprocess(
        self,
        instance: str
    ) -> List[str]:
        """
        Preprocess the given instance
        """
        instance = re.sub(r'[^a-zA-Z0-9]', ' ', instance)
        instance = instance.lower()
        words = instance.split(' ')

        return words

    def tokenize(
        self,
        instance: str
    ) -> List[int]:
        """
        Tokenize the given instance
        """
        words = self.preprocess(instance)
        tokens = []
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx['[UNK]'])

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))

        return tokens

class Atom:
    """
    The smallest unit of logic rule
    """
    def __init__(
        self,
        c_type: str, # dummy, text, categorical, numeric
        context: str, # For all kinds of dataset
        bigger: bool, # In categorical case, bigger==True means context == target
        target: str,
        **kwargs
    ):
        check_kwargs(
            ['position', 'n', 'consequent', 'atom_idx', 'pos_type'],
            kwargs=kwargs
        )

        self.c_type = c_type
        self.context = context
        self.bigger = bigger
        self.target = target

        self.position = kwargs['position']
        self.n = kwargs['n']
        self.consequent = kwargs['consequent']
        self.atom_idx = kwargs['atom_idx']
        pos_type = kwargs['pos_type']

        if c_type == 'dummy':
            self.display_str = '[DUMMY]'
        elif c_type == 'text':
            check_kwargs(
                ['tokenizer'],
                value=True,
                kwargs=kwargs,
            )
            tokenizer = kwargs['tokenizer']

            self.word = tokenizer.idx2word[int(context)]
            self.feature_id = tokenizer.vocab_size * self.position + context

            if bigger:
                self.display_str = f'({pos_type[self.position]}) {self.word} >= {target}'
            else:
                self.display_str = f'({pos_type[self.position]}) {self.word} < {target}'

        elif c_type == 'categorical':
            check_kwargs(
                ['col2feature', 'cat_map'],
                value=True,
                kwargs=kwargs,
            )
            col2feature = kwargs['col2feature']
            cat_map = kwargs['cat_map']

            self.feature_id = col2feature[(context, target)]

            target_word = cat_map[f'{context}_idx2key'][target]
            if bigger:
                self.display_str = f'{context} == {target_word}'
            else:
                self.display_str = f'{context} != {target_word}'

        elif c_type == 'numerical':
            check_kwargs(
                ['col2feature', 'numerical_max'],
                value=True,
                kwargs=kwargs,
            )
            col2feature = kwargs['col2feature']
            numerical_max = kwargs['numerical_max']

            self.feature_id = col2feature[(context, 0)]
            if bigger:
                self.display_str = f'{context} >= {round(target * numerical_max[context], 1)}'
            else:
                self.display_str = f'{context} < {round(target * numerical_max[context], 1)}'

        else:
            raise ValueError(f'Context type {c_type} is not supported.')

    def check(
        self,
        x_: np.array
    ) -> int:
        """
        Check if given x_ satisfies this atom
        """
        if len(x_.shape) == 1:
            x_ = np.array([x_])

        if self.c_type in ['text', 'numerical']:
            if self.bigger:
                ret = (x_[:, self.feature_id] >= self.target)
            else:
                ret = (x_[:, self.feature_id] < self.target)

        elif self.c_type == 'categorical':
            if self.bigger:
                ret = (x_[:, self.feature_id] >= 0.5)
            else:
                ret = (x_[:, self.feature_id] < 0.5)

        elif self.c_type == 'dummy':
            ret = np.zeros(x_.shape[0])
        else:
            raise ValueError(f'Context type {self.c_type} is not supported.')

        return ret.astype(int)

    def __repr__(self):
        disp = f'Atom {self.atom_idx}: {self.display_str}, '
        disp += f'Type: {self.c_type}, '
        disp += f'Basis: {self.n}, '
        disp += f'Consequent: {[round(p, 4) for p in self.consequent]}'

        return disp

class AtomPool:
    """
    The pool of atoms
    """
    def __init__(
        self,
        train_x: np.array,
        train_y: np.array,
        dtype: str='nlp',
        alpha: int=1,
        **kwargs,
    ):
        check_kwargs(
            ['pos_type'],
            kwargs=kwargs
        )

        self.train_x = train_x
        self.train_y = train_y

        self.alpha = alpha
        self.n_class = len(set(train_y))
        self.n_data = len(train_x)

        self.pos_type = kwargs['pos_type']

        self.atom_idx = 0
        self.atoms = OrderedDict()
        self.atom_id2key = []
        self.atom_satis_dict = {}

        if dtype == 'nlp':
            check_kwargs(
                ['tokenizer'],
                value=True,
                kwargs=kwargs
            )
            self.tokenizer = kwargs['tokenizer']

            self.cat_map = None
            self.col2feature = None
            self.numerical_max = None

        elif dtype == 'tab':
            check_kwargs(
                ['tabular_info', 'tabular_column_type'],
                value=True,
                kwargs=kwargs
            )

            self.tokenizer = None
            self.cat_map, self.numerical_threshold, self.numerical_max = kwargs['tabular_info']
            self.categorical_x_col, self.numerical_x_col, self.y_col = kwargs['tabular_column_type']

            feature_id = 0
            self.col2feature = {}
            for col in self.numerical_x_col:
                pair = (col, 0)
                self.col2feature[pair] = feature_id
                feature_id += 1

            for col in self.categorical_x_col:
                cat_dict = self.cat_map[f'{col}_idx2key']
                for i in cat_dict:
                    pair = (col, i)
                    self.col2feature[pair] = feature_id
                    feature_id += 1

        else:
            raise ValueError(f'Dataset type {dtype} is not supported.')

    def __repr__(self):
        return '\n'.join([str(atom) for key, atom in self.atoms.items()])

    def add_atom(
        self,
        c_type: str,
        context: str,
        bigger: bool,
        target: str,
        position: int=0,
    ):
        """
        Add an atom to the atom pool with given info
        """
        atom_key = (c_type, context, bigger, target, position)

        if atom_key in self.atoms:
            print(f'Atom {atom_key} already exists.')

            return

        consequent, n, ids = self.check_atom_consequent(
            c_type,
            context,
            bigger,
            target,
            position
        )

        atom = Atom(
            c_type=c_type,
            context=context,
            bigger=bigger,
            target=target,
            position=position,
            n=n,
            consequent=consequent,
            atom_idx=self.atom_idx,
            tokenizer=self.tokenizer,
            cat_map=self.cat_map,
            col2feature=self.col2feature,
            numerical_max=self.numerical_max,
            pos_type=self.pos_type,
        )

        self.atoms[atom_key] = atom
        self.atom_id2key.append(atom_key)
        self.atom_idx += 1
        self.atom_satis_dict[atom_key] = ids

    def check_atom_consequent(
        self,
        c_type: str,
        context: str,
        bigger: bool,
        target: str,
        position: int=0,
    ) -> Tuple[list, int, np.array]:
        """
        Obtain empirical consequent, number of satisfying instances,
        and their ids for given atom info.
        """
        if c_type == 'dummy':
            ids = np.array(range(len(self.train_x)))

        elif c_type == 'text':
            feature_id = self.tokenizer.vocab_size * position + context
            if bigger:
                ids = np.where(self.train_x[:, feature_id] >= target)[0]
            else:
                ids = np.where(self.train_x[:, feature_id] < target)[0]

        elif c_type == 'categorical':
            feature_id = self.col2feature[(context, target)]
            if bigger:
                ids = np.where(self.train_x[:, feature_id] >= 0.5)[0]
            else:
                ids = np.where(self.train_x[:, feature_id] < 0.5)[0]

        elif c_type == 'numerical':
            feature_id = self.col2feature[(context, 0)]
            if bigger:
                ids = np.where(self.train_x[:, feature_id] >= target)[0]
            else:
                ids = np.where(self.train_x[:, feature_id] < target)[0]
        else:
            raise ValueError(f'Context type {c_type} is not supported.')

        n = len(ids)
        count = Counter(self.train_y[ids])
        consequent = []

        for i in range(self.n_class):
            prob = (count[i] + self.alpha) / (n + self.alpha * self.n_class)
            consequent.append(prob)

        return consequent, n, ids

    def check_atoms(
        self,
        x_: np.array,
    ) -> List[int]:
        """
        Check if given x_ satisfies the atoms in this pool.
        """
        result = [v.check(x_)[0] for k, v in self.atoms.items()]

        return result

    def get_atom_consequent(
        self
    ) -> np.array:
        """
        Get consequents of atoms in this pool.
        """
        ret = []
        ret += [v.consequent for k, v in self.atoms.items()]

        return np.array(ret)

    def get_coverage_atom(
        self
    ) -> np.array:
        """
        Get coverages of atoms in this pool.
        """
        ret = []
        ret += [v.n / len(self.train_x) for k, v in self.atoms.items()]

        return np.array(ret)

    def num_atoms(
        self,
    ) -> int:
        """
        Get total number of atoms in this pool.
        """
        return len(self.atoms)

def get_true_matrix(
    atom_pool: object
) -> torch.Tensor:
    """
    Get "true matrix" for given atom pool.
    """
    num_atoms = atom_pool.num_atoms()
    satis_dict = atom_pool.atom_satis_dict
    targets = atom_pool.atoms

    true_matrix = np.zeros((num_atoms, atom_pool.n_data))

    for atom_index, satis in tqdm(satis_dict.items()):
        atom = targets[atom_index]
        for instance in satis:
            true_matrix[atom.atom_idx, instance] = 1

    return torch.Tensor(true_matrix).float().detach()

def get_word_count(
    df: pd.DataFrame,
    tokenizer: object,
    pos_type: List[str],
) -> np.array:
    """
    Get word count of given data.
    """
    word_count = np.zeros((len(df), len(pos_type) * tokenizer.vocab_size))
    for pos, col in enumerate(pos_type):
        for i, instance in enumerate(df[col]):
            tokens = tokenizer.tokenize(instance)
            counter = Counter(tokens)

            for token, count in counter.items():
                word_count[i, pos * tokenizer.vocab_size + token] = count

    return word_count
