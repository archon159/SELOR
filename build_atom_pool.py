from tqdm import tqdm
import numpy as np
import pickle
import time
import os
from sklearn.metrics import classification_report, roc_auc_score

# Import from custom files
from utils import parse_arguments, reset_seed
from atom_pool import AtomTokenizer, AtomPool, get_word_count
from dataset import get_dataset, get_class_names
from dataset import get_tabular_column_type, get_tabular_category_map, get_tabular_numerical_threshold, get_tabular_numerical_max

if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    reset_seed(seed)
    
    datasets = get_dataset(dataset=args.dataset, seed=seed)
    
    if args.dataset in ['yelp', 'clickbait']:
        train_df, valid_df, test_df = datasets
    elif args.dataset in ['adult']:
        number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df = datasets
    else:
        assert(0)

    # Create atom tokenizer. Note that we use different tokenizer for text embedding and atom creation.
    CREATE = True

    if 'save_dir' not in os.listdir('.'):
        os.system('mkdir ./save_dir')
    
    if args.dataset in ['yelp', 'clickbait']:
        if CREATE:
            atom_tokenizer = AtomTokenizer(train_df, dataset=args.dataset)
            with open(f'./save_dir/atom_tokenizer_{args.dataset}.pkl', 'wb') as f:
                pickle.dump(atom_tokenizer, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'./save_dir/atom_tokenizer_{args.dataset}.pkl', 'rb') as f:
                atom_tokenizer = pickle.load(f)

        # Create or load the word count tensor for training dataset.
        # The word count tensor indicates number of words in each sample.
        # The shape of train_x_ is (number of train samples, vocab size of tokenizer)

        train_x_ = get_word_count(
            train_df,
            tokenizer=atom_tokenizer,
            dataset=args.dataset,
            create=CREATE,
            save_path=f'./save_dir/train_x_atom_tokenizer_{args.dataset}.pkl'
        )
        train_y_ = np.array(train_df['label']).astype(int)
    elif args.dataset == 'adult':
        atom_tokenizer = None
        train_x_ = np.array(number_train_df.drop(columns=['income']))
        train_y_ = np.array(number_train_df['income']).astype(int)
    else:
        assert(0)
        

    if args.dataset in ['yelp', 'clickbait']:
        test_x_ = get_word_count(
            test_df,
            tokenizer=atom_tokenizer,
            dataset=args.dataset,
            create=CREATE,
            save_path=f'./save_dir/test_x_atom_tokenizer_{args.dataset}.pkl'
        )
        test_y_ = np.array(test_df['label']).astype(int)
    elif args.dataset == 'adult':
        test_x_ = np.array(number_test_df.drop(columns=['income']))
        test_y_ = np.array(number_test_df['income']).astype(int)
    else:
        assert(0)

    # Build atom pool.
    if args.dataset in ['yelp', 'clickbait']:
        ap = AtomPool(
            atom_tokenizer,
            None,
            None,
            train_x_,
            train_y_,
            dataset=args.dataset,
        )
        
        # Sort by frequency
        s = np.sum(train_x_, axis=0)
        a = np.argsort(s)[::-1]

        # Add dummy atom
        ap.add_atom('dummy', None, None, None, None)

        k = args.num_atoms

        # Exclude meaningless tokens
        remove_list = [
            '[PAD]', '.', ',', '', '[UNK]',
            'the', 'a', 'an',
            'i', 'my', 'me',
            'he', 'him', 'his',
            'she', 'her',
            'it', 'its',
            'we', 'our', 'us',
            'you', 'your', 
            'they', 'their', 'them', 
            'this', 'that', 'there', 'here',
            'to', 'of', 'in', 'for', 'and', 'with', 'on', 'at', 'as', 'from',
            'will', 'would',
            'is', 'was', 'are', 'were', 'be', 'been',
            'have', 'had', 'told', 'said', 'asked', 'asking',
            'given', 'telling',
        ]

        # Add atoms according to their frequency
        n = 0
        for i, feature_idx in enumerate(tqdm(a) ):
            feature_idx = int(feature_idx)
            word = feature_idx % atom_tokenizer.vocab_size
            pos = feature_idx // atom_tokenizer.vocab_size

            w = atom_tokenizer.idx2word[word]
            if w not in remove_list and len(w) > 1 and w.isalpha():
                ap.add_atom('text', word, True, 0.5, pos)
                n += 1
                if n == k:
                    break
                    
        ap.display_atoms(n)

        n_atom = ap.num_atoms()
        print(f'{n_atom} atoms added')

        with open(f'./save_dir/atom_pool_{args.dataset}_num_atoms_{args.num_atoms}.pkl', 'wb') as f:
            pickle.dump(ap, f, pickle.HIGHEST_PROTOCOL)
            
    elif args.dataset in ['adult']:
        categorical_x_col, numerical_x_col, y_col = get_tabular_column_type(dataset=args.dataset)
        cat_map = get_tabular_category_map(dataset=args.dataset)
        numerical_threshold = get_tabular_numerical_threshold(dataset=args.dataset)
        numerical_max = get_tabular_numerical_max(dataset=args.dataset)
        
        ap = AtomPool(
            None,
            number_train_df.columns.tolist(),
            cat_map,
            train_x_,
            train_y_,
            dataset=args.dataset,
        )
        
        ap.add_atom('dummy', None, None, None)
        
        for cat in categorical_x_col:
            for k in cat_map[f'{cat}_idx2key']:
                ap.add_atom('categorical', cat, True, k)
                ap.add_atom('categorical', cat, False, k)

        for cat in numerical_x_col:
            m = numerical_max[cat]
            for n in numerical_threshold[cat]:
                ap.add_atom('numerical', cat, True, n / m)
                ap.add_atom('numerical', cat, False, n / m)

        ap.display_atoms()

        n_atom = ap.num_atoms()
        print(f'{n_atom} atoms added')

        with open(f'./save_dir/atom_pool_{args.dataset}.pkl', 'wb') as f:
            pickle.dump(ap, f, pickle.HIGHEST_PROTOCOL)