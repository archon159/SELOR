import torch
import pickle
import numpy as np
import time
from tqdm import tqdm

from atom_pool import AtomPool, get_true_matrix
from utils import parse_arguments, reset_seed

def get_tm_satis(tm_left, tm_right, min_df=200):
    with torch.no_grad():
        tm_ = torch.mm(tm_left.float(), tm_right.float()).int()
        n_i, n_j = tm_.shape

        tm_satis = (tm_ > min_df).flatten().nonzero(as_tuple=True)[0]
        tm_satis = torch.stack((torch.div(tm_satis,  n_j, rounding_mode='trunc'), tm_satis % n_j))
        tm_satis = tm_satis.T

    return tm_satis

if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    reset_seed(seed)
    
    if args.dataset in ['yelp', 'clickbait']:
        with open(f'./save_dir/atom_pool_{args.dataset}_num_atoms_{args.num_atoms}.pkl', 'rb') as f:
            ap = pickle.load(f)
            
    elif args.dataset in ['adult']:
        with open(f'./save_dir/atom_pool_{args.dataset}.pkl', 'rb') as f:
            ap = pickle.load(f)
        
    gamma = args.min_df
    ap.display_atoms(n=10)
    
    # Whether each train sample satisfies each atom
    tm = get_true_matrix(ap).to(gpu)
    sampling_start = time.time()
    
    CREATE = True
    print('Create or load tm_2_satis')
    if CREATE:
        print(tm.shape)
        tm_2_satis = get_tm_satis(tm, tm.T, min_df=args.min_df)
        tm_2_satis = set([tuple(sorted(r)) for r in tqdm(tm_2_satis.tolist()) if r[0] != r[1]])
        tm_2_satis = sorted(list(tm_2_satis))
        
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_2_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_2_satis, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'./save_dir/tm_2_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_2_satis, f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_2_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_2_satis = pickle.load(f)
        else:
            with open(f'./save_dir/tm_2_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_2_satis = pickle.load(f)
    
    print('Create or load tm_2')
    if CREATE:
        tm_2 = np.zeros((len(tm_2_satis), len(ap.train_x_)))
        sd = ap.atom_satis_dict
        for k, c_list in enumerate(tqdm(tm_2_satis)):
            rc_list = [ap.atom_id2key[c] for c in c_list]
            satis_list = [set(sd[rc]) for rc in rc_list]
            satis = set.intersection(*satis_list)
            for s in satis:
                tm_2[k, s] = 1
        tm_2 = torch.tensor(tm_2)
        tm_2 = tm_2[:min(tm_2.shape[0], gamma * args.pretrain_samples), :]
        
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_2_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_2, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'./save_dir/tm_2_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_2, f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_2_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_2 = pickle.load(f)
        else:
            with open(f'./save_dir/tm_2_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_2 = pickle.load(f)
    
    print('Create or load tm_3_satis')
    if CREATE:
        tm_3_satis_ = get_tm_satis(tm.cpu(), tm_2.T, min_df=args.min_df)
        tm_3_satis = list(set([tuple(sorted([r[0]] + list(tm_2_satis[r[1]]))) for r in tqdm(tm_3_satis_.tolist())]))
        tm_3_satis = sorted(list(tm_3_satis))
        tm_3_satis = [r for r in tqdm(tm_3_satis) if len(set(r)) == len(r)]
        
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_3_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_3_satis, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'./save_dir/tm_3_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_3_satis, f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_3_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_3_satis = pickle.load(f)
        else:
            with open(f'./save_dir/tm_3_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_3_satis = pickle.load(f)

    print('Create or load tm_4_satis')
    if CREATE:
        tm_4_satis_ = get_tm_satis(tm_2, tm_2.T, min_df=args.min_df)
        tm_4_satis = list(set([tuple(sorted(list(tm_2_satis[r[0]]) + list(tm_2_satis[r[1]]))) for r in tqdm(tm_4_satis_.tolist())]))
        tm_4_satis = sorted(list(tm_4_satis))
        tm_4_satis = [r for r in tqdm(tm_4_satis) if len(set(r)) == len(r)]
        
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_4_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_4_satis, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'./save_dir/tm_4_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'wb') as f:
                pickle.dump(tm_4_satis, f, pickle.HIGHEST_PROTOCOL)
    else:
        if args.dataset in ['yelp', 'clickbait']:
            with open(f'./save_dir/tm_4_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_4_satis = pickle.load(f)
        else:
            with open(f'./save_dir/tm_4_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_4_satis = pickle.load(f)
            
    sampling_end = time.time()
    print(f'Sampling for CP Predictor: {sampling_end - sampling_start}')