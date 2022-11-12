"""
The script to sample antecedents for pretraining of consequent estimator
"""
import pickle
import os
import time
from tqdm import tqdm
import numpy as np
import torch

# Import from custom files
from selor_utils import atom as at
from selor_utils import utils
from selor_utils import dataset as ds

def get_tm_satis(tm_left, tm_right, min_df=200):
    """
    Conduct matrix multiplication to obtain the antecedents that satisfies both matrices.
    """
    with torch.no_grad():
        tm_ = torch.mm(tm_left.float(), tm_right.float()).int()
        _, n_j = tm_.shape

        tm_satis = (tm_ > min_df).flatten().nonzero(as_tuple=True)[0]
        tm_satis = torch.stack((torch.div(tm_satis,  n_j, rounding_mode='trunc'), tm_satis % n_j))
        tm_satis = tm_satis.T

    return tm_satis

if __name__ == "__main__":
    args = utils.parse_arguments()
    dtype = ds.get_dataset_type(args.dataset)
    gamma = args.min_df
    utils.reset_seed(args.seed)

    gpu = torch.device(f'cuda:{args.gpu}')

    atom_pool_path = f'./{args.save_dir}/atom_pool/atom_pool_{args.dataset}'
    if dtype == 'nlp':
        atom_pool_path += f'_num_atoms_{args.num_atoms}'
    atom_pool_path += '.pkl'
    with open(atom_pool_path, 'rb') as f:
        ap = pickle.load(f)

    # Whether each train sample satisfies each atom
    tm = at.get_true_matrix(ap).to(gpu)
    sampling_start = time.time()

    if 'tm_satis' not in os.listdir(f'./{args.save_dir}'):
        os.system(f'mkdir ./{args.save_dir}/tm_satis')

    print('Create tm_2_satis')
    print(tm.shape)
    tm_2_satis = get_tm_satis(tm, tm.T, min_df=args.min_df)
    tm_2_satis = {tuple(sorted(r)) for r in tqdm(tm_2_satis.tolist()) if r[0] != r[1]}
    tm_2_satis = sorted(list(tm_2_satis))

    tm_2_satis_path = f'./{args.save_dir}/tm_satis/tm_2_satis_dataset_{args.dataset}'
    if dtype == 'nlp':
        tm_2_satis_path += f'_num_atoms_{args.num_atoms}'
    tm_2_satis_path += f'_min_df_{args.min_df}.pkl'
    with open(tm_2_satis_path, 'wb') as f:
        pickle.dump(tm_2_satis, f, pickle.HIGHEST_PROTOCOL)


    print('Create or load tm_2')
    tm_2 = np.zeros((len(tm_2_satis), ap.n_data))
    sd = ap.atom_satis_dict
    for k, c_list in enumerate(tqdm(tm_2_satis)):
        rc_list = [ap.atom_id2key[c] for c in c_list]
        satis_list = [set(sd[rc]) for rc in rc_list]
        satis = set.intersection(*satis_list)
        for s in satis:
            tm_2[k, s] = 1
    tm_2 = torch.tensor(tm_2)
    tm_2 = tm_2[:min(tm_2.shape[0], gamma * args.pretrain_samples), :]

    tm_2_path = f'./{args.save_dir}/tm_satis/tm_2_dataset_{args.dataset}'
    if dtype == 'nlp':
        tm_2_path += f'_num_atoms_{args.num_atoms}'
    tm_2_path += f'_min_df_{args.min_df}.pkl'
    with open(tm_2_path, 'wb') as f:
        pickle.dump(tm_2, f, pickle.HIGHEST_PROTOCOL)

    print('Create or load tm_3_satis')
    tm_3_satis_ = get_tm_satis(tm.cpu(), tm_2.T, min_df=args.min_df)
    tm_3_satis = list({
        tuple(sorted(
            [r[0]] + list(tm_2_satis[r[1]])
        )) for r in tqdm(tm_3_satis_.tolist())
    })
    tm_3_satis = sorted(list(tm_3_satis))
    tm_3_satis = [r for r in tqdm(tm_3_satis) if len(set(r)) == len(r)]

    tm_3_satis_path = f'./{args.save_dir}/tm_satis/tm_3_satis_dataset_{args.dataset}'
    if dtype == 'nlp':
        tm_3_satis_path += f'_num_atoms_{args.num_atoms}'
    tm_3_satis_path += f'_min_df_{args.min_df}.pkl'
    with open(tm_3_satis_path, 'wb') as f:
        pickle.dump(tm_3_satis, f, pickle.HIGHEST_PROTOCOL)

    print('Create or load tm_4_satis')
    tm_4_satis_ = get_tm_satis(tm_2, tm_2.T, min_df=args.min_df)
    tm_4_satis = list({
        tuple(sorted(
            list(tm_2_satis[r[0]]) + list(tm_2_satis[r[1]])
        )) for r in tqdm(tm_4_satis_.tolist())
    })
    tm_4_satis = sorted(list(tm_4_satis))
    tm_4_satis = [r for r in tqdm(tm_4_satis) if len(set(r)) == len(r)]

    tm_4_satis_path = f'./{args.save_dir}/tm_satis/tm_4_satis_dataset_{args.dataset}'
    if dtype == 'nlp':
        tm_4_satis_path += f'_num_atoms_{args.num_atoms}'
    tm_4_satis_path += f'_min_df_{args.min_df}.pkl'
    with open(tm_4_satis_path, 'wb') as f:
        pickle.dump(tm_4_satis, f, pickle.HIGHEST_PROTOCOL)

    sampling_end = time.time()
    print(f'Sampling for CP Predictor: {sampling_end - sampling_start}')
