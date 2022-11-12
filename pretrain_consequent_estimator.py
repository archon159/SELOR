"""
The script to pretrain the consequent estimator
"""
import pickle
import copy
import time
import os
import torch
from prettytable import PrettyTable

# Import from custom files
from selor_utils import atom as at
from selor_utils import dataset as ds
from selor_utils import net
from selor_utils import train_eval as te
from selor_utils import utils


if __name__ == "__main__":
    args = utils.parse_arguments()

    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)

    assert dtype == btype

    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    utils.reset_seed(seed)

    if 'consequent_estimators' not in os.listdir(f'./{args.save_dir}'):
        os.system(f'mkdir ./{args.save_dir}/consequent_estimators')

    ce_path = f'ce_{args.base}_dataset_{args.dataset}_pretrain_samples_{args.pretrain_samples}'
    if dtype == 'nlp':
        ce_path += f'_num_atoms_{args.num_atoms}'

    dir_path = f'./{args.save_dir}/consequent_estimators/{ce_path}'
    class_names = ds.get_class_names(args.dataset)

    print("Loading atom_pool")
    atom_pool_path = f'./{args.save_dir}/atom_pool/atom_pool_{args.dataset}'
    if dtype == 'nlp':
        atom_pool_path += f'_num_atoms_{args.num_atoms}'
    atom_pool_path += '.pkl'
    with open(atom_pool_path, 'rb') as f:
        ap = pickle.load(f)

    # Whether each train sample satisfies each atom
    print("Loading true_matrix")
    true_matrix = at.get_true_matrix(ap)
    norm_true_matrix = true_matrix / (torch.sum(true_matrix, dim=1).unsqueeze(dim=1) + 1e-8)

    # Embedding from the base model for each train sample
    print("Loading base_embedding")
    base_embedding_path = f'./{args.save_dir}/base_models/'
    base_embedding_path += f'base_{args.base}_dataset_{args.dataset}/'
    base_embedding_path += 'train_embeddings.pt'
    base_embedding = torch.load(base_embedding_path)
    n_data, hidden_dim = base_embedding.shape

    # Obtain atom embedding
    atom_embedding = torch.mm(norm_true_matrix.to(gpu), base_embedding.to(gpu)).detach()

    ce_model = net.ConsequentEstimator(
        n_class=len(class_names),
        hidden_dim=hidden_dim,
        atom_embedding=atom_embedding,
    ).to(gpu)

    model_pretrain_dict = {}
    if args.only_eval:
        with open(f'{dir_path}/test_dataloader_dict', 'rb') as f:
            test_dataloader_dict = pickle.load(f)

        for i in range(1, args.antecedent_len + 1):
            print(f"Loading consequent estimator for length {i} antecedents")
            ce_model_path = f'{dir_path}/'
            ce_model_path += f'ce_pretrain_{i}_{args.base}_dataset_{args.dataset}.pt'
            ce_model.load_state_dict(torch.load(ce_model_path), strict=True)
            model_pretrain_dict[i] = copy.deepcopy(ce_model)

    else:
        if ce_path not in os.listdir(f'./{args.save_dir}/consequent_estimators'):
            os.system(f'mkdir {dir_path}')

        # Create datasets
        train_df, valid_df, test_df = ds.load_data(dataset=args.dataset)
        col_label = ds.get_label_column(args.dataset)
        train_y = torch.tensor(train_df[col_label]).float()

        n_atom = ap.num_atoms()

        print("Loading sampled antecedents")
        cand_dict = {}
        for i in range(1, args.antecedent_len + 1):
            if args.min_df == 0:
                cand_dict[i] = None
            else:
                if i == 1:
                    candidate = [(j,) for j in range(n_atom)]
                else:
                    tm_path = f'./{args.save_dir}/tm_satis/tm_{i}_satis'
                    tm_path += f'_dataset_{args.dataset}'
                    if dtype == 'nlp':
                        tm_path += f'_num_atoms_{args.num_atoms}'
                    tm_path += f'_min_df_{args.min_df}'
                    tm_path += '.pkl'

                    with open(tm_path, 'rb') as f:
                        candidate = pickle.load(f)

                cand_dict[i] = torch.tensor(candidate)

        sampling_start = time.time()

        pretrain_dataset_dict = {}
        print()
        for i in range(1, args.antecedent_len + 1):
            print(f"Creating datasets for antecedent length {i} antecedents")
            n_sample = min(args.pretrain_samples, len(cand_dict[i]))
            pretrain_dataset = ds.create_pretrain_dataset(
                candidate=cand_dict[i].to(gpu),
                true_matrix=true_matrix.to(gpu),
                train_y=train_y.to(gpu),
                n_sample=n_sample,
                num_classes=len(class_names),
                min_df=args.min_df,
                max_df=args.max_df,
            )
            pretrain_dataset_dict[i] = pretrain_dataset

        sampling_end = time.time()

        with open(f'{dir_path}/pretrain_dataset_dict', 'wb') as f:
            pickle.dump(pretrain_dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Create dataloaders
        train_dataloader_dict = {}
        test_dataloader_dict = {}
        w_mu_dict = {}
        w_sigma_dict = {}
        w_coverage_dict = {}

        print()
        print("Creating dataloaders")
        for i in range(1, args.antecedent_len + 1):
            train_dataloader, test_dataloader = ds.create_pretrain_dataloader(
                pretrain_dataset=pretrain_dataset_dict[i],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            train_dataloader_dict[i] = train_dataloader
            test_dataloader_dict[i] = test_dataloader

        with open(f'{dir_path}/train_dataloader_dict', 'wb') as f:
            pickle.dump(train_dataloader_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{dir_path}/test_dataloader_dict', 'wb') as f:
            pickle.dump(test_dataloader_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print()
        print("Creating weights")
        for i in range(1, args.antecedent_len + 1):
            w_mu, w_sigma, w_coverage = ds.get_weight(
                pretrain_dataset_dict[i],
                n_data
            )
            w_mu_dict[i] = w_mu
            w_sigma_dict[i] = w_sigma
            w_coverage_dict[i] = w_coverage

        training_start = time.time()

        for i in range(1, args.antecedent_len + 1):
            print()
            print(f"Pretraining consequent estimator for length {i} antecedents")
            ce_model = te.pretrain(
                ce_model=ce_model,
                train_dataloader=train_dataloader_dict[i],
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                max_antecedent_len=args.antecedent_len,
                n_data=n_data,
                w_mu=w_mu_dict[i],
                w_sigma=w_sigma_dict[i],
                w_coverage=w_coverage_dict[i],
                gpu=gpu
            )
            ce_save_path = f'{dir_path}/'
            ce_save_path += f'ce_pretrain_{i}_{args.base}_dataset_{args.dataset}.pt'
            torch.save(ce_model.state_dict(), ce_save_path)

            model_pretrain_dict[i] = copy.deepcopy(ce_model)

        training_end = time.time()
        print(f'Sampling: {sampling_end - sampling_start}')
        print(f'Training: {training_end - training_start}')

    # Evaluation
    result_table = PrettyTable()
    result_table.field_names = ["Model"] + [
        f"Length {i} Test" for i in range(1, args.antecedent_len + 1)
    ]

    for i, pm in model_pretrain_dict.items():
        row = [f'Length {i} Pretraining']
        for j, dl in test_dataloader_dict.items():
            avg_mu_err, avg_sigma_err, avg_coverage_err, f1 = te.eval_pretrain(
                pm,
                dl,
                args.antecedent_len,
                n_data,
                class_names,
                gpu
            )
            row.append(round(f1, 4))
        result_table.add_row(row)

    with open(f'{dir_path}/ce_pretrain_eval', 'w', encoding='utf-8') as feval:
        print(result_table, file=feval, flush=True)
