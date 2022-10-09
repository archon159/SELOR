import torch
from torch.utils.data import  Dataset, DataLoader, random_split
import pickle
import copy
import numpy as np
import time
from tqdm import tqdm
import os

from model import ConsequentEstimator
from dataset import get_dataset, get_class_names
from atom_pool import AtomPool, get_true_matrix
from utils import parse_arguments, reset_seed
from train_eval import pretrain, eval_pretrain

def create_pretrain_dataloader(pretrain_dataset, test_ratio=0.2):
    n_test = int(len(pretrain_dataset) * 0.2)
    n_train = len(pretrain_dataset) - n_test

    pretrain_test, pretrain_train = random_split(
        pretrain_dataset,
        [n_test, n_train],
        torch.Generator().manual_seed(args.seed)
    )
    pretrain_train_dataloader = DataLoader(
        pretrain_train,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        persistent_workers=False,
        pin_memory=False,
    )
    pretrain_test_dataloader = DataLoader(
        pretrain_test,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
    )
    
    return pretrain_train_dataloader, pretrain_test_dataloader

def get_weight(pretrain_dataset, n_data, dataset='yelp'):
    if dataset == 'yelp':
        noise_mu = torch.std(torch.tensor(pretrain_dataset.mu))
        noise_sigma = torch.std(torch.tensor(pretrain_dataset.sigma))
        noise_coverage = torch.std(torch.tensor(pretrain_dataset.n).float() / n_data)
    else:
        noise_mu = torch.mean(torch.std(torch.stack(pretrain_dataset.mu, dim=0), dim=0))
        noise_sigma = torch.mean(torch.std(torch.stack(pretrain_dataset.sigma, dim=0), dim=0))
        noise_coverage = torch.std(torch.tensor(pretrain_dataset.n).float() / n_data)
    
    weight_mu = 1 / (2 * (noise_mu ** 2))
    weight_sigma = 1 / (2 * (noise_sigma ** 2))
    weight_coverage = 1 / (2 * (noise_coverage ** 2))
    
    return weight_mu, weight_sigma, weight_coverage

def get_pretrain_dataset(
    n_sample,
    rule_length,
    true_matrix,
    train_y_,
    candidate,
    create,
    dir_path,
    args
):
    if create:
        if args.dataset == 'yelp':
            pretrain_dataset = YelpPretrainDataset(
                true_matrix,
                train_y_,
                n_sample,
                rule_length,
                candidate=candidate,
                args=args,
            )
        elif args.dataset == 'clickbait':
            pretrain_dataset = ClickbaitPretrainDataset(
                true_matrix,
                train_y_,
                n_sample,
                rule_length,
                candidate=candidate,
                args=args,
            )
        elif args.dataset == 'adult':
            pretrain_dataset = AdultPretrainDataset(
                true_matrix,
                train_y_,
                n_sample,
                rule_length,
                candidate=candidate,
                args=args,
            )
        else:
            assert(0)
        
        with open(f'{dir_path}/pretrain_dataset_{rule_length}_n_sample_{n_sample}_min_df_{args.min_df}_max_df_{args.max_df}.pkl', 'wb') as f:
            pickle.dump(pretrain_dataset, f, pickle.HIGHEST_PROTOCOL)
            
    else:
        with open(f'{dir_path}/pretrain_dataset_{rule_length}_n_sample_{n_sample}_min_df_{args.min_df}_max_df_{args.max_df}.pkl', 'rb') as f:
            pretrain_dataset = pickle.load(f)
    
    return pretrain_dataset

def get_pretrained_model(
    ce_model,
    pretrain_train_dataloader,
    rule_length,
    atom_embedding,
    n_data,
    weight_mu,
    weight_sigma,
    weight_coverage,
    create,
    dir_path,
    args
):
    if create:
        ce_model = pretrain(
            ce_model,
            pretrain_train_dataloader,
            atom_embedding,
            n_data,
            weight_mu,
            weight_sigma,
            weight_coverage,
            args
        )
        torch.save(ce_model.state_dict(), f'{dir_path}/ce_pretrain_{rule_length}_{args.base_model}_dataset_{args.dataset}.pt')
    else:
        ce_model.load_state_dict(
            torch.load(f'{dir_path}/ce_pretrain_{rule_length}_{args.base_model}_dataset_{args.dataset}.pt'),
            strict=True
        )
    
    return ce_model


if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    reset_seed(seed)
    
    datasets = get_dataset(dataset=args.dataset, seed=seed)
    if args.dataset == 'yelp':
        from dataset import YelpPretrainDataset
        pretrain_dataset_type = YelpPretrainDataset
        train_df, valid_df, test_df = datasets
    elif args.dataset == 'clickbait':
        from dataset import ClickbaitPretrainDataset
        pretrain_dataset_type = ClickbaitPretrainDataset
        train_df, valid_df, test_df = datasets
    elif args.dataset == 'adult':
        from dataset import AdultPretrainDataset
        pretrain_dataset_type = AdultPretrainDataset
        number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df = datasets
    else:
        assert(0)
        
    if args.dataset in ['yelp', 'clickbait']:
        train_y_ = train_df['label']
        with open(f'./save_dir/atom_pool_{args.dataset}_num_atoms_{args.num_atoms}.pkl', 'rb') as f:
            ap = pickle.load(f)
            
    elif args.dataset in ['adult']:
        train_y_ = number_train_df['income']
        with open(f'./save_dir/atom_pool_{args.dataset}.pkl', 'rb') as f:
            ap = pickle.load(f)
        
    n_atom = ap.num_atoms()
    
    if args.min_df == 0:
        tm_2_satis = None
        tm_3_satis = None
        tm_4_satis = None
    else:
        if args.dataset in ['yelp', 'mnli', 'clickbait']:
            with open(f'./save_dir/tm_2_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_2_satis = pickle.load(f)

            with open(f'./save_dir/tm_3_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_3_satis = pickle.load(f)

            with open(f'./save_dir/tm_4_satis_dataset_{args.dataset}_num_atoms_{args.num_atoms}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_4_satis = pickle.load(f)
        else:
            with open(f'./save_dir/tm_2_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_2_satis = pickle.load(f)

            with open(f'./save_dir/tm_3_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_3_satis = pickle.load(f)

            with open(f'./save_dir/tm_4_satis_dataset_{args.dataset}_min_df_{args.min_df}.pkl', 'rb') as f:
                tm_4_satis = pickle.load(f)
        
        print(len(tm_2_satis), len(tm_3_satis), len(tm_4_satis))
        
    # Whether each train sample satisfies each atom
    true_matrix = get_true_matrix(ap)
    norm_true_matrix = true_matrix / (torch.sum(true_matrix, dim=1).unsqueeze(dim=1) + 1e-8)

    # Embedding from the base model for each train sample
    data_embedding = torch.load(f'./save_dir/base_models/base_{args.base_model}_dataset_{args.dataset}/train_embeddings.pt')
    n_data, hidden_dim = data_embedding.shape
    
    # Obtain atom embedding
    atom_embedding = torch.mm(norm_true_matrix.to(gpu), data_embedding.to(gpu)).detach()
    
    TARGET_PATH = f'ce_{args.base_model}_dataset_{args.dataset}_pretrain_samples_{args.pretrain_samples}'
    if args.base_model in ['bert', 'roberta']:
        TARGET_PATH += f'_num_atoms_{args.num_atoms}'
        
    if 'consequent_estimators' not in os.listdir('./save_dir'):
        os.system(f'mkdir ./save_dir/consequent_estimators')
        
    DIR_PATH = f'./save_dir/consequent_estimators/{TARGET_PATH}'
    if TARGET_PATH not in os.listdir('./save_dir/consequent_estimators'):
        os.system(f'mkdir {DIR_PATH}')
        
    sampling_start = time.time()
    CREATE = True
    
    n_sample_dict = {}
    n_sample_dict[1] = min(args.pretrain_samples, n_atom)
    n_sample_dict[2] = min(args.pretrain_samples, len(tm_2_satis))
    n_sample_dict[3] = min(args.pretrain_samples, len(tm_3_satis))
    n_sample_dict[4] = min(args.pretrain_samples, len(tm_4_satis))
    
    candidate_list = [None, tm_2_satis, tm_3_satis, tm_4_satis]
    pretrain_dataset_dict = {}
    for i in range(1, args.max_rule_len + 1):
        pretrain_dataset = get_pretrain_dataset(
            n_sample_dict[i],
            i,
            true_matrix,
            train_y_,
            candidate = candidate_list[i - 1],
            create=CREATE,
            dir_path=DIR_PATH,
            args=args,
        )
        pretrain_dataset_dict[i] = pretrain_dataset
    
    sampling_end = time.time()
    
    # Create dataloaders
    pretrain_train_dataloader_dict = {}
    pretrain_test_dataloader_dict = {}
    weight_mu_dict = {}
    weight_sigma_dict = {}
    weight_coverage_dict = {}
    
    for i in range(1, args.max_rule_len + 1):
        pretrain_train_dataloader, pretrain_test_dataloader = create_pretrain_dataloader(pretrain_dataset_dict[i])
        pretrain_train_dataloader_dict[i] = pretrain_train_dataloader
        pretrain_test_dataloader_dict[i] = pretrain_test_dataloader
    
        weight_mu, weight_sigma, weight_coverage = get_weight(pretrain_dataset_dict[i], n_data, args.dataset)
        weight_mu_dict[i] = weight_mu
        weight_sigma_dict[i] = weight_sigma
        weight_coverage_dict[i] = weight_coverage
    
    class_names = get_class_names(args.dataset)
    
    ce_model = ConsequentEstimator(
        n_class=len(class_names),
        hidden_dim=hidden_dim,
        atom_embedding=atom_embedding,
        args=args,
    ).to(gpu)
    
    training_start = time.time()
    
    model_pretrain_dict = {}
    
    for i in range(1, args.max_rule_len + 1):
        model_pretrain = get_pretrained_model(
            ce_model,
            pretrain_train_dataloader_dict[i],
            i,
            atom_embedding,
            n_data,
            weight_mu_dict[i],
            weight_sigma_dict[i],
            weight_coverage_dict[i],
            create=CREATE,
            dir_path=DIR_PATH,
            args=args,
        )
        model_pretrain_dict[i] = copy.deepcopy(model_pretrain)
        
    training_end = time.time()
    
    print(f'Sampling: {sampling_end - sampling_start}')
    print(f'Training: {training_end - training_start}')

    with open(f'{DIR_PATH}/ce_pretrain_eval', 'w') as feval:
        for i, dl in pretrain_test_dataloader_dict.items():
            for j, pm in model_pretrain_dict.items():
                avg_mu_err, avg_sigma_err, avg_coverage_err, f1 = eval_pretrain(
                    pm, dl, atom_embedding, n_data, class_names, args
                )
                print(f'Dataloader {i}, Pretrained Model {j}', file=feval, flush=True)
                print(f'Avg Mu Err: {avg_mu_err:.4f}, Avg Coverage Err: {avg_coverage_err:.4f}, F1 Score: {f1:.4f}', file=feval, flush=True)