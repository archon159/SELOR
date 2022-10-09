import argparse
import random
import numpy as np
import torch

def reset_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def parse_arguments(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--max_rule_len', type=int, default=4, help='Maximum rule length')
    parser.add_argument('--hidden_dim', type=int, default=768, help='The hidden dimension of RCN model.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for preprocessing')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset')
    parser.add_argument('--base_model', type=str, default='bert', help='Base Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num_atoms', type=int, default=5000, help='Number of words to use for atoms, Only for NLP datasets.')
    parser.add_argument('--min_df', type=int, default=200, help='Minimum data frequency for a rule')
    parser.add_argument('--max_df', type=float, default=0.95, help='Maximum data frequency for a rule')
    parser.add_argument('--pretrain_samples', type=int, default=10000, help='Number of samples for pretraining of consequent estimator.')
    parser.add_argument('--only_eval', action='store_true', help='Do only evaluation')
    
    if notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    nlp_dataset = ['yelp', 'clickbait']
    nlp_base_model = ['bert', 'roberta']
    
    tabular_dataset = ['adult']
    tabular_base_model = ['dnn']
    
    if args.dataset in nlp_dataset:
        assert(args.base_model in nlp_base_model)
        
    elif args.dataset in tabular_dataset:
        assert(args.base_model in tabular_base_model)
        
    return args