VERSION_NAME = 'rule_gen'

import os
import json
import pickle
from datetime import datetime
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from custom files
from model import RuleGenerator, ConsequentEstimator
from atom_pool import AtomTokenizer, AtomPool, get_true_matrix
from dataset import get_class_names, create_dataloader, get_dataset
from utils import parse_arguments, reset_seed
from train_eval import train, eval_model

if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    reset_seed(seed)
    
    if args.base_model == 'bert':
        from transformers import BertModel, BertTokenizer, BertConfig

        MAX_LEN = 512
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        tf_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        tf_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
        config = BertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        vocab_size = tf_tokenizer.vocab_size
        pad_token_id = tf_tokenizer.pad_token_id

    elif args.base_model == 'roberta':
        from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

        MAX_LEN = 512
        PRE_TRAINED_MODEL_NAME = 'roberta-base'
        tf_tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        tf_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=True)
        config = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        vocab_size = tf_tokenizer.vocab_size
        pad_token_id = tf_tokenizer.pad_token_id

    elif args.base_model == 'lstm':
        from transformers import BertTokenizer, BertConfig

        MAX_LEN = 512
        PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
        tf_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        config = BertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
        tf_model = None
        vocab_size = tf_tokenizer.vocab_size
        pad_token_id = tf_tokenizer.pad_token_id
        
    else:
        tf_tokenizer = None
        tf_model = None
        vocab_size = 0
        pad_token_id = 0
    
    datasets = get_dataset(dataset=args.dataset)
    
    if args.dataset in ['yelp', 'clickbait']:
        with open(f'./save_dir/atom_tokenizer_{args.dataset}.pkl', 'rb') as f:
            atom_tokenizer = pickle.load(f)

        with open(f'./save_dir/atom_pool_{args.dataset}_num_atoms_{args.num_atoms}.pkl', 'rb') as f:
            ap = pickle.load(f)
            
    elif args.dataset in ['adult']:
        atom_tokenizer = None

        with open(f'./save_dir/atom_pool_{args.dataset}.pkl', 'rb') as f:
            ap = pickle.load(f)
        
    # Create datasets
    if args.dataset == 'yelp':
        train_df, valid_df, test_df = datasets
        
        from dataset import YelpDataset
        train_dataset = YelpDataset(
            train_df,
            atom_pool=ap,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
            args=args,
        )
        
        valid_dataset = YelpDataset(
            valid_df,
            atom_pool=ap,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
            args=args,
        )
        
        test_dataset = YelpDataset(
            test_df,
            atom_pool=ap,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
            args=args,
        )

        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        
    elif args.dataset == 'clickbait':
        train_df, valid_df, test_df = datasets
        
        from dataset import ClickbaitDataset
        train_dataset = ClickbaitDataset(
            train_df,
            atom_pool=ap,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
            args=args,
        )
        
        valid_dataset = ClickbaitDataset(
            valid_df,
            atom_pool=ap,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
            args=args,
        )
        
        test_dataset = ClickbaitDataset(
            test_df,
            atom_pool=ap,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
            args=args,
        )
        
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        
    elif args.dataset == 'adult':
        number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df = datasets
        
        from dataset import AdultDataset
        train_dataset = AdultDataset(
            number_train_df,
            dummy_train_df,
            atom_pool=ap,
            args=args,
        )
        
        valid_dataset = AdultDataset(
            number_valid_df,
            dummy_valid_df,
            atom_pool=ap,
            args=args,
        )
        
        test_dataset = AdultDataset(
            number_test_df,
            dummy_test_df,
            atom_pool=ap,
            args=args,
        )
        
        input_dim = train_dataset.x_dummy.shape[1]
        hidden_dim = 512
        
    train_dataloader = create_dataloader(train_dataset, args, shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset, args, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, args, shuffle=False)
        
    # Load class names
    class_names = get_class_names(args.dataset)
    
    # Whether each train sample satisfies each atom
    true_matrix = get_true_matrix(ap)
    norm_true_matrix = true_matrix / (torch.sum(true_matrix, dim=1).unsqueeze(dim=1) + 1e-8)

    # Embedding from the base model for each train sample
    data_embedding = torch.load(f'./save_dir/base_models/base_{args.base_model}_dataset_{args.dataset}/train_embeddings.pt')
        
    # Obtain atom embedding
    atom_embedding = torch.mm(norm_true_matrix.to(gpu), data_embedding.to(gpu)).detach()
    
    ce_model = ConsequentEstimator(
        n_class=len(class_names),
        hidden_dim=hidden_dim,
        atom_embedding=atom_embedding,
        args=args,
    ).to(gpu)


    CE_DIR_PATH = f'./save_dir/consequent_estimators/ce_{args.base_model}_dataset_{args.dataset}_pretrain_samples_{args.pretrain_samples}'
    if args.base_model in ['bert', 'roberta']:
        CE_DIR_PATH += f'_num_atoms_{args.num_atoms}'

    ce_model.load_state_dict(torch.load(f'{CE_DIR_PATH}/ce_pretrain_4_{args.base_model}_dataset_{args.dataset}.pt'), strict=True)

    # Freeze the consequent estimator
    for p in ce_model.parameters():
        p.requires_grad=False
        
    model = RuleGenerator(
        rule_len=args.max_rule_len,
        head=1,
        num_atoms=ap.num_atoms(),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_classes=len(class_names),
        n_data=len(train_dataset),
        atom_embedding=atom_embedding,
        consequent_estimator=ce_model,
        padding_idx=pad_token_id,
        tf_model=tf_model,
        args=args,
    )
    
    BASE_MODEL_PATH = f'./save_dir/base_models/base_{args.base_model}_dataset_{args.dataset}/model_best.pt'
    model.load_state_dict(torch.load(BASE_MODEL_PATH), strict=False)
    
    model = model.to(gpu)
    
    nll_loss_func = nn.NLLLoss(reduction='mean').to(gpu)
    if args.only_eval:
        DIR_PREFIX = f'{VERSION_NAME}_{args.base_model}_dataset_{args.dataset}_max_rule_len_{args.max_rule_len}'
        DIR_PREFIX += f'_pretrain_samples_{args.pretrain_samples}'
        if args.dataset in ['yelp', 'clickbait']:
            DIR_PREFIX += f'_num_atoms_{args.num_atoms}'
        DIR_PREFIX += f'_seed_{args.seed}'
        
        targets = [d for d in os.listdir(f'./result/{VERSION_NAME}') if d.startswith(DIR_PREFIX)]
        DIR_PATH = f'./result/{VERSION_NAME}/{targets[-1]}'
        print(f'DIR_PATH: {DIR_PATH}')
    else:
        now = datetime.now()
        CUR_TIME = now.strftime("%y%m%d:%H:%M:%S")

        if VERSION_NAME not in os.listdir('./result'):
            os.system(f'mkdir ./result/{VERSION_NAME}')

        TARGET = f'{VERSION_NAME}_{args.base_model}_dataset_{args.dataset}'
        TARGET += f'_max_rule_len_{args.max_rule_len}'
        TARGET += f'_pretrain_samples_{args.pretrain_samples}'
        if args.dataset in ['yelp', 'clickbait']:
            TARGET += f'_num_atoms_{args.num_atoms}'
        TARGET += f'_seed_{args.seed}'
        TARGET += f'_{CUR_TIME}'
        DIR_PATH = f'./result/{VERSION_NAME}/{TARGET}'
        print(f'DIR_PATH: {DIR_PATH}')

        os.system(f'mkdir {DIR_PATH}')

        LOG_PATH = f'{DIR_PATH}/log'

        flog = open(LOG_PATH, 'w')
        print(args, file=flog, flush=True)
        train(model, nll_loss_func, train_dataloader, valid_dataloader, args, DIR_PATH, flog)
        flog.close()
    
    BEST_MODEL_PATH = f'{DIR_PATH}/model_best.pt'
    EVAL_PATH = f'{DIR_PATH}/model_eval'
    
    feval = open(EVAL_PATH, 'w')
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=gpu))
    
    eval_model(model, nll_loss_func, test_dataloader, args, feval=feval, true_matrix=true_matrix.to(gpu), n_data=len(train_dataset))
        
    feval.close()