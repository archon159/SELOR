VERSION_NAME = 'base'

import os
import json
from datetime import datetime
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

# Import from custom files
from model import BaseModel
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
        
    else:
        tf_tokenizer = None
        tf_model = None
        vocab_size = 0
        pad_token_id = 0

    
    datasets = get_dataset(dataset=args.dataset)
        
    # Create datasets
    if args.dataset == 'yelp':
        train_df, valid_df, test_df = datasets
        
        from dataset import YelpDataset
        train_dataset = YelpDataset(
            train_df,
            atom_pool=None,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=None, args=args
        )
        
        valid_dataset = YelpDataset(
            valid_df,
            atom_pool=None,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=None, args=args
        )
        test_dataset = YelpDataset(
            test_df,
            atom_pool=None,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=None,
            args=args
        )
        
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        
    elif args.dataset == 'clickbait':
        train_df, valid_df, test_df = datasets
        
        from dataset import ClickbaitDataset
        train_dataset = ClickbaitDataset(
            train_df,
            atom_pool=None,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=None,
            args=args
        )
        
        valid_dataset = ClickbaitDataset(
            valid_df,
            atom_pool=None,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=None,
            args=args
        )
        
        test_dataset = ClickbaitDataset(
            test_df,
            atom_pool=None,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=None,
            args=args
        )
        
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
    
    elif args.dataset == 'adult':
        number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df = datasets
        
        from dataset import AdultDataset
        train_dataset = AdultDataset(
            number_train_df,
            dummy_train_df,
            atom_pool=None,
            args=args
        )
        
        valid_dataset = AdultDataset(
            number_valid_df,
            dummy_valid_df,
            atom_pool=None,
            args=args
        )
        
        test_dataset = AdultDataset(
            number_test_df,
            dummy_test_df,
            atom_pool=None,
            args=args
        )
        
        input_dim = train_dataset.x_dummy.shape[1]
        hidden_dim = 512
        
    train_dataloader = create_dataloader(train_dataset, args, shuffle=True)
    valid_dataloader = create_dataloader(valid_dataset, args, shuffle=False)
    test_dataloader = create_dataloader(test_dataset, args, shuffle=False)
        
    # Load class names
    class_names = get_class_names(args.dataset)
    
    model = BaseModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_classes = len(class_names),
        padding_idx=pad_token_id,
        tf_model=tf_model,
        args=args,
    ).to(gpu)
    
    nll_loss_func = nn.NLLLoss(reduction='mean').to(gpu)
    
    if args.only_eval:
        DIR_PREFIX = f'{VERSION_NAME}_{args.base_model}_dataset_{args.dataset}_seed_{args.seed}'
        targets = [d for d in os.listdir(f'./result/{VERSION_NAME}') if d.startswith(DIR_PREFIX)]
        DIR_PATH = f'./result/{VERSION_NAME}/{targets[-1]}'
        print(f'DIR_PATH: {DIR_PATH}')
    else:
        now = datetime.now()
        CUR_TIME = now.strftime("%y%m%d:%H:%M:%S")

        if 'result' not in os.listdir('.'):
            os.system(f'mkdir ./result')
        
        if VERSION_NAME not in os.listdir('./result'):
            os.system(f'mkdir ./result/{VERSION_NAME}')

        DIR_PATH = f'./result/{VERSION_NAME}/{VERSION_NAME}_{args.base_model}_dataset_{args.dataset}_seed_{args.seed}_{CUR_TIME}'
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
        
    eval_model(model, nll_loss_func, test_dataloader, args, feval=feval)
        
    feval.close()