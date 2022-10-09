VERSION_NAME = 'exp_rule_gen'

import os
import json
import pickle
import copy
from datetime import datetime
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

# Import from custom files
from model import RuleGenerator, ConsequentEstimator
from atom_pool import AtomTokenizer, AtomPool, get_true_matrix
from dataset import get_class_names, create_dataloader, get_dataset, get_tabular_numerical_max
from utils import parse_arguments, reset_seed
from train_eval import train, eval_model

def get_explanation(
    model,
    target_id,
    ap,
    test_df,
    test_dataset,
    true_matrix,
    args,
):
    model.eval()
    gpu = torch.device(f'cuda:{args.gpu}')
    
    exp = ''

    d = test_dataset[target_id]
    
    if args.dataset in ['yelp', 'clickbait']:
        target_context = test_df['text'][target_id]
        
        input_ids, attention_mask, x_, y = d
        input_ids = input_ids.to(gpu)
        attention_mask = attention_mask.to(gpu)
        x_ = x_.to(gpu).unsqueeze(dim=0)
        y = y.to(gpu)
        
        inputs = input_ids, attention_mask, x_

    elif args.dataset in ['adult']:
        target_context = test_df.loc[target_id]
        
        x, x_, y = d
        x = x.to(gpu).unsqueeze(dim=0)
        x_ = x_.to(gpu).unsqueeze(dim=0)
        y = y.to(gpu)
    
        inputs = x, x_
        
    with torch.no_grad():
        outputs, atom_prob_list, cp_list = model(
            inputs
        )
        
        exp += f'{target_context}\n\n'

        outputs = outputs.squeeze(dim=0)
        outputs = torch.exp(outputs)

        _, pred = torch.max(outputs, dim=0)

        class_names = get_class_names(args.dataset)
        label = class_names[y[0].item()]
        prediction = class_names[pred.item()]
        pred_result = (pred.item() == y[0].item())

        nc = ap.num_atoms()

        atom_list = []
        rp_list = []
        coverage_list = []

        for atom_prob in atom_prob_list:
            val, ind = torch.max(atom_prob, dim=-1)
            rp = torch.prod(val, dim=-1)
            atoms = model.ae(ind)

            ind = ind.squeeze(dim=0)
            atoms = [ap.atoms[ap.atom_id2key[i]] for i in ind]
            atom_list.append(atoms)
            rp_list.append(rp.item())

            cover_rule_prob = torch.sum(atom_prob, dim=1)
            cover_rule_prob = torch.matmul(cover_rule_prob, true_matrix)
            mat_satis = (cover_rule_prob == args.max_rule_len)
            mat_satis = torch.sum(mat_satis.float(), dim=-1)
            coverage = mat_satis / len(train_dataset)
            coverage_list.append(coverage.item())

        exp += f'Label: {label}\n'
        exp += f'Prediction: {prediction}\n'
        exp += f'Class Probability: {outputs}\n'

        for i, r in enumerate(atom_list):
            s = ' & '.join([c.display_str for c in r])
            coverage = coverage_list[i]

            exp += f'Rule {i}: {s}\n'
            exp += f'Positive Probability: [{cp_list[i].squeeze(dim=0)}]\n'
            exp += f'Coverage: [{coverage:.6f}]\n'
            exp += '\n'
        
    return exp, target_context, label, prediction, s, cp_list[i].squeeze(dim=0).cpu().tolist(), coverage

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

    datasets = get_dataset(dataset=args.dataset, seed=seed)

    if args.dataset in ['yelp', 'clickbait']:
        with open(f'./save_dir/atom_tokenizer_{args.dataset}.pkl', 'rb') as f:
            atom_tokenizer = pickle.load(f)

        with open(f'./save_dir/atom_pool_{args.dataset}.pkl', 'rb') as f:
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

        train_dataloader = create_dataloader(train_dataset, args, shuffle=True)
        valid_dataloader = create_dataloader(valid_dataset, args, shuffle=False)
        test_dataloader = create_dataloader(test_dataset, args, shuffle=False)

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

        train_dataloader = create_dataloader(train_dataset, args, shuffle=True)
        valid_dataloader = create_dataloader(valid_dataset, args, shuffle=False)
        test_dataloader = create_dataloader(test_dataset, args, shuffle=False)

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

        train_dataloader = create_dataloader(train_dataset, args, shuffle=True)
        valid_dataloader = create_dataloader(valid_dataset, args, shuffle=False)
        test_dataloader = create_dataloader(test_dataset, args, shuffle=False)

        input_dim = train_dataset.x_dummy.shape[1]
        hidden_dim = 512
    else:
        assert(0)

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
    model = model.to(gpu)
    
    DIR_PREFIX = f'rule_gen_{args.base_model}_dataset_{args.dataset}_max_rule_len_{args.max_rule_len}'
    DIR_PREFIX += f'_pretrain_samples_{args.pretrain_samples}'
    if args.dataset in ['yelp', 'clickbait']:
        DIR_PREFIX += f'_num_atoms_{args.num_atoms}'
    DIR_PREFIX += f'_seed_{args.seed}'

    targets = sorted([d for d in os.listdir(f'./result/rule_gen') if d.startswith(DIR_PREFIX)])

    MODEL_PATH = f'./result/rule_gen/{targets[-1]}/model_best.pt'
    EXP_PATH = f'./result/rule_gen/{targets[-1]}/model_explanation.csv'
    
    print(MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=gpu))
    
    if args.dataset == 'adult':
        dm = get_tabular_numerical_max(dataset=args.dataset)
        test_df = copy.deepcopy(number_test_df)
        for c in test_df.columns:
            if c in dm:
                test_df[c] = number_test_df[c] * dm[c]
            
    
    true_matrix = get_true_matrix(ap).to(gpu)

    result_df = []
    for target_id in tqdm(range(len(test_df))):
        exp, target_context, label, prediction, rule_str, class_prob, coverage = get_explanation(model, target_id, ap, test_df, test_dataset, true_matrix, args)
        row = [target_context, label, prediction, rule_str, class_prob, coverage]
        result_df.append(row)
        
    result_df = pd.DataFrame(result_df, columns=['Target', 'Label', 'Prediction', 'Explanation', 'Class_Probability', 'Coverage'])
    result_df.to_csv(EXP_PATH, index=False)