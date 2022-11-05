RUN = 'selor'

import os
import json
import pandas as pd
import pickle
from datetime import datetime
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import from custom files
from selor_utils import atom as at
from selor_utils import dataset as ds
from selor_utils import net
from selor_utils import train_eval as te
from selor_utils import utils

def get_all_explanation(
    model,
    dataset,
    test_df,
    atom_pool,
    true_matrix,
    gpu,
    tf_tokenizer=None,
    atom_tokenizer=None,
):
    exp_list = []
    result_list = []
    for target_id in tqdm(range(len(test_df)), desc='Extracting Explanations'):
        exp = ''
        
        target_context = ''
        row = test_df.iloc[target_id,:]
        class_names = ds.get_class_names(dataset=dataset)
        
        if dataset == 'yelp':
            target_context += f'text: {row["text"]}\n'
            target_context += f'label: '
        elif dataset == 'clickbait':
            target_context += f'title: {row["title"]}\n'
            target_context += f'text: {row["text"]}\n'
        elif dataset == 'adult':
            categorical_x_col, numerical_x_col, y_col = ds.get_tabular_column_type(dataset=dataset)
            total_x_col = categorical_x_col + numerical_x_col
            cat_map, numerical_threshold, numerical_max = ds.load_tabular_info(dataset=dataset)
            
            for c in total_x_col:
                if c in categorical_x_col:
                    for k in row.index:
                        if not k.startswith(f'{c}_'):
                            continue

                        v = row[k]
                        if v == 1:
                            context, target = k.split('_')
                            target_context += f'{context}: {cat_map[f"{context}_idx2key"][int(float(target))]}\n'
                elif c in numerical_x_col:
                    target_context += f'{c}: {round(v * numerical_max[c], 1)}\n'
                else:
                    assert(0)
                    
        exp += f'{target_context}\n'
                    
        inputs = ds.get_single_input(
            row,
            dataset,
            atom_pool,
            tf_tokenizer=tf_tokenizer,
            atom_tokenizer=atom_tokenizer,
        )
        
        class_probs, antecedents, coverage_list = te.get_explanation(
            model,
            atom_pool,
            true_matrix,
            inputs,
            class_names,
            gpu
        )
        
        pred = max(class_probs, key=class_probs.get)
        
        y_col = ds.get_label_column(dataset)
        y = int(row[y_col])

        label = class_names[y]
        pred_result = (pred == label)

        exp += f'Label: {label}\n'
        exp += f'Prediction: {pred}\n\n'
        exp += f'Class Probability\n'
        for k, v in class_probs.items():
            exp += f'{k}: {v}\n'

        exp += '\n'
        for i, s in enumerate(antecedents):
            coverage = coverage_list[i]

            exp += f'Explanation {i}: {s}\n'
            exp += f'Coverage: {coverage:.6f}\n'
            exp += '\n'

        rd = {
            'Id': target_id,
            'Target': target_context,
            'Label': label,
            'Prediction': pred,
            'Explanation': antecedents,
            'Class Probability': class_probs,
            'Coverage': coverage_list
        }
        
        result_list.append(rd)
        exp_list.append(exp)
            
    return exp_list, result_list

if __name__ == "__main__":
    args = utils.parse_arguments()
    
    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)
    
    assert(dtype==btype)

    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    utils.reset_seed(seed)
    
    tf_tokenizer, tf_model, config = net.get_tf_model(args.base)

    train_df, valid_df, test_df = ds.load_data(dataset=args.dataset)
    
    if dtype == 'nlp':
        with open(f'./{args.save_dir}/atom_tokenizer/atom_tokenizer_{args.dataset}.pkl', 'rb') as f:
            atom_tokenizer = pickle.load(f)

        with open(f'./{args.save_dir}/atom_pool/atom_pool_{args.dataset}_num_atoms_{args.num_atoms}.pkl', 'rb') as f:
            ap = pickle.load(f)
            
    elif args.dataset in ['adult']:
        atom_tokenizer = None

        with open(f'./{args.save_dir}/atom_pool/atom_pool_{args.dataset}.pkl', 'rb') as f:
            ap = pickle.load(f)
        
    # Create datasets
    train_dataset, valid_dataset, test_dataset = [
        ds.create_dataset(
            df,
            dataset=args.dataset,
            atom_pool=ap,
            atom_tokenizer=atom_tokenizer,
            tf_tokenizer=tf_tokenizer,
            config=config
        ) for df in [train_df, valid_df, test_df]]
    
    train_dataloader = ds.create_dataloader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True
    )
        
    valid_dataloader, test_dataloader = [
        ds.create_dataloader(
            dtset,
            args.batch_size,
            args.num_workers,
            shuffle=False
        ) for dtset in [valid_dataset, test_dataset]]
        
    if args.dataset in ds.NLP_DATASET:
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
    
    elif args.dataset in ds.TAB_DATASET:
        input_dim = train_dataset.x.shape[1]
        hidden_dim = args.hidden_dim
    else:
        raise NotImplementedError("We only support NLP and tabular dataset now.")
    
    # Load class names
    class_names = ds.get_class_names(args.dataset)
    
    # Whether each train sample satisfies each atom
    true_matrix = at.get_true_matrix(ap)
    norm_true_matrix = true_matrix / (torch.sum(true_matrix, dim=1).unsqueeze(dim=1) + 1e-8)

    # Embedding from the base model for each train sample
    data_embedding = torch.load(f'./{args.save_dir}/base_models/base_{args.base}_dataset_{args.dataset}/train_embeddings.pt')
        
    # Obtain atom embedding
    atom_embedding = torch.mm(norm_true_matrix.to(gpu), data_embedding.to(gpu)).detach()
    
    ce_model = net.ConsequentEstimator(
        n_class=len(class_names),
        hidden_dim=hidden_dim,
        atom_embedding=atom_embedding,
    ).to(gpu)

    ce_dir_path = f'./{args.save_dir}/consequent_estimators/ce_{args.base}'
    ce_dir_path += f'_dataset_{args.dataset}'
    ce_dir_path += f'_pretrain_samples_{args.pretrain_samples}'
    
    if dtype == 'nlp':
        ce_dir_path += f'_num_atoms_{args.num_atoms}'

    ce_model.load_state_dict(torch.load(f'{ce_dir_path}/ce_pretrain_4_{args.base}_dataset_{args.dataset}.pt'), strict=True)

    # Freeze the consequent estimator
    for p in ce_model.parameters():
        p.requires_grad=False
        
    model = net.AntecedentGenerator(
        dataset=args.dataset,
        base=args.base,
        antecedent_len=args.antecedent_len,
        head=1,
        num_atoms=ap.num_atoms(),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=len(class_names),
        n_data=len(train_dataset),
        atom_embedding=atom_embedding,
        consequent_estimator=ce_model,
        tf_model=tf_model,
    )
    
    base_path = f'./{args.save_dir}/base_models/base_{args.base}_dataset_{args.dataset}/model_best.pt'
    model.load_state_dict(torch.load(base_path), strict=False)
    
    model = model.to(gpu)
    
    nll_loss_func = nn.NLLLoss(reduction='mean').to(gpu)
    
    dir_prefix = f'{RUN}_{args.base}_dataset_{args.dataset}'
    dir_prefix += f'_antecedent_len_{args.antecedent_len}'
    dir_prefix += f'_pretrain_samples_{args.pretrain_samples}'
    if dtype == 'nlp':
        dir_prefix += f'_num_atoms_{args.num_atoms}'
    
    if args.only_eval:
        targets = [d for d in os.listdir(f'./result/{RUN}') if d.startswith(dir_prefix)]
        dir_path = f'./result/{RUN}/{targets[-1]}'
        print(f'Directory Path: {dir_path}')
    else:
        now = datetime.now()
        cur_time = now.strftime("%y%m%d:%H:%M:%S")

        if args.result_dir not in os.listdir('.'):
            os.system(f'mkdir ./{args.result_dir}')
        
        if RUN not in os.listdir('./result'):
            os.system(f'mkdir ./result/{RUN}')

        dir_path = f'./result/{RUN}/{dir_prefix}_seed_{args.seed}_{cur_time}'
        print(f'Directory Path: {dir_path}')

        os.system(f'mkdir {dir_path}')
        with open(f'{dir_path}/args', 'w') as f:
            print(args, file=f, flush=True)
            
        model = te.train(
            model=model,
            loss_func=nll_loss_func,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            epochs=args.epochs,
            gpu=gpu,
            class_names=class_names,
            dir_path=dir_path
        )

    best_model_path = f'{dir_path}/model_best.pt'    
    model.load_state_dict(torch.load(best_model_path, map_location=gpu))
        
    te.eval_model(
        model=model,
        loss_func=nll_loss_func,
        test_dataloader=test_dataloader,
        true_matrix=true_matrix.to(gpu),
        gpu=gpu,
        class_names=class_names,
        dir_path=dir_path
    )
    
    exp_list, result_list = get_all_explanation(
        model,
        args.dataset,
        test_df,
        atom_pool=ap,
        true_matrix=true_matrix.to(gpu),
        tf_tokenizer=tf_tokenizer,
        atom_tokenizer=atom_tokenizer,
        gpu=gpu,
    )
    
    exp_path = f'{dir_path}/model_explanation.json'

    with open(exp_path, "w") as f:
        json.dump(result_list, f)
