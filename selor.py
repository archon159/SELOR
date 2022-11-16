"""
The script to train selor
"""
import os
from pathlib import Path
import json
from datetime import datetime
import torch
from torch import nn
import pandas as pd

# Import from custom files
from selor_utils import atom as at
from selor_utils import dataset as ds
from selor_utils import net
from selor_utils import train_eval as te
from selor_utils import utils

RUN = 'selor'

if __name__ == "__main__":
    args = utils.parse_arguments()

    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)

    assert dtype == btype

    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    utils.reset_seed(seed)

    tf_tokenizer, tf_model, config = net.get_tf_model(args.base)

    train_df, valid_df, test_df = ds.load_data(dataset=args.dataset)

    atom_pool_path = f'./{args.save_dir}/atom_pool/'
    atom_pool_path += f'atom_pool_{args.dataset}'
    if dtype == 'nlp':
        atom_tokenizer_path = f'./{args.save_dir}/atom_tokenizer/'
        atom_tokenizer_path += f'atom_tokenizer_{args.dataset}_seed_{args.seed}'
        atom_tokenizer = pd.read_pickle(atom_tokenizer_path)

        atom_pool_path += f'_num_atoms_{args.num_atoms}'
    elif args.dataset in ['adult']:
        atom_tokenizer = None
    atom_pool_path += f'_seed_{args.seed}'
    ap = pd.read_pickle(atom_pool_path)

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

    if dtype == 'nlp':
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
    elif dtype == 'tab':
        input_dim = train_dataset.x.shape[1]
        hidden_dim = args.hidden_dim
    else:
        raise ValueError(f'Dataset type {dtype} is not supported.')

    # Load class names
    class_names = ds.get_class_names(args.dataset)

    # Whether each train sample satisfies each atom
    true_matrix = at.get_true_matrix(ap)
    norm_true_matrix = true_matrix / (torch.sum(true_matrix, dim=1).unsqueeze(dim=1) + 1e-8)

    # Embedding from the base model for each train sample
    data_embedding_path = f'./{args.save_dir}/base_models/'
    data_embedding_path += f'base_{args.base}_dataset_{args.dataset}_seed_{args.seed}/'
    data_embedding_path += 'train_embeddings.pt'
    data_embedding = torch.load(data_embedding_path)

    # Obtain atom embedding
    atom_embedding = torch.mm(norm_true_matrix.to(gpu), data_embedding.to(gpu)).detach()

    ce_model = net.ConsequentEstimator(
        num_classes=len(class_names),
        hidden_dim=hidden_dim,
        atom_embedding=atom_embedding,
    ).to(gpu)

    ce_dir_path = f'./{args.save_dir}/consequent_estimators/ce_{args.base}'
    ce_dir_path += f'_dataset_{args.dataset}'
    ce_dir_path += f'_pretrain_samples_{args.pretrain_samples}'

    if dtype == 'nlp':
        ce_dir_path += f'_num_atoms_{args.num_atoms}'

    ce_dir_path += f'_seed_{args.seed}'

    ce_model_path = f'{ce_dir_path}/ce_pretrain_4_{args.base}_dataset_{args.dataset}.pt'
    ce_model.load_state_dict(torch.load(ce_model_path), strict=True)

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

    base_path = f'./{args.save_dir}/base_models/'
    base_path += f'base_{args.base}_dataset_{args.dataset}_seed_{args.seed}/'
    base_path += 'model_best.pt'
    model.load_state_dict(torch.load(base_path), strict=False)

    model = model.to(gpu)

    nll_loss_func = nn.NLLLoss(reduction='mean').to(gpu)

    dir_prefix = f'{RUN}_{args.base}_dataset_{args.dataset}'
    dir_prefix += f'_antecedent_len_{args.antecedent_len}'
    dir_prefix += f'_pretrain_samples_{args.pretrain_samples}'
    if dtype == 'nlp':
        dir_prefix += f'_num_atoms_{args.num_atoms}'
    dir_prefix += f'_seed_{args.seed}'

    if args.only_eval:
        targets = [d for d in os.listdir(f'./result/{RUN}') if d.startswith(dir_prefix)]
        dir_path = f'./result/{RUN}/{targets[-1]}'
        print(f'Directory Path: {dir_path}')
        dir_path = Path(dir_path)
    else:
        now = datetime.now()
        cur_time = now.strftime("%y%m%d:%H:%M:%S")

        dir_path = f'./result/{RUN}/{dir_prefix}_{cur_time}'
        print(f'Directory Path: {dir_path}')
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        arg_path = dir_path / 'args'
        arg_path.write_text(f'{args}\n', encoding='utf-8')

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

    best_model_path = dir_path / 'model_best.pt'
    model.load_state_dict(torch.load(str(best_model_path), map_location=gpu))

    te.eval_model(
        model=model,
        loss_func=nll_loss_func,
        test_dataloader=test_dataloader,
        true_matrix=true_matrix.to(gpu),
        gpu=gpu,
        class_names=class_names,
        dir_path=dir_path
    )

    exp_list, result_list = te.get_all_explanation(
        model,
        args.dataset,
        test_df,
        atom_pool=ap,
        true_matrix=true_matrix.to(gpu),
        tf_tokenizer=tf_tokenizer,
        atom_tokenizer=atom_tokenizer,
        gpu=gpu,
        class_names=class_names,
    )

    exp_path = dir_path / 'model_explanation.json'

    with exp_path.open("w", encoding='utf-8') as f:
        json.dump(result_list, f)
