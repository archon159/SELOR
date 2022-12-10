"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The script to train base models
"""
from pathlib import Path
from datetime import datetime
import torch
from torch import nn

# Import from custom files
from selor_utils import dataset as ds
from selor_utils import net
from selor_utils import train_eval as te
from selor_utils import utils

RUN = 'base'

if __name__ == "__main__":
    args = utils.parse_arguments()

    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)
    assert dtype == btype

    seed = args.seed
    gpu = torch.device(f'cuda:{args.gpu}')
    utils.reset_seed(seed)

    tf_tokenizer, tf_model, config = net.get_tf_model(args.base)

    # Create datasets
    train_df, valid_df, test_df = ds.load_data(dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = [
        ds.create_dataset(
            df,
            dataset=args.dataset,
            atom_pool=None,
            atom_tokenizer=None,
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

    if dtype=='nlp':
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
    elif btype=='tab':
        input_dim = train_dataset.x.shape[1]
        hidden_dim = args.hidden_dim
    else:
        raise ValueError(f'Dataset type {dtype} is not supported.')

    # Load class names
    class_names = ds.get_class_names(args.dataset)

    model = net.BaseModel(
        dataset=args.dataset,
        base=args.base,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        tf_model=tf_model,
        num_classes=len(class_names),
    ).to(gpu)

    nll_loss_func = nn.NLLLoss(reduction='mean').to(gpu)

    dir_prefix = f'{RUN}_{args.base}_dataset_{args.dataset}_seed_{args.seed}'

    if args.only_eval:
        result_path = Path(f'./{args.result_dir}/{RUN}')
        targets = [d for d in result_path.iterdir() if d.name.startswith(dir_prefix)]
        dir_path = Path(f'{targets[-1]}')
        print(f'Directory Path: {str(dir_path)}')
    else:
        now = datetime.now()
        cur_time = now.strftime("%y%m%d:%H:%M:%S")

        dir_path = Path(f'./{args.result_dir}/{RUN}/{dir_prefix}_{cur_time}')
        print(f'Directory Path: {str(dir_path)}')
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
        true_matrix=None,
        gpu=gpu,
        class_names=class_names,
        dir_path=dir_path
    )
