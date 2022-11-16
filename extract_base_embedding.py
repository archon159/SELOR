"""
The script to extract embeddings from base models
"""
import shutil
from pathlib import Path
import torch

# Import from custom files
from selor_utils import dataset as ds
from selor_utils import net
from selor_utils import train_eval as te
from selor_utils import utils

def update_base_model(
    dataset: str,
    base: str,
    seed: int,
    result_dir: str,
    save_dir: str,
) -> str:
    """
    The function to update the latest base model
    """
    result_path = Path(f'./{result_dir}/base')

    prefix = f'base_{base}_dataset_{dataset}_seed_{seed}'
    cands = sorted([c for c in result_path.iterdir() if c.name.startswith(prefix)])
    target = cands[-1]

    base_update_path = Path(f'./{save_dir}/base_models/{prefix}')
    base_update_path.mkdir(parents=True, exist_ok=True)

    best_base_model_path = target / 'model_best.pt'
    eval_path = target / 'model_eval'

    shutil.copy(str(best_base_model_path), str(base_update_path / 'model_best.pt'))
    shutil.copy(str(eval_path), str(base_update_path / 'model_eval'))

    return base_update_path

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
    train_df, _, _ = ds.load_data(dataset=args.dataset)
    train_dataset = ds.create_dataset(
        train_df,
        dataset=args.dataset,
        atom_pool=None,
        atom_tokenizer=None,
        tf_tokenizer=tf_tokenizer,
        config=config
    )

    train_dataloader = ds.create_dataloader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False
    )

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

    model = net.BaseModel(
        dataset=args.dataset,
        base=args.base,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        tf_model=tf_model,
        num_classes=len(class_names),
    ).to(gpu)

    print('Update the base model to the latest one.')
    base_path = update_base_model(
        args.dataset,
        args.base,
        args.seed,
        args.result_dir,
        args.save_dir,
    )
    best_model_path = base_path / 'model_best.pt'

    # We use pre-trained model for NLP bases.
    if btype == 'tab':
        model.load_state_dict(torch.load(best_model_path.resolve()), strict=True)

    model = model.to(gpu)

    embeddings = te.get_base_embedding(
        model=model,
        train_dataloader=train_dataloader,
        gpu=gpu,
    )
    print(f'Embedding Shape: {embeddings.shape}')

    embedding_path = base_path / 'train_embeddings.pt'
    torch.save(embeddings, str(embedding_path))
