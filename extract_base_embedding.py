import os
import torch

# Import from custom files
from selor_utils import dataset as ds
from selor_utils import net
from selor_utils import train_eval as te
from selor_utils import utils

def update_base_model(
    dataset,
    base,
    result_dir,
    save_dir,
):
    result_path = f'./{result_dir}/base'
    
    if save_dir not in os.listdir('.'):
        os.system(f'mkdir ./{save_dir}')

    if 'base_models' not in os.listdir(f'./{save_dir}'):
        os.system(f'mkdir ./{save_dir}/base_models')
        
    save_path = f'./{save_dir}/base_models'
    
    s = f'base_{base}_dataset_{dataset}'
    cands = sorted([c for c in os.listdir(result_path) if c.startswith(s)])
    target = cands[-1]
    best_model_path = f'{result_path}/{target}/model_best.pt'
    eval_path = f'{result_path}/{target}/model_eval'

    if s not in os.listdir(save_path):
        os.system(f'mkdir {save_path}/{s}')

    base_path = f'{save_path}/{s}'
        
    os.system(f'cp {best_model_path} {save_path}/{s}/model_best.pt')
    os.system(f'cp {eval_path} {save_path}/{s}/model_eval')
    
    return base_path

if __name__ == "__main__":
    args = utils.parse_arguments()
    
    dtype = ds.get_dataset_type(args.dataset)
    btype = ds.get_base_type(args.base)

    assert(dtype==btype)
    
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
        raise NotImplementedError("We only support NLP and tabular dataset now.")
        
    # Load class names
    class_names = ds.get_class_names(args.dataset)
        
    model = net.BaseModel(
        dataset=args.dataset,
        base=args.base,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        tf_model=tf_model,
        num_classes=len(class_names),
        dropout=0,
    ).to(gpu)
    
    print('Update the base model to the latest one.')
    base_path = update_base_model(
        args.dataset,
        args.base,
        args.result_dir,
        args.save_dir,
    )
    best_model_path = f'{base_path}/model_best.pt'
    
    # We use pre-trained model for NLP bases. 
    if btype == 'tab':
        model.load_state_dict(torch.load(best_model_path), strict=True)
    
    model = model.to(gpu)

    embeddings = te.get_base_embedding(
        model=model,
        train_dataloader=train_dataloader,
        gpu=gpu,
    )
    print(f'Embedding Shape: {embeddings.shape}')

    torch.save(embeddings, f'{base_path}/train_embeddings.pt')