import torch

# Import from custom files
from dataset import get_class_names, create_dataloader, get_dataset
from utils import parse_arguments, reset_seed
from model import BaseModel
from train_eval import get_train_embedding

if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    reset_seed(seed)
    gpu = torch.device(f'cuda:{args.gpu}')
    
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
    
    datasets = get_dataset(dataset=args.dataset, seed=seed)
    
    # Create datasets
    if args.dataset == 'yelp':
        train_df, valid_df, test_df = datasets
        
        from dataset import YelpDataset
        train_dataset = YelpDataset(train_df, atom_pool=None, tf_tokenizer=tf_tokenizer, atom_tokenizer=None, args=args)
        
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        
    elif args.dataset == 'clickbait':
        train_df, valid_df, test_df = datasets
        
        from dataset import ClickbaitDataset
        train_dataset = ClickbaitDataset(train_df, atom_pool=None, tf_tokenizer=tf_tokenizer, atom_tokenizer=None, args=args)
        
        input_dim = config.hidden_size
        hidden_dim = config.hidden_size
        
    elif args.dataset == 'adult':
        number_train_df, dummy_train_df, number_valid_df, dummy_valid_df, number_test_df, dummy_test_df = datasets
        
        from dataset import AdultDataset
        train_dataset = AdultDataset(
            number_train_df,
            dummy_train_df,
            atom_pool=None,
            args=args,
        )

        input_dim = train_dataset.x_dummy.shape[1]
        hidden_dim = 512
        
    train_dataloader = create_dataloader(train_dataset, args, shuffle=False)
        
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
    
    if args.base_model in ['dnn']:
        BASE_MODEL_PATH = f'./save_dir/base_models/base_{args.base_model}_dataset_{args.dataset}/model_best.pt'
        model.load_state_dict(torch.load(BASE_MODEL_PATH), strict=False)
    
    model = model.to(gpu)

    embeddings = get_train_embedding(model, train_dataloader, args)
    print(f'Embedding Shape: {embeddings.shape}')

    torch.save(embeddings, f'./save_dir/base_models/base_{args.base_model}_dataset_{args.dataset}/train_embeddings.pt')