import os
from utils import parse_arguments

RESULT_BASE_DIR = './result/base'
SAVE_DIR = './save_dir/base_models'

def extract_base_model(dataset, model):
    s = f'base_{model}_dataset_{dataset}'
    cands = sorted([c for c in os.listdir(RESULT_BASE_DIR) if c.startswith(s)])
    target = cands[-1]
    best_model_path = f'{RESULT_BASE_DIR}/{target}/model_best.pt'
    eval_file = [e for e in os.listdir(f'{RESULT_BASE_DIR}/{target}') if e.startswith('model_eval')][-1]
    eval_path = f'{RESULT_BASE_DIR}/{target}/{eval_file}'

    if s not in os.listdir(SAVE_DIR):
        os.system(f'mkdir {SAVE_DIR}/{s}')

    os.system(f'cp {best_model_path} {SAVE_DIR}/{s}/model_best.pt')
    os.system(f'cp {eval_path} {SAVE_DIR}/{s}/model_eval')
    
if __name__ == "__main__":
    args = parse_arguments()

    if 'save_dir' not in os.listdir('.'):
        os.system(f'mkdir ./save_dir')

    if 'base_models' not in os.listdir('./save_dir'):
        os.system(f'mkdir ./save_dir/base_models')

    extract_base_model(args.dataset, args.base_model)