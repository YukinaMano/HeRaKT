# @AlaoCode#> This module places the methods used to record the experimental process
import json
import os
import csv
import torch
from utils._common import print_debug, check_dir

def save_exp_args(args) -> str:
    '''
    save experimental parameters to a JSON file
    '''
    log_dir = args.log_dir
    tag = args.tag
    dataset = args.dataset
    title = 'info-args'

    dataset_dir = os.path.join(log_dir, dataset)
    config_path = os.path.join(dataset_dir, f'{title}_{dataset}_{tag}.json')
    check_dir(config_path)
    config = {k: v for k, v in vars(args).items() if v is not None}
    with open(config_path, 'w') as file:
        json.dump(config, file)
    print_debug(f'\n\n>>>[Completed] save args to {config_path}')
    print_debug(f'Model Args: {config}')
    return config_path

def save_exp_data(data, args):
    '''
    save experimental data to a txt file
    '''
    log_dir = args.log_dir
    tag = args.tag 
    dataset = args.dataset
    title = 'info-data'
    dataset_dir = os.path.join(log_dir, dataset)
    tag_path = os.path.join(dataset_dir, f'{title}_{dataset}_{tag}.txt')
    check_dir(tag_path)
    with open(tag_path, 'w') as file:
        for key, value in vars(data).items():
            # Write key-value pairs to a file in the format of 'k: v'
            file.write(f"{key}: {value}\n")
    print_debug(f'\n>>>[Completed] save exp data to {tag_path}')

def save_best_model(model_params: dict, model_name: str, dataset=''):
    '''
    save the best model parameters to the checkpoint directory
    '''
    check_path = 'checkpoint'
    checkpoint_save_name = os.path.join(check_path, f'{dataset}_best_{model_name}.pth')
    check_dir(checkpoint_save_name)
    torch.save(model_params, checkpoint_save_name)
    print_debug(f'[Completed] save best model to {checkpoint_save_name}')

def log_experiment_result(model_name, dataset, auc, acc, best_epoch, tag, seed, run_time, notes='', gpu_id=0, output_file='results.csv'):
    '''
    save the experimental results to the output file
    '''
    # if the file does not exist, write it to the header first
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Model', 'Dataset', 'AUC', 'ACC', 'Best Epoch', 'Tag', 'Seed', 'Run Time(min)', 'Notes', 'GPU'])
        writer.writerow([model_name, dataset, f'{auc:.8f}', f'{acc:.8f}', best_epoch, tag, seed, f'{run_time:.2f}', notes, gpu_id])

