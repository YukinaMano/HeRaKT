# @AlaoCode#> This module places the methods used to record the experimental process
import json
import os
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