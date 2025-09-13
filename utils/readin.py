# @AlaoCode#> This module is used to read in datasets and process them into structured data
import os
import numpy as np
import pandas as pd
from datetime import datetime
from types import SimpleNamespace
from utils._common import print_debug
from utils.record import save_exp_args
from utils.seqprocess import align_structured_time_seqs, align_time_seqs, format_time_seqs, split_dataset, unpack_structured_data


def read_dataset(data_path: str) -> list:
    '''
    from the dataset readin data
    '''
    with open(data_path, 'r') as file:
        data_str = file.readlines()
    data_lis = [line.split(',') for line in data_str]
    print_debug(f'\n\n>>>[Completed] Data input: {data_path}')
    return data_lis

def serialize_dataset(meta_lines_lis: list, meta_size=4) -> list:
    '''
    serialize the readin data
    '''
    assert len(meta_lines_lis) % meta_size == 0, "The dataset does not match the data format."
    meta_lines_num = len(meta_lines_lis)
    seqs_num = meta_lines_num // meta_size

    seqs_lis = []
    for index, line in enumerate(meta_lines_lis):
        if index % meta_size == 0:
            seq_len = int(line[0])
            seq_features_lis = []
        else: 
            seq_features_lis.append(list(map(int, line[:seq_len])))
        if  index % meta_size == meta_size - 1:
            seq_steps_lis =  [list(item) for item in zip(*seq_features_lis)]
            seqs_lis.append(seq_steps_lis)

    print_debug(f'\n\n>>>[Completed] Serialize the dataset')
    print_debug(f'Exist {seqs_num} time seqs')
    return seqs_lis

def get_serialize_data(data_path: str, meta_size=4) -> list:
    '''
    obtain the original time series
    *seqs_lis: [
        seq[
            step[
                features(s, q, r), 
                ...
            ], 
            ...
        ], 
        ...
    ]
    '''
    meta_lines_lis = read_dataset(data_path)
    seqs_lis = serialize_dataset(meta_lines_lis, meta_size)
    
    return seqs_lis

# @AlaoCode#> If your data format is different from the one below, then you need to design your own method for obtaining experimental data.
def select_normal_data(dataset_name: str, dir_path: str, meta_size: int, base_offset, format_type, max_time_step, align_type, split_seed) -> list:
    '''
    obtain experimental data from datasets in different formats
    '''
    exp_data = SimpleNamespace()
    # if there is no test set as input, perform serialization, segmentation, and mapping
    if dataset_name == 'csedm' or dataset_name == 'junyi_3':
        data_path = os.path.join(dir_path, dataset_name, dataset_name + '_train.csv')
        row_seqs_lis = get_serialize_data(data_path, meta_size)
        format_seqs_lis, a_size, s_size, q_size = format_time_seqs(row_seqs_lis, base_offset, format_type)
        align_seqs_lis, align_seqs_len = align_time_seqs(format_seqs_lis, max_time_step, align_type)
        (train_data, valid_data, test_data), (train_indices, valid_indices, test_indices) = split_dataset(align_seqs_lis, [0.64, 0.16, 0.2], random_seed=split_seed)
        train_seqs_len = np.array(align_seqs_len)[train_indices].tolist()
        valid_seqs_len = np.array(align_seqs_len)[valid_indices].tolist()
        test_seqs_len = np.array(align_seqs_len)[test_indices].tolist()
    # if there is a test set as input, perform serialization and mapping centrally
    elif dataset_name == 'assist09_3' or dataset_name == 'assist12_3' or dataset_name == 'ednet_5000_3':
        train_data_path = os.path.join(dir_path, dataset_name, dataset_name + '_train.csv')
        test_data_path = os.path.join(dir_path, dataset_name, dataset_name + '_test.csv')
        train_row_seqs_lis = get_serialize_data(train_data_path, meta_size)
        test_row_seqs_lis = get_serialize_data(test_data_path, meta_size)
        test_start_index = len(train_row_seqs_lis)
        row_seqs_lis = train_row_seqs_lis + test_row_seqs_lis
        format_seqs_lis, a_size, s_size, q_size = format_time_seqs(row_seqs_lis, base_offset, format_type)
        exp_format_seqs_lis = format_seqs_lis[:test_start_index]
        test_format_seqs_lis = format_seqs_lis[test_start_index:]
        exp_align_seqs_lis, exp_seqs_len = align_time_seqs(exp_format_seqs_lis, max_time_step, align_type)
        test_data, test_seqs_len = align_time_seqs(test_format_seqs_lis, max_time_step, align_type)
        (train_data, valid_data), (train_indices, valid_indices) = split_dataset(exp_align_seqs_lis, [0.8, 0.2], random_seed=split_seed)
        train_seqs_len = np.array(exp_seqs_len)[train_indices].tolist()
        valid_seqs_len = np.array(exp_seqs_len)[valid_indices].tolist()
    # if the input is structured data, decouple it
    elif dataset_name == 'assist2017' or dataset_name == 'assist2009' or dataset_name == 'ednet':
        train_data_path = os.path.join(dir_path, dataset_name, dataset_name + '_train.csv')
        test_data_path = os.path.join(dir_path, dataset_name, dataset_name + '_test.csv')
        train_row_seqs_lis = get_data_from_structured_csv(train_data_path, ['concepts', 'questions', 'responses'])
        test_row_seqs_lis = get_data_from_structured_csv(test_data_path, ['concepts', 'questions', 'responses'])
        test_start_index = len(train_row_seqs_lis)
        row_seqs_lis = train_row_seqs_lis + test_row_seqs_lis
        format_seqs_lis, a_size, s_size, q_size = format_time_seqs(row_seqs_lis, 0, format_type, seqs_structured=True)
        exp_stuctured_data, test_stuctured_data = format_seqs_lis[:test_start_index], format_seqs_lis[test_start_index:]
        (train_stuctured_data, valid_stuctured_data), _ = split_dataset(exp_stuctured_data, [0.8, 0.2], random_seed=split_seed)
        train_align_seqs_lis, train_align_seqs_len = align_structured_time_seqs(train_stuctured_data, max_time_step, align_type)
        valid_align_seqs_lis, valid_align_seqs_len = align_structured_time_seqs(valid_stuctured_data, max_time_step, align_type)
        test_align_seqs_lis, test_align_seqs_len = align_structured_time_seqs(test_stuctured_data, max_time_step, align_type)
        train_data, graph_obj = unpack_structured_data(train_align_seqs_lis, to_get_graph=True, s_size=s_size, q_size=q_size)
        valid_data, _ = unpack_structured_data(valid_align_seqs_lis)
        test_data, _ = unpack_structured_data(test_align_seqs_lis)
        exp_data.graph_obj = graph_obj
        train_seqs_len, test_seqs_len = train_align_seqs_len, test_align_seqs_len
        valid_seqs_len = valid_align_seqs_len
    else:
        print('[Error] No available dataset selected, program terminated!!!')
    
    exp_data.train_seqs = train_data
    exp_data.valid_seqs = valid_data
    exp_data.test_seqs = test_data
    exp_data.train_seqs_len = train_seqs_len
    exp_data.valid_seqs_len = valid_seqs_len
    exp_data.test_seqs_len = test_seqs_len
    exp_data.a_size = a_size
    exp_data.s_size = s_size
    exp_data.q_size = q_size
    return exp_data

def get_data_from_structured_csv(data_path: str, structure_label: list) -> list:
    '''
    retrieve data from structured CSV files and process nested structures of ',' and '_'.
    each cell contains 200 pieces of data, separated by ',', and some data is an array separated by '_'. The first element is taken as the sequence feature.
    '''
    df = pd.read_csv(data_path)
    seq_lis = []
    for index, row in df.iterrows():
        label_lis = []
        for label in structure_label:
            if label not in df.columns:
                print(f'Error: no {label} column found, program terminated!')
                return None
            # process each cell
            processed_data = []
            for item in row[label].split(','):
                sub_items = item.split('_')  # processing arrays separated by '_'
                processed_data.append(list(map(int, (sub_items[0]))))  # only take the first element
            label_lis.append(processed_data)
        
        seq = [list(row) for row in zip(*label_lis)]
        seq_lis.append(seq)
    
    return seq_lis

def data_preprocess(args):
    '''
    Data preprocessing, obtaining standard model parameters and training data
    '''
    dir_path = args.data_dir
    dataset_name = args.dataset
    meta_size = args.meta_size
    max_time_step = args.max_step
    base_offset = args.id_offset
    id_ceiling = args.id_ceiling
    format_type = args.format_type
    align_type = args.align_type
    # split_seed = args.split_seed
    split_seed = args.random_seed
    # load experiment data
    exp_data = select_normal_data(dataset_name, dir_path, meta_size, base_offset, format_type, max_time_step, align_type, split_seed)
    q_size = exp_data.q_size
    s_size = exp_data.s_size
    a_size = exp_data.a_size
    # record experiment args
    args.a_range = a_size[1] + 1
    args.s_range = s_size[1] + 1
    args.q_range = q_size[1] + 1
    args.input_range = max(id_ceiling, a_size[1], s_size[1], q_size[1]) + 1
    args.tag = datetime.now().strftime("%Y%m%d%H%M%S")
    args.config_path = save_exp_args(args)
    return args, exp_data