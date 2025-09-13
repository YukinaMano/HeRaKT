import argparse
import os
import random
import torch
import numpy as np


def load_model_args():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    arg_parser = argparse.ArgumentParser(description='train dkt model')
    # Dataset
    arg_parser.add_argument('--data_dir', type=str, default='datasets')
    arg_parser.add_argument('--log_dir', type=str, default='logs')
    arg_parser.add_argument('--meta_size', type=int, default=4)
    arg_parser.add_argument(
        '--dataset',
        type=str,
        default='assist2009',
        choices=['assist2009', 'assist12_3', 'ednet', 'assist09_3', 'ednet_5000_3', 'csedm', 'junyi_3']
    )
    arg_parser.add_argument('--max_step', type=int, default=200)
    arg_parser.add_argument('--question_neighbor_num', type=int, default=4)
    arg_parser.add_argument('--skill_neighbor_num', type=int, default=4)    # default=10
    arg_parser.add_argument('--n_hop', type=int, default=2)
    arg_parser.add_argument('--format_type', type=str, default='separate', choices=['converge', 'separate'], help='data mapping method')
    arg_parser.add_argument('--align_type', type=str, default='cut', choices=['cut', 'whole_split', 'half_split'], help='data alignment method')
    arg_parser.add_argument('--split_seed', type=int, default=1291389, help='random seed for dataset splitting')
    arg_parser.add_argument('--random_seed', type=int, default=198136938, help='random seed for model')
    arg_parser.add_argument('--q_size', type=int, default=54000, help='total number of questions')
    arg_parser.add_argument('--s_size', type=int, default=300, help='total number of skills')
    # Train
    arg_parser.add_argument('--model', type=str, default='dkt')
    arg_parser.add_argument('--premodel', type=str, default='None')
    arg_parser.add_argument('--lr', type=float, default=0.0005)              # 0.001
    arg_parser.add_argument('--num_epochs', type=int, default=200)           # default=200
    arg_parser.add_argument('--batch_size', type=int, default=64)
    arg_parser.add_argument('--action', type=str, default='train', choices=['train', 'eval'])
    arg_parser.add_argument('--train_strategy', type=str, default='ALL_STEP', choices=['ALL_STEP', 'LAST_STEP', 'MASK_LEARNING'])
    arg_parser.add_argument('--inference_program', type=str, default='MASK_TARGET', choices=['ALL_STEP', 'MASK_TARGET', 'REVERSE_TARGET', 'RANDOM_MASK', 'SAME_MASK', 'DIFFERENT_MASK'])
    arg_parser.add_argument('--intv_ratio', type=float, default=0.15, help='the intv ratio for RANDOM_MASK, SAME_MASK, DIFFERENT_MASK')
    arg_parser.add_argument('--mask_ratio', type=float, default=0.30, help='the mask ratio for MASK_LEARNING')
    # Model
    arg_parser.add_argument('--padding_idx', type=int, default=0, help='Value used for padding')
    arg_parser.add_argument('--embedding_size', type=int, default=256)
    arg_parser.add_argument('--hidden_size', type=int, default=256)
    arg_parser.add_argument('--dropout', type=float, default=0.3, help='General dropout probability')
    arg_parser.add_argument('--mask_dropout', type=float, default=0.3, help='Probability of being masked')
    arg_parser.add_argument('--cands_size', type=int, default=16, help='Size of candidate set')
    arg_parser.add_argument('--id_ceiling', type=int, default=-1, help='Upper bound of IDs, computed from dataset by default')
    arg_parser.add_argument('--id_offset', type=int, default=1, help='ID offset, 0 means IDs start from 0')
    arg_parser.add_argument('--gcn_l', type=int, default=2, help='Number of GCN layers')
    arg_parser.add_argument('--rnn_l', type=int, default=2, help='Number of RNN layers')
    arg_parser.add_argument('--transformer_l', type=int, default=3, help='Number of Transformer layers')
    arg_parser.add_argument('--trans_head_n', type=int, default=8, help='Number of Transformer attention heads')
    arg_parser.add_argument('--trans_forward_d', type=int, default=512, help='Dimension of Transformer feed-forward network')
    arg_parser.add_argument('--encoder_l', type=int, default=3, help='Number of Transformer encoder layers')
    arg_parser.add_argument('--decoder_l', type=int, default=1, help='Number of Transformer decoder layers')
    arg_parser.add_argument("--pos_l", type=int, default=1, help='positive layer num')
    arg_parser.add_argument("--neg_l", type=int, default=1, help='negative layer num')
    arg_parser.add_argument("--q_sparsity", type=float, default=0.50, help='question sparsity')
    arg_parser.add_argument("--s_sparsity", type=float, default=0.75, help='skill sparsity')
    # Experiment Setting
    arg_parser.add_argument('--use_device', type=str, default='gpu', choices=['cpu', 'gpu'])
    arg_parser.add_argument('--device_id', type=int, default=0, help='valid only when use_device is gpu')
    arg_parser.add_argument('--exp_code', type=str, default='', help='exp appendix info')
    arg_parser.add_argument('--exp_out', type=str, default='results.csv', help='result output path')

    args = arg_parser.parse_args()
    return args
    
def init_certain_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cuda
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_device(args):
    if args.use_device == 'gpu':
        torch.cuda.set_device(args.device_id)

if __name__ == '__main__':
    args = load_model_args()
    init_certain_seed(args.random_seed)
    load_device(args)
