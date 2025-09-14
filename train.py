# @AlaoCode#> This module is used to train the model
import torch
import numpy as np
from types import SimpleNamespace
from Bigraph import BiGraph
from Q2Sgraph import Q2SGraph
from model._template_model import TrainModel
from model.dkt import DKT
from model.herakt import HeRaKT
from utils.record import save_exp_data
from utils.seqprocess import compute_q2q_sim_matrix, compute_q_difficulty

def train(args, exp_data):
    '''
    Model run, train on [exp_data.train_seqs], valid on [exp_data.valid_seqs], eval on [exp_data.test_seqs].
    '''
    # Initialize the model
    # @AlaoCode#> There the [arg.a_range] is default 3, including (0, 1, 2) and be compatible with mask tokens [UNKNOWN_ID]
    args.a_range = args.a_range + 1
    model = select_model(args)
    # Load graph data
    graph_info_func = graph_processing(exp_data, model.device)
    save_exp_data(exp_data, args)
    model.set_graph_adj(graph_info_func)
    model.show_model_info()
    # Start training
    train_seqs = exp_data.train_seqs
    valid_seqs = exp_data.valid_seqs
    test_seqs = exp_data.test_seqs
    train_seqs_len = exp_data.train_seqs_len
    valid_seqs_len = exp_data.valid_seqs_len
    test_seqs_len = exp_data.test_seqs_len
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    model.start_train_data(train_seqs, train_seqs_len, valid_seqs, valid_seqs_len, test_seqs, test_seqs_len, \
                           lr, num_epochs, batch_size)

# @AlaoCode#> Encapsulate callback functions to store graph data
def graph_processing(exp_data, device: torch.device) -> SimpleNamespace:
    '''
    Extract graph data from [exp_data.train_seqs] and encapsulate it as methods, and retrieve it as needed.
    Returns:
        graph_info: SimpleNamespace, with methods:
            - get_global_graph()
            - get_q2q_sim_matrix()
            - get_q_difficulty()
            - get_bigraph_info_matrix()
    '''
    q_size = exp_data.q_size
    s_size = exp_data.s_size
    q_range = q_size[1] + 1
    s_range = s_size[1] + 1
    train_data = exp_data.train_seqs
    graph_info = SimpleNamespace()
    def get_global_graph():
        '''
        Obtain a universal Q2S bipartite graph.
        Returns:
            tuple:
                - sparse_adj_matrix(torch.sparse_coo_tensor{Q, S}): The adjacency matrix of Q2S graph
                - q2s_edges_tensor(torch.tensor{2, Edges}): The edge list of Q2S graph
                - nz_s_weight_tensor(torch.tensor{Edges}): The weight of Q2S graph edges
        '''
        if hasattr(exp_data, 'graph_obj'):
            G_q2s = exp_data.graph_obj.G_q2s
        else:
            G_q2s = Q2SGraph(q_size, s_size).add_edges_from_seqs(train_data)
        G_q2s.get_q2s_adj_list()    # Check adj info
        q2s_edge_lis, weight_lis = G_q2s.get_q2s_sparse_adj_matrix()
        q2s_edges_tensor = torch.tensor(q2s_edge_lis, dtype=torch.long, device=device).t()
        sparse_adj_matrix = torch.sparse_coo_tensor(q2s_edges_tensor, torch.ones(len(q2s_edge_lis)), size=(q_range, s_range), device=device)
        nz_s_weight_tensor = torch.tensor(weight_lis, dtype=torch.float, device=device)
        return sparse_adj_matrix, q2s_edges_tensor, nz_s_weight_tensor
    def get_q2q_sim_matrix():
        '''
        Obtain a question similarity matrix for NMI.
        Returns:
            torch.tensor{3, Pairs}: The (q_i, q_j, sim_value) pairs with sim_value > τ
        '''
        q2q_sim_matrix = compute_q2q_sim_matrix(train_data, q_range)
        q2q_sim_pairs = extract_matrix_pairs(q2q_sim_matrix, tau=0.5)
        q2q_sim_pairs_tensor = torch.from_numpy(q2q_sim_pairs).to(torch.float32).to(device)
        return q2q_sim_pairs_tensor
    def get_q_difficulty():
        '''
        Obtain a question difficulty list.
        Returns:
            torch.tensor{Q}: The difficulty of each question
        '''
        q_difficulty = compute_q_difficulty(train_data, q_size[1] + 1)
        q_difficulty_tensor = torch.tensor(q_difficulty, dtype=torch.float, device=device)
        return q_difficulty_tensor
    def get_bigraph_info_matrix():
        '''
        Obtain a Q2S bipartite graph with double nodes for DGEKT.
        Returns:
            tuple:
                - q2s_adj_matrix(torch.sparse_coo_tensor{Q*2, S*2}): The adjacency matrix of Q2S graph
                - q2q_adj_matrix(torch.sparse_coo_tensor{Q*2, Q*2}): The adjacency matrix of Q2Q graph
        '''
        if hasattr(exp_data, 'graph_obj'):
            B_G = exp_data.graph_obj.G_bi
        else:
            B_G = BiGraph(q_size, s_size).add_edges_from_seqs(train_data)
        q2s_edges, _ = B_G.get_q2s_sparse_adj_matrix()
        q2q_edges, _ = B_G.get_q2q_sparse_nmi_matrix()
        q2s_edges_tensor = torch.tensor(q2s_edges, dtype=torch.long, device=device).t()
        q2s_adj_matrix = torch.sparse_coo_tensor(q2s_edges_tensor, torch.ones(len(q2s_edges)), size=(q_range*2, s_range), device=device)
        q2q_edges_tensor = torch.tensor(q2q_edges, dtype=torch.long, device=device).t()
        q2q_adj_matrix = torch.sparse_coo_tensor(q2q_edges_tensor, torch.ones(len(q2q_edges)), size=(q_range*2, q_range*2), device=device)
        return q2s_adj_matrix, q2q_adj_matrix
    graph_info.get_global_graph = get_global_graph
    graph_info.get_q2q_sim_matrix = get_q2q_sim_matrix
    graph_info.get_q_difficulty = get_q_difficulty
    graph_info.get_bigraph_info_matrix = get_bigraph_info_matrix
    return graph_info

def extract_matrix_pairs(matrix: np.ndarray, tau: float = 0.5) -> np.ndarray:
    '''
    Extract (q_i, q_j, sim_value) triples from a q2q similarity matrix where sim_value > tau.
    Args:
        matrix (np.ndarray{Q, Q}): Question–question similarity matrix
        tau (float): Threshold; only pairs with sim_value > tau are kept
    Returns:
        np.ndarray{3, Pairs}: Stacked triples (q_i, q_j, sim_value) of all pairs
            with similarity above the threshold and excluding the diagonal
    '''

    matrix_numpy = np.array(matrix)
    # Locate indices where similarity exceeds the threshold
    q_i, q_j = np.where(matrix_numpy > tau)
    # Exclude diagonal entries (q_i == q_j)
    mask = q_i != q_j
    q_i, q_j = q_i[mask], q_j[mask]
    sim_values = matrix_numpy[q_i, q_j]
    high_sim_pairs = np.stack([q_i, q_j, sim_values], axis=0)

    return high_sim_pairs


def select_model(args) -> TrainModel:
    '''
    Select a model by [args.model]
    '''
    if args.model == 'dkt':
        model = DKT(args)
    elif args.model == 'herakt':
        model = HeRaKT(args)
    else:
        print('[Error] No available model selected, program terminated!')
        print('[Error] No available model selected, program terminated!')
        print('[Error] No available model selected, program terminated!')
        return
    return model.to(model.device)