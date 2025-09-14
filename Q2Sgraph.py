# @AlaoCode#> This module places universal graph structure.
from collections import Counter
from utils._common import print_debug

class Q2SGraph:
    '''
    The bipartite graph object of q2s.
    '''
    def __init__(self, q_size, s_size):
        super(Q2SGraph, self).__init__()
        self.q_size = q_size
        self.s_size = s_size

        self.edges_lis = []
        self.q_nodes_set = Counter()
        self.s_nodes_set = Counter()
        self.edges_set = Counter()
        self.no_load_data = True
        self.no_build_graph = True

    def add_edges_from_seqs(self, seqs_lis):
        '''
        Storing graphs with edge weights.
        Args:
            seqs_lis(list): A list of sequences, each sequence is a list of triples (s, q, a).
        '''
        if seqs_lis == []:
            return
        self.no_load_data = False
        edges_lis = self.edges_lis
        q_nodes_set = self.q_nodes_set
        s_nodes_set = self.s_nodes_set

        for seq in seqs_lis:
            for s, q, a in seq:
                edges_lis.append((q, s))
                q_nodes_set[q] += 1
                s_nodes_set[s] += 1

        self.edges_lis = edges_lis
        self.q_nodes_set = q_nodes_set
        self.s_nodes_set = s_nodes_set
        return self

    def get_q2s_sparse_adj_matrix(self):
        '''
        Obtain the sparse adjacency matrix of the Q2S graph.
        Returns:
            tuple:
                - q2s_edge_lis(list{2, Edges}): The edge list of Q2S graph
                - nz_s_weight(list{Edges}): The weight of Q2S graph edges
        '''
        if self.no_build_graph:
            self.build_graph()
            self.no_build_graph = False

        edges_set = self.edges_set
        q2s_edge_lis = []
        weight_lis = []
        s_weight_lis = [0 for _ in range(self.s_size[-1] + 1)]
        for (q, s), weight in edges_set.items():
            q2s_edge_lis.append([q, s])
            weight_lis.append(weight)
            s_weight_lis[s] += weight
            
        nz_s_weight = [weight_lis[i] / s_weight_lis[q2s_edge_lis[i][1]] for i in range(len(q2s_edge_lis))] 
        return q2s_edge_lis, nz_s_weight

    def get_q2s_adj_list(self):
        '''
        Obtain the q2s adjacency table.
        Returns:
            tuple:
                - q2s_adj_lis(list{Q, list{Neighbors}}): The adjacency list of Q2S graph
                - s2q_adj_lis(list{S, list{Neighbors}}): The adjacency list of S2Q graph
        '''
        if self.no_build_graph:
            self.build_graph()
            self.no_build_graph = False

        edges_set = self.edges_set
        q_size = self.q_size[-1] + 1
        s_size = self.s_size[-1] + 1
        q2s_adj_lis = [[] for _ in range(q_size)]
        s2q_adj_lis = [[] for _ in range(s_size)]
        for (q, s), weight in edges_set.items():
            q2s_adj_lis[q].append(s)
            s2q_adj_lis[s].append(q)
        q_neighbors_num = [len(q_neighbors) for q_neighbors in q2s_adj_lis]
        s_neighbors_num = [len(s_neighbors) for s_neighbors in s2q_adj_lis]

        print_debug('\n\n>>>[Complete] Adjacency information export')
        print_debug(f'Questions/Skills Node Number is: {len(q2s_adj_lis)}/{len(s2q_adj_lis)}')
        print_debug(f'Questions/Skills Max of Number-Neighbors is: {max(q_neighbors_num)}/{max(s_neighbors_num)}')
        return q2s_adj_lis, s2q_adj_lis

    def build_graph(self):
        '''
        Build the graph by edges and weights.
        '''
        if self.no_load_data:
            print('[Error] The data in the picture has not been loaded yet, stop running.')
            return
        edges_lis = self.edges_lis
        edges_set = Counter(edges_lis)
        q_nodes_set = self.q_nodes_set
        s_nodes_set = self.s_nodes_set

        print_debug('\n\n>>>[Complete] Graph building')
        print_debug(f'Valid Edge Number is: {len(edges_set)}')
        print_debug(f'Valid Question Node Number is: {len(q_nodes_set)}')
        print_debug(f'Valid Skill Node Number is: {len(s_nodes_set)}')
        self.edges_set = edges_set
        self.no_build_graph = False
        return self

    def add_edges_from_structured_seqs(self, seqs_lis):
        '''
        Add edges through structured sequences.
        Args:
            seqs_lis(list): A list of sequences, each sequence is a list of triples (s_, q_, a_).
                s_(list): A list of skills
                q_(list): A list of questions
                a_(list): A list of answers
        '''
        if seqs_lis == []:
            return
        self.no_load_data = False
        edges_lis = self.edges_lis
        q_nodes_set = self.q_nodes_set
        s_nodes_set = self.s_nodes_set

        for seq in seqs_lis:
            for s_, q_, a_ in seq:
                q_nodes_set.update(q_)
                s_nodes_set.update(s_)
                q = q_[0]
                for s in s_:
                    edges_lis.append((q, s))

        self.edges_lis = edges_lis
        self.q_nodes_set = q_nodes_set
        self.s_nodes_set = s_nodes_set
        return self