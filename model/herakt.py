# @AlaoCode#> Our Model: Heterogeneous Bidirectional Modeling for Response-aware Knowledge Tracing(HeRaKT)
import torch
import torch.nn as nn
from model._template_model import TrainModel

class HeRaKT(TrainModel):
    def __init__(self, args, model_name='HeRaKT'):
        super(HeRaKT, self).__init__(args, model_name)
        # Data control
        # where data control changed as dataset
        self.zero2zero = True
        self.q_range = args.q_range
        self.s_range = args.s_range
        self.a_range = args.a_range
        # Model architecture hyperparameters
        self.q_sparsity = args.q_sparsity
        self.s_sparsity = args.s_sparsity
        self.UNKNOWN_ID = 3
        self.embedding_size = args.embedding_size
        self.pos_l = args.pos_l
        self.neg_l = args.neg_l
        self.rnn_l = args.rnn_l
        self.dropout = args.dropout
        self.padding_idx = args.padding_idx
        self.max_step = args.max_step
        self.train_strategy = 'MASK_LEARNING'
        # Model architecture
        # Embedding layer
        self.question_embedding_layer = SemanticEmbedding(self.q_range, self.embedding_size, self.q_sparsity, padding_idx=self.padding_idx)
        self.skill_embedding_layer = SemanticEmbedding(self.s_range, self.embedding_size, self.s_sparsity, padding_idx=self.padding_idx)
        self.answer_embedding_layer = nn.Embedding(self.a_range, self.embedding_size, padding_idx=self.padding_idx)
        self.avg_gcn_layer = AvgGCN(self.question_embedding_layer, self.skill_embedding_layer, layer_num=1, zero2zero=True)
        # Enhancer
        self.enhancer = BiLSTM(
            self.embedding_size, 
            self.embedding_size,
            num_layers=self.pos_l,
            batch_first=True
        )
        # Denoiser
        self.denoiser = LFFTStack(seq_len=self.max_step, d_model=self.embedding_size, n_layers=self.neg_l, channelwise=False)
        # Propagation layer
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, self.rnn_l, dropout=self.dropout, batch_first=True)
        # Prediction layer
        self.biline_predict_layer = nn.Bilinear(self.embedding_size, self.embedding_size, 1)
        # FC units
        self.pos_fc = nn.Linear(self.embedding_size*3, self.embedding_size)
        self.neg_fc = nn.Linear(self.embedding_size*3, self.embedding_size)
        self.row_fc = nn.Linear(self.embedding_size*3, self.embedding_size)
        # Norm&Add
        self.norm_layer = nn.LayerNorm(self.embedding_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        # Nonlinear unit
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # loss - BCEWithLogit
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, seqs):
        s = seqs[:, :, 0]
        q = seqs[:, :, 1]
        a = seqs[:, :, 2]
        # Emb
        s_embedding = self.avg_gcn_layer(q, self.sparse_graph)
        q_embedding = self.question_embedding_layer(q)
        a_embedding = self.answer_embedding_layer(a)
        # Correctness assumption
        t = a.clone()
        t[a != 0] = 2
        t_embedding = self.answer_embedding_layer(t)
        # Heterogeneous modeling
        e_pos = self.relu(self.pos_fc(torch.cat((q_embedding, s_embedding, a_embedding), dim=-1)))
        e_neg = self.relu(self.neg_fc(torch.cat((q_embedding, s_embedding, a_embedding), dim=-1)))
        e_row = self.relu(self.row_fc(torch.cat((q_embedding, s_embedding, a_embedding), dim=-1)))
        e_pos_enh = self.enhancer(e_pos)
        e_neg_enh = self.denoiser(e_neg)
        f_mask = (a == self.id_offset).float().unsqueeze(-1)
        t_mask = (a == (self.id_offset + 1)).float().unsqueeze(-1)
        e_embedding = self.norm_layer(self.dropout_layer(e_pos_enh) * t_mask + self.dropout_layer(e_neg_enh) * f_mask + e_row)
        r_embedding = self.relu(self.pos_fc(torch.cat((q_embedding, s_embedding, t_embedding), dim=-1)))
        # Rnn
        outputs, _ = self.lstm(e_embedding)
        # Output P{h_t => r_t}
        h_t = outputs[:, :-1, :]
        r_t = r_embedding[:, 1:, :]
        logits = self.biline_predict_layer(h_t, r_t)[:, :, -1]
        return logits

    def set_graph_adj(self, graph_info_func):
        '''
        Get global Q2S graph.
        '''
        sparse_graph, _, _ = graph_info_func.get_global_graph()
        self.sparse_graph = sparse_graph.to(self.device)

class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers=1, dropout=0.0, batch_first=True, bidirectional=True):
        '''
        The BiLSTM module to enhance positive embeddings.
        Args:
            num_layers(int): The number of layers
            dropout(float): The dropout rate
            batch_first(bool): Whether the input tensor is in batch first format
            bidirectional(bool): Whether the LSTM is bidirectional
        Inputs:
            x{B, S, D}: Input tensor to be enhanced
        Outputs:
            output{B, S, D}: The enhanced output tensor
        '''
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)
        self.hidden_size = hidden_size
        self.direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.direction, embedding_size)
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out)
        return out

class LFFTBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        channelwise: bool = False
    ):
        '''
        The LFFT layer, FFT -> Learnable filtering -> iFFT.
        Args:
            seq_len(int): sequence length
            d_model(int): The number of channels, equivalent to the embedding dimension
            channelwise(bool): Whether to learn frequency weights per channel, 
                learn (S, D) weights if True; learn (S, 1) weights if False, share weights across channels
        Inputs:
            x{B, S, D}: Input tensor to be denoised
        Outputs:
            output{B, S, D}: The denoised output tensor
        '''
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.channelwise = channelwise
        if channelwise:
            self.filter_weights = nn.Parameter(torch.ones(seq_len, d_model))
        else:
            self.filter_weights = nn.Parameter(torch.ones(seq_len, 1))
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        assert S == self.seq_len and D == self.d_model, \
            f"Input shape (S={S}, D={D}) must match configured (S={self.seq_len}, D={self.d_model})."
        x_freq = torch.fft.fft(x, dim=1)    # FFT
        W = self.act(self.filter_weights)   # Learnable filtering
        W = W.unsqueeze(0)                  # -> (1, S, 1) or (1, S, D)
        x_freq_filt = x_freq * W            # Take the actual part
        x_denoised = torch.fft.ifft(x_freq_filt, dim=1).real    # IFFT
        return x_denoised


class LFFTStack(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_layers: int = 1,
        channelwise: bool = False,
        share_weights: bool = False
    ):
        '''
        The LFFTStack module to denoise negative embeddings.
        Args:
            seq_len(int): sequence length
            d_model(int): The number of channels, equivalent to the embedding dimension
            n_layers(int): The number of LFFT layers
            channelwise(bool): Whether to learn frequency weights per channel, 
                learn {S, D} weights if True; learn {S, 1} weights if False, share weights across channels
            share_weights(bool): Whether to share weights across layers
        Inputs:
            x{B, S, D}: Input tensor to be denoised
        Outputs:
            output{B, S, D}: The denoised output tensor
        '''
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layers = n_layers

        if share_weights:
            block = LFFTBlock(seq_len, d_model, channelwise)
            self.blocks = nn.ModuleList([block] + [block for _ in range(n_layers - 1)])
        else:
            self.blocks = nn.ModuleList([
                LFFTBlock(seq_len, d_model, channelwise)
                for _ in range(n_layers)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

class SemanticEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, t=0.5, padding_idx=None, **kwargs):
        '''
        Enhance the embedding layer using heuristic methods, initialize the embedding with soft one-hot.
        Args:
            t(float): Sparsity threshold, values below t are set to 0, values above t are kept
        Inputs:
            x{B, S}: Input question ID sequences
        Outputs:
            output{B, S, D}: The embedding tensor
        '''
        super(SemanticEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.threshold = t
        self.tau = 1 - t

        init_weight = torch.randn(num_embeddings, embedding_dim)  # (Q or S, D)
        # # Gumbel Softmax generates sparse representations
        # sparse_weight = F.gumbel_softmax(init_weight, tau=self.tau, hard=False)
        # Thresholding generates sparse representations
        init_weight = torch.abs(init_weight)
        mask =  torch.abs(init_weight) >= self.threshold 
        sparse_weight = init_weight * mask
        self.weight = nn.Parameter(sparse_weight)

    def forward(self, x):
        return self.weight[x]


class AvgGCN(nn.Module):
    # input_size:   {batch, nodes_num}
    # output_size:  {batch, nodes_num, embedding_size}
    def __init__(self, q_embedding_layer: nn.Embedding, s_embedding_layer: nn.Embedding, layer_num=1, zero2zero=False):
        '''
        Average convolutional neighbor vector.
        Args:
            q_embedding_layer(nn.Embedding): The query embedding layer
            s_embedding_layer(nn.Embedding): The semantic embedding layer
            layer_num(int): The number of layers
            zero2zero(bool): Whether to set the embedding of padding token to 0
        Inputs:
            x{B, N}: Input question ID sequences
        Outputs:
            output{B, N, D}: The output tensor
        '''
        super(AvgGCN, self).__init__()
        self.q_embedding_layer = q_embedding_layer
        self.s_embedding_layer = s_embedding_layer
        self.layer_num = layer_num
        self.zero2zero = zero2zero
        self.gcn_q_Ematrix = None
        self.gcn_s_Ematrix = None

    def forward(self, x: torch.Tensor, q2s_A_matrix: torch.sparse.Tensor):
        q_Ematrix = self.q_embedding_layer.weight
        s_Ematrix = self.s_embedding_layer.weight
        # Compute the degree of d_q and d_s
        d_q = torch.sparse.sum(q2s_A_matrix, dim=1).to_dense().clamp(min=1)
        d_s = torch.sparse.sum(q2s_A_matrix, dim=0).to_dense().clamp(min=1)
        # Compute normalization factor ω=1/d
        omega_q = 1 / d_q.unsqueeze(1)
        omega_s = 1 / d_s.unsqueeze(1)        
        for _ in range(self.layer_num):
            # Compute E'=A'*X
            l_q_Ematrix = torch.sparse.mm(q2s_A_matrix, s_Ematrix)
            l_s_Ematrix = torch.sparse.mm(q2s_A_matrix.t(), q_Ematrix)
            # Compute E=ωE'
            q_Ematrix = l_q_Ematrix * omega_q
            s_Ematrix = l_s_Ematrix * omega_s

        self.gcn_q_Ematrix = q_Ematrix
        self.gcn_s_Ematrix = s_Ematrix
        x_embedding = self.gcn_q_Ematrix[x]
        if self.zero2zero:
            mask = (x == 0).unsqueeze(-1).to(torch.float32)
            x_embedding = (1 - mask) * x_embedding
        return x_embedding