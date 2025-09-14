# [DKT](https://dl.acm.org/doi/abs/10.5555/2969239.2969296)
import torch
import torch.nn as nn
from model._template_model import TrainModel

class DKT(TrainModel):
    def __init__(self, args, model_name = 'DKT'):
        '''
        DKT:
        - Embedding layer   => Random initialization i&q i=(q, r)
        - Propagation layer => h = LSTM(i)
        - Prediction layer  => p_t = h_t * q_t
        '''
        super(DKT, self).__init__(args, model_name)
        # data control
        # where data control changed as dataset
        self.q_range = args.q_range
        self.a_range = args.a_range
        # model architecture hyperparameters
        self.embedding_size = args.embedding_size
        self.dropout = args.dropout
        self.rnn_l = args.rnn_l
        self.padding_idx = args.padding_idx
        # model architecture
        # embedding layer
        self.question_embedding_layer = nn.Embedding(self.q_range, self.embedding_size, padding_idx=self.padding_idx)
        self.interaction_embedding_layer = nn.Embedding(self.q_range*self.a_range, self.embedding_size, padding_idx=self.padding_idx)
        # propagation layer
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, self.rnn_l, dropout=self.dropout, batch_first=True)
        # prediction layer
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()
        # loss - BCEWithLogit
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, seqs):
        q = seqs[:, :, 1]
        a = seqs[:, :, 2]
        Q = self.q_range
        i = q + a * Q
        # Emb i = q + a * Q
        i_embedding = self.interaction_embedding_layer(i)
        q_embedding = self.question_embedding_layer(q)
        # Rnn
        rnn_outs, _ = self.lstm(i_embedding)
        # Output
        q_t = q_embedding[:, 1:, :]
        h_t = rnn_outs[:, :-1, :]
        h_t = self.dropout_layer(h_t)   # paper step
        # # @AlaoCode#> There is a modification because the sum of multiple y_t · σ(q_t) in the original text cannot be guaranteed to be between [0,1]
        # y_t = self.sigmoid(self.fc2to1(h_t))
        # p_t = torch.sum(y_t * self.sigmoid(q_t), dim=-1)
        y_t = self.fc(h_t)
        logits = torch.sum(y_t * q_t, dim=-1)
        return logits