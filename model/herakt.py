# @AlaoCode#> Our Model: Heterogeneous Bidirectional Modeling for Response-aware Knowledge Tracing(HeRaKT)
import torch
import torch.nn as nn
from model._template_model import TrainModel
from model.dkt import DKT

class HeRaKT(DKT):
    def __init__(self, args, model_name='HeRaKT'):
        super(HeRaKT, self).__init__(args, model_name)
        pass