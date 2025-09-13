# @AlaoCode#> This module serves as a template for the model
# @AlaoCode#> connot use, should extend
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from utils._common import get_pytorch_device
from utils._common import print_debug
from utils.record import log_experiment_result, save_best_model

class TrainModel(nn.Module):
    def __init__(self, args, model_name='train_model'):
        self.model_name = model_name
        super(TrainModel, self).__init__()
        # data control
        self.id_offset = args.id_offset
        self.batch_size = args.batch_size
        self.batch_guide = torch.arange(args.batch_size)
        self.max_step = args.max_step
        self.hist_neighbor_num = args.skill_neighbor_num
        self.question_neighbor_num = args.question_neighbor_num
        self.device_id = args.device_id
        self.device = get_pytorch_device(args.use_device, args.device_id)
        # where data control changed as dataset
        self.dataset = args.dataset
        self.input_range = args.input_range
        self.q_range = args.q_range
        self.s_range = args.s_range
        self.a_range = args.a_range
        # model architecture hyperparameters
        self.padding_idx = args.padding_idx
        self.embedding_size = args.embedding_size
        self.predict_size = 1
        self.dropout = args.dropout
        # experiment setting
        self.UNKNOWN_ID = args.a_range
        self.train_strategy = args.train_strategy
        self.inference_program = args.inference_program
        self.mask_ratio = args.mask_ratio
        self.intv_ratio = args.intv_ratio
        self.tag = args.tag
        self.exp_code = args.exp_code
        self.exp_out = args.exp_out
        self.seed = args.random_seed
        # @AlaoCode#> [subclasses override] model architecture

        # loss - BCEWithLogit
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        
    # @AlaoCode#> [subclasses override] forward
    def forward(self, x):
        pass
    # @AlaoCode#> [subclasses override] get graph info
    def set_graph_adj(self, graph_info_func):
        pass
    def compute_loss(self, target_probability, target_answer):
        '''
        Args:
            target_probability: {batch_size, N}
            target_answer: {batch_size, N}
        where N will change based on [train_strategy] and [inference_program]
        Returns:
            loss: {1}
        '''
        pt = target_probability
        at = target_answer
        loss = self.loss_fn(pt, at)     # loss: {batch_size, 1}
        return loss
    def forward_batch_data(self, batch, action_mode='eval'):
        '''
        forward [batch]
        '''
        seqs, seqs_len = batch
        seqs, seqs_len = seqs.to(self.device), seqs_len.to(self.device)
        '''
        Tensor[seqs]{B, S, 3}, the data structure is as follows:
        1 ,  2, 11, 45, ... => skill
        11, 23, 57, 49, ... => question
        1 ,  2,  1,  0, ... => answer
        '''
        a_t = seqs[:, 1:, -1].float() - 1   # pass first timestep
        # lock active position
        no_padding_mask = (a_t >= 0).long()
        # lock target position
        last_valid_mask = (no_padding_mask.sum(dim=1) - 1).long()
        # get mask token
        UNKNOWN_ID = self.UNKNOWN_ID
        '''
        SELECT[action_mode]===>
        *train: training model parameters
        *eval: fixed model for inference
        '''
        if action_mode == 'train':
            '''
            SELECT[train_strategy]===>
            *ALL_STEP: full sequence for training
            *LAST_STEP: only last step for training
            *MASK_LEARNING: only mask step for training
            '''
            if self.train_strategy == 'ALL_STEP':
                logit_t = self.forward(seqs)
                valid_logit_t = logit_t[no_padding_mask == 1]
                valid_a_t = a_t[no_padding_mask == 1]
            elif self.train_strategy == 'LAST_STEP':
                logit_t = self.forward(seqs)
                valid_logit_t = logit_t[self.batch_guide, last_valid_mask]
                valid_a_t = a_t[self.batch_guide, last_valid_mask]
            else:
                mask_ratio = 0.30 if self.mask_ratio is None else self.mask_ratio
                mutation_mask = (torch.rand_like(a_t) < mask_ratio) & (a_t >= 0)
                sample_mask = (torch.rand_like(a_t) < 0.70) & (a_t >= 0)
                seqs[:, 1:, -1] = torch.where(mutation_mask, torch.tensor(UNKNOWN_ID, device=self.device), seqs[:, 1:, -1])
                logit_t = self.forward(seqs)
                valid_logit_t = logit_t[mutation_mask == 1]
                valid_a_t = a_t[mutation_mask == 1]
        else:
            r_last = seqs[self.batch_guide, last_valid_mask + 1, -1].unsqueeze(1)
            r_all = seqs[self.batch_guide, :, -1]
            r_com = (r_all == r_last)
            r_unc = (r_all != r_last)
            r_random = (r_all > 0)
            r_mark = torch.zeros_like(r_all).bool()
            intv_ratio = 0.15 if self.intv_ratio is None else self.intv_ratio
            '''
            SELECT[inference_program]===>
            *RANDOM_MASK: mask random responses
            *SAME_MASK: mask only responses of the same-type
            *DIFFERENT_MASK: mask only responses of the different-type
            *REVERSE_TARGET: reverse the last response
            *MASK_TARGET: mask the last response
            '''
            if self.inference_program == 'RANDOM_MASK':
                r_mark = r_random & (torch.rand_like(r_all.float()) < intv_ratio)
            elif self.inference_program == 'SAME_MASK':
                r_mark = r_com & (torch.rand_like(r_all.float()) < intv_ratio)
            elif self.inference_program == 'DIFFERENT_MASK':
                r_mark = r_unc & (torch.rand_like(r_all.float()) < intv_ratio)
            elif self.inference_program == 'REVERSE_TARGET':
                last_a_reverse = (r_last == 1).long() * 2 + (r_last == 2).long() * 1
                seqs[self.batch_guide, last_valid_mask + 1, -1] = last_a_reverse
            else:
                pass
            seqs[:, :, 2] = torch.where(r_mark, torch.tensor(UNKNOWN_ID, device=self.device), r_all)
            seqs[self.batch_guide, last_valid_mask + 1, -1] = UNKNOWN_ID
            logit_t = self.forward(seqs)
            valid_logit_t = logit_t[self.batch_guide, last_valid_mask]
            valid_a_t = a_t[self.batch_guide, last_valid_mask]
            
        batch_loss = self.compute_loss(valid_logit_t, valid_a_t)
        return batch_loss, self.sigmoid(valid_logit_t), valid_a_t
    def start_train_batch(self, optimizer, batch):
        '''
        train [batch] through [optimizer]
        '''
        self.train()
        scaler = GradScaler()
        optimizer.zero_grad()
        with autocast(device_type=f'cuda:{self.device_id}', dtype=torch.float32):
            loss, p_t, a_t = self.forward_batch_data(batch, action_mode='train')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss, p_t, a_t = self.forward_batch_data(batch, action_mode='train')
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        return loss, p_t, a_t
    def start_eval_batch(self, _, batch):
        '''
        eval [batch]
        '''
        self.eval()
        with torch.no_grad():
            loss, p_t, a_t = self.forward_batch_data(batch)
        return loss, p_t, a_t
    def start_run_epoch(self, optimizer, data_loader, action_mode='train'):
        '''
        train or evaluate a single epoch
        '''
        overall_loss = 0
        preds, targets = [], []
        if action_mode == 'train':
            batch_forward_func = self.start_train_batch
        else:
            batch_forward_func = self.start_eval_batch
        for batch in data_loader:
            loss, p_t, a_t = batch_forward_func(optimizer, batch)
            overall_loss += loss.item()
            preds += p_t.detach().tolist()
            targets += a_t.detach().tolist()
        # compute AUC & ACC
        auc = roc_auc_score(targets, preds)
        binary_preds = [1 if i >= 0.5 else 0 for i in preds]
        binary_targets = [1 if i >= 0.5 else 0 for i in targets]
        PNTF_cnt = Counter(zip(binary_targets, binary_preds))
        print_debug(f'PN:TF ===> {PNTF_cnt}')
        acc = accuracy_score(binary_targets, binary_preds)
        return overall_loss, auc, acc
    def start_train_data(self, train_seqs, train_seqs_len, valid_seqs, valid_seqs_len, test_seqs, test_seqs_len, lr, num_epochs, batch_size):
        '''
        train model with [train_seqs], evaluate [valid_seqs] and [test_seqs]
        '''
        optimizer = optim.Adam(self.parameters(), lr)
        # load data
        train_dataset = SequenceDataset(train_seqs, train_seqs_len)
        valid_dataset = SequenceDataset(valid_seqs, valid_seqs_len)
        test_dataset = SequenceDataset(test_seqs, test_seqs_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
        # start train
        patience = 15
        best_auc = 0
        best_acc = 0
        best_loss = 1e6
        best_test_auc = 0
        best_test_acc = 0
        best_epoch = 0
        early_stop = 0
        start_time = time.time()
        for epoch in tqdm(range(num_epochs)):
            # train
            early_stop += 1
            print(f' ')
            print(f'Epoch-{epoch}')
            overall_loss, train_auc, train_acc = self.start_run_epoch(optimizer, train_loader, action_mode='train')
            print(f'--train loss: {overall_loss:.8f} auc: {train_auc:.8f}, acc: {train_acc:.8f}')
            # eval
            valid_loss, valid_auc, valid_acc = self.start_run_epoch(None, valid_loader, action_mode='eval')
            print(f'--valid loss: {valid_loss:.8f} auc: {valid_auc:.8f}, acc: {valid_acc:.8f}')
            test_loss, test_auc, test_acc = self.start_run_epoch(None, test_loader, action_mode='eval')
            print(f'---test loss: {test_loss:.8f} auc: {test_auc:.8f}, acc: {test_acc:.8f}')
            # select best model
            # if best_loss - valid_loss >= 0.0 or best_auc - valid_auc <= 0.0:
            if best_loss - valid_loss >= 0.0:
                early_stop = 0
                best_loss = valid_loss
                best_auc = valid_auc
                best_acc = valid_acc
                best_epoch = epoch
                best_test_auc = test_auc
                best_test_acc = test_acc
                save_best_model(self.state_dict(), self.model_name, self.dataset)
            print(f'->>best valid: {best_epoch} loss: {best_loss:.8f} auc: {best_auc:.8f} acc: {best_acc:.8f}')
            print(f'->>test performance auc: {best_test_auc:.8f} acc: {best_test_acc:.8f}')
            if early_stop >= patience:
                break
        run_time = (time.time() - start_time) / 60
        print(f"Early stopping at epoch {best_epoch}")
        print(f"Run time: {run_time:.2f} min")
        print(f"Model-{self.model_name} Dataset-{self.dataset} best auc: {best_test_auc:.8f} best acc: {best_test_acc:.8f}")
        # save results to csv
        log_experiment_result(
            model_name=self.model_name,
            dataset=self.dataset,
            auc=best_test_auc,
            acc=best_test_acc,
            best_epoch=best_epoch,
            tag=self.tag,
            seed=self.seed,
            run_time=run_time,
            notes=self.exp_code,
            gpu_id=self.device_id,
            output_file=self.exp_out
        )
    def predict_data(self, seqs):
        '''
        predict [seqs]
        '''
        self.eval()
        with torch.no_grad():
            p_t = self(seqs)
        return p_t
    def show_model_info(self):
        '''
        print model tips
        '''
        print(' ')
        print(' ')
        print(' ')
        try:
            print(f'****** [Model]: {self.model_name}  [Dataset]: {self.dataset} ******')
        except:
            Exception('!!!Error!!! No model or dataset selected')
        try:
            print(f'****** [Train Strategy]: {self.train_strategy}  [Inference Program]: {self.inference_program} ******')
        except:
            Exception('!!!Error!!! No train strategy or inference program selected')

class SequenceDataset(Dataset):
    '''
    Combining sequence data with sequence length
    '''
    def __init__(self, seqs, seqs_len):
        self.seqs = seqs
        self.seqs_len = seqs_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seqs_len[idx]

def collate_fn(batch):
    '''
    ensure that the batch is sorted by sequence length
    '''
    batch.sort(key=lambda x: x[1], reverse=True)
    seqs, seqs_len = zip(*batch)
    # Convert to Tensor
    seqs = torch.tensor(seqs, dtype=torch.long)
    seqs_len = torch.tensor(seqs_len, dtype=torch.long)
    return seqs, seqs_len