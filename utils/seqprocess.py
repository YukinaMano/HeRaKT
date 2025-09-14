# @AlaoCode#> This module places the methods used to process sequences
import numpy as np
from collections import Counter
from types import SimpleNamespace
from Q2Sgraph import Q2SGraph
from Bigraph import BiGraph
from utils._common import print_debug


def align_structured_time_seqs(seqs_lis: list, max_time_step=200, align_type='cut', padding_value=0) -> tuple[list, list]:
    '''
    Align structured time series length (using post padding)
    Args:
        seqs_lis: list of sequences, each sequence is a list of time steps, each time step is a list of features
        max_time_step: maximum time step
    Returns:
        align_seqs: list of aligned sequences
        seqs_len_lis: list of sequence lengths, each length is the active length of the sequence
    '''
    align_seqs = []
    seqs_len_lis = []

    for seq in seqs_lis:
        seq_len = sum(1 for step in seq if all(all(feature) != 0 for feature in step))
        feature_num = len(seq[0])
        if seq_len <= max_time_step:        # when the sequence length is insufficient, use post padding
            padding_count = max_time_step - seq_len
            padding = [[[padding_value]] * feature_num for _ in range(padding_count)]
            tmp_seq = seq[:seq_len] + padding
            align_seqs.append(tmp_seq)      # **post padding**
            seqs_len_lis.append(seq_len)
        else:                               # when the sequence length is excessive, directly truncate
            align_seqs.append(seq[seq_len-max_time_step:seq_len])
            seqs_len_lis.append(max_time_step)

    print_debug(f'\n\n>>> [Completed] structured time series alignment length control is: {max_time_step}')
    print_debug(f'Processing before and after time series quantity change {len(seqs_lis)} => {len(align_seqs)}')
    
    return align_seqs, seqs_len_lis

def align_time_seqs(seqs_lis: list, max_time_step=200, align_type='cut', padding_value=0) -> tuple[list, list]:
    '''
    Align time series length (using post padding)
    Args:
        seqs_lis: list of sequences, each sequence is a list of time steps, each time step is a list of features
        max_time_step: maximum time step
        align_type: alignment type, 'cut' for direct truncation, 'whole_split' for whole segment split, 'half_split' for half overlap split
    Returns:
        align_seqs: list of aligned sequences
        seqs_len_lis: list of sequence lengths, each length is the active length of the sequence
    '''
    align_seqs = []
    seqs_len_lis = []
    
    for seq in seqs_lis:
        seq_len = len(seq)
        feature_num = len(seq[0])
        if seq_len <= max_time_step:            # when the sequence length is insufficient, use post padding
            padding_count = max_time_step - seq_len
            padding = [[padding_value] * feature_num for _ in range(padding_count)]
            align_seqs.append(seq + padding)    # **post padding**
            seqs_len_lis.append(seq_len)
        else:                                   # when the sequence length is excessive, use the segmentation strategy
            if align_type == 'whole_split':     # **whole segment split**
                n_split = (seq_len + max_time_step - 1) // max_time_step
                for k in range(n_split):
                    start_index = k * max_time_step
                    end_index = (k + 1) * max_time_step
                    if k == n_split - 1:
                        split = seq[-max_time_step:]
                    else:
                        split = seq[start_index:end_index]
                    align_seqs.append(split)
                    seqs_len_lis.append(len(split))
            elif align_type == 'half_split':     # **semi overlapping segmentation**
                n_split = (seq_len + max_time_step - 1) // (max_time_step // 2)
                for k in range(n_split):
                    start_index = k * (max_time_step // 2)
                    end_index = start_index + max_time_step
                    if end_index > seq_len:
                        split = seq[-max_time_step:]
                    else:
                        split = seq[start_index:end_index]
                    align_seqs.append(split)
                    seqs_len_lis.append(len(split))
            else:                               # **direct truncation**
                align_seqs.append(seq[-max_time_step:])
                seqs_len_lis.append(max_time_step)
    
    print_debug(f'\n\n>>> [Completed] time series alignment length control is: {max_time_step}')
    print_debug(f'Processing before and after time series quantity change {len(seqs_lis)} => {len(align_seqs)}')
    
    return align_seqs, seqs_len_lis

def check_knowledge_tracking_data(seqs_lis: list, seqs_structured=False):
    '''
    verify data range distribution
    '''
    SKILL_TAG = 0
    QUESTION_TAG = 1
    ANSWER_TAG = 2

    seq_len_lis = [len(seq) for seq in seqs_lis]
    if seqs_structured:
        skl_fea_lis = [skill for seq in seqs_lis for step in seq for skill in step[SKILL_TAG]]
        qst_fea_lis = [question for seq in seqs_lis for step in seq for question in step[QUESTION_TAG]]
        ans_fea_lis = [answer for seq in seqs_lis for step in seq for answer in step[ANSWER_TAG]]
    else:
        skl_fea_lis = [step[SKILL_TAG] for seq in seqs_lis for step in seq]
        qst_fea_lis = [step[QUESTION_TAG] for seq in seqs_lis for step in seq]
        ans_fea_lis = [step[ANSWER_TAG] for seq in seqs_lis for step in seq]

    seq_len_cnt = Counter(seq_len_lis)
    ans_fea_cnt = Counter(ans_fea_lis)
    skl_fea_cnt = Counter(skl_fea_lis)
    qst_fea_cnt = Counter(qst_fea_lis)

    q_max = max(qst_fea_cnt.keys())
    q_min = min(qst_fea_cnt.keys())
    s_max = max(skl_fea_cnt.keys())
    s_min = min(skl_fea_cnt.keys())
    a_max = max(ans_fea_cnt.keys())
    a_min = min(ans_fea_cnt.keys())
    # conduct scope information statistics
    print_debug('\n\n----------- Data Check ------------')
    print_debug(f'The range of seq-len is [{min(seq_len_cnt.keys())}:{max(seq_len_cnt.keys())}]')
    print_debug(f'The range of skill ID is [{min(skl_fea_cnt.keys())}:{max(skl_fea_cnt.keys())}]')
    print_debug(f'The range of questions ID is [{min(qst_fea_cnt.keys())}:{max(qst_fea_cnt.keys())}]')
    print_debug(f'The range of answer tag is [{min(ans_fea_cnt.keys())}:{max(ans_fea_cnt.keys())}]')
    # conduct count information statistics
    print_debug(f'The number of seq-len is {len(seq_len_cnt)}')
    print_debug(f'The number of skills is {len(skl_fea_cnt)}')
    print_debug(f'The number of questions is {len(qst_fea_cnt)}')
    print_debug(f'The number of answer is {ans_fea_cnt}')
    # conduct sequence information statistics
    print_debug(f'The number of seq is {sum(seq_len_cnt.values())}')
    print_debug(f'The number of (s, q, a) is {sum(qst_fea_cnt.values())}')

    return q_max, q_min, s_max, s_min, a_max, a_min


def format_time_seqs(seqs_lis: list, base_offset, format_type='separate', seqs_structured=False, question_size=None, skill_size=None, answer_size=None) -> tuple[list, tuple, tuple, tuple]:
    '''
    Mapping time series to the specified feature space
    '''
    print_debug(f'\n\n>>> [Check] sequence mapping before')
    q_max, q_min, s_max, s_min, a_max, a_min = check_knowledge_tracking_data(seqs_lis, seqs_structured)
    base_idx = base_offset

    # assert a_min >= 0 , 'answer feature error'
    if answer_size == None:
        answer_size = (a_min, a_max)
    answer_offset = base_idx - answer_size[0]
    base_idx = answer_size[1] + answer_offset + 1 
    if format_type == 'separate':
        base_idx = base_offset

    # assert s_min >= 0 , 'skill feature error'
    if skill_size == None:
        skill_size = (s_min, s_max)
    skill_offset = base_idx - skill_size[0]
    base_idx = skill_size[1] + skill_offset + 1
    if format_type == 'separate':
        base_idx = base_offset

    # assert q_min >= 0 , 'question feature error'
    if question_size == None:
        question_size = (q_min, q_max)
    question_offset = base_idx - question_size[0]
    base_idx = question_size[1] + question_offset + 1

    seqs_format_lis = []
    for seq in seqs_lis:
        seq_format = []
        for s, q, a in seq:
            if seqs_structured:
                s_f = [s_ + skill_offset for s_ in s]
                q_f = [q_ + question_offset for q_ in q]
                a_f = [a_ + answer_offset for a_ in a]
            else:
                s_f = s + skill_offset
                q_f = q + question_offset
                a_f = a + answer_offset
            seq_format.append([s_f, q_f, a_f])
        seqs_format_lis.append(seq_format)

    print_debug(f'\n\n>>> [Completed] sequence mapping after')
    check_knowledge_tracking_data(seqs_format_lis, seqs_structured)
    a_range = tuple(x + answer_offset for x in answer_size)
    s_range = tuple(y + skill_offset for y in skill_size)
    q_range = tuple(z + question_offset for z in question_size)
    return seqs_format_lis, a_range, s_range, q_range

def split_dataset(seqs_lis: list, split_ratios = [0.7, 0.2, 0.1], split_names = None, random_seed=42) -> tuple[list, list]:
    '''
    Split the dataset by split_ratios
    '''
    import random
    random.seed(random_seed)
    # verify segmentation ratio
    assert abs(sum(split_ratios) - 1.0) <= 1e-6, "split_ratios must sum to 1"
    # shuffle the serial rank
    total_samples = len(seqs_lis)
    indices = list(range(total_samples))
    random.shuffle(indices)
    # calculate the number of samples for each section
    split_nums = [int(total_samples * ratio) for ratio in split_ratios]
    # add the difference from the last part to ensure that the total sum is totaled. samples
    split_nums[-1] = total_samples - sum(split_nums[:-1])
    # split data and indices
    split_data = []
    split_indices = []
    start = 0
    for count in split_nums:
        end = start + count
        part_indices = indices[start:end]
        split_indices.append(part_indices)
        split_data.append([seqs_lis[i] for i in part_indices])
        start = end
    # save to file
    if split_names:
        assert len(split_names) == len(split_ratios), "The number of filenames must match split_ratios"
        for path, data in zip(split_names, split_data):
            with open(path, 'w') as f:
                for seq in data:
                    line = ';'.join([','.join(map(str, triplet)) for triplet in seq])
                    f.write(line + '\n')
    # print debug information
    print_debug('\n\n>>>[Completed] data segmentation')
    for i, data in enumerate(split_data):
        print_debug(f'Split Data {i}:')
        print_debug(f'  Number of sequences: {len(data)}')
        # print_debug(f'  Sample: {data[0]}')

    return split_data, split_indices

def unpack_structured_data(structured_seqs_lis: list, to_get_graph=False, q_size=None, s_size=None):
    '''
    in structured data, one question can connect multiple skills, so additional methods for unpacking and compatibility graph processing need to be designed
    '''
    seqs_lis = []
    graph_obj = SimpleNamespace()
    G_q2s = None
    G_bi = None
    for seq in structured_seqs_lis:
        seq_lis = []
        for s_, q_, a_ in seq:
            seq_lis.append([s_[0], q_[0], a_[0]])
        seqs_lis.append(seq_lis)
    if to_get_graph:
        G_q2s = Q2SGraph(q_size, s_size).add_edges_from_structured_seqs(structured_seqs_lis)
        G_bi = BiGraph(q_size, s_size).add_edges_from_structured_seqs(structured_seqs_lis)
    graph_obj.G_q2s = G_q2s
    graph_obj.G_bi = G_bi
    return seqs_lis, graph_obj

def compute_q_difficulty(seqs_lis: list, q_size: int, default_difficulty=0.5) -> list[float]:
    '''
    compute the difficulty of all questions from the KT sequence [seqs_lis].
    Args:
        seqs_lis(list{S, L, 3}(int)): the id of (s, q, a) in KT seqs
        q_size(int): the number of questions
        default_difficulty(float): the default difficulty of the question, apply it if no data is available
    Returns:
        list{Q}(float): the difficulty of each question
    '''
    q_difficulty = [default_difficulty] * q_size
    q_num_cnt = [0] * q_size
    a_num_cnt = [0] * q_size
    for seq in seqs_lis:
        for s, q, a in seq:
            if a not in (0, 1):
                raise ValueError(f"a must be 0 or 1, but got {a} in question {q}")
            q_num_cnt[q] += 1
            a_num_cnt[q] += a
    for q in range(q_size):
        if q_num_cnt[q] == 0:
            continue
        q_difficulty[q] = 1 - a_num_cnt[q] / q_num_cnt[q]

    return q_difficulty

def compute_q2q_sim_matrix(seqs_lis: list[list[tuple[int, int, int]]], q_size: int) -> np.ndarray:
    '''
    Compute the question-question (q2q) similarity matrix using a LightGCN-style normalized co-occurrence adjacency.
    Args:
        seqs_lis (list{S, L, 3}(int)): The id of (s, q, a) in KT seqs, where only q is used
        q_size (int): Number of questions Q
    Returns:
        np.ndarray{Q, Q}(float32): Normalized q2q similarity matrix where
            S[i, j] = co_occurrence(i, j) / sqrt(freq(i) * freq(j)),
            and the diagonal is set to 1.0.
    '''
    q2q_count = np.zeros((q_size, q_size), dtype=np.float32)
    q_freq = np.zeros(q_size, dtype=np.int32)

    for seq in seqs_lis:
        # All questions can be counted at most once per sequence
        unique_qs = np.fromiter({q for _, q, _ in seq}, dtype=np.int64)
        if unique_qs.size == 0:
            continue
        q_freq[unique_qs] += 1
        q2q_count[np.ix_(unique_qs, unique_qs)] += 1.0
    # Avoid division by zero in normalization
    denom = np.sqrt(q_freq, dtype=np.float32)
    denom[denom == 0] = 1.0
    # Normalize: S[i, j] = count[i, j] / sqrt(freq[i] * freq[j])
    q2q_sim_matrix = q2q_count / (denom[:, None] * denom[None, :])
    np.fill_diagonal(q2q_sim_matrix, 1.0)
    # (Development) Print a rough distribution of similarity values for quick inspection
    bins = [0.0, 1e-8, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(q2q_sim_matrix, bins=bins)
    distribution = {f'[{bins[i]:.1f}, {bins[i+1]:.1f})': int(hist[i]) for i in range(len(hist))}
    print_debug(distribution)

    return q2q_sim_matrix
