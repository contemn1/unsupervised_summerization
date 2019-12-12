from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, GPT2Tokenizer

from preprocessor import Preprocessor


class CNNDailyMailDataset(Dataset):
    def __init__(self, content_list: List[Tuple[List[str], List[str]]], tokenizer: PreTrainedTokenizer,
                 max_seq_length: int = 512, tokenized: bool = False):
        self.content_list = content_list
        self.tokenizer = tokenizer
        self.tokenized = tokenized
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.content_list)

    def __getitem__(self, idx):
        document_list, summary_list = self.content_list[idx]
        document_id_list = [idx for doc_sent in document_list for idx in self.__encode_sentence(doc_sent)]
        document_id_list = document_id_list[:self.max_seq_length]
        tl_dr_ids = self.tokenizer.encode("TL;DR:")
        document_id_list.extend(tl_dr_ids)
        summary_id_list = [idx for sum_sent in summary_list for idx in self.__encode_sentence(sum_sent)]
        return document_id_list, summary_id_list

    def __encode_sentence(self, sent):
        return self.tokenizer.encode(sent) if not self.tokenized else self.tokenizer.convert_tokens_to_ids(sent)

    def collate(self, batch_data):
        document_ids, summary_ids = zip(*batch_data)
        document_ids = list(document_ids)

        summary_ids = list(summary_ids)
        padded_doc_ids, doc_attention_mask = pad_sequence(document_ids, 0)
        padded_sum_ids, _ = pad_sequence(summary_ids, 0)
        return torch.from_numpy(padded_doc_ids), torch.from_numpy(doc_attention_mask),\
               torch.from_numpy(padded_sum_ids)


def pad_sequence(seq_list, padding_index, output_attention_mask=True):

    seq_length_list = [len(ele) for ele in seq_list]
    max_seq_length = max(seq_length_list)
    padding_matrix = np.full((len(seq_list), max_seq_length), fill_value=padding_index, dtype=np.int64)
    if output_attention_mask:
        mask_matrix = np.zeros((len(seq_list), max_seq_length), dtype=np.int64)
    for idx, seq in enumerate(seq_list):
        padding_matrix[idx][:seq_length_list[idx]] = np.array(seq, dtype=np.int64)
        if output_attention_mask:
            mask_matrix[idx][:seq_length_list[idx]] = 1
    return padding_matrix, mask_matrix
