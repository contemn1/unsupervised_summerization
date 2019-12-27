import math
import os
import re
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from config import init_argument_parser
from dataset import CNNDailyMailDataset
from io_util import output_iterator
from preprocessor import Preprocessor


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
        1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


@torch.no_grad()
def sample_sequence(model, length, context, attention_mask, method, num_samples=1, temperature: float = 1.0, top_k=0,
                    top_p=0.0,
                    repetition_penalty=1.0, min_seq_length=25, eos_idx=50256
                    ) -> torch.LongTensor:
    context = context.repeat(num_samples, 1)
    previous_ids = context[:, :-1]
    batch_size = context.size()[0]
    input_ids = context
    past = None
    result = None
    previous_embeddings = None
    data_type = next(model.parameters()).dtype
    penalty_mask = torch.zeros(
        batch_size, 50257, device=context.device, dtype=data_type)
    penalty_value = torch.zeros(batch_size, 1, device=context.device, dtype=data_type).fill_(
        -math.log(repetition_penalty))
    for idx in trange(length):
        inputs = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'past': past}
        outputs = model(
            **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
        past = outputs[1]

        if method == "extractive":
            output_embeddings = outputs[2][-1]
            current_embeddings = output_embeddings[:, -1, :]
            if previous_embeddings is None:
                previous_embeddings = output_embeddings[:, :-1, :]

            attention_logits = torch.bmm(previous_embeddings, current_embeddings.unsqueeze(2)).squeeze(2)
            attention_logits = attention_logits / current_embeddings.size(1)
            attention_scores = torch.nn.Softmax(dim=-1)(attention_logits)
            next_token_scores = torch.zeros(batch_size, 50257, device=input_ids.device,
                                            dtype=data_type).scatter_add_(1, previous_ids, attention_scores)
            next_token_scores = next_token_scores / torch.sum(next_token_scores, dim=1, keepdim=True)
            next_token_logits = torch.log(next_token_scores)

            previous_embeddings = torch.cat((previous_embeddings, current_embeddings.unsqueeze(1)), dim=1)
            previous_ids = input_ids if idx == 0 else torch.cat((previous_ids, input_ids), dim=1)
        else:
            next_token_logits = outputs[0][:, -1, :]
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.)

        if idx < min_seq_length:
            next_token_logits[:, eos_idx] = -1e4
        # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
        if result is not None and repetition_penalty != 1.0:
            penalty_mask.scatter_add_(1, input_ids, penalty_value)
            negative_indices = next_token_logits < 0
            actual_mask = penalty_mask.clone()
            actual_mask[negative_indices] = - actual_mask[negative_indices]
            next_token_logits = next_token_logits * torch.exp(actual_mask)
        filtered_logits = top_k_top_p_filtering(
            next_token_logits, top_k=top_k, top_p=top_p, filter_value=-1e4)
        filtered_logits = filtered_logits.float()
        if temperature == 0:  # greedy sampling:
            next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
        else:
            filtered_logits[filtered_logits == float('inf')] = 1e8
            distribution = Categorical(logits=filtered_logits)
            next_token = distribution.sample().unsqueeze(1)
        result = torch.cat((result, next_token),
                           dim=1) if result is not None else next_token
        attention_mask = torch.ones_like(next_token)
        input_ids = next_token
    return result


def decode_id_array(id_list: List[np.ndarray]) -> List[str]:
    flattened_id_list = np.vstack(id_list)
    return [gpt_tokenizer.decode(arr.tolist()) for arr in flattened_id_list]


if __name__ == '__main__':
    args = init_argument_parser().parse_args()
    test_dir = args.input_dir
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)  # type: GPT2Tokenizer
    output_hidden = False if args.method == "abstractive" else True
    gpt_model = GPT2LMHeadModel.from_pretrained(
        args.model_name, output_hidden_states=output_hidden)  # type: GPT2LMHeadModel
    use_cuda = torch.cuda.is_available()

    cnn_preprocessor = Preprocessor(test_dir, gpt_tokenizer)
    tokenize = False
    batch_size = args.batch_size
    content_list = cnn_preprocessor.get_document_summary(tokenize)
    summary_list = [" ".join(tup[1]) for tup in content_list]
    cnn_dataset = CNNDailyMailDataset(
        content_list, gpt_tokenizer, 512, cnn_preprocessor.tokenized)
    cnn_dataloader = DataLoader(cnn_dataset, shuffle=False, num_workers=8,
                                batch_size=batch_size,
                                collate_fn=cnn_dataset.collate,
                                pin_memory=torch.cuda.is_available())

    gpt_model.eval()

    if args.half_precision:
        gpt_model.half()

    if use_cuda:
        gpt_model = gpt_model.to("cuda")

    if args.use_multiple_gpu:
        gpt_model = DataParallel(gpt_model)

    sample_id_list = []
    for ele in cnn_dataloader:
        input_ids, attention_mask, output_ids = ele
        if use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            result = sample_sequence(gpt_model, 100, input_ids, attention_mask, args.method,
                                     repetition_penalty=1.2, top_p=0.9, temperature=0.9,
                                     eos_idx=gpt_tokenizer.eos_token_id)
            sample_id_list.append(result)

    sample_id_list = [ele.cpu().numpy() for ele in sample_id_list]
    sample_list = decode_id_array(sample_id_list)
    sample_list = [re.sub("\n+", " ", ele) for ele in sample_list]

    output_iterator(os.path.join(args.output_dir,
                                 "generated_summaries.txt"), sample_list)

    output_iterator(os.path.join(args.output_dir,
                                 "actual_summaries.txt"), summary_list)
