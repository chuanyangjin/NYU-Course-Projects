import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu


## Dummy functions defined to use the same function run_epoch() during eval
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        """
        Parameters:
            d_model: dimension of input embeddings
            dropout (you don't need to use this, already in forward()): nn.Dropout
            max_len: maximum length of input
        Set variable value:
            pe: torch.tensor of size (max_len, d_model)
            
        """

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        
 
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)   


    
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    
    "Implement beam search decoding with 'beam_size' width"
    
    """
    Parameters:
        model: instance of EncoderDecoder()
        src: input tensor of size (1, inp_seq_len)
        src_mask: input mask of size (1, 1, inp_seq_len)
        max_len: int, maximum output length
        start_symbol: int, index of start symbol
        beam_size: int, width of beam in beam search
        end_idx: int, index of end of sentence symbol
        
    Return:
        ys: output tensor of size (1, out_seq_len)
        
    """
    
    
    # Output of encoder
    memory = model.encode(src, src_mask)
    
    # Initialize start of sequence with start symbol
    ys = torch.zeros(1,1).fill_(start_symbol).type_as(src.data).cuda()
    
    # We will use this to store log_prob of the sequences so far
    scores = torch.Tensor([0.]).cuda()
    
    for i in range(max_len - 1):
        
        # Compute the output using the decoder
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        
        # Compute the log_prob for the next token
        prob = model.generator(out[:,-1])
        vocab_size = prob.shape[-1]
        
        # For sequences which have finished, set log_prob to zero
        prob[ys[:,-1] == end_idx,:] = 0
        


        
        # Combine the log_prob of next token with our scores so far
        log_prob = torch.nn.functional.softmax(prob, dim = -1)       # (beam_size, vocab_size)

        # Obtain new scores
        new_scores = (scores.unsqueeze(1) + log_prob).view(-1)       # (beam_size * vocab_size,)

        # Select topk scores and indices
        scores, topk_indices = torch.topk(new_scores, k = beam_size, dim = -1)              # (beam_size,)
        ys_indices, token_indices = topk_indices // vocab_size, topk_indices % vocab_size

        # Construct variable ys
        ys = torch.cat([ys[ys_indices], token_indices.unsqueeze(-1)], dim = -1)               # (beam_size, i+2)
        

        
        # If all beams are finished, exit
        if (ys[:, -1]==end_idx).sum() == beam_size:
            break

        # Encoder output expansion from the second time step to the beam size
        # This is needed because for the first time step we only have the start token as ys
        if i==0:
            memory = memory.expand(beam_size, *memory.shape[1:])
            src_mask = src_mask.expand(beam_size, *src_mask.shape[1:])
    
    # convert the top scored sequence to a list of text tokens
    ys, _ = max(zip(ys, scores), key=lambda x: x[1])
    ys = ys.unsqueeze(0)
    
    return ys
    

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def remove_start_end_tokens(sent):
    
    if sent.startswith('<s>'):
        sent = sent[3:]
        
    if sent.endswith('</s>'):
        sent = sent[:-4]
                
    return sent


def compute_corpus_level_bleu(refs, hyps):
    
    refs = [remove_start_end_tokens(sent) for sent in refs]
    hyps = [remove_start_end_tokens(sent) for sent in hyps]
    
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    
    return bleu.score

