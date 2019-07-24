import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Attn_for_rnn(nn.Module):
    def __init__(self, hidden_size, batch_size, cuda):
        super(Attn_for_rnn, self).__init__()

        self.hidden_size = hidden_size
        self.lin = nn.Linear(self.hidden_size, self.hidden_size)

        self.weight_vec = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        torch.nn.init.xavier_normal_(self.weight_vec)

        self.batch_size = batch_size
        self.to_cuda = cuda

    def forward(self, rnn_outputs, questions_lens):#, output_log=False):
        max_question_len = max(questions_lens)

        rnn_outputs = rnn_outputs.contiguous().view(-1, self.hidden_size ) # *for bidirectional

        attn_energies = self.score(rnn_outputs)

        if self.to_cuda:
            attn_energies = attn_energies.cuda()

        # attn_energies = attn_energies.view(self.batch_size, max_question_len)
        attn_energies = attn_energies.view(-1, max_question_len)

        masks = [self.masker(qlen, max_question_len) for qlen in questions_lens]
        masks = torch.stack(masks)

        if self.to_cuda:
            masks = masks.cuda()

        attn_energies.masked_fill_(masks, float('-inf'))

        res = F.softmax(attn_energies, dim=1)
        return res

    def score(self, word_embed):

        energy = self.lin(word_embed)

        energy = torch.matmul(energy, self.weight_vec)

        return energy


    def masker(self, qlen, max_question_len):
        if qlen == max_question_len:
            if self.to_cuda:
                return torch.zeros(max_question_len).byte().cuda()
            return torch.zeros(max_question_len).byte()
        if self.to_cuda:
            return torch.zeros(max_question_len).scatter_(0, torch.LongTensor(list(range(qlen, max_question_len))), 1.0).byte().cuda()
        return torch.zeros(max_question_len).scatter_(0, torch.LongTensor(list(range(qlen, max_question_len))), 1.0).byte()