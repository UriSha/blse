from torch.autograd import Variable
from Utils.utils import *
from abstract_blse import Abstract_Blse
from attn_for_rnn import Attn_for_rnn


class Rnn_Attn_BLSE(Abstract_Blse):
    def __init__(self, src_vecs, trg_vecs, pdataset,
                 cdataset, trg_dataset,
                 projection_loss='mse',
                 output_dim=4,
                 hidden_size=800,
                 n_layers=2,
                 batch_size=16,
                 to_cuda=True,
                 src_syn1=None, src_syn2=None, src_neg=None,
                 trg_syn1=None, trg_syn2=None, trg_neg=None,
                 ):
        super(Rnn_Attn_BLSE, self).__init__(src_vecs, trg_vecs, pdataset,
                                            cdataset, trg_dataset,
                                            projection_loss,
                                            output_dim,
                                            hidden_size,
                                            n_layers,
                                            batch_size,
                                            to_cuda,
                                            src_syn1, src_syn2, src_neg,
                                            trg_syn1, trg_syn2, trg_neg,
                                            )
        self.padding_idx = 0
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.embedding_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.attn = Attn_for_rnn(self.hidden_size * 2, batch_size, to_cuda)
        self.cls = nn.Linear(self.hidden_size * 2, output_dim)

    def forward(self, input_text, text_lens, proj_X, proj_Y):
        x_proj, y_proj = self.project(proj_X, proj_Y)

        embedded_words = self.semb(Variable(input_text))
        projected_embedded_words = self.m(embedded_words)

        rnn_outputs = self.forward_rnn(projected_embedded_words, text_lens)

        attn_weighted_ouputs = self.forward_attn(rnn_outputs, text_lens)

        outputs = self.cls(attn_weighted_ouputs)

        return outputs, x_proj, y_proj

    def forward_without_proj(self, input_text, text_lens, src_lang):
        if src_lang:
            embedded_words = self.semb(Variable(input_text))
            projected_embedded_words = self.m(embedded_words)
        else:
            embedded_words = self.temb(Variable(input_text))
            projected_embedded_words = self.mp(embedded_words)

        rnn_outputs = self.forward_rnn(projected_embedded_words, text_lens)

        attn_weighted_ouputs = self.forward_attn(rnn_outputs, text_lens)

        outputs = self.cls(attn_weighted_ouputs)

        return outputs

    def init_hidden(self):
        if self.to_cuda:
            return (torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda(),
                    torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        return (torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size),
                torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size))


    def forward_attn(self, rnn_outputs, text_lens):
        energies = self.attn(rnn_outputs, text_lens)
        energies = energies.unsqueeze(1)
        attn_weighted_ouputs = torch.bmm(energies, rnn_outputs)
        attn_weighted_ouputs = attn_weighted_ouputs.squeeze(1)
        return attn_weighted_ouputs

    def forward_rnn(self, projected_embedded_words, text_lens):
        embedded_words = torch.nn.utils.rnn.pack_padded_sequence(projected_embedded_words, text_lens, batch_first=True)
        last_hidden = self.init_hidden()
        rnn_outputs, hidden = self.lstm(embedded_words, last_hidden)
        rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs