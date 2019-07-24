from torch.autograd import Variable
from bi_lstm import BiLstm
from abstract_blse import Abstract_Blse


class RNN_BLSE(Abstract_Blse):
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
        super(RNN_BLSE, self).__init__(src_vecs, trg_vecs, pdataset,
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
        self.lstm = BiLstm(hidden_size, src_vecs.vector_size, output_dim, n_layers,
                          self.batch_size, self.padding_idx, to_cuda)

    def forward(self, input_text, text_lens, proj_X, proj_Y):
        embedded_words = self.semb(Variable(input_text))
        projected_embedded_words = self.m(embedded_words)

        lstm_preds = self.lstm(projected_embedded_words, text_lens)

        x_proj, y_proj = self.project(proj_X, proj_Y)

        return lstm_preds, x_proj, y_proj

    def forward_without_proj(self, input_text, text_lens, src_lang):
        if src_lang:
            embedded_words = self.semb(Variable(input_text))
            projected_embedded_words = self.m(embedded_words)
        else:

            embedded_words = self.temb(Variable(input_text))
            projected_embedded_words = self.mp(embedded_words)

        lstm_preds = self.lstm(projected_embedded_words, text_lens)

        return lstm_preds
