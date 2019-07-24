from Utils.utils import *


class BiLstm(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers, batch_size, padding_idx, to_cuda):
        super(BiLstm, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.padding_idx = padding_idx
        self.to_cuda = to_cuda

        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, embedded_words, text_lens):

        embedded_words = torch.nn.utils.rnn.pack_padded_sequence(embedded_words, text_lens, batch_first=True)

        last_hidden = self.init_hidden()

        rnn_outputs, hidden = self.lstm(embedded_words, last_hidden)

        rnn_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)

        rnn_outputs = torch.sum(rnn_outputs, -2)

        output = self.out(rnn_outputs)

        return output

    def init_hidden(self):
        return (torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
