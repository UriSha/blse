from torch.autograd import Variable
from Utils.WordVecs import *
from Utils.utils import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score, confusion_matrix
import abc


class Abstract_Blse(nn.Module):
    """
    Bilingual Sentiment Embeddings

    Parameters:

        src_vecs: WordVecs instance with the embeddings from the source language
        trg_vecs: WordVecs instance with the embeddings from the target language
        pdataset: Projection_Dataset from source to target language
        cdataset: Source sentiment dataset
        projection_loss: the distance metric to use for the projection loss
                         can be either mse (default) or cosine
        output_dim: the number of class labels to predict (default: 4)

    Optional:

        src_syn1: a list of positive sentiment words in the source language
        src_syn2: a second list of positive sentiment words in the
                  source language. This must be of the same length as
                  src_syn1 and should not have overlapping vocabulary
        src_neg : a list of negative sentiment words in the source language
                  this must be of the same length as src_syn1
        trg_syn1: a list of positive sentiment words in the target language
        trg_syn2: a second list of positive sentiment words in the
                  target language. This must be of the same length as
                  trg_syn1 and should not have overlapping vocabulary
        trg_neg : a list of negative sentiment words in the target language
                  this must be of the same length as trg_syn1



    """

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
        super(Abstract_Blse, self).__init__()

        # Embedding matrices
        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.sw2idx = src_vecs._w2idx
        self.sidx2w = src_vecs._idx2w
        self.temb = nn.Embedding(trg_vecs.vocab_length, trg_vecs.vector_size)
        self.temb.weight.data.copy_(torch.from_numpy(trg_vecs._matrix))
        self.tw2idx = trg_vecs._w2idx
        self.tidx2w = trg_vecs._idx2w

        # Projection vectors
        self.embedding_size = src_vecs.vector_size
        self.m = nn.Linear(src_vecs.vector_size,
                           src_vecs.vector_size,
                           bias=False)
        self.mp = nn.Linear(trg_vecs.vector_size,
                            trg_vecs.vector_size,
                            bias=False)

        self.to_cuda = to_cuda
        self.batch_size = batch_size

        # Datasets
        self.pdataset = pdataset
        self.cdataset = cdataset
        self.trg_dataset = trg_dataset
        self.src_syn1 = src_syn1
        self.src_syn2 = src_syn2
        self.src_neg = src_neg
        self.trg_syn1 = trg_syn1
        self.trg_syn2 = trg_syn2
        self.trg_neg = trg_neg

        # Trg Data
        if (self.trg_dataset != None and
                    self.trg_syn1 != None and
                    self.trg_syn2 != None and
                    self.trg_neg != None):
            self.trg_data = True
        else:
            self.trg_data = False

        # History
        self.history = {'loss': [], 'dev_cosine': [], 'dev_f1': [], 'cross_f1': [],
                        'syn_cos': [], 'ant_cos': [], 'cross_syn': [], 'cross_ant': []}

        # Do not update original embedding spaces
        self.semb.weight.requires_grad = False
        self.temb.weight.requires_grad = False

    @abc.abstractmethod
    def forward(self, input_text, text_lens, proj_X, proj_Y):
        pass

    @abc.abstractmethod
    def forward_without_proj(self, input_text, text_lens, src_lang):
        pass

    def project(self, X, Y):
        """
        Project X and Y into shared space.
        X is a list of source words from the projection lexicon,
        and Y is the list of single word translations.
        """
        if self.to_cuda:
            x_lookup = torch.cuda.LongTensor(np.array([self.sw2idx[w] for w in X]))
            y_lookup = torch.cuda.LongTensor(np.array([self.tw2idx[w] for w in Y]))
        else:
            x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in X]))
            y_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in Y]))
        x_embedd = self.semb(Variable(x_lookup))
        y_embedd = self.temb(Variable(y_lookup))
        x_proj = self.m(x_embedd)
        y_proj = self.mp(y_embedd)
        return x_proj, y_proj

    def project_one(self, x, src=True):
        """
        Project only a single list of words to the shared space.
        """
        if src:
            if self.to_cuda:
                x_lookup = torch.cuda.LongTensor(np.array([self.sw2idx[w] for w in x]))
            else:
                x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in x]))
            x_embedd = self.semb(Variable(x_lookup))
            x_proj = self.m(x_embedd)
        else:
            if self.to_cuda:
                x_lookup = torch.cuda.LongTensor(np.array([self.tw2idx[w] for w in x]))
            else:
                x_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in x]))
            x_embedd = self.temb(Variable(x_lookup))
            x_proj = self.mp(x_embedd)
        return x_proj

    # def projection_loss(self, x, y):
    #     """
    #     Find the loss between the two projected sets of translations.
    #     The loss is the proj_criterion.
    #     """
    #
    #     x_proj, y_proj = self.project(x, y)
    #
    #     # distance-based loss (cosine, mse)
    #     loss = self.proj_criterion(x_proj, y_proj)
    #
    #     return loss

    def idx_vecs(self, sentence, model):
        """
        Converts a tokenized sentence to a vector
        of word indices based on the model.
        """
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except KeyError:
                # continue
                sent.append(0)
        return torch.LongTensor(np.array(sent))

    def lookup(self, X, model):
        """
        Converts a batch of tokenized sentences
        to a matrix of word indices from model.
        """
        return [self.idx_vecs(s, model) for s in X]

    def predict(self, X, Y, src_lang=True):

        num_batches = int(len(X) / self.batch_size)
        # if not src_lang:
        #     print()
        #     print("in predict")
        #     print("num_batches: {}".format(num_batches))
        idx = 0

        preds = []
        sorted_y = []

        for j in range(num_batches):
            # preds, sorted_y ,idx= self.pred_batch(X, Y, src_lang, preds, sorted_y, idx, self.batch_size)
            cx = X[idx:idx + self.batch_size]
            cy = Y[idx:idx + self.batch_size]
            idx += self.batch_size

            if src_lang:
                cx = [self.idx_vecs(s, self.sw2idx) for s in cx]
            else:
                cx = [self.idx_vecs(s, self.tw2idx) for s in cx]

            x_batch, y_batch = sort_batch_by_sent_lens(cx, cy)
            sorted_y.extend(cy)
            batch, lens = prepare_batch(x_batch)

            batch_preds = self.forward_without_proj(batch, lens, src_lang=src_lang)
            preds.extend(batch_preds)
        #
        # if not src_lang:
        #     print("len(preds): {}".format(len(preds)))
        return torch.stack(preds), np.array(sorted_y)

    def pred_batch(self, X, Y, src_lang, preds, sorted_y, idx, batch_size):
        cx = X[idx:idx + batch_size]
        cy = Y[idx:idx + batch_size]
        idx += self.batch_size

        if src_lang:
            cx = [self.idx_vecs(s, self.sw2idx) for s in cx]
        else:
            cx = [self.idx_vecs(s, self.tw2idx) for s in cx]

        x_batch, y_batch = sort_batch_by_sent_lens(cx, cy)
        sorted_y.extend(cy)
        batch, lens = prepare_batch(x_batch)

        batch_preds = self.forward_without_proj(batch, lens, src_lang=src_lang)

        preds.extend(batch_preds)

        return preds, sorted_y, idx

    # def plot(self, title=None, outfile=None):
    #     """
    #     Plots the progression of the model. If outfile != None,
    #     the plot is saved to outfile.
    #     """
    #
    #     h = self.history
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot(h['dev_cosine'], label='translation_cosine')
    #     ax.plot(h['dev_f1'], label='source_f1', linestyle=':')
    #     ax.plot(h['cross_f1'], label='target_f1', linestyle=':')
    #     ax.plot(h['syn_cos'], label='source_synonyms', linestyle='--')
    #     ax.plot(h['ant_cos'], label='source_antonyms', linestyle='-.')
    #     ax.plot(h['cross_syn'], label='target_synonyms', linestyle='--')
    #     ax.plot(h['cross_ant'], label='target_antonyms', linestyle='-.')
    #     ax.set_ylim(-.5, 1.4)
    #     ax.legend(
    #         loc='upper center', bbox_to_anchor=(.5, 1.05),
    #         ncol=3, fancybox=True, shadow=True)
    #     if title:
    #         ax.title(title)
    #     if outfile:
    #         plt.savefig(outfile)
    #     else:
    #         plt.show()

    def confusion_matrix(self, X, Y, src=True, results_file=None):
        """
        Prints a confusion matrix for the model
        """
        pred, cy = self.predict(X, Y, src_lang=src)
        pred = pred.cpu().data.numpy().argmax(1)
        cm = confusion_matrix(cy, pred, sorted(set(cy)))
        print(cm)
        if results_file:
            results_file.write("{}".format(cm))

    def evaluate(self, X, Y, results_file=None, src=True, preds_outfile_name=None):
        """
        Prints the accuracy, macro precision, macro recall,
        and macro F1 of the model on X. If outfile != None,
        the predictions are written to outfile.
        """
        pred, cy = self.predict(X, Y, src_lang=src)
        pred = pred.cpu().data.numpy().argmax(1)
        acc = accuracy_score(cy, pred)
        prec = per_class_prec(cy, pred).mean()
        rec = per_class_rec(cy, pred).mean()
        f1 = macro_f1(cy, pred)
        if preds_outfile_name:
            with open(preds_outfile_name, 'w') as out:
                out.write(' Prediction |  True Label \n')
                for p, y in zip(pred, cy):
                    out.write('     {}      |      {}     \n'.format(p, y))

        if results_file:
            print('Test Set:')
            print(
                'acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}'.format(acc, prec, rec, f1))
            results_file.write('\n')
            results_file.write('Test Set:\n')
            results_file.write(
                'acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}\n\n'.format(acc, prec, rec,
                                                                                                       f1))
            results_file.write('test labels:\n')
            results_file.write('{}\n'.format(cy))
            results_file.write('test preds:\n')
            results_file.write('{}\n\n'.format(pred))
        return acc, prec, rec, f1

    def eval_of_dev(self, outfile=None):
        self.evaluate(self.trg_dataset._Xdev, self.trg_dataset._ydev, src=False, preds_outfile_name=outfile)

    def __str__(self):
        return "better_blse"
