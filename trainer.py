from torch.autograd import Variable
from Utils.utils import *
import torch
import torch.nn as nn


class Trainer():
    def __init__(self, model, alpha, optimizer, learning_rate, class_criterion, proj_criterion, epochs, batch_size,
                 results_file,
                 weight_dir='models',
                 to_cuda=False):
        self.model = model
        self.alpha = alpha
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.class_criterion = class_criterion
        self.proj_criterion = proj_criterion
        self.epochs = epochs
        self.batch_size = batch_size
        self.results_file = results_file
        self.weight_dir = weight_dir
        self.to_cuda = to_cuda

    def train(self, proj_X, proj_Y, class_X, class_Y):
        print("creating {}".format(self.weight_dir))
        os.makedirs(self.weight_dir, exist_ok=True)
        print("{} created".format(self.weight_dir))

        best_model_weight_file = self.weight_dir
        num_batches = int(len(class_X) / self.batch_size)
        best_cross_f1 = 0
        num_epochs = 0
        print("loss alpha is ", self.alpha)
        # results_file = open(self.results_file_name, "w+")
        old_file = None
        proj_X = np.array(proj_X)
        proj_Y = np.array(proj_Y)

        for i in range(self.epochs):
            idx = 0
            num_epochs += 1

            class_X, class_Y = shuffle_data(class_X, class_Y)
            for j in range(num_batches):
                # Get batch
                cx = class_X[idx:idx + self.batch_size]
                cy = class_Y[idx:idx + self.batch_size]
                idx += self.batch_size

                cx = [self.model.idx_vecs(s, self.model.sw2idx) for s in cx]

                x_batch, y_batch = sort_batch_by_sent_lens(cx, cy)
                x_batch, lens = prepare_batch(x_batch)

                self.optimizer.zero_grad()
                preds, x_projected, y_projected = self.model(x_batch, lens, proj_X, proj_Y)

                # calculate loss
                proj_loss = self.proj_criterion(x_projected, y_projected)
                if self.to_cuda:
                    labels = Variable(torch.LongTensor(y_batch)).cuda()
                else:
                    labels = Variable(torch.LongTensor(y_batch))
                classification_loss = self.class_criterion(preds, labels)
                loss = proj_loss * self.alpha + classification_loss * (1 - self.alpha)

                # Backprop
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    # check cosine distance between dev translation pairs
                    xdev = self.model.pdataset._Xdev
                    ydev = self.model.pdataset._ydev
                    xp, yp = self.model.project(xdev, ydev)
                    score = self.cos(xp, yp)

                    # check source dev f1
                    xdev = self.model.cdataset._Xdev
                    ydev = self.model.cdataset._ydev
                    xp, ydev_sorted = self.model.predict(xdev, ydev, src_lang=True)
                    xp = xp.cpu().data.numpy().argmax(1)

                    # xp = model.predict(xdev).data.numpy().argmax(1)

                    # macro f1
                    dev_f1 = macro_f1(ydev_sorted, xp)

                    # check cosine distance between source sentiment synonyms
                    p1 = self.model.project_one(self.model.src_syn1)
                    p2 = self.model.project_one(self.model.src_syn2)
                    syn_cos = self.cos(p1, p2)

                    # check cosine distance between source sentiment antonyms
                    p3 = self.model.project_one(self.model.src_syn1)
                    n1 = self.model.project_one(self.model.src_neg)
                    ant_cos = self.cos(p3, n1)

            if self.model.trg_data:
                with torch.no_grad():
                    # check target dev f1
                    crossx = self.model.trg_dataset._Xdev
                    crossy = self.model.trg_dataset._ydev
                    xp, crossy_sorted = self.model.predict(crossx, crossy, src_lang=False)
                    xp = xp.cpu().data.numpy().argmax(1)

                    # macro f1
                    cross_f1 = macro_f1(crossy_sorted, xp)

                if cross_f1 > best_cross_f1:
                    self.results_file.write("\n")
                    self.results_file.write("found new best cross_f1: {} in epoch {}\n".format(cross_f1, num_epochs))

                    self.results_file.write("True labels:\n")
                    self.results_file.write("{}\n".format(crossy_sorted))
                    self.results_file.write("Preds:\n")
                    self.results_file.write("{}\n".format(xp))

                    best_cross_f1 = cross_f1
                    best_model_weight_file = os.path.join(self.weight_dir,
                                                          '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}crossf1-{4}lr.pth'.format(
                                                              num_epochs,
                                                              self.batch_size,
                                                              self.alpha,
                                                              best_cross_f1,
                                                              '{0:.15f}'.format(self.learning_rate).rstrip('0').rstrip(
                                                                  '.')))
                    if old_file != None:
                        os.remove(old_file)
                    torch.save(self.model, best_model_weight_file)
                    old_file = best_model_weight_file

            with torch.no_grad():
                # check cosine distance between target sentiment synonyms
                cp1 = self.model.project_one(self.model.trg_syn1, src=False)
                cp2 = self.model.project_one(self.model.trg_syn2, src=False)
                cross_syn_cos = self.cos(cp1, cp2)

                # check cosine distance between target sentiment antonyms
                cp3 = self.model.project_one(self.model.trg_syn1, src=False)
                cn1 = self.model.project_one(self.model.trg_neg, src=False)
                cross_ant_cos = self.cos(cp3, cn1)
                self.results_file.write(
                    'epoch {0} loss: {1:.6f}  trans: {2:.3f}  src_f1: {3:.3f}  trg_f1: {4:.3f}  src_syn: {5:.3f}  src_ant: {6:.3f}  cross_syn: {7:.3f}  cross_ant: {8:.3f}\n'.format(
                        num_epochs, loss.item(), score.item(), dev_f1,
                        cross_f1, syn_cos.item(), ant_cos.item(),
                        cross_syn_cos.item(), cross_ant_cos.item()))

                sys.stdout.write(
                    '\r epoch {0} loss: {1:.6f}  trans: {2:.3f}  src_f1: {3:.3f}  trg_f1: {4:.3f}  src_syn: {5:.3f}  src_ant: {6:.3f}  cross_syn: {7:.3f}  cross_ant: {8:.3f}\n'.format(
                        num_epochs, loss.item(), score.item(), dev_f1,
                        cross_f1, syn_cos.item(), ant_cos.item(),
                        cross_syn_cos.item(), cross_ant_cos.item()))
                sys.stdout.flush()
                self.model.history['loss'].append(loss.item())
                self.model.history['dev_cosine'].append(score.item())
                self.model.history['dev_f1'].append(dev_f1)
                self.model.history['cross_f1'].append(cross_f1)
                self.model.history['syn_cos'].append(syn_cos.item())
                self.model.history['ant_cos'].append(ant_cos.item())
                self.model.history['cross_syn'].append(cross_syn_cos.item())
                self.model.history['cross_ant'].append(cross_ant_cos.item())

        return best_model_weight_file

    def cos(self, x, y):
        """
        This returns the mean cosine similarity between two sets of vectors.
        """
        c = nn.CosineSimilarity()
        return c(x, y).mean()
