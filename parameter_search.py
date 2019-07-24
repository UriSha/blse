from Utils.Datasets import *
from Utils.WordVecs import *
from Utils.utils import *

from rnn_blse import RNN_BLSE
from trainer import Trainer
from rnn_attn_blse import Rnn_Attn_BLSE


def train_model(model, dataset, cross_dataset,
                src_vecs, trg_vecs, synonyms1, synonyms2,
                neg, cross_syn1, cross_syn2, cross_neg,
                pdataset, weight_dir, proj_loss, alpha, learning_rate, batch_size, output_dim, b, params):
    results_file_name = "results/report_{}_{}_proj_loss_{}_alpha-{}_batch_size-{}_epochs-{}_lr-{}.txt".format(model,
                                                                                                              b,
                                                                                                              proj_loss,
                                                                                                              alpha,
                                                                                                              batch_size,
                                                                                                              params.epochs,
                                                                                                              '{0:.15f}'.format(
                                                                                                                  learning_rate).rstrip(
                                                                                                                  '0').rstrip(
                                                                                                                  '.'))

    # import datasets (representation will depend on final classifier)
    print()
    print('training model')
    print('Parameters:')
    print('model:     {0}'.format(model))
    print('binary:     {0}'.format(b))
    print('epochs:      {0}'.format(params.epochs))
    print('alpha (projection loss coef):      {0}'.format(alpha))
    print('batchsize:  {0}'.format(batch_size))
    print('learning rate:  {0}'.format(learning_rate))
    print('weight_dir:  {0}'.format(weight_dir))
    print('results_file_name:  {0}'.format(results_file_name))
    print()

    # Set up model
    if model == 'rnn_blse':
        model = RNN_BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                         projection_loss=proj_loss,
                         output_dim=output_dim,
                         to_cuda=params.to_cuda,
                         batch_size=batch_size,
                         src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                         trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg,
                         )
    elif model == 'rnn_attn_blse':
        model = Rnn_Attn_BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                              projection_loss=proj_loss,
                              output_dim=output_dim,
                              to_cuda=params.to_cuda,
                              batch_size=batch_size,
                              src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                              trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg,
                              )

    if torch.cuda.is_available():
        print("cuda is available")
        model.cuda()
    else:
        print("cuda is not available")

    # Loss Functions
    class_criterion = nn.CrossEntropyLoss()
    proj_criterion = nn.MSELoss()

    if proj_loss == 'mse':
        proj_criterion = nn.MSELoss()
    elif proj_loss == 'cosine':
        proj_criterion = cosine_loss
    else:
        print("no projection criterion supported: {}".format(proj_loss))
        exit(1)
    #
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    # Fit model
    results_file = open(results_file_name, "w+")
    trainer = Trainer(model, alpha, optim, learning_rate, class_criterion, proj_criterion, params.epochs,
                      batch_size, results_file, weight_dir, params.to_cuda)

    best_model_file_path = trainer.train(pdataset._Xtrain, pdataset._ytrain,
                                         dataset._Xtrain, dataset._ytrain)

    # Get best dev f1 and weights
    print("looking in dir: {}".format(weight_dir))
    best_f1, best_params = get_best_model_params(best_model_file_path)
    # best_f1, best_params, best_weights_path = get_best_run(weight_dir)
    best_model = torch.load(best_model_file_path)
    state_dict = best_model.state_dict()
    model.load_state_dict(state_dict)

    print()
    print('Dev set')
    print('best dev f1: {0:.3f}'.format(best_f1))
    print('parameters: epochs {0} batch size {1} alpha {2} learning rate {3}'.format(*best_params))

    results_file.write('\n')
    results_file.write('Dev set\n')
    results_file.write('best dev f1: {0:.3f}\n'.format(best_f1))
    results_file.write('parameters: epochs {0} batch size {1} alpha {2}\n'.format(*best_params))

    # Evaluate on test set
    model.eval()

    acc, prec, rec, f1 = model.evaluate(cross_dataset._Xtest, cross_dataset._ytest, results_file=results_file,
                                        src=False)
    model.confusion_matrix(cross_dataset._Xtest, cross_dataset._ytest, src=False, results_file=results_file)

    results_file.close()

    return best_model_file_path, acc, prec, rec, f1, results_file_name


def train_model_with_different_params(params):
    # Train a certain model (rnn_blse / rnn_attn_blse) for a certain target language
    #  for different combinations of hyper parameters

    if params.model not in ['rnn_attn_blse', 'rnn_blse']:
        print("no such model: {}".format(params.model))
        exit(1)

    # If there's no savedir, create it
    os.makedirs(params.savedir, exist_ok=True)

    if params.binary:
        output_dim = 2
        b = 'bi'
    else:
        output_dim = 4
        b = '4cls'

    weight_dir = "{}/{}/{}-{}-{}".format(params.savedir, params.model, params.dataset, params.target_lang,
                                         b)
    best_params_file_name = "results/best_params_report_{}_{}_{}.txt".format(params.model, params.target_lang, b)
    best_params_file = open(best_params_file_name, "w+")

    best_params_file.write("Start parameter search:\n")
    best_params_file.write("Model: {}\n".format(params.model))
    best_params_file.write("is_binary: {}\n".format(params.binary))
    best_params_file.write("target_lang: {}\n".format(params.target_lang))

    best_f1 = 0.0
    best_params = None
    old_file_name = None
    old_results_file_name = None
    rest_of_scores = []

    print('importing datasets')

    dataset = General_Dataset(os.path.join('datasets', params.source_lang, params.dataset),
                              None,
                              binary=params.binary,
                              rep=words,
                              one_hot=False)

    cross_dataset = General_Dataset(os.path.join('datasets', params.target_lang, params.dataset),
                                    None,
                                    binary=params.binary,
                                    rep=words,
                                    one_hot=False)

    # Import monolingual vectors
    print('importing word embeddings')
    trg_vecs_file_path = "embeddings/original/sg-300-{}.txt".format(params.target_lang)
    print("trg_vecs_file_path: {}".format(trg_vecs_file_path))
    src_vecs = WordVecs(params.src_vecs)
    trg_vecs = WordVecs(trg_vecs_file_path)

    # Get sentiment synonyms and antonyms to check how they move during training
    synonyms1, synonyms2, neg = get_syn_ant(params.source_lang, src_vecs)
    cross_syn1, cross_syn2, cross_neg = get_syn_ant(params.target_lang, trg_vecs)

    # Import translation pairs
    translation_file_path = "lexicons/{}/en-{}.txt".format(params.trans, params.target_lang)
    print("translation_file_path: {}".format(translation_file_path))
    pdataset = ProjectionDataset(translation_file_path, src_vecs, trg_vecs)

    for proj_loss in params.proj_losses:
        for alpha in params.alphas:
            for learning_rate in params.learning_rates:
                for batch_size in params.batch_sizes:
                    best_model_file_path, acc, prec, rec, f1, results_file_name = train_model(params.model, dataset,
                                                                                              cross_dataset,
                                                                                              src_vecs, trg_vecs,
                                                                                              synonyms1, synonyms2,
                                                                                              neg, cross_syn1,
                                                                                              cross_syn2, cross_neg,
                                                                                              pdataset,
                                                                                              weight_dir,
                                                                                              proj_loss, alpha,
                                                                                              learning_rate,
                                                                                              batch_size, output_dim, b,
                                                                                              params)

                    if f1 > best_f1:
                        print()
                        print("Found new set of best hyper params:")
                        print("f1:      {0:.3f}".format(f1))
                        print("acc:      {0:.3f}".format(acc))
                        print("prec:      {0:.3f}".format(prec))
                        print("rec:      {0:.3f}".format(rec))
                        print('model:     {0}'.format(params.model))
                        print('is_binary:     {0}'.format(params.binary))
                        print('epochs:      {0}'.format(params.epochs))
                        print('proj_loss:      {0}'.format(proj_loss))
                        print('alpha (projection loss coef):      {0}'.format(alpha))
                        print('batch size:  {0}'.format(batch_size))
                        print('learning rate:  {0}'.format(learning_rate))
                        print('weight_dir:  {0}'.format(weight_dir))
                        print('best_model_file_path:  {0}'.format(best_model_file_path))
                        print()

                        best_params_file.write("\n")
                        best_params_file.write("Found new set of best hyper params:\n")
                        best_params_file.write("f1       {0:.3f}:\n".format(f1))
                        best_params_file.write("acc       {0:.3f}:\n".format(acc))
                        best_params_file.write("prec       {0:.3f}:\n".format(prec))
                        best_params_file.write("rec       {0:.3f}:\n".format(rec))
                        best_params_file.write('model:     {0}\n'.format(params.model))
                        best_params_file.write('is_binary:     {0}\n'.format(params.binary))
                        best_params_file.write('epochs:      {0}\n'.format(params.epochs))
                        best_params_file.write('proj_loss:      {0}\n'.format(proj_loss))
                        best_params_file.write("alpha (projection loss coef):      {0}\n".format(alpha))
                        best_params_file.write('batch size:  {0}\n'.format(batch_size))
                        best_params_file.write('learning:  {0}\n'.format(learning_rate))
                        best_params_file.write('weight_dir:  {0}\n'.format(weight_dir))
                        best_params_file.write('best_model_file_path:  {0}\n'.format(best_model_file_path))

                        if old_file_name != None:
                            os.remove(old_file_name)

                        if old_results_file_name != None:
                            os.remove(old_results_file_name)

                        torch.save(params.model, best_model_file_path)
                        old_file_name = best_model_file_path
                        old_results_file_name = results_file_name
                        best_f1 = f1
                        rest_of_scores = [acc, prec, rec]

                        best_params = [proj_loss, alpha, learning_rate, batch_size]

                    else:
                        os.remove(results_file_name)
                        os.remove(best_model_file_path)

    print("")
    print("Done parameters search")
    print("best f1: {0:.3f}".format(best_f1))
    print("its acc: {0:.3f}".format(rest_of_scores[0]))
    print("its prec: {0:.3f}".format(rest_of_scores[1]))
    print("its rec: {0:.3f}".format(rest_of_scores[2]))
    print("best_params:")
    print('model:     {0}'.format(params.model))
    print('is_binary:     {0}'.format(params.binary))
    print('proj_loss:     {0}'.format(best_params[0]))
    print('alpha (projection loss coef):      {0}'.format(best_params[1]))
    print('learning rate:  {0}'.format(best_params[2]))
    print('batch size:  {0}'.format(best_params[3]))
    print("")

    best_params_file.write("\n")
    best_params_file.write("Done parameters search\n")
    best_params_file.write("best f1: {0:.3f}\n".format(best_f1))
    best_params_file.write("its acc: {0:.3f}\n".format(rest_of_scores[0]))
    best_params_file.write("its prec: {0:.3f}\n".format(rest_of_scores[1]))
    best_params_file.write("its rec: {0:.3f}\n".format(rest_of_scores[2]))
    best_params_file.write('model:     {0}\n'.format(params.model))
    best_params_file.write('is_binary:     {0}\n'.format(params.binary))
    best_params_file.write('proj_loss:      {0}\n'.format(best_params[0]))
    best_params_file.write("alpha (projection loss coef):      {0}\n".format(best_params[1]))
    best_params_file.write('learning:  {0}\n'.format(best_params[2]))
    best_params_file.write('batch size:  {0}\n'.format(best_params[3]))
    best_params_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang',
                        help="source language: es, ca, eu, en (default: en)",
                        default='en')
    parser.add_argument('-tl', '--target_lang',
                        help="target language: es, ca, eu, en (default: es)",
                        default='es')
    parser.add_argument('-bi', '--binary',
                        help="binary or 4-class (default: True)",
                        default=True,
                        type=str2bool)
    parser.add_argument('-e', '--epochs',
                        help="training epochs (default: 200)",
                        default=200,
                        type=int)
    parser.add_argument('-a', '--alphas',
                        help="trade-off between projection and classification objectives (default: .001)",
                        nargs='+', default=[.01, .001],
                        type=float)
    parser.add_argument('-pl', '--proj_losses',
                        help="projection loss: mse, cosine (default: cosine)",
                        nargs='+', default=['mse'])
    parser.add_argument('-bs', '--batch_sizes',
                        help="classification batch size (default: 50)",
                        nargs='+', default=[21],
                        type=int)
    parser.add_argument('-sv', '--src_vecs',
                        help=" source language vectors (default: GoogleNewsVecs )",
                        default='embeddings/original/google.txt')
    parser.add_argument('-tr', '--trans',
                        help='translation pairs (default: Bing Liu Sentiment Lexicon Translations)',
                        default='bingliu')
    parser.add_argument('-da', '--dataset',
                        help="dataset to train and test on (default: opener_sents)",
                        default='opener_sents', )
    parser.add_argument('-sd', '--savedir',
                        help="where to dump weights during training (default: ./models)",
                        default='models')
    parser.add_argument('-lr', '--learning_rates',
                        help="where to dump weights during training (default: 0.0001)",
                        nargs='+', default=[0.0001], type=float)
    parser.add_argument('-m', '--model',
                        help="where to dump weights during training (default: attn_blse)",
                        default='rnn_attn_blse')
    parser.add_argument('-cu', '--to_cuda',
                        help="where to dump weights during training (default: True)",
                        default=True, type=bool)
    # parser.add_argument('-te', '--temp',
    #                     help="where to dump weights during training (default: True)",
    #                     default=[], nargs='+')
    args = parser.parse_args()

    print("")
    print("Start parameters search")
    print('model:     {0}'.format(args.model))
    print('is_binary:     {0}'.format(args.binary))
    print('proj_losses:     {0}'.format(args.proj_losses))
    print('alphas (projection loss coef):      {0}'.format(args.alphas))
    print('learning rates:  {0}'.format(args.learning_rates))
    print('batch sizes:  {0}'.format(args.batch_sizes))
    print("")

    train_model_with_different_params(args)


if __name__ == '__main__':
    main()
