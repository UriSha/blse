from Utils.Datasets import *
from Utils.WordVecs import *
from Utils.utils import *

from rnn_blse import RNN_BLSE
from trainer import Trainer
from rnn_attn_blse import Rnn_Attn_BLSE


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
    parser.add_argument('-a', '--alpha',
                        help="trade-off between projection and classification objectives (default: .001)",
                        default=.1,
                        type=float)
    parser.add_argument('-pl', '--proj_loss',
                        help="projection loss: mse, cosine (default: cosine)",
                        default='cosine')
    parser.add_argument('-bs', '--batch_size',
                        help="classification batch size (default: 50)",
                        default=21,
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
    parser.add_argument('-lr', '--learning_rate',
                        help="where to dump weights during training (default: 0.0001)",
                        default=0.0001, type=float)
    parser.add_argument('-m', '--model',
                        help="where to dump weights during training (default: attn_rnn_blse)",
                        default='rnn_attn_blse')
    parser.add_argument('-cu', '--to_cuda',
                        help="where to dump weights during training (default: True)",
                        default=True, type=bool)
    args = parser.parse_args()

    # If there's no savedir, create it
    if args.model not in ['rnn_attn_blse', 'rnn_blse']:
        print("no such model: {}".format(args.model))
        exit(1)

    os.makedirs(args.savedir, exist_ok=True)

    if args.binary:
        output_dim = 2
        b = 'bi'
    else:
        output_dim = 4
        b = '4cls'

    weight_dir = "{}/{}/{}-{}-{}".format(args.savedir, args.model, args.dataset, args.target_lang, b)

    results_file_name = "results/report_{}_alpha-{}_batch_size-{}_epochs-{}_lr-{}.txt".format(args.model,
                                                                                              args.alpha,
                                                                                              args.batch_size,
                                                                                              args.epochs,
                                                                                              '{0:.15f}'.format(
                                                                                                  args.learning_rate).rstrip(
                                                                                                  '0').rstrip('.'))

    # import datasets (representation will depend on final classifier)
    print()
    print('training model')
    print('Parameters:')
    print('model:     {0}'.format(args.model))
    print('binary:     {0}'.format(b))
    print('epochs:      {0}'.format(args.epochs))
    print('alpha (projection loss coef):      {0}'.format(args.alpha))
    print('batchsize:  {0}'.format(args.batch_size))
    print('learning rate:  {0}'.format(args.learning_rate))
    print('weight_dir:  {0}'.format(weight_dir))
    print('results_file_name:  {0}'.format(results_file_name))
    print()

    print('importing datasets')

    dataset = General_Dataset(os.path.join('datasets', args.source_lang, args.dataset),
                              None,
                              binary=args.binary,
                              rep=words,
                              one_hot=False)

    cross_dataset = General_Dataset(os.path.join('datasets', args.target_lang, args.dataset),
                                    None,
                                    binary=args.binary,
                                    rep=words,
                                    one_hot=False)
    # print("len(cross_dataset._Xdev): {}".format(len(cross_dataset._Xdev)))
    # print("len(cross_dataset._Xtest): {}".format(len(cross_dataset._Xtest)))

    # Import monolingual vectors
    print('importing word embeddings')
    trg_vecs_file_path = "embeddings/original/sg-300-{}.txt".format(args.target_lang)
    print("trg_vecs_file_path: {}".format(trg_vecs_file_path))
    src_vecs = WordVecs(args.src_vecs)
    trg_vecs = WordVecs(trg_vecs_file_path)

    # Get sentiment synonyms and antonyms to check how they move during training
    synonyms1, synonyms2, neg = get_syn_ant(args.source_lang, src_vecs)
    cross_syn1, cross_syn2, cross_neg = get_syn_ant(args.target_lang, trg_vecs)

    # Import translation pairs
    translation_file_path = "lexicons/{}/en-{}.txt".format(args.trans, args.target_lang)
    print("translation_file_path: {}".format(translation_file_path))
    pdataset = ProjectionDataset(translation_file_path, src_vecs, trg_vecs)

    # Set up model
    if args.model == 'rnn_blse':
        model = RNN_BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                         projection_loss=args.proj_loss,
                         output_dim=output_dim,
                         batch_size=args.batch_size,
                         to_cuda=args.to_cuda,
                         src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                         trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg,
                         )
    elif args.model == 'rnn_attn_blse':
        model = Rnn_Attn_BLSE(src_vecs, trg_vecs, pdataset, dataset, cross_dataset,
                              projection_loss=args.proj_loss,
                              output_dim=output_dim,
                              to_cuda=args.to_cuda,
                              batch_size=args.batch_size,
                              src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg,
                              trg_syn1=cross_syn1, trg_syn2=cross_syn2, trg_neg=cross_neg,
                              )

    if torch.cuda.is_available() and args.to_cuda:
        print("cuda is available")
        model.cuda()
    else:
        print("cuda is not available")

    # Loss Functions
    class_criterion = nn.CrossEntropyLoss()
    proj_criterion = nn.MSELoss()

    if args.proj_loss == 'mse':
        proj_criterion = nn.MSELoss()
    elif args.proj_loss == 'cosine':
        proj_criterion = cosine_loss
    else:
        print("no projection criterion supported: {}".format(args.proj_loss))
        exit(1)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Fit model
    results_file = open(results_file_name, "w+")
    trainer = Trainer(model, args.alpha, optim, args.learning_rate, class_criterion, proj_criterion, args.epochs,
                      args.batch_size, results_file, weight_dir, args.to_cuda)

    best_model_file_path = trainer.train(pdataset._Xtrain, pdataset._ytrain,
                                         dataset._Xtrain, dataset._ytrain)

    # Get best dev f1 and weights
    print("looking in dir: {}".format(weight_dir))
    best_f1, best_params = get_best_model_params(best_model_file_path)
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

    model.evaluate(cross_dataset._Xtest, cross_dataset._ytest, results_file=results_file, src=False)

    model.confusion_matrix(cross_dataset._Xtest, cross_dataset._ytest, src=False, results_file=results_file)

    results_file.close()

if __name__ == '__main__':
    main()
