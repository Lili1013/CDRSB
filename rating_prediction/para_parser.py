import argparse

def parse():
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec source')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='gnn embedding size')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--dec_embed_dim', type=int, default=128, metavar='N', help='embedding size of decomposing variable')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='N',
                        help='hyper parameter of c loss')
    parser.add_argument('--beta', type=float, default=0.001, metavar='N',
                        help='hyper parameter of mi loss')
    parser.add_argument('--save_model_path', type=str, default='../best_models/rating_prediction/dianping_best_model_beta_0.001_dec-128', metavar='N',
                        help='the best source path')
    parser.add_argument('--train_model_weights_path', type=str, default='../best_models/rating_prediction/train_model_weights_dianping_beta_0.001_dec-128', metavar='N',
                        help='the best source path')


    args = parser.parse_args()
    return args