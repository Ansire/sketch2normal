from NormalNet import *
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='4legs', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=150, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--output_size', dest='output_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--lr', dest='lr', type=float, default=5e-5, help='initial learning rate for adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=500, help='weight on L1 term in objective')
parser.add_argument('--gradient_penalty_coefficient', dest='gradient_penalty_coefficient', type=float, default=100.0, help='coefficient of the gradient penalty')
parser.add_argument('--n_critic', dest='n_critic', type=int, default=5, help='#n_critic')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = normalnet(sess, image_size=args.output_size, batch_size=args.batch_size,
                        output_size=args.output_size, L1_lambda=args.L1_lambda, n_critic=args.n_critic,
                        dataset_name=args.dataset_name, coefficient=args.gradient_penalty_coefficient)

        if args.phase == 'train':
            model.train(args)
        else:
            model.test(args)

if __name__ == '__main__':
    tf.app.run()
