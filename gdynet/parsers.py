import argparse


# parser for the main.py
main_parser = argparse.ArgumentParser(
    description='The main training function for Graph Dynamical Networks')
main_parser.add_argument('--train-flist', nargs='+', required=True,
                         help='a list of file paths to training data files.')
main_parser.add_argument('--val-flist', nargs='+', required=True,
                         help='a list of file paths to validation data files.')
main_parser.add_argument('--test-flist', nargs='+', required=True,
                         help='a list of file paths to testing data files.')
main_parser.add_argument('--job-dir', type=str, default='./',
                         help='the directory that saves training results '
                         '(default: ./)')
main_parser.add_argument('--mode', choices=['kdtree', 'direct', 'vanilla'],
                         default='kdtree', help='choose the type of model'
                         'to build (default: kdtree)')
main_parser.add_argument('--tau', type=int, default=1,
                         help='lag time steps (default: 1)')
main_parser.add_argument('--n-classes', type=int, default=2,
                         help='number of classes (default: 2')
main_parser.add_argument('--k-eig', type=int, default=0,
                         help='number of eigenvalues to use (default: 0)')
main_parser.add_argument('--atom-fea-len', type=int, default=16,
                         help='length of the atom feature vector '
                         '(default: 16)')
main_parser.add_argument('--n-conv', type=int, default=3,
                         help='number of convolutional layers (default: 3)')
main_parser.add_argument('--lr', type=float, default=0.0005,
                         help='learning rate (default: 0.0005)')
main_parser.add_argument('--batch-size', '-b', type=int, default=16,
                         help='batch size (default: 16)')
main_parser.add_argument('--no-bn', action='store_true',
                         help='whether to use batch normalization (default: '
                         'False)')
main_parser.add_argument('--n-epoch', '-e', type=int, default=10,
                         help='number of epochs for training (default: 10)')
main_parser.add_argument('--no-shuffle', action='store_true',
                         help='whether to shuffle the data (default: shuffle)')
main_parser.add_argument('--random-seed', type=int, default=123,
                         help='random seed for shuffling the data '
                         '(default: 123)')

# parser for the preprocess.py
prep_parser = argparse.ArgumentParser(
    description='The preprocess function for the Graph Dynamical Networks')
prep_parser.add_argument('input_file', help='path to the input file')
prep_parser.add_argument('output_file', help='path to the output file')
prep_parser.add_argument('--n-workers', type=int, default=0,
                         help='number of workers to preprocess data, 0 means '
                         'do not use multiprocessing. (default: 0)')
prep_parser.add_argument('--n-nbrs', type=int, default=20,
                         help='number of neighbors used for construct graph '
                         '(default: 20)')
prep_parser.add_argument('--radius', type=float, default=7.,
                         help='search radius for finding nearest neighbors '
                         '(default: 7.)')
prep_parser.add_argument('--backend', choices=['kdtree', 'direct', 'ndirect'],
                         default='kdtree', help='"kdtree", "direct" or "ndirect" available, '
                         'the backend used to search for nearest neighbors. '
                         '"kdtree" has linear scaling but only works for '
                         'orthogonal lattices. "direct" works for trigonal '
                         'lattices but has quadratic scaling. "ndirect" is '
                         'an enhanced method for "direct" which could '
                         'accelarate the process ofdealing with large lattices.'
                         '(default: "kdtree")')
