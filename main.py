from __future__ import print_function, division

import sys
from gdynet.model import GDyNet
from gdynet.parsers import main_parser as parser


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print(args)
    gdynet = GDyNet(train_flist=args.train_flist,
                    val_flist=args.val_flist,
                    test_flist=args.test_flist,
                    job_dir=args.job_dir,
                    mode=args.mode,
                    tau=args.tau,
                    n_classes=args.n_classes,
                    k_eig=args.k_eig,
                    atom_fea_len=args.atom_fea_len,
                    n_conv=args.n_conv,
                    learning_rate=args.lr,
                    batch_size=args.batch_size,
                    use_bn=not args.no_bn,
                    n_epoch=args.n_epoch,
                    shuffle=not args.no_shuffle,
                    random_seed=args.random_seed)
    gdynet.train_model()
