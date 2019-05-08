from __future__ import print_function, division

import sys
import os
import shutil
import multiprocessing as mp
from gdynet.preprocess import Preprocess
from gdynet.parsers import prep_parser as parser
from gdynet.utils import split_traj_files, combine_split_files


def preprocess(input_file, output_file, n_nbrs, radius, backend):
    if input_file[-4:] != '.npz':
        input_file += '.npz'
    prep = Preprocess(input_file=input_file,
                      output_file=output_file,
                      n_nbrs=n_nbrs,
                      radius=radius,
                      backend=backend)
    prep.preprocess()


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print(args)
    if args.n_workers == 0:
        preprocess(input_file=args.input_file,
                   output_file=args.output_file,
                   n_nbrs=args.n_nbrs,
                   radius=args.radius,
                   backend=args.backend)
    else:
        split_dir = '_temp'
        if not os.path.exists(split_dir):
            os.mkdir(split_dir)
        basename = split_traj_files(args.input_file, split_dir, args.n_workers)
        workers = []
        for i in range(args.n_workers):
            split_file = os.path.join(split_dir, basename + str(i))
            split_output = os.path.join(split_dir, basename + '-graph' + str(i))
            workers.append(mp.Process(target=preprocess,
                                      args=(split_file,
                                            split_output,
                                            args.n_nbrs,
                                            args.radius,
                                            args.backend)))
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        combine_split_files(split_dir, args.output_file, basename + '-graph',
                            args.n_workers, zip=True)
        shutil.rmtree(split_dir)
