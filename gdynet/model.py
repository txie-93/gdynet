import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense, Lambda, Embedding,\
    BatchNormalization, Activation, Add, concatenate, Permute
from .data import MDStackGenerator, MDStackGenerator_direct,\
    MDStackGenerator_vannila
from .vampnet import VampnetTools


def _concat_nbrs(inps):
    """
    Concatenate neighbor features based on graph structure into a full
    feature vector

    B: Batch size
    N: Number of atoms in each crystal
    M: Max number of neighbors
    atom_fea_len: the length of atom features
    bond_fea_len: the length of bond features

    Parameters
    ----------
    atom_fea: (B, N, atom_fea_len)
    bond_fea: (B, N, M, bond_fea_len)
    nbr_list: (B, N, M)

    Returns
    -------
    total_fea: (B, N, M, 2 * atom_fea_len + bond_fea_len)
    """
    atom_fea, bond_fea, nbr_list = inps
    M = tf.shape(nbr_list)[-1]
    N = tf.shape(nbr_list)[1]
    B = tf.shape(nbr_list)[0]
    batch_idx = tf.reshape(tf.range(0, B), (B, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, N, M))
    full_idx = tf.stack((batch_idx, nbr_list), axis=-1)
    atom_nbr_fea = tf.gather_nd(atom_fea, full_idx)
    atom_self_fea = tf.tile(tf.expand_dims(atom_fea, 2), [1, 1, M, 1])
    full_fea = tf.concat([atom_self_fea, atom_nbr_fea, bond_fea], axis=-1)
    return full_fea


def _concat_nbrs_output_shape(input_shapes):
    """
    Needed since the inference of output shape fail in Keras for non-float
    tensors.
    https://github.com/keras-team/keras/issues/8437
    """
    B = input_shapes[0][0]
    N = input_shapes[0][1]
    M = input_shapes[2][-1]
    atom_fea_len = input_shapes[0][-1]
    bond_fea_len = input_shapes[1][-1]
    return (B, N, M, 2 * atom_fea_len + bond_fea_len)


def _pooling(inps):
    """
    Pool the atom_fea of target atoms together using their indexes.

    Parameters
    ----------
    atom_fea: (B, N, atom_fea_len)
    target_index: (B, N0)

    Returns
    -------
    crys_fea: (B, N0, atom_fea_len)
    """
    atom_fea, target_index = inps
    B = tf.shape(atom_fea)[0]
    N0 = tf.shape(target_index)[1]
    batch_idx = tf.reshape(tf.range(0, B), (B, 1))
    batch_idx = tf.tile(batch_idx, (1, N0))
    full_idx = tf.stack((batch_idx, target_index), axis=-1)
    return tf.gather_nd(atom_fea, full_idx)


def _pooling_output_shape(input_shapes):
    atom_fea_shape, target_index_shape = input_shapes
    assert atom_fea_shape[0] == target_index_shape[0]
    B, N0 = target_index_shape
    atom_fea_len = atom_fea_shape[-1]
    return (B, N0, atom_fea_len)


class _PreProcessCGCNN(object):
    """
    Pre-process data into a form for CGCNN.

    Parameters
    ----------
    num_atom: int, number of atoms in each cluster
    dmin: float, Gaussian filter minimum distance
    dmax: float, Gaussian filter maximum distace
    step: float, Gaussian filter step size
    var: float, Gaussian filter variance
    """
    def __init__(self, num_atom, dmin, dmax, step, var=None):
        self.num_atom = num_atom
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.var = var if var is not None else step
        self.bond_fea_len = len(np.arange(self.dmin, self.dmax + self.step,
                                          self.step))

    def gaussian_expand(self, nbr_dist):
        """
        Expand distances to gaussian basis

        Parameters
        ----------
        nbr_dist: shape (B, num_atom, M)

        Returns
        -------
        bond_fea: shape (B, num_atom, M, bond_fea_len)
        """
        gfilter = tf.range(self.dmin, self.dmax + self.step, self.step)
        return tf.exp(-(tf.expand_dims(nbr_dist, axis=-1) - gfilter)**2 /
                      (self.var**2))

    def pre_process(self, inps):
        """
        Pre-process input data.

        Parameters
        ----------
        stacked_coords: shape (B, num_atom, 3, 2)
        stacked_lattices: shape (B, 3, 2)
        stacked_nbr_lists: shape (B, num_atom, M, 2)

        Returns
        -------
        nbr_lists_1: shape (B, num_atom, M)
        bond_fea_1: shape (B, num_atom, M, bond_fea_len)
        """
        stacked_coords, stacked_lattices, stacked_nbr_lists = inps

        def pdc_dist(atom_coords, nbr_coords, lattice):
            """
            atom_coords: shape (B, N, 3)
            nbr_coords: shape (B, N, M, 3)
            lattice: shape (B, 3)
            """
            atom_coords = tf.expand_dims(atom_coords, axis=2)
            delta = tf.abs(atom_coords - nbr_coords)
            lattice = tf.expand_dims(tf.expand_dims(lattice, axis=1), axis=2)
            delta = tf.where(delta > 0.5 * lattice, delta - lattice, delta)
            return tf.sqrt(tf.reduce_sum(delta**2, axis=-1))

        def tile_dim0(nbr_list, atom_types):
            return tf.tile(tf.expand_dims(nbr_list, axis=0),
                           [tf.shape(atom_types)[0], 1, 1])

        def batch_gather(atom_coords, nbr_list):
            """
            atom_coords: shape (B, N, 3)
            nbr_list: shape (B, N, M)
            """
            batch_size = tf.shape(nbr_list)[0]
            num_atom = tf.shape(nbr_list)[1]
            num_nbr = tf.shape(nbr_list)[2]
            batch_idx = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1))
            batch_idx = tf.tile(batch_idx, (1, num_atom, num_nbr))
            full_idx = tf.stack((batch_idx, nbr_list), axis=-1)
            return tf.gather_nd(atom_coords, full_idx)

        def process_one(atom_coords, lattice, nbr_lists):
            nbr_coords = batch_gather(atom_coords, nbr_lists)
            nbr_dist = pdc_dist(atom_coords, nbr_coords, lattice)
            bond_fea = self.gaussian_expand(nbr_dist)
            return bond_fea

        atom_coords_1, atom_coords_2 = tf.unstack(stacked_coords, axis=-1)
        lattice_1, lattice_2 = tf.unstack(stacked_lattices, axis=-1)
        nbr_lists_1, nbr_lists_2 = tf.unstack(stacked_nbr_lists, axis=-1)
        bond_fea_1 = process_one(atom_coords_1, lattice_1, nbr_lists_1)
        bond_fea_2 = process_one(atom_coords_2, lattice_2, nbr_lists_2)
        return [nbr_lists_1, bond_fea_1, nbr_lists_2, bond_fea_2]


def load_keras_optimizer(model, filepath):
    """
    Load the state of optimizer.

    Parameters
    ----------
    optimizer: keras optimizer
      the optimizer for loading the state
    filepath: str
      the path of the pickle (.pkl) file that saves the optimizer state
    """
    with open(filepath, 'rb') as f:
        weight_values = pickle.load(f)
    model._make_train_function()
    model.optimizer.set_weights(weight_values)


class SaveOptimizerState(keras.callbacks.Callback):
    """
    Save the state of optimizer at the end of each epoch.

    Parameters
    ----------
    filepath: str
      the path of the pickle (.pkl) file that saves the optimizer state
    """
    def __init__(self, filepath):
        super(SaveOptimizerState, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = keras.backend.batch_get_value(symbolic_weights)
        with open(self.filepath, 'wb') as f:
            pickle.dump(weight_values, f)


class EpochCounter(keras.callbacks.Callback):
    """
    Count the number of epochs and save it.

    Parameters
    ----------
    filepath: str
      the path of the json file that saves the current number of epochs and
      number of training stage
    train_stage: int
      current training stage, starting with 0
    """
    def __init__(self, filepath, train_stage=0):
        super(EpochCounter, self).__init__()
        self.filepath = filepath
        self.train_stage = train_stage

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'w') as f:
            json.dump({'epoch': epoch,
                       'stage': self.train_stage}, f)


def reorder_predictions(raw_predictions, num_data, tau):
    """
    Reorder raw prediction array

    Parameters
    ----------
    raw_predictions: shape (num_data * (F - tau), num_atom, 2 * n_classes)
    predictions: shape (num_data, F, num_atom, n_classes)
    """
    if (raw_predictions.shape[0] % num_data != 0 or
            len(raw_predictions.shape) != 3 or
            raw_predictions.shape[2] % 2 != 0):
        raise ValueError('Bad format!')
    n_classes = raw_predictions.shape[2] // 2
    num_atom = raw_predictions.shape[1]
    raw_predictions = raw_predictions.reshape(num_data, -1, num_atom,
                                              n_classes * 2)
    assert np.allclose(raw_predictions[:, tau:, :, :n_classes],
                       raw_predictions[:, :-tau, :, n_classes:])
    predictions = np.concatenate([raw_predictions[:, :, :, :n_classes],
                                  raw_predictions[:, -tau:, :, n_classes:]],
                                 axis=1)
    return predictions


class GDyNet(object):
    """
    Graph Dynamical Network with VAMP loss to analyze time series data.
    """
    def __init__(self, train_flist, val_flist, test_flist,
                 job_dir='./', mode='kdtree',
                 tau=1, n_classes=2, k_eig=0, no_pool=False,
                 atom_fea_len=16, n_conv=3,
                 train_n_li=None, val_n_li=None, test_n_li=None,
                 dmin=0., dmax=7., step=0.2,
                 learning_rate=0.0005, batch_size=16,
                 use_bn=True, n_epoch=10, shuffle=True, random_seed=123):
        if mode not in ['kdtree', 'direct', 'vanilla']:
            raise ValueError('`mode` must in `kdtree`, `direct`, or `vanilla`')
        self.train_flist = train_flist
        self.val_flist = val_flist
        self.test_flist = test_flist
        self.job_dir = job_dir
        self.mode = mode
        self.tau = tau
        self.n_classes = n_classes
        self.k_eig = k_eig
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.dmin = dmin
        self.dmax = dmax
        self.step = step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.n_epoch = n_epoch
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.vamp = VampnetTools(epsilon=1e-5, k_eig=self.k_eig)

    def build_cgcnn_layer(self, atom_fea, bond_fea, nbr_list):
        total_fea = Lambda(_concat_nbrs, output_shape=_concat_nbrs_output_shape
                           )([atom_fea, bond_fea, nbr_list])
        # total_fea shape (None, N, M, 2 * atom_fea_len + bond_fea_len)
        nbr_core = Dense(self.atom_fea_len)(total_fea)
        nbr_filter = Dense(1)(total_fea)
        if self.use_bn:
            nbr_core = BatchNormalization(axis=-1)(nbr_core)
        nbr_filter = Permute((1, 3, 2))(nbr_filter)
        nbr_filter = Activation('softmax')(nbr_filter)
        nbr_filter = Permute((1, 3, 2))(nbr_filter)
        # nbr_filter = keras.activations.softmax(nbr_filter, axis=-2)
        nbr_core = Activation('relu')(nbr_core)
        nbr_sumed = Lambda(lambda x: tf.reduce_mean(x[0] * x[1], axis=2))(
            [nbr_filter, nbr_core])
        if self.use_bn:
            nbr_sumed = BatchNormalization(axis=-1)(nbr_sumed)
        out = Activation('relu')(Add()([atom_fea, nbr_sumed]))
        return out

    def build_cgcnn_model(self):
        atom_types_inp = Input(shape=(self.num_atom, ), dtype='int32')
        bond_fea_inp = Input(shape=(self.num_atom, self.num_nbr,
                                    self.bond_fea_len))
        nbr_list_inp = Input(shape=(self.num_atom, self.num_nbr),
                             dtype='int32')
        target_index_inp = Input(shape=(self.num_target, ), dtype='int32')
        atom_fea = Embedding(input_dim=100, output_dim=self.atom_fea_len)(
            atom_types_inp)
        bond_fea, nbr_list = bond_fea_inp, nbr_list_inp
        for _ in range(self.n_conv):
            atom_fea = self.build_cgcnn_layer(atom_fea, bond_fea, nbr_list)
        crys_fea = Lambda(_pooling, output_shape=_pooling_output_shape)(
            [atom_fea, target_index_inp])
        crys_fea = Activation('relu')(crys_fea)
        crys_fea = Activation('relu')(Dense(self.atom_fea_len)(crys_fea))
        out = Dense(self.n_classes, activation='softmax')(crys_fea)
        cgcnn_model = keras.Model(inputs=[atom_types_inp,
                                          bond_fea_inp,
                                          nbr_list_inp,
                                          target_index_inp],
                                  outputs=out)
        return cgcnn_model

    def build_gdynet(self):
        """build gdynet using graphs constructed using the `kdtree` backend"""
        stacked_coords_inp = Input(shape=(self.num_atom, 3, 2))
        stacked_lattices_inp = Input(shape=(3, 2))
        stacked_nbr_lists_inp = Input(shape=(self.num_atom, self.num_nbr, 2),
                                      dtype='int32')
        atom_types_inp = Input(shape=(self.num_atom, ), dtype='int32')
        target_index_inp = Input(shape=(self.num_target, ), dtype='int32')
        nbr_lists_1, bond_fea_1, nbr_lists_2, bond_fea_2 =\
            Lambda(self.prep.pre_process)(
                [stacked_coords_inp, stacked_lattices_inp,
                 stacked_nbr_lists_inp])
        cgcnn_model = self.build_cgcnn_model()
        branch_1 = cgcnn_model([atom_types_inp,
                                bond_fea_1,
                                nbr_lists_1,
                                target_index_inp])
        branch_2 = cgcnn_model([atom_types_inp,
                                bond_fea_2,
                                nbr_lists_2,
                                target_index_inp])
        merged = concatenate([branch_1, branch_2])
        self.model = keras.Model(inputs=[stacked_coords_inp,
                                         stacked_lattices_inp,
                                         stacked_nbr_lists_inp,
                                         atom_types_inp,
                                         target_index_inp],
                                 outputs=merged)
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.losses = [self.vamp.loss_VAMP2_autograd,
                       self.vamp._loss_VAMP_sym,
                       self.vamp.loss_VAMP2_autograd]

    def build_gdynet_direct(self):
        """build gdynet with graphs constructed using the `direct` backend"""
        atom_types_inp = Input(shape=(self.num_atom, ), dtype='int32')
        target_index_inp = Input(shape=(self.num_target, ), dtype='int32')
        bond_dist_1_inp = Input(shape=(self.num_atom, self.num_nbr))
        bond_dist_2_inp = Input(shape=(self.num_atom, self.num_nbr))
        nbr_list_1_inp = Input(shape=(self.num_atom, self.num_nbr),
                               dtype='int32')
        nbr_list_2_inp = Input(shape=(self.num_atom, self.num_nbr),
                               dtype='int32')
        bond_fea_1 = Lambda(self.prep.gaussian_expand)(bond_dist_1_inp)
        bond_fea_2 = Lambda(self.prep.gaussian_expand)(bond_dist_2_inp)
        cgcnn_model = self.build_cgcnn_model()
        branch_1 = cgcnn_model([atom_types_inp,
                                bond_fea_1,
                                nbr_list_1_inp,
                                target_index_inp])
        branch_2 = cgcnn_model([atom_types_inp,
                                bond_fea_2,
                                nbr_list_2_inp,
                                target_index_inp])
        merged = concatenate([branch_1, branch_2])
        self.model = keras.Model(inputs=[atom_types_inp,
                                         target_index_inp,
                                         bond_dist_1_inp,
                                         nbr_list_1_inp,
                                         bond_dist_2_inp,
                                         nbr_list_2_inp],
                                 outputs=merged)
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.losses = [self.vamp.loss_VAMP2_autograd,
                       self.vamp._loss_VAMP_sym,
                       self.vamp.loss_VAMP2_autograd]

    def build_vanilla(self):
        """build a vanilla VAMPnet as baseline"""
        traj_coords_1_inp = Input(shape=(self.num_atom, 3))
        traj_coords_2_inp = Input(shape=(self.num_atom, 3))
        bn_layer = BatchNormalization()
        dense_layers = [Dense(self.atom_fea_len, activation='relu')
                        for _ in range(self.n_conv)]

        # building network
        lx_branch = bn_layer(traj_coords_1_inp)
        rx_branch = bn_layer(traj_coords_2_inp)

        for layer in dense_layers:
            lx_branch = layer(lx_branch)
            rx_branch = layer(rx_branch)

        softmax = Dense(self.n_classes, activation='softmax')

        lx_branch = softmax(lx_branch)
        rx_branch = softmax(rx_branch)

        merged = concatenate([lx_branch, rx_branch])
        self.model = keras.Model(inputs=[traj_coords_1_inp,
                                         traj_coords_2_inp],
                                 outputs=merged)
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.losses = [self.vamp.loss_VAMP2_autograd,
                       self.vamp._loss_VAMP_sym,
                       self.vamp.loss_VAMP2_autograd]

    def load_data(self):
        self.train_generator = MDStackGenerator(self.train_flist,
                                                tau=self.tau,
                                                batch_size=self.batch_size,
                                                random_seed=self.random_seed,
                                                shuffle=self.shuffle)
        self.val_generator = MDStackGenerator(self.val_flist,
                                              tau=self.tau,
                                              batch_size=self.batch_size,
                                              random_seed=self.random_seed,
                                              shuffle=self.shuffle)
        self.num_atom = self.train_generator[0][0][0].shape[1]
        self.num_nbr = self.train_generator[0][0][2].shape[2]
        self.num_target = self.train_generator[0][0][4].shape[1]
        self.prep = _PreProcessCGCNN(num_atom=self.num_atom, dmin=self.dmin,
                                     dmax=self.dmax, step=self.step)
        self.bond_fea_len = self.prep.bond_fea_len

    def load_data_direct(self):
        self.train_generator = MDStackGenerator_direct(
            self.train_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.val_generator = MDStackGenerator_direct(
            self.val_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.num_atom = self.train_generator[0][0][0].shape[1]
        self.num_nbr = self.train_generator[0][0][2].shape[2]
        self.num_target = self.train_generator[0][0][1].shape[1]
        self.prep = _PreProcessCGCNN(num_atom=self.num_atom, dmin=self.dmin,
                                     dmax=self.dmax, step=self.step)
        self.bond_fea_len = self.prep.bond_fea_len

    def load_data_vanilla(self):
        self.train_generator = MDStackGenerator_vannila(
            self.train_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.val_generator = MDStackGenerator_vannila(
            self.val_flist,
            tau=self.tau,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            shuffle=self.shuffle)
        self.num_atom = self.train_generator[0][0][0].shape[1]

    def check_train_state(self):
        try:
            with open(os.path.join(self.job_dir, 'train_state.json')) as f:
                data = json.load(f)
                init_epoch, init_stage = data['epoch'], data['stage']
            if init_epoch == self.n_epoch - 1:
                init_epoch = 0
                init_stage += 1
            else:
                init_epoch += 1
            weights_file = os.path.join(self.job_dir,
                                        'last_model.hdf5'.format(
                                            init_stage))
            self.model.load_weights(weights_file)
            is_continue = True
        except IOError:
            init_epoch, init_stage, is_continue = 0, 0, False
        return init_epoch, init_stage, is_continue

    def train_model(self):
        if self.mode == 'kdtree':
            self.load_data()
            self.build_gdynet()
        elif self.mode == 'direct':
            self.load_data_direct()
            self.build_gdynet_direct()
        elif self.mode == 'vanilla':
            self.load_data_vanilla()
            self.build_vanilla()
        else:
            raise ValueError

        init_epoch, init_stage, is_continue = self.check_train_state()
        for l_index, loss_function in enumerate(self.losses):
            # skip this round if previous training records are found
            if l_index < init_stage:
                continue
            self.model.compile(optimizer=self.optimizer,
                               loss=loss_function,
                               metrics=[self.vamp.metric_VAMP,
                                        self.vamp.metric_VAMP2])
            if is_continue:
                opti_state_file = os.path.join(self.job_dir, 'opti_state.pkl')
                load_keras_optimizer(self.model, opti_state_file)
                is_continue = False
            best_model_path = os.path.join(
                self.job_dir, 'best_model_{}.hdf5'.format(l_index))
            last_model_path = os.path.join(
                self.job_dir, 'last_model.hdf5'.format(l_index))
            train_logger_path = os.path.join(
                self.job_dir, 'train_{}.log'.format(l_index))
            opti_state_path = os.path.join(
                self.job_dir, 'opti_state.pkl')
            train_state_path = os.path.join(
                self.job_dir, 'train_state.json')
            callbacks = [keras.callbacks.TerminateOnNaN(),
                         keras.callbacks.ModelCheckpoint(
                         best_model_path,
                         monitor='val_metric_VAMP2',
                         save_best_only=True, save_weights_only=True,
                         mode='max'),
                         keras.callbacks.ModelCheckpoint(
                         last_model_path,
                         save_weights_only=True),
                         keras.callbacks.CSVLogger(
                         train_logger_path,
                         separator=',', append=True),
                         EpochCounter(train_state_path, train_stage=l_index),
                         SaveOptimizerState(opti_state_path)]
            self.model.fit_generator(generator=self.train_generator,
                                     validation_data=self.val_generator,
                                     use_multiprocessing=False,
                                     epochs=self.n_epoch,
                                     callbacks=callbacks,
                                     initial_epoch=init_epoch)
        # evaluate_generator seems to evaluate on batches instead of all
        # predictions. keras bug?

        # delete training and validation data to save memory
        del self.train_generator
        del self.val_generator
        del self.model
        self.predict_model(self.job_dir)

    def predict_model(self, result_dir):
        if self.mode == 'kdtree':
            self.test_generator = MDStackGenerator(
                self.test_flist,
                tau=self.tau,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                shuffle=False)
            self.num_atom = self.test_generator[0][0][0].shape[1]
            self.num_nbr = self.test_generator[0][0][2].shape[2]
            self.num_target = self.test_generator[0][0][4].shape[1]
            self.prep = _PreProcessCGCNN(num_atom=self.num_atom,
                                         dmin=self.dmin, dmax=self.dmax,
                                         step=self.step)
            self.bond_fea_len = self.prep.bond_fea_len
            self.build_gdynet()
        elif self.mode == 'direct':
            self.test_generator = MDStackGenerator_direct(
                self.test_flist,
                tau=self.tau,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                shuffle=False)
            self.num_atom = self.test_generator[0][0][0].shape[1]
            self.num_nbr = self.test_generator[0][0][2].shape[2]
            self.num_target = self.test_generator[0][0][1].shape[1]
            self.prep = _PreProcessCGCNN(num_atom=self.num_atom,
                                         dmin=self.dmin, dmax=self.dmax,
                                         step=self.step)
            self.bond_fea_len = self.prep.bond_fea_len
            self.build_gdynet_direct()
        elif self.mode == 'vanilla':
            self.test_generator = MDStackGenerator_vannila(
                self.test_flist,
                tau=self.tau,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                shuffle=False)
            self.num_atom = self.train_generator[0][0][0].shape[1]
            self.build_vanilla()
        else:
            raise ValueError
        # load weights
        weights_file = os.path.join(self.job_dir, 'last_model.hdf5')
        self.model.load_weights(weights_file)
        raw_preds = self.model.predict_generator(generator=self.test_generator,
                                                 use_multiprocessing=False)
        preds = reorder_predictions(raw_preds, len(self.test_flist),
                                    self.tau)
        np.save(os.path.join(result_dir, 'test_pred.npy'), preds)
        preds_placeholder = tf.placeholder(raw_preds.dtype,
                                           shape=raw_preds.shape)
        metric_vamp = self.vamp.metric_VAMP(None, preds_placeholder)
        metric_vamp2 = self.vamp.metric_VAMP2(None, preds_placeholder)
        with tf.Session() as sess:
            results = sess.run([metric_vamp, metric_vamp2],
                               feed_dict={preds_placeholder: raw_preds})
        np.savetxt(os.path.join(result_dir, 'test_eval.csv'),
                   np.array(results), delimiter=',')
