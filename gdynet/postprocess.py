from __future__ import print_function, division

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from .vampnet import VampnetTools
from .utils import batch_pdist_pbc


def plot_timescales(predictions, lags, n_splits,
                    split_axis=0, time_unit_in_ns=1.):
    """
    Plot timescales implied by the Koopman dynamical model.

    Parameters
    ----------
    predictions: np.array, shape (F, num_atom, n_classes)
        probability of each state
    lags: np.array, shape (N,)
        lag times where timescales are computed
    n_splits: int
        number of splits to divide the trajectory to estimate uncertainty
    split_axis: int
        choose the axis to split the trajectory
    time_unit_in_ns: float
        the unit of each time step (0.1 means 0.1 ns/timestep)
    """
    if split_axis not in [0, 1]:
        raise ValueError('Split can only happen along axis 0 (time) or '
                         '1 (atom)')
    vamp = VampnetTools(epsilon=1e-5)
    splited_preds = np.array_split(predictions, n_splits, axis=split_axis)
    splited_its = np.stack([vamp.get_its(p, lags) for p in splited_preds])

    # time unit conversion
    lags = lags * time_unit_in_ns
    splited_its = splited_its * time_unit_in_ns

    its_log_mean = np.mean(np.log(splited_its), axis=0)
    its_log_std = np.std(np.log(splited_its), axis=0)
    its_mean = np.exp(its_log_mean)
    its_upper = np.exp(its_log_mean + its_log_std * 1.96 /
                       np.sqrt(n_splits))
    its_lower = np.exp(its_log_mean - its_log_std * 1.96 /
                       np.sqrt(n_splits))

    plt.figure(figsize=(4, 3))
    for i in range(its_mean.shape[0]):
        plt.semilogy(lags, its_mean[i])
        plt.fill_between(lags, its_lower[i], its_upper[i], alpha=0.2)
    plt.semilogy(lags, lags, 'k')
    plt.fill_between(lags, lags, lags[0], alpha=0.2, color='k')
    plt.xlabel('Lag time (ns)')
    plt.ylabel('Timescales (ns)')


def plot_ck_tests(predictions, tau_msm, steps, n_splits,
                  split_axis=0, time_unit_in_ns=1.):
    """
    Plot CK tests by the Koopman dynamical model.

    Parameters
    ----------
    predictions: np.array, shape (F, num_atom, n_classes)
        probability of each state
    tau_msm: int
        lag time used to build the Koopman model
    steps: int
        number of steps to validate the Koopman model
    n_splits: int
        number of splits to divide the trajectory to estimate uncertainty
    split_axis: int
        choose the axis to split the trajectory
    time_unit_in_ns: float
        the unit of each time step (0.1 means 0.1 ns/timestep)
    """
    if split_axis not in [0, 1]:
        raise ValueError('Split can only happen along axis 0 (time) or '
                         '1 (atom)')
    n_states = predictions.shape[-1]
    vamp = VampnetTools(epsilon=1e-5)
    splited_preds = np.array_split(predictions, n_splits, axis=split_axis)
    splited_predicted, splited_estimated = [], []
    for p in splited_preds:
        predicted, estimated = vamp.get_ck_test(p, steps, tau_msm)
        splited_predicted.append(predicted)
        splited_estimated.append(estimated)
    splited_predicted = np.stack(splited_predicted)
    splited_estimated = np.stack(splited_estimated)

    # time unit conversion
    tau_msm = tau_msm * time_unit_in_ns

    fig, ax = plt.subplots(n_states, n_states, sharex=True, sharey=True,
                           figsize=(4, 3))
    for i in range(n_states):
        for j in range(n_states):
            pred_mean = splited_predicted[:, i, j].mean(axis=0)
            pred_std = splited_predicted[:, i, j].std(axis=0)
            ax[i][j].plot(np.arange(0, steps * tau_msm, tau_msm),
                          pred_mean, color='b')
            ax[i][j].fill_between(
                np.arange(0, steps * tau_msm, tau_msm),
                pred_mean - pred_std * 1.96 / np.sqrt(n_splits),
                pred_mean + pred_std * 1.96 / np.sqrt(n_splits),
                alpha=0.2, color='b')
            ax[i][j].errorbar(
                np.arange(0, steps * tau_msm, tau_msm),
                splited_estimated[:, i, j].mean(axis=0),
                splited_estimated[:, i, j].std(axis=0) * 1.96 /
                np.sqrt(n_splits),
                color='r', linestyle='--')
            ax[i][j].set_title(str(i) + '->' + str(j))
    ax[0][0].set_ylim((-0.1, 1.1))
    ax[0][0].set_xlim((0, steps * tau_msm))
    ax[0][0].axes.get_xaxis().set_ticks(
        np.linspace(0, steps * tau_msm, 3))
    fig.text(0.5, 0.0, '(ns)', ha='center')
