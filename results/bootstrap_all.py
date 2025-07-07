# %%
import numpy as np
import pandas as pd
from astropy.io import fits
from eval_utils import label_field, stellar_metrics
from scipy.stats import norm

import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

hdul = fits.open('data/dataset3_subset2.fits')
dataset = hdul[1].data
hdul.close()

test_ind = np.load('data/3_2_test.npy')

label_f = label_field(dataset, regr=False, new=True)

classes = label_f.classes
indices = label_f.ord_to_idx(classes)

y_true = indices[test_ind]
with open('ultimate_dict.pkl', 'rb') as f:
    ultimate_dict = pickle.load(f)

print(f'Model list available: {list(ultimate_dict.keys())}')

def stratified_indices(y, n_reps=1000, rng=None):
    """Return a list of length `n_reps`, each an array of bootstrap indices
        drawn with replacement *within* every class of `y`."""
    rng = np.random.default_rng(rng)
    class_to_idx = {c: np.where(y == c)[0] for c in np.unique(y)}
    reps = []
    for _ in range(n_reps):
        sample = np.concatenate([
            rng.choice(idxs, size=len(idxs), replace=True)
            for idxs in class_to_idx.values()
        ])
        reps.append(sample)
    return reps

idx_boot = stratified_indices(y_true, rng=1337)

def jackknife_indices(n_samples):
    """
    Generator that yields the index array for each leave-one-out replicate.

    """
    full = np.arange(n_samples)
    for i in range(n_samples):
        # everything except position i
        yield np.concatenate((full[:i], full[i+1:]))

def bca_ci(boot_vals, jack_vals, true_val, alpha=0.95):
    """
    Bias-Corrected & Accelerated (BCa) bootstrap CI.

    Parameters
    ----------
    boot_vals : 1-D array
        Bootstrap distribution of a statistic (length B).
    jack_vals : 1-D array
        Jack-knife distribution of the same statistic (length N).
    alpha : float
        Confidence level (0.95 → 95 % CI).

    Returns
    -------
    (lower, upper) tuple
        End-points of the two-sided BCa confidence interval.
    """
    boot_vals = np.asarray(boot_vals)
    jack_vals = np.asarray(jack_vals)

    B = boot_vals.size           # # bootstrap resamples
    z0 = norm.ppf(               # bias-correction
        (boot_vals < true_val).mean()
    )

    # acceleration from jack-knife influence values
    jack_mean = jack_vals.mean()
    numer = np.sum((jack_mean - jack_vals)**3)
    denom = 6.0 * (np.sum((jack_mean - jack_vals)**2) ** 1.5)
    a = numer / denom

    # helper: transform nominal ↦ adjusted percentiles
    def _pct(q):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return 100 * adj  # convert to percentile (0–100)

    q_lo, q_hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
    lo = np.percentile(boot_vals, _pct(q_lo))
    hi = np.percentile(boot_vals, _pct(q_hi))
    return lo, hi

def metrics_for_model(actuals, preds, idx_boot):
        def one_rep(idx):
            yt_t, yp_t = actuals[idx], preds[idx]
            sm = stellar_metrics(label_f, yp_t, yt_t, True, False)
    
            return (sm.f1_macro(),
                    sm.mae(),
                    sm.two_stage_qwk("q")[0],
                    sm.two_stage_qwk("q")[1])
        return np.vstack([one_rep(idx) for idx in idx_boot])

def bootstrap_pvalue(gaps, delta_hat, two_sided=True, add_one=True):
    """
    Bootstrap p-value for H0: Δ = 0 using the paired bootstrap distribution.

    Parameters
    ----------
    gaps      : 1-D array of bootstrap gaps (A − B)  – shape (B,)
    delta_hat : Observed gap on the full sample.
    two_sided : If True, return a two-sided p-value.  If False, upper-tail.
    add_one   : Whether to apply the common (+1)/(B+1) finite-sample correction.

    Returns
    -------
    p         : float   – bootstrap p-value.
    """
    B = len(gaps)
    # centre the distribution so the null is at 0
    centred = gaps - delta_hat

    if two_sided:
        extreme = np.abs(centred) >= abs(delta_hat)
    else:  # upper-tail test (Δ > 0)
        extreme = centred >= delta_hat

    count = extreme.sum()
    if add_one:          # prevents a p-value of exactly 0
        count += 1
        B += 1
    return count / B
# %%
def one_model(model_name):
    print(f'Performance metrics and CI for {model_name}')
    preds = ultimate_dict[model_name]['preds']

    true_val = metrics_for_model(y_true, preds, (np.arange(len(test_ind)),))[0]
    
    print(f'F1 Macro: {true_val[0]:.2%}')
    print(f'Mean Absolute Error: {true_val[1]:.4f}')
    print(f'Q-Weighted Kappa: {true_val[2]:.2%}, {true_val[3]:.2%}')
    
    boot_vals = metrics_for_model(y_true, preds, idx_boot)

    jack_vals = metrics_for_model(y_true, preds, jackknife_indices(len(y_true)))
    
    ci = [bca_ci(boot_vals[:,i], jack_vals[:,i], true_val[i]) for i in range(4)]
    print('\n Confidence intervals:')
    for (lo, hi) in ci:
        print(f"\t {lo:.4f}  –  {hi:.4f}")
    print('--------------------------------\n\n')

# %%
for key in ultimate_dict.keys():
    one_model(key)

# %%

def two_models(model1, model2):
    preds_a = ultimate_dict[model1]['preds']
    preds_b = ultimate_dict[model2]['preds']

    def paired_bootstrap_gap(actuals, preds_a, preds_b, indices):
        """Return bootstrap vector of metric differences (A - B)."""
        def one_rep(idx):
            yt_t, yp_t_a, y_pt_b = actuals[idx], preds_a[idx], preds_b[idx]
            sm_a = stellar_metrics(label_f, yp_t_a, yt_t, True, False)
            sm_b = stellar_metrics(label_f, y_pt_b, yt_t, True, False)

            return (sm_a.f1_macro()-sm_b.f1_macro(),
                    sm_a.mae()-sm_b.mae(),
                    sm_a.two_stage_qwk("q")[0]-sm_b.two_stage_qwk("q")[0],
                    sm_a.two_stage_qwk("q")[1]-sm_b.two_stage_qwk("q")[1])
        return np.vstack([one_rep(idx) for idx in indices])
    
    true_gap = paired_bootstrap_gap(y_true, preds_a, preds_b, (np.arange(len(test_ind)),))[0]

    print(f'F1 Macro Δ: {true_gap[0]:.2%}')
    print(f'Mean Absolute Error Δ: {true_gap[1]:.4f}')
    print(f'Q-Weighted Kappa Δ: {true_gap[2]:.2%}, {true_gap[3]:.2%}')

    gaps_full = paired_bootstrap_gap(y_true, preds_a, preds_b, idx_boot)

    gaps_jack = paired_bootstrap_gap(y_true, preds_a, preds_b, jackknife_indices(len(y_true)))


    ci = [bca_ci(gaps_full[:,i], gaps_jack[:,i], true_gap[i]) for i in range(4)]
    p_two_sided = [bootstrap_pvalue(gaps_full[:,i], true_gap[i]) for i in range(4)]

    print('\nConfidence intervals for Δ:')
    for (lo, hi), p in zip(ci,p_two_sided):
        print(f"\t {lo:.4f}  –  {hi:.4f} (p = {p:.4%})")
    print('--------------------------------\n\n')
# %%
two_models('conv1d_best','resnet50_1d')
# %%
two_models('conv1d_best','conv1d_no_ord')
# %%
from scipy.stats import mode

def ensemble_models(models, method):
    preds = np.empty((1295,len(models),39))
    for i,model in enumerate(models):
            preds[:,i,:] = ultimate_dict[model]['scores']
    if method == 'mean':
        return preds.sum(axis=1).argmax(axis=-1)
    elif method == 'vote':
        return mode(preds.argmax(axis=-1),axis=-1)[0]

hey = ensemble_models(['conv1d_best', 'resnet50_1d'], 'mean')

# %%

ultimate_dict.update({'ensemble_best':{'preds':hey,'scores':None}})

# %%
sm = stellar_metrics(label_f, hey, y_true, idx=True, regr=False)

sm.draw_cm('Ensemble of Conv1D(α=0.75) and ResNet50', 300, False)
# %%
sm.report('', False)
sm.report('temp', False)
sm.report('lum', False)
# %%
sm.draw_ord_cm('',300)
# %%
def ensemble_models(models, method):
    preds = np.empty((1295,len(models),39))
    for i,model in enumerate(models):
            preds[:,i,:] = ultimate_dict[model]['scores']
    if method == 'mean':
        return preds.sum(axis=1).argmax(axis=-1)
    elif method == 'vote':
        return mode(preds.argmax(axis=-1),axis=-1)[0]

hey = ensemble_models(['conv1d_no_ord', 'resnet50_1d'], 'mean')

true_val = metrics_for_model(y_true, hey, (np.arange(len(test_ind)),))[0]
    
print(f'F1 Macro: {true_val[0]:.2%}')
print(f'Mean Absolute Error: {true_val[1]:.4f}')
print(f'Q-Weighted Kappa: {true_val[2]:.2%}, {true_val[3]:.2%}')
# %%
ultimate_dict.update({'ensemble_no_ord+resnet':{'preds':hey,'scores':None}})
# %%
hey = ensemble_models(['conv1d_best', 'conv1d_half_ord'], 'mean')

true_val = metrics_for_model(y_true, hey, (np.arange(len(test_ind)),))[0]
    
print(f'F1 Macro: {true_val[0]:.2%}')
print(f'Mean Absolute Error: {true_val[1]:.4f}')
print(f'Q-Weighted Kappa: {true_val[2]:.2%}, {true_val[3]:.2%}')
# %%
ultimate_dict.update({'ensemble_best+half':{'preds':hey,'scores':None}})
# %%
