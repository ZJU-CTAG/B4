from collections import Counter
import numpy as np
import scipy.special as sc

def bernoulli_with_beta_prior(n1, n, alpha=1, beta=1):
    alpha_2 = n1 + alpha
    beta_2 = n - n1 + beta
    return sc.betaln(alpha_2, beta_2)

def B4(E, beta_0, alpha_xy):

    test_case_layouts = [tuple(E[i, :].tolist()) for i in range(len(E))]
    counter = Counter(test_case_layouts)

    pred_x_loss = []
    for test_case_layout in counter:
        pred_y = np.asarray(test_case_layout)
        pred_x = np.zeros(len(E))
        cur_x = []

        for i in range(len(E)):
            if tuple(E[i].tolist()) == test_case_layout:
                pred_x[i] = 1
                cur_x.append(i)

        log_posterior_py = bernoulli_with_beta_prior(np.sum(pred_y), len(pred_y), alpha=alpha_xy)
        log_posterior_px = bernoulli_with_beta_prior(np.sum(pred_x), len(pred_x), alpha=alpha_xy)

        E1 = E[pred_x == 0][:, pred_y == 1]
        log_posterior_p1 = bernoulli_with_beta_prior(np.sum(E1), E1.shape[0] * E1.shape[1])

        E0 = E[pred_x == 0][:, pred_y == 0]
        log_posterior_p0 = bernoulli_with_beta_prior(np.sum(E0), E0.shape[0] * E0.shape[1], beta=beta_0)

        p_posterior = log_posterior_px + log_posterior_py + log_posterior_p1 + log_posterior_p0

        pred_x_loss.append((cur_x, p_posterior))
    
    return sorted(pred_x_loss, key=lambda x: x[1], reverse=True)


def CodeT(E):

    test_case_layouts = [tuple(E[i, :].tolist()) for i in range(len(E))]
    counter = Counter(test_case_layouts)

    pred_x_loss = []

    for test_case_layout in counter:
        pred_x = np.zeros(len(E))
        cur_x = []
        for i in range(len(E)):
            if tuple(E[i].tolist()) == test_case_layout:
                pred_x[i] = 1
                cur_x.append(i)
        
        res = sum(test_case_layout) * (counter[test_case_layout] ** 0.5)
        pred_x_loss.append((cur_x, res))
    return sorted(pred_x_loss, key=lambda x: x[1], reverse=True)


def MBR_exec(E):

    test_case_layouts = [tuple(E[i, :].tolist()) for i in range(len(E))]
    counter = Counter(test_case_layouts)

    pred_x_loss = []

    for test_case_layout in counter:
        pred_x = np.zeros(len(E))
        cur_x = []
        for i in range(len(E)):
            if tuple(E[i].tolist()) == test_case_layout:
                pred_x[i] = 1
                cur_x.append(i)
        
        res = counter[test_case_layout]
        pred_x_loss.append((cur_x, res))
    return sorted(pred_x_loss, key=lambda x: x[1], reverse=True)

def MaxPass(E):

    test_case_layouts = [E[i, :].sum() for i in range(len(E))]
    counter = Counter(test_case_layouts)

    pred_x_loss = []

    for test_case_layout in counter:
        pred_x = np.zeros(len(E))
        cur_x = []
        for i in range(len(E)):
            if E[i, :].sum() == test_case_layout:
                pred_x[i] = 1
                cur_x.append(i)
        pred_x_loss.append((cur_x, test_case_layout))
    return sorted(pred_x_loss, key=lambda x: x[1], reverse=True)

def Random(E):
    return [(list(range(len(E))), 1)]


