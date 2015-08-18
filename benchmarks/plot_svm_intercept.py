#
#  File to reproduce the SVM illustrations in the paper
#  A Coordinate Descent Primal-Dual Algorithm with Large
#  Step Size and Possibly Non Separable Functions by
#  O. Fercoq and P. Bianchi
#


import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, sparse, io

from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import logsumexp, safe_sparse_dot

from lightning.classification import SDCAClassifier


class Callback(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.Y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y)
        self.obj = []
        self.dual_obj = []
        self.gap = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, clf, t=None):
        test_time = time.clock()

        if hasattr(clf, "_finalize_coef"):
            clf._finalize_coef()

        Y = self.Y
        n_samples = Y.shape[0]
        w = clf.coef_
        XTw = safe_sparse_dot(X, w.T)

        # recover intercept
        valp = np.sort(1. - XTw[Y==1])
        valn = np.sort(1. + XTw[Y==-1])
        val = np.sort(np.append(valp, valn))

        ind_l = 0
        ind_u = val.shape[0]-1
        while ind_l < ind_u-1:
            ind_t = np.floor(ind_l/2.+ind_u/2.)
            w0_t = val[ind_t];
            derivative = np.sum(valn+w0_t>=0) - np.sum(valp-w0_t>=0);
            if derivative < 0:
                ind_l = ind_t
            else:
                ind_u = ind_t
        if derivative == 0:
            w0 = val[ind_t]
        else:  # necessarily derivative == -1
            w0 = val[ind_u]
        
        alpha = clf.alpha * (1.-clf.l1_ratio)
        C = clf.C[Y]

        loss = (alpha * 0.5 * np.sum(w ** 2)
                + np.dot(C.T,np.maximum(0., 1. - Y*XTw - Y*w0)))

        scale = C / alpha
        dual_coef = clf.dual_coef_.T * scale
        for kk in range(30):
            dual_coef = dual_coef - np.mean(dual_coef)
            dual_coef[Y>0] = np.maximum(0., np.minimum(scale[Y>0], dual_coef[Y>0]))
            dual_coef[Y<0] = np.maximum(-scale[Y<0], np.minimum(0, dual_coef[Y<0]))
        w2 = (X.T * dual_coef)
        dual_loss = - 0.5 * (w2.T.dot(w2)) * alpha + Y.T.dot(dual_coef) * alpha - 100 * np.abs(np.sum(dual_coef))   # to check

        #print loss.squeeze(), dual_loss.squeeze(), loss.squeeze() - dual_loss.squeeze(), w0

        self.obj.append(loss.squeeze())
        self.dual_obj.append(dual_loss.squeeze())
        self.gap.append(loss.squeeze() - dual_loss.squeeze())
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)

for dataset in [1]:
    if dataset == 0:
        iris = load_iris()
        X, y = iris.data, iris.target
    elif dataset == 1:
        #X, y = make_classification(n_samples=100,
        #                           n_features=300,
        #                           n_classes=3,
        #                           n_informative=50,
        #                           random_state=0)
        dataset_name = 'rcv1'

        data = io.loadmat('/cal/homes/ofercoq/scikit_learn_data/rcv1_train.binary.mat')
        #y, X = svm_read_problem('/cal/homes/ofercoq/datasets/rcv1_train.binary')
        X = data['X'].astype(np.float)
        y = data['y'].astype(np.float).ravel()
        eps = 1e-4 # the smaller it is the longer is the path
    elif dataset >= 2 and dataset <=4:
        data = io.loadmat('/cal/homes/ofercoq/datasets/orange_kdd_preprocessed.mat')
        X = data['X'].astype(np.float)

        mean = np.array(X.mean(axis=0))
        X.data[:] = X.data[:]**2
        X2mean = np.array(X.mean(axis=0))

        X = data['X'].astype(np.float)

        std = np.sqrt(X2mean - mean**2).T.squeeze()
        std[std<1e-5] = np.inf  # We eliminate these variables
        inv_std = sparse.spdiags(1. / std, 0, std.shape[0], std.shape[0])

        X = X * inv_std / np.sqrt(X.shape[0])
        # We do not substract the mean to keep the sparsity structure of the matrix

        X = sparse.csr_matrix(X)

        if dataset == 2:
            dataset_name = 'kdd_orange_appentency'
            y = np.loadtxt('/cal/homes/ofercoq/datasets/orange_large_train_appetency.labels')
        elif dataset == 3:
            dataset_name = 'kdd_orange_churn'
            y = np.loadtxt('/cal/homes/ofercoq/datasets/orange_large_train_churn.labels')
        elif dataset == 4:
            dataset_name = 'kdd_orange_upselling'
            y = np.loadtxt('/cal/homes/ofercoq/datasets/orange_large_train_upselling.labels')


    #C = 4
    #alpha = 1. / C / X.shape[0]
    alpha = 1. / X.shape[0]
    C = np.zeros(3)
    C[1] = 1. / X.shape[0]
    C[-1] = 1. / X.shape[0] * sum(y == 1) / sum(y == -1)

    times = [[],[],[]]
    obj = [[],[],[]]
    dual_obj = [[],[],[]]
    gap = [[],[],[]]

    for intercept in [0,1]:  # [0, 1, 2]:
        # intercept == 0: sdca
        # intercept == 1: primal-dual cd
        # intercept == 2: necoara's constrained cd
        print intercept
        clf = SDCAClassifier(loss="hinge", alpha=alpha, C=C,
                       max_iter=200, n_calls=X.shape[0], random_state=0,
                       l1_ratio=0, verbose=0, tol=0, 
                       intercept=intercept)

        cb = Callback(X, y)
        clf.callback = cb

        clf.fit(X.tocsr(), y)

        times[intercept] = cb.times
        obj[intercept] = cb.obj
        dual_obj[intercept] = cb.dual_obj
        gap[intercept] = cb.gap

    plt.figure()
    plt.plot(times[0], gap[0], '.-',
             times[1], gap[1], '-',
             times[2], gap[2], '--', linewidth=2)

    plt.yscale("log")
    plt.xlabel("CPU time (s)")
    plt.ylabel("Duality gap")
    plt.legend(["SDCA (Shalev-Shwartz & Zhang)",
                "Primal-dual coordinate descent"])
#                "RCD (Necoara & Patrascu)"])

    plt.savefig('%s_comp_svm_intercept.pdf' % dataset_name)

plt.show()
