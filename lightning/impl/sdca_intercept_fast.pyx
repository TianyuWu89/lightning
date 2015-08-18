# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Olivier Fercoq from Mathieu Blondel's sdca
# License: BSD

import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG

from libc.math cimport fabs

from lightning.impl.dataset_fast cimport RowDataset


cdef void _add_l2(double* data,
             int* indices,
             int n_nz,
             double* w,
             double update,
             double* regul) nogil:

    cdef int j, jj
    cdef double delta, w_old

    for jj in xrange(n_nz):
        j = indices[jj]
        delta = update * data[jj]
        w_old = w[j]
        w[j] += delta
        regul[0] += delta * (2 * w_old + delta)


cdef inline double _truncate(double v,
                             double sigma) nogil:
    if v > sigma:
        return v - sigma
    elif v < -sigma:
        return v + sigma
    else:
        return 0


cdef void _add_elastic(double* data,
                  int* indices,
                  int n_nz,
                  double*w,
                  double* v,
                  double update,
                  double* regul,
                  double sigma)nogil :

    cdef int j, jj
    cdef double delta, w_old, v_old

    for jj in xrange(n_nz):
        j = indices[jj]
        delta = update * data[jj]
        v_old = v[j]
        w_old = w[j]
        v[j] += delta
        w[j] = _truncate(v[j], sigma)
        regul[0] -= v_old * w_old
        regul[0] += v[j] * w[j]


cdef _sqnorms(RowDataset X,
              np.ndarray[double, ndim=1, mode='c'] sqnorms):

    cdef int n_samples = X.get_n_samples()
    cdef int i, j
    cdef double dot

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for i in xrange(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        dot = 0
        for jj in xrange(n_nz):
            dot += data[jj] * data[jj]
        sqnorms[i] = dot


cdef double _pred(double* data,
                  int* indices,
                  int n_nz,
                  double* w)nogil :

    cdef int j, jj
    cdef double dot = 0

    for jj in xrange(n_nz):
        j = indices[jj]
        dot += w[j] * data[jj]

    return dot


cdef void _solve_subproblem(double*data,
                            int* indices,
                            int n_nz,
                            double y,
                            double* w,
                            double* v,
                            double* dcoef,
                            double* multiplier,
                            double* ydm,  # y.T dcoef and y.T multiplier 
                            int n_samples,
                            int loss_func,
                            double sqnorm,
                            double scale,
                            double balance,
                            double sigma,
                            double gamma,
                            double* primal,
                            double* dual,
                            double* regul):

    cdef double pred, dcoef_old, residual, error, loss, update
    cdef double multiplier_old, multiplier_update, d_stepsize,
    cdef double inv_d_stepsize, mult_stepsize, stepsize_factor

    dcoef_old = dcoef[0]
    multiplier_old = multiplier[0]

    if y>0:
        stepsize_factor = 10.
    else:
        stepsize_factor = 10. * balance

    mult_stepsize = sqnorm * stepsize_factor  # is it the best?
    inv_d_stepsize = (sqnorm + mult_stepsize) / 0.95

    # i-th element of the projection of 
    # mutiplier + mult_stepsize * dcoef on y
    multiplier_update = (ydm[1] + mult_stepsize * ydm[0]) / n_samples
    multiplier_update -= multiplier_old

    pred = _pred(data, indices, n_nz, w)

    if loss_func == 0:  # square loss
        residual = pred - y
        loss = 0.5 * residual * residual
        update = -(dcoef_old + residual) / (1 + inv_d_stepsize * scale)
        dual[0] += update * (y - dcoef_old - 0.5 * update)

    elif loss_func == 1:  # absolute loss
        residual = y - pred
        loss = fabs(residual)
        update = residual / (inv_d_stepsize * scale) + dcoef_old
        update = min(1.0, update)
        update = max(-1.0, update)
        update -= dcoef_old
        dual[0] += y * update

    elif loss_func == 2:  # hinge loss
        error = 1 - y * pred
        loss = max(0.0, error)
        update = (dcoef_old * y + (error - (multiplier_old + 2. *  multiplier_update) * y)
                                  / (inv_d_stepsize * scale))
        update = min(1.0, update)
        update = max(0.0, update)
        update = y * update - dcoef_old
        dual[0] += y * update

    elif loss_func == 3:  # smooth hinge loss
        error = 1 - y * pred

        if error < 0:
            loss = 0
        elif error > gamma:
            loss = error - 0.5 * gamma
        else:
            loss = 0.5 / gamma * error * error

        update = (error - gamma * dcoef_old * y) / (inv_d_stepsize * scale + gamma)
        update += dcoef_old * y
        update = min(1.0, update)
        update = max(0.0, update)
        update = y * update - dcoef_old
        dual[0] += y * update
        dual[0] -= gamma * dcoef_old * update
        dual[0] -= 0.5 * gamma * update * update

    elif loss_func == 4:  # squared hinge loss
        # Update is the same as squared loss but with a truncation.
        residual = pred - y
        update = -(dcoef_old + residual) / (1 + inv_d_stepsize * scale)
        if (dcoef_old + update) * y < 0:
            update = -dcoef_old

        error = 1 - y * pred
        if error >= 0:
            loss = residual * residual

        dual[0] += ((y - dcoef_old) * update - 0.5 * update * update) * scale

    # Use accumulated loss rather than true primal objective value, which is
    # expensive to compute.
    primal[0] += loss * scale

    if update != 0:
        dcoef[0] += update
        if sigma > 0:
            _add_elastic(data, indices, n_nz, w, v, update * scale, regul,
                         sigma)
        else:
            _add_l2(data, indices, n_nz, w, update * scale, regul)
        ydm[0] += update * scale

    if multiplier_update != 0:
        multiplier[0] += multiplier_update
        ydm[1] += multiplier_update


def _prox_sdca_intercept_fit(self,
                   RowDataset X,
                   np.ndarray[double, ndim=1]y,
                   np.ndarray[double, ndim=1]coef,
                   np.ndarray[double, ndim=1]dual_coef,
                   double alpha1,
                   double alpha2,
                   np.ndarray[double, ndim=1] C,
                   int loss_func,
                   double gamma,
                   int max_iter,
                   double tol,
                   callback,
                   int n_calls,
                   int verbose,
                   rng):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables
    cdef double sigma, scale, primal, dual, regul, gap
    cdef int it, ii, i
    cdef int has_callback = callback is not None
    cdef LONG t

    # Pre-compute square norms.
    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
    sqnorms = np.zeros(n_samples, dtype=np.float64)
    _sqnorms(X, sqnorms)

    # Pointers
    cdef double* w = <double*>coef.data
    cdef double* dcoef = <double*>dual_coef.data

    cdef np.ndarray[double, ndim=1] v_data
    v_data = np.zeros(n_features, dtype=np.float64)
    cdef double* v = <double*>v_data.data

    cdef np.ndarray[double, ndim=1] multiplier_data
    multiplier_data = np.zeros(n_samples, dtype=np.float64)
    cdef double* multiplier = <double*>multiplier_data.data

    cdef np.ndarray[double, ndim=1] ydm_data
    ydm_data = np.zeros(2, dtype=np.float64)
    cdef double* ydm = <double*>ydm_data.data
    ydm[0] = np.sum(dual_coef)

    cdef np.ndarray[int, ndim=1] sindices
    sindices = np.arange(n_samples, dtype=np.int32)

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    if alpha1 > 0:  # Elastic-net case
        sigma = alpha1 / alpha2
    else:  # L2-only case
        sigma = 0

    scale = 1. / (alpha2 * n_samples)

    balance = abs(sum(y[y<0])) / sum(y[y>0])

    dual = 0
    regul = 0

    t = 0
    for it in xrange(max_iter):
        primal = 0

        rng.shuffle(sindices)

        for ii in xrange(n_samples):

            i = sindices[ii]

            if sqnorms[i] == 0:
                continue

            # Retrieve row.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            if y[i] == 1:
                scale = C[1] / alpha2
            else:
                scale = C[2] / alpha2  # C[-1]

            _solve_subproblem(data, indices, n_nz, y[i], w, v, dcoef + i,
                              multiplier + i, ydm, n_samples,
                              loss_func, sqnorms[i], scale, balance, sigma, gamma,
                              &primal, &dual, &regul)

            if has_callback and t % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    break

            t += 1

        # end for ii in xrange(n_samples)

        gap = (primal - dual + abs(ydm[0])*100.) * alpha2 + alpha2 * regul
        # gap = max(0, gap)

        if verbose:
            print "iter", it + 1, gap, primal * alpha2 + alpha2 * regul
            print ydm[0], ydm[1]

        #if gap <= tol:  # this is not the good gap
        #    if verbose:
        #        print "Converged"
        #    break

    # for it in xrange(max_iter)

    for i in xrange(n_samples):
        if y[i] == 1:
            scale = C[1] / alpha2
        else:
            scale = C[2] / alpha2  # C[-1]
        dcoef[i] *= scale


cdef void _solve_subproblem_necoara(double* data1,
                               int* indices1,
                               int n_nz1,
                               double* data2,
                               int* indices2,
                               int n_nz2,
                               double y1,
                               double y2,
                               double* w,
                               double* v,
                               double* dcoef1,
                               double* dcoef2,
                               int loss_func,
                               double sqnorm,  # scales are included in sqnorm
                               double scale1,
                               double scale2,
                               double sigma,
                               double gamma,
                               double* primal,
                               double* dual,
                               double* regul) nogil:

    cdef double pred1, pred2, dcoef1_old, dcoef2_old, residual, error1, error2, loss,
    cdef double update1, update2

    dcoef1_old = dcoef1[0]
    dcoef2_old = dcoef2[0]

    pred1 = _pred(data1, indices1, n_nz1, w)
    pred2 = _pred(data2, indices2, n_nz2, w)

    if loss_func == 2:  # hinge loss
        error1 = 1 - y1 * pred1
        error2 = 1 - y2 * pred2
        loss = max(0.0, error1) * scale1 + max(0.0, error2) * scale2
        update1 = dcoef1_old + (y1 * error1 - y2 * error2 * scale2 / scale1) / sqnorm
        # print y1, update1, error1, error2, error1 - y1 * y2 * error2
        update1 = min(1.0, y1 * update1)
        update1 = max(0.0, update1)
        if y1 == y2:
            update1 = max(y1 * (dcoef1_old + dcoef2_old) - scale2 / scale1, update1)
            update1 = min(y1 * (dcoef1_old + dcoef2_old), update1)
        elif y1 == -y2:
            update1 = max(y1 * (dcoef1_old + dcoef2_old), update1)
            update1 = min(y1 * (dcoef1_old + dcoef2_old) + scale2 / scale1, update1)
        else:
            update1 = 1e10  # this should never happen if y in {-1, 1}

        update1 = y1 * update1 - dcoef1_old
        update2 = - update1
        dual[0] += y1 * update1 * scale1 + y2 * update2 * scale2

    # Use accumulated loss rather than true primal objective value, which is
    # expensive to compute.
    primal[0] += loss

    if update1 != 0:
        # print "up", update1
        dcoef1[0] += update1
        _add_l2(data1, indices1, n_nz1, w, update1 * scale1, regul)
        dcoef2[0] += update2
        _add_l2(data2, indices2, n_nz2, w, update2 * scale2, regul)


def necoara_sdca(self,
                 RowDataset X,
                 np.ndarray[double, ndim=1] y,
                 np.ndarray[double, ndim=1] coef,
                 np.ndarray[double, ndim=1] dual_coef,
                 double alpha1,
                 double alpha2,
                 np.ndarray[double, ndim=1] C,
                 int loss_func,
                 double gamma,
                 int max_iter,
                 double tol,
                 callback,
                 int n_calls,
                 int verbose,
                 rng):
    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables
    cdef double sigma, scale, primal, dual, regul, gap
    cdef int it, ii, i1, i2
    cdef int has_callback = callback is not None
    cdef LONG t

    # Pre-compute square norms.
    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
    sqnorms = np.zeros(n_samples, dtype=np.float64)
    _sqnorms(X, sqnorms)

    # Pointers
    cdef double* w = <double*>coef.data
    cdef double* dcoef = <double*>dual_coef.data

    cdef np.ndarray[double, ndim=1] v_data
    v_data = np.zeros(n_features, dtype=np.float64)
    cdef double* v = <double*>v_data.data

    cdef np.ndarray[double, ndim=1] multiplier_data
    multiplier_data = np.zeros(n_samples, dtype=np.float64)
    cdef double* multiplier = <double*>multiplier_data.data

    cdef np.ndarray[double, ndim=1] ydm_data
    ydm_data = np.zeros(2, dtype=np.float64)
    cdef double* ydm = <double*>ydm_data.data
    ydm[0] = np.sum(dual_coef)

    cdef np.ndarray[int, ndim=1] sindices
    sindices = np.arange(n_samples, dtype=np.int32)

    # Data pointers.
    cdef double* data1
    cdef int* indices1
    cdef int n_nz1
    cdef double* data2
    cdef int* indices2
    cdef int n_nz2


    if alpha1 > 0:  # Elastic-net case
        sigma = alpha1 / alpha2
    else:  # L2-only case
        sigma = 0

    scale = 1. / (alpha2 * n_samples)

    dual = 0
    regul = 0

    t = 0
    for it in xrange(max_iter):
        primal = 0

        rng.shuffle(sindices)

        for ii in xrange(n_samples/2):

            i1 = sindices[2 * ii]
            i2 = sindices[2 * ii + 1]
            
            if y[i1] == 1:
                scale1 = C[1] / alpha2
            else:
                scale1 = C[2] / alpha2  # C[-1]
            if y[i2] == 1:
                scale2 = C[1] / alpha2
            else:
                scale2 = C[2] / alpha2  # C[-1]

            # Retrieve rows.
            X.get_row_ptr(i1, &indices1, &data1, &n_nz1)
            X.get_row_ptr(i2, &indices2, &data2, &n_nz2)

            # print "dual", dcoef[i1]
            _solve_subproblem_necoara(data1, indices1, n_nz1, data2, indices2, n_nz2,
                                      y[i1], y[i2], w, v, dcoef + i1,
                                      dcoef + i2,
                                      loss_func, sqnorms[i1] * scale1 + sqnorms[i2] * scale2,
                                      scale1, scale2, sigma,
                                      gamma, &primal, &dual, &regul)
            # print "dual", dcoef[i1]

            if has_callback and t % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    break

            t += 1

        # end for ii in xrange(n_samples)

        gap = (primal - dual + abs(ydm[0])*100.) * alpha2 + alpha2 * regul
        # gap = max(0, gap)

        if verbose:
            print "iter", it + 1, gap, primal * alpha2 + alpha2 * regul
            print ydm[0], ydm[1]

        #if gap <= tol:
        #    if verbose:
        #        print "Converged"
        #    break

    # for it in xrange(max_iter)

    for i in xrange(n_samples):
        if y[i] == 1:
            scale = C[1] / alpha2
        else:
            scale = C[2] / alpha2  # C[-1]
        dcoef[i] *= scale
