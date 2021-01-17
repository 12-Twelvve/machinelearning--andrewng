##################################################################################################
##################################################################################################
"""                  ex8- AnomalyDetection and Recommendation system                            """
"""                  AnomalyDetection and Recommendation system                        """
##################################################################################################
# importing the modeules
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
##################################################################################################
"""             AnomalyDetection    """
# In this exercise, you will implement an anomaly detection algorithm to detect anomalous behavior in server computers.
# The features measure the through-put (mb/s) and latency (ms) of response of each server.
data = io.loadmat('./data/ex8data1.mat')
print(data.keys())
X = data['X']
X_val = data['Xval']
y_val = data['yval']
# print(X.shape) 307 * 2


def visdata():
    plt.figure()
    plt.scatter(X.T[0], X.T[1], c='r', s=5)
    plt.xlabel('latecy(ms)')
    plt.ylabel('through-put (mb/s)')
    plt.title('first dataset')
    plt.show()
# visData()
# ######################  1.1 gaussianDistribution
# The function "multivariate_normal" imported from scipy.stats above will be used in place of "multivariateGaussian.m"
# #######    1.2 estimating the parameters for the gausssian
# i.e   =   sigma square  and mean(Mu)


def estimate_gaussian(x):
    mu = np.mean(x, axis=0)           # mean
    var = np.var(x, axis=0, ddof=1)   # variance
    return mu, var


def gaussiandist():
    mu, var = estimate_gaussian(X)
    rv = multivariate_normal(mu, np.diag(var))
    xs, ys = np.mgrid[0:30:0.1, 0:30:0.1]
    pos = np.empty(xs.shape + (2,))
    pos[:, :, 0] = xs
    pos[:, :, 1] = ys

    plt.figure(figsize=(8, 6))
    plt.plot(X.T[0], X.T[1], 'bx', ms=4)
    plt.contour(xs, ys, rv.pdf(pos), 10. ** np.arange(-21, -2, 3))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()

# gaussiandist()
# ###########        1.3  Selecting the threshold, $\epsilon$


def select_threshold(y, p_val):
    best_epsilon, best_f1 = 0, 0
    step_size = (max(p_val) - min(p_val)) / 1000
    for epsilon in np.arange(min(p_val), max(p_val), step_size)[1:]:
        cv_predictions = p_val > epsilon

        tp = np.sum((cv_predictions == 1) & (y == 1))
        fp = np.sum((cv_predictions == 1) & (y == 0))
        fn = np.sum((cv_predictions == 0) & (y == 1))
        # tn = np.sum((cv_predictions == 0) & (y == 0))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_epsilon = epsilon
            best_f1 = f1
    return best_epsilon, best_f1


def selthersold():
    #
    mu, var = estimate_gaussian(X)
    rv = multivariate_normal(mu, np.diag(var))
    xs, ys = np.mgrid[0:30:0.1, 0:30:0.1]
    pos = np.empty(xs.shape + (2,))
    pos[:, :, 0] = xs
    pos[:, :, 1] = ys

    #
    p_val = rv.pdf(X_val)   #what it does????
    best_epsilon, best_f1 = select_threshold(y_val, p_val)
    print("best epsilon =", best_epsilon)
    # best_epsilon = 8.96156768719e-05
    outliers = X[rv.pdf(X) < best_epsilon]   # """ oooops """
    plt.figure(figsize=(8, 6))
    plt.plot(X.T[0], X.T[1], 'bx', ms=4)
    plt.plot(outliers.T[0], outliers.T[1], 'ro', ms=8, mfc='none', mec='r')
    plt.contour(xs, ys, rv.pdf(pos), 10. ** np.arange(-21, -2, 3))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
# your anomaly detection code and circle the anomalies in the plot
selthersold()
# ##############         1.4 High dimensional dataset


def highdimdata():
    data_2_dict = io.loadmat('./data/ex8data2.mat')
    X = data_2_dict['X']
    X_val = data_2_dict['Xval']
    y_val = data_2_dict['yval'].T[0]
    # M = len(X)
    # M_val = len(X_val)
    # parameter
    mu, var = estimate_gaussian(X)
    rv = multivariate_normal(mu, np.diag(var))
    p_val = rv.pdf(X_val)
    best_epsilon, best_f1 = select_threshold(y_val, p_val)
    outliers = X[rv.pdf(X) < best_epsilon]
    print("best epsilon = ", best_epsilon)
    print("best F1 =", best_f1)
    print("number of outliers =", len(outliers))
# highdimdata()