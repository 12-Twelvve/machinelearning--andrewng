##################################################################################################
##################################################################################################
"""                         ex7- unsupervised machine learning                                  """
"""                  K-means Clustering and Principal Component Analysis                        """
##################################################################################################
#importing the modeules
import numpy as np
import scipy as sp
import scipy.io
from scipy.optimize import minimize
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib import scimath #


##################################################################################################
##################   2. pca ( principle component Analysis )      ################################
#######################  2.1 Example Dataset
# get the data2
data = scipy.io.loadmat('./data/ex7data1.mat')
X = data['X']
M = len(X)
def data1show():
    plt.figure(figsize=(6, 6))
    plt.plot(X.T[0], X.T[1], 'bo', mfc='none', mec='b', ms=8)
    plt.xlim(0.5, 6.5)
    plt.ylim(2, 8)
    # plt.savefig('fig4.jpg')
    plt.show()

###################    2.2 implementing PCA
# feature Normalize
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sig = np.std(X, axis=0)
    X_norm = (X - mu) / sig
    return X_norm, mu, sig

def pca(X_norm):
    Sigma = 1 / M * X_norm.T.dot(X_norm)
    U, S, V = la.svd(Sigma)
    return U, S
def implementingPCA():
    X_norm, mu, sig = feature_normalize(X)
    U, S = pca(X_norm)
    # print ('top eigenvector:', U.T[0])
    U_0_end_pts = np.vstack((mu, mu + 1.5 * S[0] * U.T[0]))
    U_1_end_pts = np.vstack((mu, mu + 1.5 * S[1] * U.T[1]))
    plt.figure(figsize=(6, 6))
    plt.plot(X.T[0], X.T[1], 'bo', mfc='none', mec='b', ms=8)
    plt.plot(U_0_end_pts.T[0], U_0_end_pts.T[1], 'k-', lw=2)
    plt.plot(U_1_end_pts.T[0], U_1_end_pts.T[1], 'k-', lw=2)
    plt.xlim(0.5, 6.5)
    plt.ylim(2, 8)
    # plt.savefig('fig5.jpg')
    # plt.show()
    return  X_norm , U


####################       2.3 dimensionReduction

#####        2.3.1 Projecting the data onto the principal components
def project_data(X, U, K):
    return X.dot(U[:, :K])
#####       2.3.2 Reconstructing an approximation of the data
def recover_data(Z, U, K):
    return Z.dot(U[:, :K].T)

def defReduction():
    X_norm, U = implementingPCA()
    Z = project_data(X_norm, U, 1)
    X_rec = recover_data(Z, U, 1)
    # print (project_data(X_norm, U, 1)[0])
    # print (X_rec[0])
    # 2.3.3 Visualizing the projections
    plt.figure(figsize=(6, 6))
    plt.plot(X_norm.T[0], X_norm.T[1], 'bo', mfc='none', mec='b', ms=8)
    plt.plot(X_rec.T[0], X_rec.T[1], 'ro', mfc='none', mec='r', ms=8)
    for (x, y), (x_rec, y_rec) in zip(X_norm, X_rec):
        plt.plot([x, x_rec], [y, y_rec], 'k--', lw=1)
    plt.xlim(-4, 3)
    plt.ylim(-4, 3)
    # plt.savefig('fig6.jpg')
    plt.show()


##########       2.4  Face  image dataset
data_faces_dict = scipy.io.loadmat('data/ex7faces.mat')
X = data_faces_dict['X']
M, N = X.shape
def grid_plot(X, N, dim, file_name):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(dim, dim)
    gs.update(bottom=0.01, top=0.99, left=0.01, right=0.99,
              hspace=0.05, wspace=0.05)
    k = 0
    for i in range(dim):
        for j in range(dim):
            ax = plt.subplot(gs[i, j])
            ax.axis('off')
            ax.imshow(X[k+100].reshape(int(scimath.sqrt(N)), int(scimath.sqrt(N))).T,
                      cmap=plt.get_cmap('Greys'),  # vmin=-1, vmax=1,
                      interpolation='nearest')  # , alpha = 1.0)
            k += 1

    # plt.savefig('' + file_name, dpi=300)
    plt.show()
    pass

# grid_plot(-X, N, 10, 'fig7.jpg')

##############       2.4.1 PCA on faces
X_norm, mu, sig = feature_normalize(X)
U, S = pca(X_norm)
# grid_plot(-U.T, N, 6, 'fig8.jpg')

##################      2.4.2 Dimensionality reduction
Z = project_data(X_norm, U, 100)
X_rec = recover_data(Z, U, 100)
grid_plot(-X_rec, N, 10, 'fig9.jpg')

#################      2.5 PCA for visualization
M = len(X)
rand_indices = np.arange(M)
sp.random.shuffle(rand_indices)
X_sub = X[rand_indices[:1000]]

X_sub_norm, mu, sig = feature_normalize(X_sub)
U, S = pca(X_sub_norm)
Z = project_data(X_sub_norm, U, 2)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.scatter(-Z.T[0], -Z.T[1],  s=10)
# plt.savefig('fig_11.jpg', dpi=300)
plt.show()


