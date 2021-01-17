##################################################################################################
##################################################################################################
"""                         ex7- unsupervised machine learning                                  """
"""                  K-means Clustering and Principal Component Analysis                        """
##################################################################################################
#importing the modeules
import numpy as np
import numpy.linalg as nlinalg #for    =>  svd()
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import scipy.io as io
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import axes3d   #interesting
# from bqplot import pyplot as bqp  #not installled
import seaborn as sns

# seaborn setiing   > no idea
sns.set_context('notebook')
sns.set_style('white')
"""                                     1. k-means Clustering                                   """
##################################################################################################
#  1.1 implementing k-means
# loading ist data and implementing k-means
def firstData():
    data1 = io.loadmat('./data/ex7data2.mat')
    X1 = data1['X']   #300  and  300 * 2
    # M = len(X1)
    # data vis
    plt.scatter(X1[:,0], X1[:,1], s=40,  cmap=plt.cm.prism)
    plt.show()
    # kmeans
    km1 = KMeans(3)   #where 3 = K
    km1.fit(X1)
    # clustring
    plt.scatter(X1[:,0], X1[:,1], s=40, c=km1.labels_, cmap=plt.cm.prism)
    plt.title('K-Means Clustering Results with K=3')
    plt.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);
    plt.show()

# Image compression with K-means
def imageComp():
    img = plt.imread('data/bird_small.png') #image is read by imread of plt
    img_shape= img.shape # (128, 128, 3)
    print(img_shape)
    A = img/255
    AA = A.reshape(128 *128,3) #(16384, 3)
    km2 = KMeans(16)
    km2.fit(AA)
    # cluster centers
    B = km2.cluster_centers_[km2.labels_].reshape(img_shape[0], img_shape[1], 3)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))

    ax1.imshow(img)
    ax1.set_title('Original')
    ax2.imshow(B*255)
    ax2.set_title('Compressed, with 16 colors')
    plt.show()

    for ax in fig.axes:
        ax.axis('off')
# pca on dataset example -2
def pcaData2():
    data = io.loadmat('./data/ex7data1.mat')
    X = data['X']
    # print(X[:5,:])
    # preprocessing the data by sklearn preprocesssing standardScalar
    scaler= StandardScaler()
    scaler.fit(X)
    # print(scaler.transform(X[:5,:]))
    # print(scaler.mean_)
    U, S, V =slinalg.svd(scaler.transform(X).T)
    print(U)
    plt.scatter(X[:, 0], X[:, 1], s=30, edgecolors='b', facecolors='None', linewidth=1);
    # setting aspect ratio to 'equal' in order to show orthogonality of principal components in the plot
    plt.gca().set_aspect('equal')
    plt.quiver(scaler.mean_[0], scaler.mean_[1], U[0, 0], U[0, 1], scale=S[1], color='r')
    plt.quiver(scaler.mean_[0], scaler.mean_[1], U[1, 0], U[1, 1], scale=S[0], color='r');
    plt.show()

pcaData2()