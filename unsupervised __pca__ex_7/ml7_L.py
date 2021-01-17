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
from numpy.lib import scimath
# from bqplot import pyplot

##################################################################################################

##################      1 k-means clustering            #########################################
####################### 1.1 implementing k-means
# get the data2
data2 = scipy.io.loadmat('./data/ex7data2.mat')
X= data2['X'] #300 * 2
######   1.1.1 finding the closest centroid
def findClosetCentroid_v1(X, centroids):
    centroidIndices = []
    for example in X:
        distances = []
        for centroid in centroids:
            distances.append(scimath.sqrt((example[0] - centroid[0]) ** 2 + (example[1] - centroid[1]) ** 2))
        centroidIndices.append(np.array(distances).argmin())
        return np.array(centroidIndices)
def findClosestCentroids_v2(X, centroids):
    distances = np.array([scimath.sqrt((X.T[0] - centroid[0])**2 + (X.T[1] - centroid[1])**2) for centroid in centroids])
    return distances.argmin(axis=0)
# timeit findClosetCentroid_v1(X, centroids)
# timeit findClosetCentroid_v2(X, centroids) #faster
find_closest_centroids = findClosestCentroids_v2
centroids = np.array([[3, 3], [6, 2], [8, 5]]) #say
centroid_indices = find_closest_centroids(X, centroids)


####        1.1.2 Computing centroid means
def compute_means(X, centroid_indices, K):
    centroids = []
    for k in range(K):
        centroids.append(np.mean(X[centroid_indices == k], axis=0))
    return np.array(centroids)
K=3
# centroids = compute_means(X, centroid_indices, K)

##################      1.2 K-means on example dataset
def run_k_means(X, K, centroids, iterations):
    centroids_history = [centroids]
    for i in range(iterations):
        centroid_indices = find_closest_centroids(X, centroids)
        centroids = compute_means(X, centroid_indices, K)
        centroids_history.append(centroids)
    return centroids, centroid_indices, centroids_history
centroids = np.array([[3, 3], [6, 2], [8, 5]])
iterations = 10
centroids, centroid_indices, centroids_history = run_k_means(X, K, centroids, iterations)

###########
def initialize_viz(fig_num, X, M, centroids_history, colors):
    iteration_number = 0
    fig = plt.figure(fig_num)
    plt.title('Iteration number ' + str(iteration_number))
    for k, color in enumerate(colors):
        X_cluster = X[centroid_indices == k]
        plt.plot(X_cluster.T[0], X_cluster.T[1], color + 'o') #, default_size=16, default_opacities=[0.5] * M
    centroid_snap_T = centroids_history[0].T
    centroids_dict = {}
    for k, color in enumerate(colors):
        centroids_dict[color] = np.array([centroid_snap_T[0][k], centroid_snap_T[1][k]])
        plt.plot(centroids_dict[color].T[0].reshape(-1), centroids_dict[color].T[1].reshape(-1), color + '+')
    plt.show()
    return iteration_number, centroids_dict

#########
def update_viz(fig_num, centroids_history, colors, iteration_number):
    iteration_number += 1
    if iteration_number >= iterations: iteration_number = 0
    plt.figure(fig_num)
    plt.title('Iteration number ' + str(iteration_number))
    centroid_snap_T = centroids_history[iteration_number].T
    centroids_dict = {}
    for k, color in enumerate(colors):
        centroids_dict[color] = \
            np.vstack((centroids_dict[color], np.array([centroid_snap_T[0][k], centroid_snap_T[1][k]])))
        plt.plot(centroids_dict[color].T[0], centroids_dict[color].T[1], 'k-')
        plt.plot(centroids_dict[color].T[0], centroids_dict[color].T[1], color + '+')
    plt.show()
    return iteration_number, centroids_dict

colors = ['r', 'g', 'b']
fig_num = 1
M = len(X)
# iteration_number, centroids_dict = initialize_viz(fig_num, X, M, centroids_history, colors)
# iteration_number, centroids_dict = update_viz(fig_num, centroids_history, colors, iteration_number)

###########         1.3 Random initialization
def run_k_means_rand_init(X, K, iterations):
    rand_indices = np.arange(len(X))
    sp.random.shuffle(rand_indices)
    centroids = X[rand_indices][:K]
    centroids_history = [centroids]
    for i in range(iterations):
        centroid_indices = find_closest_centroids(X, centroids)
        centroids = compute_means(X, centroid_indices, K)
        centroids_history.append(centroids)
    return centroids, centroid_indices, centroids_history

# centroids, centroid_indices, centroids_history = run_k_means_rand_init(X, K, iterations)
# fig_num = 2
# iteration_number, centroids_dict = initialize_viz(fig_num, X, M, centroids_history, colors)
# iteration_number, centroids_dict = update_viz(fig_num, centroids_history, colors, iteration_number)


###########     1.4 Image compression with K-means
######     1.4.1 K-means on pixels
A = plt.imread('data/bird_small.png')
A = A.reshape(-1, 3)
A = A.astype('float') / 255.
K = 16
iterations = 10
centroids, centroid_indices, centroids_history = run_k_means_rand_init(A, K, iterations)
centroid_indices = find_closest_centroids(A, centroids)
A_recon = centroids[centroid_indices]
A_recon = A_recon.reshape(-1, 128, 3)
# plt.imsave('bird_small_recon.png', A_recon)


#########       2.5 PCA for visualization
X = A.reshape(-1, 3)
M = len(X)
rand_indices = np.arange(M)
sp.random.shuffle(rand_indices)
centroid_indices_sub = centroid_indices[rand_indices[:1000]]
X_sub = X[rand_indices[:1000]]
colors = plt.cm.rainbow(np.linspace(0, 1, 16))
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_sub[:, 0], X_sub[:, 1], X_sub[:, 2], c=colors[centroid_indices_sub], s=10,
          edgecolor=colors[centroid_indices_sub])
ax.set_title('All pixels plotted in 3D. Color shows centroid memberships.')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
plt.savefig('fig_10.jpg', dpi=300)
plt.show()

