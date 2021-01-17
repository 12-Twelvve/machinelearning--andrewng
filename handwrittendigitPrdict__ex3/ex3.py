###############################################################
            # hand written digit predict
            # by one vs all logistic regression and neural network
            # By @jack12
###############################################################

# from __future__ import division
# this should be in the first line of the file
# dont kknow what it does ???????????????/
# soo i commented it /////////
##################################
# used for manipulating directory paths
import os
import numpy as np
from scipy.optimize import minimize
import scipy.io # used to load matlab-formatted .m files.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# statistical data visualization using seaborn
import seaborn as sns

# bqlot and IPython.display are used to create an interactive plot in the final section below
# bqplot is to plot in the jupyter notebook
# import bqplot as bq
from IPython.display import display

from sklearn.linear_model import LogisticRegression
# used following the first section of the notebook

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("notebook")

def sns_reset():
    """Call this function to toggle back to the sns plotting environment from the matplotlib environment."""
    sns.reset_defaults()
    sns.set_style("white")
    sns.set_style("ticks")
    sns.set_context("notebook")

# Call these three functions at the top of the notebook to allow toggling between sns and matplotlib
# environments while maintaining a uniform plot style throughout.
sns.reset_orig()
sns_reset()
plt.ion()
#################### load the data   ####################
# load the Data from the mat file
data_dict = scipy.io.loadmat(os.path.join('Data','ex3data1.mat'))
print(data_dict.keys())
X = data_dict['X']
y = data_dict['y'].ravel()

M = X.shape[0]   # 0 gives the rows
N = X.shape[1]   # 1 gives the column
# m, n = X.shape
num_labels = len(np.unique(y)) # = 10

# Add a column of ones to the data matrix that allows us to treat the intercept parameter as a feature.
X = np.hstack((np.ones((M, 1)), X))

#################   Define the sigmoid function.     ################################
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# def square(x):
#     return x * x
# and
# sqr_fun = lambda x: x * x
# are same
# called lambda function
###################### end function #################

###################   multiclass classification ###############################################

#########################   1.2 visualizing the data   ######################
# Draw dim * dim random examples of images from the dataset and
# display them in a grid. Note: Drop column of ones from X here.
dim = 10
# M = 5000
examples = X[:, 1:][np.random.randint(M, size=dim * dim)]
print(np.shape(examples))
# 100 * 400
######### displayData function #########
"""Python version of displayData.m."""
def displayData(examples,dim):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(dim, dim)
    gs.update(bottom=0.01, top=0.99, left=0.01, right=0.99,
              hspace=0.05, wspace=0.05)
    k = 0
    for i in range(dim):
        for j in range(dim):
            ax = plt.subplot(gs[i, j])
            # ????
            ax.axis('off')
            # N = 400
            ax.imshow(examples[k].reshape(int(np.sqrt(N)), int(np.sqrt(N))).T,
                cmap=plt.get_cmap('Greys'), interpolation='nearest')
            k += 1
    plt.savefig('fig_1.png', dpi=300)
############# end function ###########

# displayData(examples,dim)


''' ###################        one vs all           #####################'''
# set the regularization parameter.
lam = 1
# Inialize all model parameters to zero.
# theta   or weight
theta_0 = np.zeros(N  + 1)
# Define the cost function for regularized logistic regression.
# (The exact same cost function is used in Exercise 2.3.)
cost_history = []  # Use this array to log costs calculated in the call to scipy's "minimize" below.
############  cost function regLogReggression#############3
def cost_function(theta, X, y, M, lam):
    """Python version of lrCostFunction.m."""
    # vactor implimentation
    global cost_history
    cost = 1 / M * (- y.dot(np.log(sigmoid(theta.dot(X.T)))) - (1 - y).dot(np.log(1 - sigmoid(theta.dot(X.T))))) \
           + lam / 2 / M * np.sum(theta[1:] ** 2)
    grad = 1 / M * (sigmoid(theta.dot(X.T)) - y).dot(X) + lam / M * np.hstack(([0], theta[1:]))
    cost_history.append(cost)
    return cost, grad

# testing the cost
# cost_0, _ = cost_function(theta_0, X, y, M, lam)
# print('Cost at initial theta (zeros): %.3f' % cost_0)
# Cost at initial theta (zeros): 160.394


########################### train the 1's classifier #######################################333
# Use a one-hot encoding of the target variable.s
def oneClassifier():
        y_1_hot = (y == 1).astype('float')
        # basicallly it does that : getting the ones value only for optimizing the
        # 1 data and train the data for 1 - value only  from 0-9
        # here we can take 2 ,3 4 or 5 or others
        # to make one -vs - other

        ######## Run the optimization.
        cost_history = []
        """This call to scipy's "minimize" is a Python version of the Octave call to "fminunc"."""
        res = minimize(cost_function, theta_0, method='L-BFGS-B', args=(X, y_1_hot, M, lam), jac=True)
        # check the result
        print(res.keys())
        # dict_keys(['fun', 'jac', 'nfev', 'njev', 'nit', 'status', 'message', 'x', 'success', 'hess_inv'])
        # fun has the cost for best fit data
        theta = res['x']
        print('Cost at best-fit theta: %.3f' % res['fun'])
        # Cost at best-fit theta: 0.027

        num_steps = len(cost_history)
        # how many times does the minimizer (optimation algorithm or method)
        # run to obtain the best fit data
        plt.figure()
        # it helps to plot the data into the fig
        plt.scatter(np.arange(num_steps), cost_history, c='k', marker='o')
        plt.xlabel('Steps')
        plt.ylabel('Cost')
        plt.xlim(-num_steps * 0.05, num_steps * 1.05)
        plt.ylim(0, max(cost_history) * 1.05)
        plt.savefig('cost_vs_steps_train_1s.png', dpi=300)
#########################################################################################
"""   train all the classifiers    """

# Run the same optimization as above for all 10 classes of digits.
def oneVsAll(theta_0, X, y, M, num_labels, lam):
    """Python version of oneVsAll.m."""
    # N =400
    # num_labels = 10
    all_theta = np.zeros((num_labels, N + 1))
    for i in range(1, num_labels + 1): # note that 0s are labeled with a y-value of 10 in this dataset.
        y_i_hot = (y == i).astype(np.float64)
        cost_history = [] # reset cost_history for each call to cost_function (even though cost_history not used here)
        res = minimize(cost_function, theta_0, method='L-BFGS-B', args=(X, y_i_hot, M, lam), jac=True)
        all_theta[i - 1] = res['x']
    #      1 - axis from 1-10
    #      x is the theta values

    return all_theta
# all_theta = oneVsAll(theta_0, X, y, M, num_labels, lam)

##################-- 1.4.1 One-vs-all prediction -------- ##############
def prediction(all_theta, X):
    """Python version of predictOneVsAll.m."""
    return sigmoid(all_theta.dot(X.T)).T.argmax(axis=1) + 1
def accuracy(all_theta, X, y, M):
    return np.mean(prediction(all_theta, X) == y)
# accuracy
# print('Train Accuracy default(one vs all) : %.4f' % accuracy(all_theta, X, y, M))
# Train Accuracy: 0.9446

###############################################################################################
'''   Multi-class classification with scikit-learn  '''
# The 'LogisticRegression' classifier is generated here with the regularization parameter,
# $C=1/\lambda$, set to '1'. Setting 'penalty' to 'l2' specifies the use of L2 regularization.
# Setting 'multi_class' to 'ovr' ('one-vs-rest') specifies the use of the cross-entropy cost function. (n
# These settings also happen to be the default parameters for the LogisticRegression classifier,
# so 'clf = LogisticRegression()' is equivalent to the first line below.
# Note: The data matrix, $X$, should have dimensions num_samples x num_features and
# should omit the initial column of ones. The target vector, $y$, does not need to be one-hot encoded.

def mulclsSckit():
    clf = LogisticRegression(C=1, penalty='l2', multi_class='ovr')
    clf.fit(X[:, 1:], y)
    print('Train Accuracy (sklearn-1): %.4f' % clf.score(X[:, 1:], y))
    # Train Accuracy (sklearn): 0.9440


    # The latter LogisticRegression classifier uses the 'liblinear' library to solve the optimization problem.
    # The following version uses the 'lbfgs' solver.
    # Notice that the resulting accuracy is identical to that in section 1.4.1 to 4 significant digits.


    clf = LogisticRegression(C=1, penalty='l2', multi_class='ovr', solver='lbfgs')
    clf.fit(X[:, 1:], y)
    print('Train Accuracy (sklearn -2): %.4f' % clf.score(X[:, 1:], y))
    # Train Accuracy (sklearn): 0.9446
    # little bit  page of warning ..............................................................s

# mulclsSckit()

#######################################################################################################
'''                      neural networks                                          '''

#just implimenting the neural networks of  3 layers  in which theta1 and theta2 is already given
#

#     2.1   neural   networks
# Load the saved neural network parameters.
weights_dict = scipy.io.loadmat('Data/ex3weights.mat')
# print(weights_dict.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'Theta1', 'Theta2'])
theta_1 = weights_dict['Theta1']
theta_2 = weights_dict['Theta2']
# print(np.shape(theta_1))
# 25 *  401
# print(np.shape(theta_2))
# 10 * 26

# 2.2 Feedforward propagation and prediction
# Calculate the activations of the hidden layer based on the data of the input layer.
a_2 = sigmoid(theta_1.dot(X.T))
a_2 = np.vstack((np.ones(M), a_2))
# print(np.shape(a_2))
# 26 * 5000
# Calculate the activations of the output layer based on the activations of the hidden layer.
a_3 = sigmoid(theta_2.dot(a_2))
# Make predictions and estimate accuracy.
# a3 = 10 * 5000
# print(np.shape(a_3))
def nn_prediction(a_3):
    """Python version of predict.m"""
    return a_3.argmax(axis=0) + 1
# it does that anmong the ten rows it gives the max row num.indexing from 0;
def nn_accuracy(a_3):
    return np.mean(nn_prediction(a_3) == y)
print('Training set accuracy: %.3f' % nn_accuracy(a_3))
# Training set accuracy: 0.975




#######################################################################################################
"""                              bqplot-----NOT FOUND--------                                      """

# Use bqplot to display a sequence of randomly selected images from the dataset and the predicted classification.

# nn_predictions = nn_prediction(a_3)

# This following code block plots a single example from the dataset.
# train_ex_num = np.random.randint(0, M)
# train_ex = X[train_ex_num][1:]
# train_ex = train_ex.reshape((20, 20)).T[::-1].flatten()
# train_ex += abs(train_ex.min())
# train_ex /= train_ex.max()
# bqplot isnot installed...............................??????????????????????????????????????????????
# xs = bq.LinearScale()
# ys = bq.LinearScale()
# x_vals, y_vals = np.meshgrid(np.arange(20), np.arange(20))
# scatt_1 = bq.Scatter(x=x_vals.ravel(), y=y_vals.ravel(), default_opacities=[1.0],
#     scales={'x': xs, 'y': ys}, marker='square', default_colors=['white'], stroke_width=6.7)
# scatt_2 = bq.Scatter(x=x_vals.ravel(), y=y_vals.ravel(), default_opacities=train_ex.tolist(),
#     scales={'x': xs, 'y': ys}, marker='square', default_colors=['black'], stroke_width=6.7)
# fig = bq.Figure(marks=[scatt_1, scatt_2], min_width=400, min_height=400, preserve_aspect=True,
#     title='Dataset label: %d; NN prediction: %d' % (y[train_ex_num] % 10, nn_predictions[train_ex_num] % 10))
# display(fig)

# The following code block can be executed multiple times to update the above plot with other examples.
# train_ex_num = np.random.randint(0, M)
# train_ex = X[train_ex_num][1:]
# train_ex = train_ex.reshape((20, 20)).T[::-1].flatten()
# train_ex += abs(train_ex.min())
# train_ex /= train_ex.max()
#
# scatt_2.default_opacities = train_ex.tolist()
# fig.title = 'Dataset label: %d; NN prediction: %d' \
#     % (y[train_ex_num] % 10, nn_predictions[train_ex_num] % 10)


#############################################################################################################
"""----------------------------------------the end--------------------------------------------------------"""
#############################################################################################################