############################     exercise 4         ##############################################
############################     neural network           ##############################################
# forward and back propagation
# andrew ng ml course in python

###############################################################################################################
""" import the module """
import  os
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import matplotlib.gridspec as gspec
import seaborn as sea


data = io.loadmat('Data/ex4data1.mat')
# handwritten digits. example
print(data.keys())
X , y = data['X'],data['y'].ravel()
print(np.shape(X))       #datasets = 5000 * 400
print(np.shape(y))       #labels = 5000 * 1

M = X.shape[0] # = 5000 samples
N = X.shape[1] # = 400 pixels per sample
L = 26 # = number of nodes in the hidden layer (including bias node)
K = len(np.unique(y)) # = 10 distinct classes for this example.
print(M,N,K)

# add the ones feature in the data
# print(float(X[0,0]))
# suprise but has 0.0 values  idkwhy?
X = np.hstack((np.ones((M, 1)), X))
############# visialize the data    #######################

###########################################################
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# print(sigmoid(12))




# Generate a one-hot encoding, Y, of the target vector, y.

Y = np.zeros((M, K), dtype='uint8')
for i, row in enumerate(Y):
    Y[i, y[i] - 1] = 1
# something like puting ones to every sample  which value is high or something like that


# load the saved neural network parameters (the trained neural network weights).
weights_dict = io.loadmat(os.path.join('Data','ex4weights.mat'))

theta_1 = weights_dict['Theta1'] # theta_1.shape = (25, 401)
theta_2 = weights_dict['Theta2'] # theta_2.shape = (10, 26)

# print(np.shape(theta_1.flatten()))   -> 10025 flatten data
nn_params_saved = np.concatenate((theta_1.flatten(), theta_2.flatten()))
# allthe flatten data


###################################################################################
"""                  neural Network                                          """
###################################################################################

"""      1.2  visualizing the data  and save in the .png file                """
# use only once because we get the idea what is the file looks like
def visualizeData():
    # draw a dim * dim random examples of images from the datasets  and display them in a grid
    # so drop 1 from the X here
    dim  = K
    examples = X[:,1:][np.random.randint(M,size=dim * dim)]
    #radnomly select the size(10*10)sample from the 5000 sample
    fig = plt.figure(figsize= (5,5))
    # exactly idk the size meaning

    gs = gspec.GridSpec(dim ,dim)
    # specing rows and cols

    # make the grid of 10 * 10
    gs.update(bottom =0.01, top = 0.99, left= 0.01, right=0.99, hspace=0.05, wspace=0.05)
    # make space inbetween

    k= 0
    #for the samples iteration or increment

    # run the iteration machine
    for i in range(dim):
        for j in range(dim):
            ax= plt.subplot(gs[i,j])
            ax.axis('off')
    #       no idea about axis
            ax.imshow(examples[k].reshape(int(np.sqrt(N)), int(np.sqrt(N))).T, cmap=plt.get_cmap('Greys'), interpolation='nearest')
            k+= 1

    plt.savefig('showfig.png',dpi= 300)


#######################################################################################
"""                    feedForward and costFunction                                 """

def nn_costFunction(nn_params, X, Y, M, N, L, K):
    theta_1  = nn_params[:(L-1) * (N+1)].reshape(L-1, N+1)
#    0 -10024   i.e 25 * 401 =10025
    theta_2  = nn_params[(L-1) * (N+1):].reshape(K,L)
#    10025-10284  i.e  10 * 26 =260
#    calculate the activation functions inthe second layer .
    a_2 = sigmoid(theta_1.dot(X.T))
#     25 * 401 dot* 401 * 5000  = 25 * 5000
    a_2_b = np.vstack((np.ones(M), a_2))
#     26 * 5000
#     again find thne a3 i.e the output layer for this exercise
#     10 * 26 dot* 26 * 5000
    a_3 = sigmoid(theta_2.dot(a_2_b))
#     output = 10 * 5000
#     Y = m=5000 * 10
#     now calculate the cost function
    cost = 1 / M * np.trace(- Y.dot(np.log(a_3)) - (1 - Y).dot(np.log(1 - a_3)))
    # trace adds diagonally ........................
    # cost = 1 / M * np.trace(- Y.T.dot(np.log(a_3).T) - (1 - Y).T.dot(np.log(1 - a_3).T))
    #same thing
    return cost

# nn_params_saved = get from the weights rolled weight both theta1 and theta2
# X = whole data with bias feature
# Y = labels which is classed i.e. m = high for only desire output only.
# M = X.shape[0] # = 5000 samples
# N = X.shape[1] # = 400 pixels per sample
# L = 26 # = number of nodes in the hidden layer (including bias node)
# K = len(np.unique(y)) # = 10 distinct classes for this example.
# nn_params_saved = 10285
# print(np.size(nn_params_saved))


costSaved = nn_costFunction(nn_params_saved,X, Y, M, N, L, K)
print('cost at parameters (loaded form the file ): %.6f ' %costSaved)
print('the predictaef value should be 0.287629')


####################################################################################################
"""              regularized CostFunction                                           """

# Regularized cost function Add a regularization term to the cost function.Note: Bias parameters are not regularized.
###########################################################################################
"""                not this                         """
#             its so long double coded use bottom one...............................
def nn_cost_function_reg(nn_params, X, Y, M, N, L, K, lam):
    # Unroll the parameter vector.
    theta_1 = nn_params[:(L - 1) * (N + 1)].reshape(L - 1, N + 1)
    theta_2 = nn_params[(L - 1) * (N + 1):].reshape(K, L)
    # same as above ....
    # Calculate activations in the second layer.
    a_2 = sigmoid(theta_1.dot(X.T))
    # Add the second layer's bias node.
    a_2_p = np.vstack((np.ones(M), a_2))
    # Calculate the activation of the third layer.
    a_3 = sigmoid(theta_2.dot(a_2_p))
    regCost = lam / 2 / M * (np.sum(theta_1[:, 1:]**2)+ np.sum(theta_2[:, 1:]**2))
    # Calculate the cost function with the addition of a regularization term.
    cost = 1 / M * np.trace(- Y.dot(np.log(a_3)) - (1 - Y).dot(np.log(1 - a_3))) \
           + regCost
    # \ is for continue ........
    return cost
####################################################################################
# lamda is passed 1.
# cost_saved_reg = nn_cost_function_reg(nn_params_saved, X, Y, M, N, L, K, 1)
# print('Regularized cost at parameters (loaded from ex4weights): %.6f' % cost_saved_reg)

# J(\theta) = \frac{1}{m} \sum_{i=1}^{m}\sum_{k=1}^{K} \left[ - y_k^{(i)} \log \left( \left( h_\theta \left( x^{(i)} \right) \right)_k \right)
#             - \left( 1 - y_k^{(i)} \right) \log \left( 1 - \left( h_\theta \left( x^{(i)} \right) \right)_k \right) \right]
#             + \frac{\lambda}{2 m} \left[ \sum_{j=1}^{25} \sum_{k=1}^{400} \left( \Theta_{j,k}^{(1)} \right)^2
#             + \sum_{j=1}^{10} \sum_{k=1}^{25} \left( \Theta_{j,k}^{(2)} \right)^2 \right]


####################################################################################
"""                 this one                """
# same func with less code
def regCost(nn_params, M, N, L, K, lam):
    theta_1 = nn_params[:(L - 1) * (N + 1)].reshape(L - 1, N + 1)
    theta_2 = nn_params[(L - 1) * (N + 1):].reshape(K, L)
    # why we are unrolling this theta we just need to add them right
    # to avoide the bias unit
    regCost = lam / 2 / M * (np.sum(theta_1[:, 1:]**2)+ np.sum(theta_2[:, 1:]**2))
    # sum of every layer in here there is 3 layers so we got to add (1 - (L-1)) i.e 2 layers
    # and we have to add every rows and cols of squared theta.
    return  regCost
totalCost = regCost(nn_params_saved, M, N, L, K, 1) + costSaved
print('Regularized cost at parameters (loaded from ex4weights): %.6f' % totalCost)
print('(this value should be about 0.383770)')



############################################################################################
"""                 2. Backpropagation                   """
##############################################################################################
# In this part of the exercise, you will implement the backpropagation algorithm
# to compute the gradient for the neural network cost function.
# You will need to update the function nnCostFunction so that
# it returns an appropriate value for grad.
# Once you have computed the gradient, you will be able to train the neural network
# by minimizing the cost function $J(\theta)$ using an advanced optimizer such as scipy's optimize.minimize.
# You will first implement the backpropagation algorithm to compute the gradients
# for the parameters for the (unregularized) neural network. After you have verified that
# your gradient computation for the unregularized case is correct,
# you will implement the gradient for the regularized neural network.


##############      2.1 Sigmoid gradient                ###########################
# i really dont get it why does we calculated that part ??????????????????????
# ?????????????????????????????????????????????????????????????????????????
"""
   Computes the gradient of the sigmoid function evaluated at z. 
   This should work regardless if z is a matrix or a vector. 
   In particular, if z is a vector or matrix, you should return
   the gradient for each element.

   Parameters
   ----------
   z : array_like
       A vector or matrix as input to the sigmoid function. 

   Returns
   --------
   g : array_like
       Gradient of the sigmoid function. Has the same shape as z. 

   Instructions
   ------------
   Compute the gradient of the sigmoid function evaluated at
   each value of z (z can be a matrix, vector or scalar).
   """
# got that ....use for back propagation......> g(z(3)).prime = a(3) .* (1-a(3))
sigmoid_gradient = lambda x: sigmoid(x) * (1 - sigmoid(x))
# ?????????????
# sigmoid = lambda x: 1 / (1 + np.exp(-x))

# When you are done, the following cell call sigmoidGradient on a given vector z.
# Try testing a few values by calling sigmoidGradient(z). For large values (both positive and negative) of z,
# the gradient should be close to 0. When $z = 0$, the gradient should be exactly 0.25.
# Your code should also work with vectors and matrices. For a matrix,
# your function should perform the sigmoid gradient function on every element.


print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:', \
    ', '.join('%.3f' % item for item in sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]: 0.197, 0.235, 0.250, 0.235, 0.197')
# print(sigmoid_gradient(0.4)) # = 0.2406
#######################################################################################################################

# load the data
#randomly initialize the data
# implement forward propagation
# implement backward propagation to compute the partial derivatives
# use gradient checking to comfirm back propagation is working
# use gradient descent or optimal algorithm to minimize the cost function with the weights in theta


#############           2.2 Random initialization           ####################################

# Draw initial values of the neural network parameters from
# a uniform distribution on the open interval $(-\epsilon_{\rm init}, \epsilon_{\rm init})$.

# When training neural networks, it is important to randomly initialize the parameters for symmetry breaking.
# One effective strategy for random initialization is to randomly select values for
# $\Theta^{(l)}$ uniformly in the range $[-\epsilon_{init}, \epsilon_{init}]$.
# You should use $\epsilon_{init} = 0.12$.
# This range of values ensures that the parameters are kept small and makes the learning more efficient.

# Your job is to complete the function randInitializeWeights to initialize the weights for $\Theta$.
# Modify the function by filling in the following code:

# Randomly initialize the weights to small values
# W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
# Note that we give the function an argument for $\epsilon$ with default value epsilon_init = 0.12.
eps_init = 0.12

theta_1_0 = np.random.uniform(-eps_init, eps_init, theta_1.shape)
theta_2_0 = np.random.uniform(-eps_init, eps_init, theta_2.shape)
# initialized.........
nn_params_0 = np.concatenate((theta_1_0.flatten(), theta_2_0.flatten()))
# flatten the data.........

# next way of doing it ................
# making a function .............
# def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
"""
    Randomly initialize the weights of a layer in a neural network.
    Parameters
    ----------
    L_in : int
        Number of incomming connections.
    L_out : int
        Number of outgoing connections. 
    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.
    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.

    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds 
    to the parameters for the bias unit.
    
     # You need to return the following variables correctly 
     # W = np.zeros((L_out, 1 + L_in))
"""
#
# print('Initializing Neural Network Parameters ...')
# initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
# initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# # Unroll parameters
# initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

##################################################################################################################


########################   2.3 backpropagation costFunction and gradient       #############################################

"""                  2.3  backpropagation algorithm                   """
# Backpropagation Expand on the definition of the cost function
# defined above such that both cost and gradient are returned.

# Now, you will implement the backpropagation algorithm. Recall that the intuition behind the backpropagation algorithm
# is as follows.
# Given a training example $(x^{(t)}, y^{(t)})$, we will first run a “forward pass” to compute all the activations
# throughout the network, including the output value of the hypothesis $h_\theta(x)$.
# Then, for each node $j$ in layer $l$, we would like to compute an “error term” $\delta_j^{(l)}$ that measures
# how much that node was “responsible” for any errors in our output.
#
# For an output node, we can directly measure the difference between the network’s activation and the true target value,
# and use that to define $\delta_j^{(3)}$ (since layer 3 is the output layer). For the hidden units,
# you will compute $\delta_j^{(l)}$ based on a weighted average of the error terms of the nodes in layer $(l+1)$.
# In detail, here is the backpropagation algorithm . You should implement steps 1 to 4 in a loop
# that processes one example at a time. Concretely, you should implement a for-loop for t in range(m)
# and place steps 1-4 below inside the for-loop, with the $t^{th}$ iteration performing the calculation on the $t^{th}$
# training example $(x^{(t)}, y^{(t)})$.
# Step 5 will divide the accumulated gradients by $m$ to obtain the gradients for the neural network cost function.
#
# 1. Set the input layer’s values $(a^{(1)})$ to the $t^{th }$training example $x^{(t)}$. Perform a feedforward pass,
#    computing the activations $(z^{(2)}, a^{(2)}, z^{(3)}, a^{(3)})$ for layers 2 and 3. Note that you need to add a +1 term
#    to ensure that the vectors of activations for layers $a^{(1)}$ and $a^{(2)}$ also include the bias unit.
#    In numpy, if a 1 is a column matrix, adding one corresponds to a_1 = np.concatenate([np.ones((m, 1)), a_1], axis=1).
#
# 2. For each output unit $k$ in layer 3 (the output layer), set :
#    $$\delta_k^{(3)} = \left(a_k^{(3)} - y_k \right)$$
#    where $y_k \in \{0, 1\}$ indicates whether the current training example belongs to class $k$ $(y_k = 1)$, or
#    if it belongs to a different class $(y_k = 0)$. You may find logical arrays helpful for this task.
#
# 3. For the hidden layer $l = 2$, set $$ \delta^{(2)} = \left( \Theta^{(2)} \right)^T \delta^{(3)} * g'\left(z^{(2)} \right)$$
#    Note that the symbol $*$ performs element wise multiplication in numpy.
#
# 4. Accumulate the gradient from this example using the following formula. Note that you should skip or remove $\delta_0^{(2)}$.
#    In numpy, removing $\delta_0^{(2)}$ corresponds to delta_2 = delta_2[1:].
#    $$ \Delta^{(l)} = \Delta^{(l)} + \delta^{(l+1)} (a^{(l)})^{(T)} $$
#
# 5. Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients
#    by $\frac{1}{m}$:
#    $$ \frac{\partial}{\partial \Theta_{ij}^{(l)}} J(\Theta) = D_{ij}^{(l)} = \frac{1}{m} \Delta_{ij}^{(l)}$$



def nn_cost_function_grad(nn_params, X, Y, M, N, L, K, lam):
    """Python version of nnCostFunction.m after completing 'Part 2'
    (the cost function is regularized here but not the gradient)."""

    # Unroll the parameter vector.
    theta_1 = nn_params[:(L - 1) * (N + 1)].reshape(L - 1, N + 1)
    theta_2 = nn_params[(L - 1) * (N + 1):].reshape(K, L)

    """Feedforward pass."""
    # Calculate activations in the second layer (as well as z_2, which is needed below).
    z_2 = theta_1.dot(X.T)
    a_2_p = np.vstack((np.ones(M), sigmoid(z_2)))

    # Calculate the activation of the third layer.
    a_3 = sigmoid(theta_2.dot(a_2_p))

    # Calculate the cost function with the addition of a regularization term.
    cost = 1 / M * np.trace(- Y.dot(np.log(a_3)) - (1 - Y).dot(np.log(1 - a_3))) \
           + lam / 2 / M * (np.sum(theta_1[:, 1:] * theta_1[:, 1:]) + np.sum(theta_2[:, 1:] * theta_2[:, 1:]))

    """Backpropagation (use the chain rule)."""
    # Calculate the gradient for parameters in the third layer.
    grad_theta_2 = 1 / M * (a_3 - Y.T).dot(a_2_p.T)
    # ????????????????????/ whtis that?????????
    # grad_theta = D(l)...
    # i think it is  D(l) = 1/m capitalDelta(l) + lamda  * thata(l+1) if   j != 0
    #                D(l) = 1/m capitalDelta(l)                       if   j = 0  ........<><>

    #  capitalDelta(l) = delta(l+1) dot*(a(l))
    # i.e third layer ......
    # delta(3)..
    # delta(l+1=L) = a(L) - y

    # Calculate the gradient for parameters in the second layer.

    theta_delta = theta_2_0[:, 1:].T.dot(a_3 - Y.T)
    s_g_z_2 = sigmoid_gradient(z_2)
    # delta(2) =  (theta(2)).T dot(delta(3)) .* sigmoid_grad(z(2))

    grad_theta_1 = 1 / M * np.array([[np.sum(theta_delta[p] * s_g_z_2[p] * X.T[q])
                                      for q in range(N + 1)] for p in range(L - 1)])
    # grad_theta = D(l)..
    # D(l) :=
    # capitalDelta(l) := capitalDelta(l) + a(l) dot*(delta(3))

    # Roll the two gradient vectors into a single vector and return.
    return cost, np.concatenate((grad_theta_1.flatten(), grad_theta_2.flatten()))


#############################################################################################################
"""                 2.4 Gradient checking           """



# After you have implemented the backpropagation algorithm, we will proceed to run gradient checking on your implementation.
# The gradient check will allow you to increase your confidence that your code is computing the gradients correctly.

# In your neural network, you are minimizing the cost function $J(\Theta)$.
# To perform gradient checking on your parameters, you can imagine “unrolling” the parameters $\Theta^{(1)}$, $\Theta^{(2)}$
# into a long vector $\theta$. By doing so, you can think of the cost function being $J(\Theta)$
# instead and use the following gradient checking procedure.
#
# Suppose you have a function $f_i(\theta)$ that purportedly computes $\frac{\partial}{\partial \theta_i} J(\theta)$;
# you’d like to check if $f_i$ is outputting correct derivative values.
# $$ \text{Let }
# \theta^{(i+)} =
#    \theta + \begin{bmatrix} 0 \\ 0 \\ \vdots \\ \epsilon \\ \vdots \\ 0 \end{bmatrix} \quad
# \text{and}
# \quad \theta^{(i-)} =
#    \theta - \begin{bmatrix} 0 \\ 0 \\ \vdots \\ \epsilon \\ \vdots \\ 0 \end{bmatrix} $$
#
# So, $\theta^{(i+)}$ is the same as $\theta$, except its $i^{th}$ element has been incremented by $\epsilon$.
# Similarly, $\theta^{(i−)}$ is the corresponding vector with the $i^{th}$ element decreased by $\epsilon$.
# You can now numerically verify $f_i(\theta)$’s correctness by checking, for each $i$, that:
# $$ f_i\left( \theta \right) \approx \frac{J\left( \theta^{(i+)}\right) - J\left( \theta^{(i-)} \right)}{2\epsilon} $$
#
# The degree to which these two values should approximate each other will depend on the details of $J$.
# But assuming $\epsilon = 10^{-4}$, you’ll usually find that the left- and right-hand sides of the above will agree
# to at least 4 significant digits (and often many more).
#
# We have implemented the function to compute the numerical gradient for you in computeNumericalGradienit.....littebit different........
#
# In the next cell we will run the provided function checkNNGradients which will create a small neural network
# and dataset that will be used for checking your gradients. If your backpropagation implementation is correct,
# you should see a relative difference that is less than 1e-9.
# **Practical Tip**: When performing gradient checking, it is much more efficient to use a small neural network
# with a relatively small number of input units and hidden units, thus having a relatively small number of parameters.
# Each dimension of $\theta$ requires two evaluations of the cost function and this can be expensive.
# In the function `checkNNGradients`, our code creates a small random model and dataset which is used with
# `computeNumericalGradient` for gradient checking. Furthermore, after you are confident
# that your gradient computations are correct, you should turn off gradient checking before running your learning algorithm.
# Practical Tip: Gradient checking works for any function where you are computing the cost and the gradient.
# Concretely, you can use the same `computeNumericalGradient` function to check if your gradient implementations
# for the other exercises are correct too (e.g., logistic regression’s cost function).

def gradChecking():
    # Compare the gradient as computed from the analytical expression above
    # to the gradient computed numerically based on estimates of the cost function
    # (without regularization) at two nearby points in the neural network parameter space.
    def compute_numerical_gradient(indicies_to_check, eps, nn_params, X, Y, M, N, L, K):
        """Python version of computeNumericalGradient.m."""
        numerical_grad = np.zeros(len(indicies_to_check))
        unit_vector = np.zeros(nn_params.shape)
        k = 0
        for i in indicies_to_check:
            unit_vector[i] = eps
            loss_1 = nn_costFunction(nn_params + unit_vector, X, Y, M, N, L, K)
            loss_2 = nn_costFunction(nn_params - unit_vector, X, Y, M, N, L, K)
            numerical_grad[k] =  (loss_1 - loss_2) / (2 * eps)
            unit_vector[i] = 0
            k += 1
        return numerical_grad

    # Because checking the gradient vector would be somewhat computationally intensive,
    # we perform the comparison of analytical and numerical gradient
    # calculations only on a random subset of gradient vector elements.

    num_to_check = 20

    indicies_to_check = np.arange(len(nn_params_0))
    np.random.shuffle(indicies_to_check)
    indicies_to_check = indicies_to_check[:num_to_check]

    # The following few lines calculate the difference between
    # the analytical and numerical computation of random gradient vector elements.

    """Together with the next few lines, this is a Python check similar to the check in checkNNGradients.m."""

    _, analytical_grad_random_subset = nn_cost_function_grad(nn_params_0, X, Y, M, N, L, K, 0)
    analytical_grad_random_subset = analytical_grad_random_subset[indicies_to_check]
    # takes ~20sec to run on my laptop.

    eps = 1e-4
    # numerical_grad_random_subset = compute_numerical_gradient(indicies_to_check, eps, nn_params_0, X, Y, M, N, L, K)
    # diff = 2 * (analytical_grad_random_subset - numerical_grad_random_subset) / (analytical_grad_random_subset + numerical_grad_random_subset)

    # This line demonstrates that the largest difference is $O( 10\times\epsilon)$.
    # print(np.sum(abs(diff) < 1e-4) == len(diff))
    # True

###################################################################################################################
'''                       we done checking                      '''

#################################      regularized NeuralNetwork     ############################################



# After you have successfully implemented the backpropagation algorithm, you will add regularization to the gradient.
# To account for regularization, it turns out that you can add this as an additional term
# after computing the gradients using backpropagation.
#
# Specifically, after you have computed $\Delta_{ij}^{(l)}$ using backpropagation, you should add regularization using
# $$ \begin{align} &amp; \frac{\partial}{\partial \Theta_{ij}^{(l)}} J(\Theta) =
#           D_{ij}^{(l)} =
#           \frac{1}{m} \Delta_{ij}^{(l)} &amp; \qquad \text{for }                                                j = 0 \\
#           &amp; \frac{\partial}{\partial \Theta_{ij}^{(l)}} J(\Theta) =
#           D_{ij}^{(l)} =
#           \frac{1}{m} \Delta_{ij}^{(l)} + \frac{\lambda}{m} \Theta_{ij}^{(l)} &amp; \qquad \text{for }          j \ge 1
#           \end{align} $$
#
# Note that you should not be regularizing the first column of $\Theta^{(l)}$ which is used for the bias term.
# Furthermore, in the parameters $\Theta_{ij}^{(l)}$, $i$ is indexed starting from 1,
# and $j$ is indexed starting from 0. Thus,
# $$ \Theta^{(l)} =
#       \begin{bmatrix}
#           \Theta_{1,0}^{(i)} &amp; \Theta_{1,1}^{(l)} &amp; \cdots \\
#           \Theta_{2,0}^{(i)} &amp; \Theta_{2,1}^{(l)} &amp; \cdots \\
#           \vdots &amp; ~ &amp; \ddots
#       \end{bmatrix} $$

cost_history = []  # Use this array to log costs calculated in the call to scipy's "minimize" below.


def nn_cost_function_grad_reg(nn_params, X, Y, M, N, L, K, lam):
    """Python version of nnCostFunction.m after completing 'Part 3'."""

    # Unroll the parameter vector.
    theta_1 = nn_params[:(L - 1) * (N + 1)].reshape(L - 1, N + 1)
    theta_2 = nn_params[(L - 1) * (N + 1):].reshape(K, L)

    """Feedforward pass."""
    # Calculate activations in the second layer (as well as z_2, which is needed below).
    z_2 = theta_1.dot(X.T)
    a_2_p = np.vstack((np.ones(M), sigmoid(z_2)))

    # Calculate the activation of the third layer.
    a_3 = sigmoid(theta_2.dot(a_2_p))

    # Calculate the cost function with the addition of a regularization term.
    cost = 1 / M * np.trace(- Y.dot(np.log(a_3)) - (1 - Y).dot(np.log(1 - a_3))) \
           + lam / 2 / M * (np.sum(theta_1[:, 1:] * theta_1[:, 1:]) + np.sum(theta_2[:, 1:] * theta_2[:, 1:]))

    """Backpropagation (use the chain rule)."""
    # Calculate the gradient for parameters in the third layer.
    grad_theta_2 = 1 / M * (a_3 - Y.T).dot(a_2_p.T) \
                   + lam / M * np.hstack(
        (np.zeros(K).reshape(-1, 1), theta_2[:, 1:]))  # this is the theta_2 grad reg term

    # Calculate the gradient for parameters in the second layer.
    theta_delta = theta_2_0[:, 1:].T.dot(a_3 - Y.T)
    s_g_z_2 = sigmoid_gradient(z_2)
    grad_theta_1 = 1 / M * np.array([[np.sum(theta_delta[p] * s_g_z_2[p] * X.T[q])
                                      for q in range(N + 1)] for p in range(L - 1)]) \
                   + lam / M * np.hstack(
        (np.zeros(L - 1).reshape(-1, 1), theta_1[:, 1:]))  # this is the theta_1 grad reg term

    # Roll the two gradient vectors into a single vector.
    grad = np.concatenate((grad_theta_1.flatten(), grad_theta_2.flatten()))

    cost_history.append(cost)
    return cost, grad

#####################################################################################################################

###########################      2.6  learning parameter        ###################################################
#minimize the num_params

# For a regularization parameter set to 1 and random initial neural network parameter values as drawn above, train the network.
lam = 1
cost_history = []
nn_params_learned=np.array([])
# minimize the params................................................
def minPara():
    res = minimize(nn_cost_function_grad_reg, nn_params_0,
                   method='L-BFGS-B', args=(X, Y, M, N, L, K, lam), jac=True, options={'maxiter': 100})

    nn_params_learned = res['x']
    np.savetxt('nn_params_learned.txt', nn_params_learned)
    print('Cost at best-fit theta: %.3f' % res['fun'])
    # Cost at best-fit theta: 0.670   but my output is 0.689  need to be corrected....
#
# plot the cost and steps................................................
def plotCostvsSteps():
    num_steps = 100
    plt.figure()
    plt.scatter(np.arange(num_steps), np.array(cost_history[:num_steps]), c='k', marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.xlim(-num_steps * 0.05, num_steps * 1.05)
    plt.ylim(0, max(cost_history[:num_steps]) * 1.05)
    plt.savefig('cost_vs_steps_learned.png', dpi=300)

# load from the saved params.........................................................

nn_params_learned = np.loadtxt('nn_params_learned.txt')

# find the accuracy of the learned parameters........................................
def nn_accuracy(nn_params, X, y):
    theta_1 = nn_params[:(L - 1) * (N + 1)].reshape(L - 1, N + 1)
    theta_2 = nn_params[(L - 1) * (N + 1):].reshape(K, L)

    z_2 = theta_1.dot(X.T)
    a_2 = sigmoid(z_2)
    a_2_p = np.vstack((np.ones(M), a_2))
    a_3 = sigmoid(theta_2.dot(a_2_p))

    return np.sum((a_3.argmax(axis=0) + 1) == y) / M


# Accuracy of the neural net trained here:

print(nn_accuracy(nn_params_learned, X, y))

# 0.93640000000000001

print(nn_accuracy(np.concatenate((theta_1.flatten(), theta_2.flatten())), X, y))
# 0.97519999999999996


################################################################################################################


"""                                            the End                                                      """