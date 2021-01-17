##########################################################################################
"""Programming Exercise 5 - Regularized Linear Regression and Bias v.s. Variance"""
#############################################################################################
"""                                     @jack12                                           """
"""                           completed but error values                                  """
# In this exercise, you will implement regularized linear regression and use it to study models
# with different bias-variance properties.
#############################################################################################

##################      import things to do task                ###################
import numpy as np
import scipy.io as io
from scipy.optimize import minimize
import matplotlib.gridspec as gspec
import matplotlib.pyplot as plt
import seaborn as sns

##################      make the data > ready to drill                 ##############

# load the datasets
data = io.loadmat('./data/ex5data1.mat')
# print(data.keys())
X, y, Xtest, ytest, Xval, yval = data['X'], data['y'], data['Xtest'], data['ytest'], data['Xval'], data['yval']
# this is train data =12 1
# this is new test value = 21 1
# this is cross validation value= 21 1
M = len(X)
# length of the train data ie samples = 12
M_val = len(Xval)
# length of the cv data ie samples = 21
M_test = len(Xtest)
# length of the test data ie samples = 12

# Add a column of ones to the data matrix that allows us to treat the intercept parameter as a feature.
X = np.hstack((np.ones((M,1)),X))
Xval = np.hstack((np.ones((M_val,1)),Xval))
Xtest = np.hstack((np.ones((M_test,1)),Xtest))

##############################################################################################
##################      1. regullarized linear regression               ###################
#################      1.1 visualizing the Datasets               ###################
def visualizeData():
    plt.plot(X.T[1], y, 'ro', ms=10, mec='k', mew=1) #ro = red_ and _ shapeOf_ o , #ms  = size , mec= border , mew =?
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)');
    plt.show()
# visualizeData()
########################################################################################

##############      1.2 Regularized linear regression cost function      ###############
lin_theta_0 = np.ones((2, 1))    # not do the ones(2)...
lam = 1

def linearRegCost_Grad(theta, X, y, reg):
    m = y.size
    h = X.dot(theta.reshape(-1, 1))
    grad = (1 / m) * (X.T.dot(h - y)) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    J = (1 / (2 * m)) * np.sum(np.square(h - y)) + (reg / (2 * m)) * np.sum(np.square(theta[1:]))
    return J, grad.flatten()

# cost, grad = linearRegCost_Grad(lin_theta_0, X, y, lam)
# print('Cost at theta = [1, 1]: %.3f' %cost)
# print('(this value should be about 303.993)')
# print('Gradient at theta = [1, 1]:', grad)
# print('(this value should be about [-15.303016, 598.250744])')

##############################################################################################################
##############      1.4 Fitting linear regression      ###############
# we set regularization parameter $\lambda$ to zero.Because our current implementation of linear regression is
# trying to fit a 2-dimensional $\theta$,  regularization willnot be incredibly helpful for a $\theta$ of such low dimension.
#
lam = 0
initial_theta = np.ones((2,1))
# initial_theta = np.array([[15], [15]])

def trainLinearReg(theta, X, y, lam):
    # theta = np.zeros((X.shape[1],1))
    # For some reason the minimize() function does not converge when using
    # zeros as initial theta.
    # res = minimize(linearRegCostFunction, theta, args=(X, y, reg), method=None, jac=lrgradientReg, )
    res = minimize(linearRegCost_Grad, theta, method='L-BFGS-B', args=(X, y, lam), jac=True, options={'maxiter': 5000})
    return (res)

# fit = trainLinearReg(initial_theta, X, y, 0)  # reg = 0
# print(fit)
##########################################################################################################
###############     plot the fit line           #################################
#  Plot fit over the data
def plotFitData():
    plt.plot(X[:,1], y, 'ro', ms=10, mew=1.5) #mec = 'k'
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    # plt.plot(X[:,1], np.dot(X, fit['x']), '--', lw=2);
    plt.show()

#####################################################################################

#########################################################################################################################
"""                         2.  bais vs variance                                    """
###############         2.1 learning curves                    #################################
def learningCurve(lin_theta_0, X, y, Xval, yval, lam=0):
    m = y.size
    error_train, error_val = np.zeros(m - 1), np.zeros(m - 1)
    # ====================== YOUR CODE HERE ======================
    for i in range(2, m+1):
        res_train = trainLinearReg(lin_theta_0, X[:i, :], y[:i], 0)
        theta = res_train['x']
        error_train[i-2], _= linearRegCost_Grad(theta, X[:i, :], y[:i], 0)
        error_val[i - 2], _= linearRegCost_Grad(theta, Xval, yval, 0)
    # =============================================================
    return error_train, error_val

lin_theta_0 = np.array([1., 1.])
# error_train, error_val = learningCurve(lin_theta_0, X, y, Xval, yval)

###############    2.2 plotting the learning Curve    #############################

def plotLearningCurve():
    plt.figure()
    # plt.plot(np.arange(2, M + 1), error_train, 'b-', label='Train')
    # plt.plot(np.arange(2, M + 1), error_val, 'g-', label='Cross validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.xlim(0, 13)
    plt.ylim(-10, 150)
    plt.title('Learning curve for linear regression')
    plt.legend(numpoints=1, loc=0)
    plt.show()
# plotLearningCurve()
##################################################################################################################

###########################        3. ploynomial Regreesion     ##################################################
"""                            3. Polynomial Regression                                """
####################        3.1  polyfeatures and normalize the datasets  upto --p   #######################
# deleting the ones first column
X = X[:,1:]
Xval = Xval[:,1:]
Xtest = Xtest[:,1:]

##########              polyfeatures                ######
def polyFeatures(X_poly, p):
    X_l = X_poly[:,0:]
    for i in range(2, p+1):
        X_poly = np.hstack((X_poly, X_l**i ))
    return X_poly
##########             normalize the features                ######
def feature_normalize(X_poly):
    X_norm = np.zeros_like(X_poly)
    mu = np.mean(X_poly)   #mean
    sigma = np.std(X_poly)  #standard deviation
    for i, feature in enumerate(X_poly[:]):
        feature = (feature - mu) / sigma
        X_norm[i] = feature
    return X_norm, mu, sigma

def val_or_test_feature_normalize(X, mu, sigma):
    X_poly = np.zeros_like(X)
    for i, feature in enumerate(X[:]):
        feature = (feature - mu) / sigma
        X_poly[i] = feature
    return X_poly
######################################################
poly_deg = 8
X_polyNorm, mu, sigma = feature_normalize(polyFeatures(X, poly_deg))
X_polyNorm_val = val_or_test_feature_normalize(polyFeatures(Xval, poly_deg), mu, sigma)
X_polyNorm_test = val_or_test_feature_normalize(polyFeatures(Xtest, poly_deg), mu, sigma)
##############################################################################################
X_polyNorm = np.hstack((np.ones((len(X_polyNorm),1)), X_polyNorm))
X_polyNorm_val = np.hstack((np.ones((len(X_polyNorm_val),1)), X_polyNorm_val))
X_polyNorm_test = np.hstack((np.ones((len(X_polyNorm_test),1)), X_polyNorm_test))
####################        3.2 learning polynomial regression   #######################
def learningPolyreg(lam=0):
    lin_theta_0 = np.ones(poly_deg+1)
    res = trainLinearReg(lin_theta_0, X_polyNorm, y, lam)
    print (res)
    theta = res['x']

    plt.figure()
    plt.plot(theta, 'ko', ms=8)
    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.show()
    # plt.savefig('poly_reg_costs_lam_0.png', dpi=300)

    num_pts = 100
    x_pts = np.linspace(-110, 50, num_pts)
    y_pts = theta[0] * np.ones(num_pts)
    for i in range(1, poly_deg+1):
        y_pts += theta[i] * (x_pts**i - mu) / sigma

    plt.figure()
    plt.plot(X_polyNorm, y, 'rx', ms=8)
    plt.plot(x_pts, y_pts, 'b--')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial regression fit ' r'($\lambda=%d$)' % lam)
    # plt.xlim(-110, 50)
    # plt.ylim(-250, 100)
    plt.show()
    # plt.savefig('fig4.png', dpi=300)

    error_train, error_val = learningCurve(lin_theta_0, X_polyNorm, y, X_polyNorm_val, yval, lam)

    plt.figure()
    plt.plot(np.arange(2, M+1), error_train, 'b-', label='Train')
    plt.plot(np.arange(2, M+1), error_val, 'g-', label='Cross validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Polynomial regression learning curve ' r'($\lambda=%d$)' % lam)
    # plt.xlim(0, 13)
    # plt.ylim(-10, 180)
    plt.legend(numpoints=1, loc=9)
    plt.show()
    # # plt.savefig('fig5.png', dpi=300)
####################        3.2 adjusting the regularization parameter   #######################
# applying lam = 0 , 1 and 100  and check the result

# lam =0
# lam= 1
lam = 100
# learningPolyreg(lam)
##############################################################################################
############     3.3 Selecting lam using a cross validation set    #######################
def cvLamserror():
    lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    lin_theta_0 = np.ones(poly_deg+1)

    errors_train = []
    errors_val = []
    for lam in lams:
        res_train = trainLinearReg(lin_theta_0, X_polyNorm, y, lam)
        theta = res_train['x']
        errors_train.append(linearRegCost_Grad(theta, X_polyNorm, y, 0))
        errors_val.append(linearRegCost_Grad(theta, X_polyNorm_val, yval, 0))

    plt.figure()
    plt.plot(lams, errors_train, 'b-', label='Train')
    plt.plot(lams, errors_val, 'g-', label='Cross validation')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Error')
    plt.xlim(0, 10)
    plt.ylim(0, 20)
    plt.legend(numpoints=1, loc=9)

############     3.4 Computing test set error    #######################
def compTestset():
    lam = 3
    lin_theta_0 = np.ones(poly_deg+1)

    res_train = trainLinearReg(lin_theta_0, X_polyNorm, y, lam)
    theta = res_train['x']
    error_val = linearRegCost_Grad(theta, X_polyNorm_val, yval, 0)
    print ("validation error =", error_val)
    error_test = linearRegCost_Grad(theta, X_polyNorm_test, ytest, 0)
    print ("test error =", error_test)

# validation error = 6.70622046004
# test error = 6.25034833059

#####################################       errror    ################################################