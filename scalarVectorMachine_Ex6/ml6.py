####################################################################################
"""     Machine Learning Online Class Exercise 6 | Support Vector Machines      """
"""     Machine Learning Online Class Exercise 6 | Support Vector Machines      """
####################################################################################
# import modules
import scipy.io as io
from  scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm       #SVM software

####################################################################################
# =============== Part 1: Loading and Visualizing Data ================
#  loading and visualizing the dataset.
print('Loading  Data ...\n')
data = io.loadmat('./data/ex6data1.mat')
X, y = data['X'], data['y']
# print(X.shape)  #51 * 2
# print(y.shape) #51 * 1

#NOT inserting a column of 1's in case SVM software does it for me automatically...
#X =     np.insert(X    ,0,1,axis=1)

#Divide the sample into two: ones with positive classification, one with null classification
pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])


def plotData(pos, neg):
    print('Visualizing data ......')
    # plt.figure(figsize=(10,6))
    plt.plot(pos[:, 0], pos[:, 1], '+', label = 'positive sample')
    plt.plot(neg[:, 0], neg[:, 1], 'ro', label = 'negative sample')
    plt.xlabel('column 1 variable')
    plt.ylabel('column 2 variable')
    plt.legend()
    # plt.grid(True)
    # plt.show()

# plotData(pos, neg)
####################################################################################
####################           Function to draw the SVM boundary        #############
def plotBoundary(my_svm, xmin, xmax, ymin, ymax):
    """
    Function to plot the decision boundary for a trained SVM .It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the SVM classifies that point as True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    xvals = np.linspace(xmin, xmax,100)
    yvals =np.array( np.linspace(ymin, ymax,100))
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(my_svm.predict(np.array([xvals[i],yvals[j]]).reshape(1, -1)))
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
        # UserWarning: (c = 1 and 100)  No contour levels were found within the data range.
    plt.title("Decision Boundary")
    plt.show()

# Run the SVM training (with C = 1) using SVM software. When C = 1, you should find that the SVM puts the decision boundary
# in the gap between the two datasets and misclassifies the data point on the far left .First we make an instance of an SVM with C=1 and 'linear' kernel
def c1():
    linear_svm = svm.SVC(C=1, kernel = 'linear')
    #Now we fit the SVM to our X matrix (no bias unit)
    linear_svm.fit( X, y.flatten() )
    #Now we plot the decision boundary
    plotData(pos, neg)
    plotBoundary(linear_svm,0,4.5,1.5,5)
# c1()

# When C = 100, you should find that the SVM now classifies every single example correctly, but has
# a decision boundary that does not appear to be a natural fit for the data.
def c100():
    linear_svm = svm.SVC(C=100, kernel='linear')
    linear_svm.fit( X, y.flatten() )
    plotData(pos, neg)
    plotBoundary(linear_svm,0,4.5,1.5,5)
# c100()

####################################################################################
###############     1.2 SVM with Gaussian Kernels       ###########################
# Here's how to use this SVM software with a custom kernel:
# http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html
# similarity (x, l(i)) = f(i) = >

def svmGauss():
    def gaussKernel(x1, x2, sigma):
        sigmasquared = np.power(sigma,2)
        return np.exp(-(x1-x2).T.dot(x1-x2)/(2*sigmasquared))
    #
    # x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
    # sim = gaussianKernel(x1, x2, sigma);
    # this value should be about 0.324652
    print(gaussKernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.))
# svmGauss()

####################################################################################
# Now that I've shown I can implement a gaussian Kernel, I will use the of-course built-in gaussian kernel in my SVM software
# because it's certainly more optimized than mine. It is called 'rbf' and instead of dividing by sigmasquared,
# it multiplies by 'gamma'. As long as I set gamma = sigma^(-2), it will work just the same.

################          1.2.2 Example Dataset 2        ############################
def data2():
    print('Loading  Data 2...\n')
    data = io.loadmat('./data/ex6data2.mat')
    X, y = data['X'], data['y']
    #Divide the sample into two: ones with positive classification, one with null classification
    pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
    # plotData(pos, neg)
    # # Train the SVM with the Gaussian kernel on this dataset.
    sigma = 0.1
    gamma = np.power(sigma,-2.)
    gaus_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)
    gaus_svm.fit( X, y.flatten() )
    plotData(pos, neg)
    plotBoundary(gaus_svm,0,1,.4,1.0)

# data2()
####################################################################################
##########      1.2.3 Example Dataset 3             ######################
def data3():
    print('Loading  Data 3...\n')
    data = io.loadmat('./data/ex6data3.mat')
    X, y = data['X'], data['y']
    Xval, yval = data['Xval'], data['yval']
    # Divide the sample into two: ones with positive classification, one with null classification
    pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])
    # plotData(pos, neg)

    # Your task is to use the cross validation set Xval, yval to determine the best C and σ parameter to use.

    # The score() function for a trained SVM takes in X and y to test the score on, and the (float)
    # value returned is "Mean accuracy of self.predict(X) wrt. y"

    Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
    sigmavalues = Cvalues
    best_pair, best_score = (0, 0), 0

    for Cvalue in Cvalues:
        for sigmavalue in sigmavalues:
            gamma = np.power(sigmavalue,-2.)
            gaus_svm = svm.SVC(C=Cvalue, kernel='rbf', gamma=gamma)
            gaus_svm.fit( X, y.flatten() )
            this_score = gaus_svm.score(Xval, yval)
            #print this_score
            if this_score > best_score:
                best_score = this_score
                best_pair = (Cvalue, sigmavalue)

    print ("Best C, sigma pair is (%f, %f) with a score of %f."%(best_pair[0],best_pair[1],best_score))
    # Best C, sigma pair is (0.300000, 0.100000) with a score of 0.965000.
    gaus_svm = svm.SVC(C=best_pair[0], kernel='rbf', gamma = np.power(best_pair[1],-2.))
    gaus_svm.fit( X, y.flatten())
    plotData(pos, neg)
    plotBoundary(gaus_svm,-.5,.3,-.8,.6)
# data3()
####################################################################################
