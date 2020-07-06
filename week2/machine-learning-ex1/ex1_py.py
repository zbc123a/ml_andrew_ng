#%%
import numpy as np

# Identity Matrix
def warmUpExercise():
    A =np.eye(5)
    return A

# %%

#load data
import numpy as np
wdir = 'C:\\Users\\Beichen\\Desktop\\ML_NG\\week2\\machine-learning-ex1\\ex1\\ex1data1.txt'
data = np.loadtxt(wdir, delimiter = ',')
X,y = data[:,0], data[:,1]
print(X,y)

# %%
#plot data
def plotData(X,y):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.scatter(X,y,s=100, c = 'green', marker = 'o',edgecolor = 'black',alpha=0.75)
    plt.show()

# %%
# Compute Cost function
# first, we need to contruct a X matrix (theta_0+theta1*X)
m = len(X)
ones = np.ones(m)
X = np.stack([np.ones(m),X],axis = 1)

#define cost function

def computeCost(X,y,theta):
    """
    X: matrix (array like)
       The shape of matrix is m x (n+1), where m is the number of sample size, and n is the
       number of variables, 1 is for the one vector.
    y: vector
       shape: m x 1
    theta: vector
           The parameters for the regression function, shape (n+1) x 1, 1 is for the one vector

    returns
    J the value of the regression cost function
    """
    import numpy as np
    m = len(y)
    h = np.dot(X,theta)
    J = (1/(2*m))*np.sum(np.multiply((h-y),(h-y)),axis=0)
    
    return J
#%%
#test cost function   
J = computeCost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# %%
#define the graident descent algorithm
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    
    Parameters
    ----------
    alpha : float
        The learning rate.
    
    num_iters : int
        The number of iterations for gradient descent. 
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, 1).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    J_history = []
    #gradient descent algorithm
    for i in range(num_iters):
        print('This is the %dth iteration'%i)
        theta = theta - (alpha/m)*(np.dot(X,theta)-y).dot(X)
        J_history.append(computeCost(X,y,theta))
        print('the cost is %.2f'% computeCost(X,y,theta))
    return theta, J_history

#%%
#test gradient ascend function

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

# %%
# plot the linear fit
import matplotlib.pyplot as plt
plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.legend(['Training data', 'Linear regression'])

# %%
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

# %%
from matplotlib import pyplot
import numpy as np

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])
        
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = pyplot.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pass

# %%
# Linear Regression with multiple variables

import numpy as np
# Load data
wdir = 'C:\\Users\\Beichen\\Desktop\\ML_NG\\week2\\machine-learning-ex1\\ex1\\ex1data2.txt'
data = np.loadtxt(wdir, delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
'''
# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
'''

# Normalizes the features in X
def featureNormalize(X):
    '''
    X: matrix
        The data set of shape m x n
    return
    X_norm: matrix
        the data set of shape m x n
    '''
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X,axis = 0)
    sigma = np.std(X, axis = 0)

    X_norm = (X-mu)/sigma

    return X_norm, mu, sigma

# compute the cost function of multiple linear regression

def computeCostMulti(X, y, theta):
    import numpy as np
    m = y.shape[0]
    J = 0
    h = np.dot(X,theta)
    J = (1/(2*m))*np.sum(np.square(h-y))

    return J



# Perform gradient descent to learn theta

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    
    print(X)
    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha/m)*(np.dot(X,theta)-y).dot(X)
        J_history.append(computeCostMulti(X,y,theta))
    return theta, J_history


"""
Instructions
------------
We have provided you with the following starter code that runs
gradient descent with a particular learning rate (alpha). 

Your task is to first make sure that your functions - `computeCost`
and `gradientDescent` already work with  this starter code and
support multiple variables.

After that, try running gradient descent with different values of
alpha and see which one gives you the best result.

Finally, you should complete the code at the end to predict the price
of a 1650 sq-ft, 3 br house.

Hint
----
At prediction, make sure you do the same feature normalization.
"""
from matplotlib import pyplot
X_norm, mu, sigma = featureNormalize(X)
m = X.shape[0]
ones = np.reshape(np.ones(m),[m,1])
X_norm = np.concatenate([ones,X_norm],axis = 1)
# Choose some alpha value - change this
alpha = 0.1
num_iters = 400

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))

# Estimate the price of a 1650 sq-ft, 3 br house
# ======================= YOUR CODE HERE ===========================
# Recall that the first column of X is all-ones. 
# Thus, it does not need to be normalized.

X_array = [1, 1650, 3]
X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta)

# ===================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))

# %%
