import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def read_dataset_from_h5(train_filepath, test_filepath):
    train_dataset = h5py.File(train_filepath, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_filepath, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def preprocess_dataset(train_x_orig, test_x_orig, m_train, m_test, w_dim):
    """ 
    Preprocess:
        1. Flatten
        2. Standardize
    
    # Detailed description of Preprocessing Step:
    #   - to center and standardize your dataset
    #   - e.g. substract the mean of the whole numpy array from each example, 
    #          and then divide each example by the standard deviation of the whole numpy array. 
    #   - for picture datasets, divide every pixel by 255 (the maximum value of a pixel channel).
    """
    train_X = train_x_orig.reshape(m_train, w_dim).T / 255.
    test_X = test_x_orig.reshape(m_test, w_dim).T / 255.
    return train_X, test_X


def load_dataset(train_filepath, test_filepath, print_info=True, testing_mode=False):
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = read_dataset_from_h5(train_filepath, test_filepath)
    
    if testing_mode:
        print (classes)                 # [b'non-cat' b'cat']
        print (train_x_orig.shape)      # (209, 64, 64, 3)
        print (train_y_orig.shape)      # (1, 209)
        print (test_x_orig.shape)       # (50, 64, 64, 3)
        print (test_y_orig.shape)       # (1, 50)
    
    assert (train_y_orig.shape[1] == train_x_orig.shape[0])     # m_train
    assert (test_y_orig.shape[1] == test_x_orig.shape[0])       # m_test
    assert (train_x_orig.shape[1:] == test_x_orig.shape[1:])    # image size

    if testing_mode:
        # Showing the 25th picture for example.
        index = 25
        plt.imshow(train_x_orig[index])
        print ("y = " + str(train_y_orig[:, index]) + ", it's a '" + classes[np.squeeze(train_y_orig[:, index])].decode("utf-8") +  "' picture.")
        plt.show()

    m_train = train_y_orig.shape[1]
    m_test = test_y_orig.shape[1]
    w_dim = train_x_orig.shape[1] * train_x_orig.shape[2] * train_x_orig.shape[3]

    train_Y, test_Y = train_y_orig, test_y_orig
    train_X, test_X = preprocess_dataset(train_x_orig, test_x_orig, m_train, m_test, w_dim)

    if print_info:
        print ("Each image is of size: " + str(train_x_orig.shape[1:]))
        print ("Number of training examples: m_train = " + str(m_train))
        print ("Number of testing  examples: m_test  = " + str(m_test))
        print ("train_Y shape: " + str(train_Y.shape))
        print ("test_Y  shape: " + str(test_Y.shape))
        print ("train_X shape: " + str(train_X.shape))
        print ("test_X  shape: " + str(test_X.shape))
        print ("sanity check after reshaping: " + str(train_X[0:5,0]))

    return train_X, train_Y, test_X, test_Y, classes


sigmoid = lambda z: ( 1 / (1 + np.exp(-z)) )

"""
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
"""

def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights / size (dim, 1)
    b -- bias / a scalar
    X -- data / size (dim, m)
    Y -- ground truth / size (1, m) / either 0 (not-cat) or 1 (cat)
    
    # w = (img_bytes, 1)
    # b = scalar
    # X = (img_bytes, m_train), (img_bytes, m_test)
    # A = (1, m_train)
    # Y = (1, m_train)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    # dw = (img_bytes, 1), (neuron_num, 1)
    # db = scalar
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = np.sum(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T)) / -m
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

"""
    L0 = -np.log( 1 / (1+np.exp(-7)) )
    L1 = -np.log( np.exp(-13) / (1+np.exp(-13)) )
    print ((L0 + L1) / 2)
"""

"""
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
"""

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    Returns:
    params -- model parameters, containing the weights `w` and bias `b`
    grads -- gradients, containing `dw` and `db`
    cost_records -- costs in every interation, used to plot the learning curve.
    """
    
    cost_records = []
    
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)
        
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        
        # Record the costs.
        if i % 100 == 0:
            cost_records.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    return params, grads, cost_records

"""
params, grads, cost_records = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
"""

def predict(w, b, X):
    '''
    Arguments:
    w -- weights / size (w_dim, 1)
    b -- bias / a scalar
    X -- data / size (w_dim, m)
    A -- answers, probabilities / size (1, m)
    
    Returns:
    C -- category / size (1, m) / each value is of (0/1)
    '''
    
    assert (w.shape == (X.shape[0], 1))
    A = sigmoid(np.dot(w.T, X) + b)

    C_list = map(lambda p: (1 if p > 0.5 else 0), A[0])
    C = np.array(C_list).reshape((1, X.shape[1]))
    return C

"""
print ("predictions = " + str(predict(w, b, X)))
"""

def model(train_X, train_Y, test_X, test_Y, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Build and run the logistic regression model.
    
    Arguments:
    train_X -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    train_Y -- training labels represented by a numpy array (vector) of shape (1, m_train)
    test_X -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    test_Y -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # Initialize model parameters.
    w = np.zeros((train_X.shape[0], 1))
    b = 0.0

    # Use gradient descent to optimize (minimize the cost) and generate a model (weights and bias).
    parameters, grads, costs = optimize(w, b, train_X, train_Y, num_iterations, learning_rate, print_cost)
    

    # Predict train/test set examples
    w = parameters["w"]
    b = parameters["b"]
    train_C = predict(w, b, train_X)
    test_C = predict(w, b, test_X)

    # Print train/test Errors
    print("train accuracy: {} % images are correct".format(100 - np.mean(np.abs(train_C - train_Y)) * 100))
    print("test accuracy: {} % images are correct".format(100 - np.mean(np.abs(test_C - test_Y)) * 100))


    d = {"costs": costs,
         "test_C": test_C, 
         "train_C": train_C, 
         "w": w, 
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    return d


if __name__ == '__main__':

    train_X, train_Y, test_X, test_Y, classes = load_dataset('datasets/train_catvnoncat.h5', 'datasets/test_catvnoncat.h5')

    d = model(train_X, train_Y, test_X, test_Y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
