'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    logistic = 1 / (1 + np.exp(-(z)))

    return  logistic# your code here
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
#    params = np.zeros(shape = (n_hidden * (n_input + 1) + n_class * (n_hidden + 1),)) 
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#    
#    training_data = train_data
#    training_label = train_label
    o_pred = np.zeros(shape = [training_data.shape[0],2])
    

    #prediction = sigmoid(np.dot(w2,np.append(sigmoid(np.dot(w1,np.transpose(train_data))),1)))
    # Your code here

#    
    x_train_constant = [1]*training_data.shape[0]
    final_train_data = np.column_stack([training_data, x_train_constant])

    gradient_w1 = np.zeros(shape = w1.shape)
    gradient_w2 = np.zeros(shape = w2.shape)
    """
    calculate gradience.
    """
    gradient_w1_final = np.zeros(shape = w1.shape)
    gradient_w2_final = np.zeros(shape = w2.shape)
    for i in range(0,final_train_data.shape[0]):
        z = sigmoid(np.dot(w1,np.transpose(final_train_data[i,:])))  #
        z = np.append(z, 1)
        o = sigmoid(np.dot(w2,z))
        label = training_label[i,]
        true_label_code = np.zeros(shape = [2,])
        true_label_code[(label),] = 1

#        sigma_w2 = (o-true_label_code) *o*(1-o)
#        gradient_w2 = np.outer(sigma_w2,z)+lambdaval*w2
#        sigma_w1 = np.transpose(w2[:,0:w1.shape[0]]).dot(sigma_w2) * (z[0:w1.shape[0]] * (1-z)[0:w1.shape[0]])
#        gradient_w1 = np.outer(sigma_w1,final_train_data[i,]) + lambdaval*w1

        sigma_w2 = o - true_label_code
        gradient_w2 = np.outer(sigma_w2, z)
        sigma_w1 = (1-z)*z*np.dot(np.transpose(sigma_w2),w2)
        gradient_w1 = np.outer(sigma_w1[0:n_hidden,] , final_train_data[i,:])

        gradient_w2_final = gradient_w2_final + gradient_w2
        gradient_w1_final = gradient_w1_final + gradient_w1
        
    gradient_w2_final = gradient_w2_final/final_train_data.shape[0]
    gradient_w1_final = gradient_w1_final/final_train_data.shape[0]
#        w1 = w1 - 0.01*gradient_w1
#        w2 = w2 - 0.01*gradient_w2
#        for k in range(0,w2.shape[0]):
#            for n in range(0,w2.shape[1]):
#                gradient_w2[k,n] = (o - true_label_code)[k] * o[k] * (1 - o[k]) * z[n] + lambdaval * w2[k,n]
#                #w2[k,n] = w2[k,n] - learning_rate * gradient_w2
#
#
#            for m in range(0,w1.shape[0]):
#                for f in range(0,w1.shape[1]):
#                    gradient_w1[m,f] = ((o - true_label_code)[k] * o[k] * (1 - o[k]))*(w2[k,m])* z[m] * (1 - z[m]) * final_train_data[i,f] + lambdaval * w1[m,f]
#                    w1[m, f] = w1[m, f]-learning_rate * gradient_w1
    obj_grad = np.concatenate((gradient_w1_final.flatten(),gradient_w2_final.flatten()),0)
    """
    calculate the objective value
    """
    obj_val = 0
    
    for i in range(0,final_train_data.shape[0]):
        z_pred = sigmoid(np.dot(w1,np.transpose(final_train_data[i,:])))  #
        z_pred = np.append(z_pred, 1)
        o_pred[i,] = sigmoid(np.dot(w2,z_pred)) 
        label = training_label[i]
        label_vector = np.zeros(shape = [2,])
        label_vector[label]=1
        tmp_error = -((np.dot(np.log(o_pred[i,]),np.transpose(label_vector))+np.dot((1-label_vector),np.log(1-o_pred[i,]))))
        obj_val = obj_val + tmp_error
        
    obj_val = obj_val/final_train_data.shape[0] + (lambdaval * (np.sum(w1**2)+np.sum(w2**2)))    
    obj_val
    
       
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    
    return (obj_val, obj_grad)
# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
    
    x_data_constant = [1]*data.shape[0]
    final_data = np.column_stack([data, x_data_constant])
    labels = np.array([])
    # Your code here
#    error_number = 0
    for i in range(0,final_data.shape[0]):
        layer1  = sigmoid(np.dot(w1,np.transpose(final_data[i,:])))  #
        layer1 = np.append(layer1, 1)
        label_tmp = sigmoid(np.dot(w2,layer1))
        prediction = np.argmax(label_tmp)
        labels = np.append(labels,prediction)
#        true = train_label[i]
#        if(true != prediction):
#            error_number = error_number + 1

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('D:\\Study\\SUNY-Buffalo\\CSE574\\assignment1\\face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 0.0002;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
