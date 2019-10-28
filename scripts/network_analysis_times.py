#-------------------------------------------------------------------------
# Copyright (C) 2017, All rights reserved
#
# JENSEN HUGHES
#
# 3610 Commerce Blvd, Suite 817
#
# Baltimore, MD 21227
#
# http://www.jensenhughes.com
#
# JENSEN HUGHES. Copyright Information
#
#-------------------------------------------------------------------------
#=========================================================================
# #
# # DESCRIPTION:
# #        Contains performance analysis code for 1-D signal scoring.
#=========================================================================

#=========================================================================
# # IMPORTS
#=========================================================================

import numpy as np
import yaml
from collections import defaultdict
import util_common as uc
import tensorflow as tf
import pickle
import glob
import sys
import generate_dataset as gd
from generate_dataset import GriddedMeasurementPair
import psutil
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal as scsi
import skimage
from networkDesign import cnnModel3
from skimage.segmentation import chan_vese
from skimage import img_as_float
import os

class tfTrainedVars(object):
    
    def __init__(self,modelDir,modelFnc,normalize=True):
        classifier = tf.estimator.Estimator(model_fn=modelFnc, model_dir=modelDir)
        weightNames = classifier.get_variable_names()
        
        #conv1_kernel = np.empty((10,10,1,8))
        for name in weightNames:
            newName = name.replace('/','_')
            value = classifier.get_variable_value(name)
            if normalize:
                value = (value-value.min())/(value.max()-value.min())
            if 'bias' in name:
                setattr(self,newName,value)
            elif 'kernel' in name and len(value.shape) <= 2:
                setattr(self,newName,value)
            elif 'kernel' in name:
                if value.shape[2] == 1:
                    if 'conv1_kernel' not in locals():
                        conv1_kernel = np.array(value)
                    else:
                        conv1_kernel = np.append(conv1_kernel,value,axis=2)
                    #conv1_kernel.extend(value,axis=2)
                else:
                    setattr(self,newName,value)
            print(name,np.shape(value))
        if len(conv1_kernel) > 0:
            setattr(self,'conv1_kernel',np.squeeze(conv1_kernel))
            
    def plotWeight(self,name,channel,kernel):
        
        if len(channel) == 1:
            img = getattr(self,name)
            if kernel is not None:
                img = img[:,:,channel[0],kernel]
                plt.imshow(img,cmap='gray')
            else:
                img = img[:,:,channel[0],:]
                totalLen = img.shape[2]
                rootSize = np.ceil(totalLen**0.5)
                plt.figure(figsize=(12,12))
                for i in range(1,totalLen+1):
                    plt.subplot(rootSize,rootSize,i)
                    plt.imshow(img[:,:,i-1],cmap='gray')
                    plt.tick_params(bottom='off',labelbottom='off',left='off',labelleft='off')
                    
        elif len(channel) == 3:
            sz = np.shape(getattr(self,name))
            if kernel is not None:
                img = np.zeros((sz[0],sz[1],3))
                for i in range(0,3):
                    img[:,:,i] = getattr(self,name)[:,:,channel[i],kernel]
                    plt.imshow(img)
            else:
                img = np.zeros((sz[0],sz[1],3,sz[3]))
                for i in range(0,3):
                    img[:,:,i,:] = getattr(self,name)[:,:,channel[i],:]
                totalLen = img.shape[3]
                rootSize = np.ceil(totalLen**0.5)
                plt.figure(figsize=(12,12))
                for i in range(1,totalLen+1):
                    plt.subplot(rootSize,rootSize,i)
                    plt.imshow(img[:,:,:,i-1])
                    plt.tick_params(bottom='off',labelbottom='off',left='off',labelleft='off')
        
        
        #plt.colorbar()
        #self.Value = classifier.get_variable_value(weightNames[0])
        #self.Names = weightNames
        


def splitdata_tf(data,test_number=None,fakeRandom=False):
    ''' splitdata: This function will split the data into test and training
        sets.
        
        Inputs:
          data: tuple of data in tensorflow format
          test_number: number of samples to withold for testing. If none, half
            the data is used.
        Outputs:
          test_data: portion of input data for testing
          training_data: portion of input data for training
    '''
    if fakeRandom:
        np.random.seed(1)
    total_len = data[0].shape[0]
    
    if test_number is None:
        random_inds = np.array(np.round(np.random.rand(int(total_len/2),1)*total_len,decimals=0)-1,dtype=np.int64)
    else:
        random_inds = np.array(np.round(np.random.rand(test_number,1)*total_len,decimals=0)-1,dtype=np.int64)
    random_inds = np.array(np.unique(random_inds),dtype=np.int64)
    mask = np.ones(data[0].shape[0],dtype=bool)
    mask[random_inds] = False
    training_data = (data[0][mask,:],data[1][mask,:])
    test_data = (data[0][~mask,:],data[1][~mask,:])
    return test_data, training_data

def extract_wb(w1,b1,sess):
    ''' extract_wb: This function extract weights and biases from tensorflow
        session.
        
        Inputs:
          w1: list of tensorflow weights
          b1: list of tensorflow biases
          sess: tensorflow session
          
        Outputs:
          w2: list of numpy weights
          b2: list of numpy biases
    '''
    w2 = []
    b2 = []
    for w0 in w1:
        w = sess.run(w0)
        w2.append(w)
    for b0 in b1:
        b = sess.run(b0)
        b2.append(b)
    return w2, b2

def import_wb(file,sess):
    ''' import_wb: This function imports pickled weights and biases and
        initializes the tensorflow network variables
        
        Inputs:
          file: name of pickled file containing weights and biases
          sess: tensorflow session
          
        Outputs:
          w: list of tensorflow weights
          b: list of tensorflow biases
          af: activation function
          dims: number of inputs to neural network
          ydim: number of outputs from neural network
    '''
    f = open(file,'rb')
    w2, b2, af, epoch = pickle.load(f)
    neurons = []
    dims = w2[0].shape[0]
    ydim = w2[-1].shape[1]
    for i in range(0,len(w2)):
        neurons.append(w2[i].shape[0])
    neurons.append(w2[-1].shape[1])
    w,b = init_network(neurons)
    
    for i in range(0,len(w2)):
        sess.run(w[i].assign(w2[i]))
    for i in range(0,len(b2)):
        sess.run(b[i].assign(b2[i]))
    return w,b,af,dims,ydim,epoch

def import_data(file):
    f = open(file,'rb')
    test_data, training_data = pickle.load(f)
    return test_data, training_data

def tensorflow_network(data,num=1001,neurons=None,test_number=None,ns='',ds='',
                       train=True,learning_rate=0.00001,continue_train=True,
                       fakeRandom=False,
                       activation_function='relu',
                       comparison_function='rmse'):
    ''' tensorflow_network: This function defines a tensorflow network. It can
        be used to train or test the network.
        
        Inputs:
          data: input data in tensorflow format
            training format:
              type(data) = tuple
              len(data) = 2
              type(data[0]) = numpy.ndarray
              data[0].shape = (number of samples, number of inputs)
              type(data[1]) = numpy.ndarray
              data[1].shape = (number of samples, number of outputs)
            test format: (Note: can also accept training format)
              type(data) = numpy.ndarray
              data.shape = (number of samples, number of inputs)
          num: number of epochs to train
          train: whether or not to train the network
          ns: namespace
          neurons: list of neurons to use in fully connected hidden layers
          test_number: number of samples to withold for testing. If none, half
            the data is used.
          learning_rate: learning rate for neural network
          activation_function: type of activation function to use. Valid
            arguments are 'relu' and 'sigmoid'
          
        Outputs (train):
          training_data: subset of data used to train the network
          test_data: subset of data used to test the network
          save_path: pickled network weights and biases
          test_prediction: network predictions of test data
        
        Outputs (no train):
          test_data: data used to test the network
          test_prediction: network predictions of test data
          
    '''
    
    # Check data format
    if type(data) is tuple:
        assert type(data) is tuple, 'type(data) should be tuple'
        assert len(data) == 2, 'len(data) should be 2'
        assert type(data[0]) is np.ndarray, 'type(data[0]) should be numpy.ndarray'
        assert data[0].shape[0] == data[1].shape[0], 'data[0].shape[0] should be the same as data[1].shape[0]'
    elif type(data) is np.ndarray and not train:
        assert len(data.shape) == 2, 'len(data.shape) should be 2'
    elif continue_train and data is None:
        print("Loading data from file %s"%(ds))
    else:
        #print("Did not recognize input format. See documentation.")
        assert False, 'Did not recognize input format. See documentation.'
    
    if glob.glob(ns+'model.pkl') and glob.glob(ds) and continue_train and train:
        continue_train = True
    else:
        continue_train = False
    # Start tensorflow session
    sess = tf.Session()
    
    if train and not continue_train:
        # Determine input and output layer dimensions
        dims = data[0].shape[1]
        ydim = data[1].shape[1]
        
        # Split and arrange data
        test_data, training_data = splitdata_tf(data,test_number=test_number,fakeRandom=fakeRandom)
        
        # Define layers
        if neurons is None:
            neurons = [dims, ydim]
        else:
            neu = [dims]
            neu.extend(neurons)
            neu.extend([ydim])
            neurons = neu
        print("NN layers:",neurons)
        
        # Weight initializations
        w1,b1 = init_network(neurons)
        old_epoch = 0
    elif continue_train and train:
        # Import saved network parameters
        w1,b1,activation_function,dims,ydim, old_epoch = import_wb("./"+ns+"model.pkl",sess)
        test_data, training_data = splitdata_tf(data,test_number=test_number,fakeRandom=fakeRandom)
        #test_data, training_data = import_data("./"+ds+"data.out")
        w2,b2 = extract_wb(w1,b1,sess)
    elif not train:
        # Import saved network parameters
        w1,b1,activation_function,dims,ydim, old_epoch = import_wb("./"+ns+"model.pkl",sess)
        w2,b2 = extract_wb(w1,b1,sess)
        if type(data) is tuple:
            test_data = data[0]
        else:
            test_data = data
    
    X = tf.placeholder("float", shape=[None, dims])
    y = tf.placeholder("float", shape=[None, ydim])
    
    # Forward propagation
    
    if activation_function == 'sigmoid':
        yhat = forwardprop_sigmoid(X, w1, b1)
    elif activation_function == 'relu':
        yhat = forwardprop_relu(X, w1, b1)
    elif activation_function == 'tanh':
        yhat = forwardprop_tanh(X, w1, b1)
    else:
        yhat = forwardprop(X, w1, b1)
    
    if train:
        # Backward propagation
        cost = (y-yhat)**2
        updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        # Initialize everything else
        if not continue_train:
            init = tf.global_variables_initializer()
            sess.run(init)
            #with open("./"+ds+"data.out",'wb') as f:
            #    pickle.dump([test_data,training_data],f)
        # Perform num training epochs
        if num < 10:
            modNum = 1
        else:
            modNum = int(num/10)
        for epoch in range(num):
            sess.run(updates, feed_dict={X: training_data[0], y: training_data[1]})
            if epoch % modNum == 0:
                if comparison_function == 'rmse':
                    train_accuracy = np.mean(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]}))**2)**0.5
                    test_accuracy  = np.mean(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]}))**2)**0.5
                if comparison_function == 'mae':
                    train_accuracy = np.mean(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]})))
                    test_accuracy  = np.mean(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]})))
                elif comparison_function == 'sae':
                    train_accuracy = np.sum(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]})))/training_data[1].shape[0]
                    test_accuracy  = np.sum(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]})))/test_data[1].shape[0]
                #print(test_data[1][0])
                #print(sess.run(yhat, feed_dict={X: test_data[0]})[0])
                print("Epoch = %d, train rmse = %.2f, test rmse = %.2f"
                      % (old_epoch+epoch + 1, train_accuracy, test_accuracy))
        
                # Save network parameters using pickle
                if ns[0:2] == "..":
                    save_path = "./"+ns+"model.pkl"
                else:
                    save_path = ns+"model.pkl"
                    
                with open(save_path,'wb') as f:
                    w2,b2 = extract_wb(w1,b1,sess)
                    pickle.dump([w2,b2,activation_function,epoch+old_epoch+1],f)
                    
        # Generate test prediction
        test_prediction = sess.run(yhat, feed_dict={X: test_data[0]})
        
        # Close session
        sess.close()
        
        # Return training and test datas, path to network parameters, test predictions
        return training_data, test_data, save_path, test_prediction
    else:
        # Generate test prediction
        test_prediction = sess.run(yhat, feed_dict={X: test_data})
        w2,b2 = extract_wb(w1,b1,sess)
        
        # Close session
        sess.close()
        
        # Return training and test datas, path to network parameters, test predictions
        return test_prediction, test_data












def tensorflow_network_conv(data,num=1001,neurons=None,test_number=None,ns='',ds='',
                       train=True,learning_rate=0.00001,continue_train=True,
                       fakeRandom=False,
                       activation_function='relu',
                       comparison_function='rmse'):
    ''' tensorflow_network: This function defines a tensorflow network. It can
        be used to train or test the network.
        
        Inputs:
          data: input data in tensorflow format
            training format:
              type(data) = tuple
              len(data) = 2
              type(data[0]) = numpy.ndarray
              data[0].shape = (number of samples, number of inputs)
              type(data[1]) = numpy.ndarray
              data[1].shape = (number of samples, number of outputs)
            test format: (Note: can also accept training format)
              type(data) = numpy.ndarray
              data.shape = (number of samples, number of inputs)
          num: number of epochs to train
          train: whether or not to train the network
          ns: namespace
          neurons: list of neurons to use in fully connected hidden layers
          test_number: number of samples to withold for testing. If none, half
            the data is used.
          learning_rate: learning rate for neural network
          activation_function: type of activation function to use. Valid
            arguments are 'relu' and 'sigmoid'
          
        Outputs (train):
          training_data: subset of data used to train the network
          test_data: subset of data used to test the network
          save_path: pickled network weights and biases
          test_prediction: network predictions of test data
        
        Outputs (no train):
          test_data: data used to test the network
          test_prediction: network predictions of test data
          
    '''
    
    # Check data format
    if type(data) is tuple:
        assert type(data) is tuple, 'type(data) should be tuple'
        assert len(data) == 2, 'len(data) should be 2'
        assert type(data[0]) is np.ndarray, 'type(data[0]) should be numpy.ndarray'
        assert data[0].shape[0] == data[1].shape[0], 'data[0].shape[0] should be the same as data[1].shape[0]'
    elif type(data) is np.ndarray and not train:
        assert len(data.shape) == 2, 'len(data.shape) should be 2'
    elif continue_train and data is None:
        print("Loading data from file %s"%(ds))
    else:
        #print("Did not recognize input format. See documentation.")
        assert False, 'Did not recognize input format. See documentation.'
    
    if glob.glob(ns+'model.pkl') and glob.glob(ds) and continue_train and train:
        continue_train = True
    else:
        continue_train = False
    # Start tensorflow session
    sess = tf.Session()
    
    if train and not continue_train:
        # Determine input and output layer dimensions
        dims = data[0].shape[1]
        ydim = data[1].shape[1]
        
        # Split and arrange data
        test_data, training_data = splitdata_tf(data,test_number=test_number,fakeRandom=fakeRandom)
        
        # Define layers
        if neurons is None:
            neurons = [dims, ydim]
        else:
            neu = [dims]
            neu.extend(neurons)
            neu.extend([ydim])
            neurons = neu
        print("NN layers:",neurons)
        
        # Weight initializations
        w1,b1 = init_network(neurons)
        old_epoch = 0
    elif continue_train and train:
        # Import saved network parameters
        w1,b1,activation_function,dims,ydim, old_epoch = import_wb("./"+ns+"model.pkl",sess)
        test_data, training_data = splitdata_tf(data,test_number=test_number,fakeRandom=fakeRandom)
        #test_data, training_data = import_data("./"+ds+"data.out")
        w2,b2 = extract_wb(w1,b1,sess)
    elif not train:
        # Import saved network parameters
        w1,b1,activation_function,dims,ydim, old_epoch = import_wb("./"+ns+"model.pkl",sess)
        w2,b2 = extract_wb(w1,b1,sess)
        if type(data) is tuple:
            test_data = data[0]
        else:
            test_data = data
    
    X = tf.placeholder("float", shape=[None, dims])
    y = tf.placeholder("float", shape=[None, ydim])
    
    # Forward propagation
    
    if activation_function == 'sigmoid':
        yhat = forwardprop_sigmoid(X, w1, b1)
    elif activation_function == 'relu':
        yhat = forwardprop_relu(X, w1, b1)
    elif activation_function == 'tanh':
        yhat = forwardprop_tanh(X, w1, b1)
    else:
        yhat = forwardprop(X, w1, b1)
    
    if train:
        # Backward propagation
        cost = (y-yhat)**2
        updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        # Initialize everything else
        if not continue_train:
            init = tf.global_variables_initializer()
            sess.run(init)
            #with open("./"+ds+"data.out",'wb') as f:
            #    pickle.dump([test_data,training_data],f)
        # Perform num training epochs
        if num < 10:
            modNum = 1
        else:
            modNum = int(num/10)
        for epoch in range(num):
            sess.run(updates, feed_dict={X: training_data[0], y: training_data[1]})
            if epoch % modNum == 0:
                if comparison_function == 'rmse':
                    train_accuracy = np.mean(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]}))**2)**0.5
                    test_accuracy  = np.mean(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]}))**2)**0.5
                if comparison_function == 'mae':
                    train_accuracy = np.mean(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]})))
                    test_accuracy  = np.mean(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]})))
                elif comparison_function == 'sae':
                    train_accuracy = np.sum(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]})))/training_data[1].shape[0]
                    test_accuracy  = np.sum(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]})))/test_data[1].shape[0]
                #print(test_data[1][0])
                #print(sess.run(yhat, feed_dict={X: test_data[0]})[0])
                print("Epoch = %d, train rmse = %.2f, test rmse = %.2f"
                      % (old_epoch+epoch + 1, train_accuracy, test_accuracy))
        
                # Save network parameters using pickle
                if ns[0:2] == "..":
                    save_path = "./"+ns+"model.pkl"
                else:
                    save_path = ns+"model.pkl"
                    
                with open(save_path,'wb') as f:
                    w2,b2 = extract_wb(w1,b1,sess)
                    pickle.dump([w2,b2,activation_function,epoch+old_epoch+1],f)
                    
        # Generate test prediction
        test_prediction = sess.run(yhat, feed_dict={X: test_data[0]})
        
        # Close session
        sess.close()
        
        # Return training and test datas, path to network parameters, test predictions
        return training_data, test_data, save_path, test_prediction
    else:
        # Generate test prediction
        test_prediction = sess.run(yhat, feed_dict={X: test_data})
        w2,b2 = extract_wb(w1,b1,sess)
        
        # Close session
        sess.close()
        
        # Return training and test datas, path to network parameters, test predictions
        return test_prediction, test_data











def init_weights(shape, stddev=0.1):
    ''' init_weights: This function creates a tensorflow variable of specified
        size. Values are initialized using a normal distribution.
        
        Inputs:
          shape: tuple of desired size
          stddev: standard deviation to use in normal distribution
          
        Outputs:
          tensorflow variable
    '''
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

def init_network(neurons):
    ''' init_network: This function initializes the weights and biases for the
        network with fully connected layers specified by the list of neurons.
        
        Inputs:
          neurons: list containing number of neurons for each hidden layer
        
        Outputs:
          w: tensorflow variable for network weights
          b: tensorflow variable for network biases
    '''
    # First set of weights correspond to input->first hidden layer
    w = [init_weights((neurons[0],neurons[1]))]
    b = []
    
    # Loop through hidden layers
    for i in range(1,len(neurons)-1):
        w.append(init_weights((neurons[i],neurons[i+1])))
        b.append(init_weights((neurons[i],)))
    return w,b

def forwardprop(X, w, b):
    ''' forwardprop: This function propogates inputs to outputs using the
        weights and biases from the neural network using sigmoid activation
        function.
        
        Inputs:
          X: tensorflow variable for neural network inputs
          w: tensorflow variable for network weights
          b: tensorflow variable for network biases
        
        Outputs:
          yhat: tensorflow variable for neural network outputs
    '''
    h = tf.nn.leaky_relu(tf.add(tf.matmul(X,w[0]),b[0]))
    if len(w)-1>1:
        for i in range(1,len(w)-2):
            h = tf.nn.leaky_relu(tf.add(tf.matmul(h,w[i]),b[i]))
        h = tf.nn.tanh(tf.add(tf.matmul(h,w[-2]),b[-1]))
        print("Many relu!")
    else:
        print("Only one relu")
        for i in range(1,len(w)-1):
            h = tf.nn.tanh(tf.add(tf.matmul(h,w[i]),b[i]))
    yhat = tf.matmul(h, w[-1])
    return yhat

def forwardprop_sigmoid(X, w, b):
    ''' forwardprop: This function propogates inputs to outputs using the
        weights and biases from the neural network using sigmoid activation
        function.
        
        Inputs:
          X: tensorflow variable for neural network inputs
          w: tensorflow variable for network weights
          b: tensorflow variable for network biases
        
        Outputs:
          yhat: tensorflow variable for neural network outputs
    '''
    h = tf.nn.sigmoid(tf.add(tf.matmul(X,w[0]),b[0]))
    for i in range(1,len(w)-1):
        h = tf.nn.sigmoid(tf.add(tf.matmul(h,w[i]),b[i]))
    yhat = tf.matmul(h, w[-1])
    return yhat

def forwardprop_tanh(X, w, b):
    ''' forwardprop: This function propogates inputs to outputs using the
        weights and biases from the neural network using sigmoid activation
        function.
        
        Inputs:
          X: tensorflow variable for neural network inputs
          w: tensorflow variable for network weights
          b: tensorflow variable for network biases
        
        Outputs:
          yhat: tensorflow variable for neural network outputs
    '''
    h = tf.nn.tanh(tf.add(tf.matmul(X,w[0]),b[0]))
    for i in range(1,len(w)-1):
        h = tf.nn.tanh(tf.add(tf.matmul(h,w[i]),b[i]))
    yhat = tf.matmul(h, w[-1])
    return yhat

def forwardprop_relu(X, w, b):
    ''' forwardprop_relu: This function propogates inputs to outputs using the
        weights and biases from the neural network using Relu activation
        function.
        
        Inputs:
          X: tensorflow variable for neural network inputs
          w: tensorflow variable for network weights
          b: tensorflow variable for network biases
        
        Outputs:
          yhat: tensorflow variable for neural network outputs
    '''
    h = tf.nn.relu(tf.add(tf.matmul(X,w[0]),b[0]))
    for i in range(1,len(w)-1):
        h = tf.nn.relu(tf.add(tf.matmul(h,w[i]),b[i]))
    yhat = tf.matmul(h, w[-1])
    return yhat

def swap_datainds(data):
    ''' swap_datainds: This function swaps the indices of the data
    
        Inputs:
          data: list of data in raw format
        
        Outputs:
          data2: list of data with swapped indices
    '''
    data2 = []
    for d in data:
        data2.append([d[1],d[0]])
    return data2

def scale_datay(data):
    ''' scale_datay: This function scales the output data between 0 and 1
    
        Inputs:
          data: list of data in raw format
        
        Outputs:
          data2: list of data with scaled output data
          scalefactor: list with scale factors
    '''
    params_min = np.zeros(len(data[0][1]),)+99999
    params_max = np.zeros(len(data[0][1]),)-99999
    data2 = []
    for d in data:
        inds = np.argwhere(d[1][:,0]-params_min < 0)
        params_min[inds] = d[1][inds,0]
        inds = np.argwhere(d[1][:,0]-params_max > 0)
        params_max[inds] = d[1][inds,0]
    for d in data:
        dscaled = (d[1][:,0]-params_min)/(params_max-params_min)
        data2.append([np.array(d[0]),np.reshape(dscaled,(len(dscaled),))])
    return data2, [params_min,params_max]



def network_wildfire_train(data,ns,ds,af,neu=[100,100,100],tn=10,num=11,lr=10**-7):
    
    cf='sae'
    #for n in neu:
    #    ns = ns+'_'+str(n)
    t1 = uc.tic()
    train_data, test_data, save_path, tp2 = tensorflow_network(
            data,ns=ns,neurons=neu,num=num,test_number=tn,learning_rate=lr,
            activation_function=af,comparison_function=cf,
            fakeRandom=True,ds=ds)
    uc.toc(t1)
    #print("Tensor flow param->score time:",uc.toc(t1))
    plt.figure()
    plt.plot(test_data[1][0],test_data[1][0])
    plt.xlabel('True Scaled Score')
    plt.ylabel('Pred Scaled Score')
    plt.title('Score Estimate (TensorFlow)')
    return test_data, train_data

def network_wildfire_test(data,ns):
    #t1 = uc.tic()
    test_prediction, test_data2 = tensorflow_network(data,train=False,ns=ns)
    #print("Tensor flow retest param->score time:",uc.toc(t1))
    #plt.figure(figsize=(12,8))
    #d = data[1][0].copy()
    #d[d<7] = 0
    #d[d>=7] = 1
    #plt.plot(d,test_prediction[0])
    #plt.xlabel('Measured Active Fire Index')
    #plt.ylabel('Predicted Active Fire Index')
    #plt.title('Score Estimate (TensorFlow)')
    #for i in range(0,len(test_prediction)):
    #    plt.scatter(d[i],test_prediction[i])
    return test_prediction

def plotWildfireTest(datas,names,
                     clims=None,labels=None,closeFig=None,
                     saveFig=False,saveName='',
                     gridOn=True):
    totalPlots = np.ceil(float(len(datas))**0.5)
    colPlots = totalPlots
    rowPlots = np.ceil((float(len(datas)))/colPlots)
    currentPlot = 0
    
    if saveFig:
        fntsize = 32
        lnwidth = 5
        fig = plt.figure(figsize=(colPlots*12,rowPlots*10))#,tight_layout=True)      
        if closeFig is None:
            closeFig = True
    else:
        fig = plt.figure(figsize=(colPlots*6,rowPlots*5))#,tight_layout=True)
        fntsize = 20
        lnwidth = 2
        if closeFig is None:
            closeFig = False
        
    xmin = 0
    xmax = datas[0].shape[1]
    xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/10)+1))
    ymin = 0
    ymax = datas[0].shape[0]
    yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/10)+1))

    for i in range(0,len(names)):
        key = names[i]
        currentPlot = currentPlot+1

        ax = fig.add_subplot(rowPlots,colPlots,currentPlot)
        ax.tick_params(axis='both',labelsize=fntsize)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('km',fontsize=fntsize)
        plt.ylabel('km',fontsize=fntsize)
        #plt.xlabel('Longitude',fontsize=fntsize)
        #plt.ylabel('Latitude',fontsize=fntsize)
        plt.title(key,fontsize=fntsize)

        if clims is None:
            clim = np.linspace(0,1,10)
            label = ''
        else:
            clim = clims[i]
        if labels is None:
            label = ''
        else:
            label = labels[i]
        img = ax.imshow(datas[i],cmap='jet',vmin=clim[0],vmax=clim[-1])#,vmin=0,vmax=1)
        #img = ax.contourf(self.longitude,self.latitude,getattr(self,key),levels=clim,cmap=cmap)
        img_cb = plt.colorbar(img,ax=ax,label=label)

        img_cb.set_label(label=label,fontsize=fntsize)
        img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        if gridOn: ax.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        for ln in ax.lines:
            ln.set_linewidth(lnwidth)
    if saveFig:
        fig.savefig(saveName,dpi=300)
        
    if closeFig:
        plt.clf()
        plt.close(fig)


def plotIndividualChannels(datas,names,
                     clims=None,closeFig=None,
                     saveFig=False,saveName=''):
    totalPlots = np.ceil(float(len(datas))**0.5)
    colPlots = totalPlots
    rowPlots = np.ceil((float(len(datas)))/colPlots)
    currentPlot = 0
    
    if saveFig:
        #fntsize = 20
        #lnwidth = 5
        #fig = plt.figure(figsize=(colPlots*12,rowPlots*10))#,tight_layout=True)      
        if closeFig is None:
            closeFig = True
    else:
        #fig = plt.figure(figsize=(colPlots*6,rowPlots*5))#,tight_layout=True)
        #fntsize = 20
        #lnwidth = 2
        if closeFig is None:
            closeFig = False
    
    fntsize = 20
    lnwidth = 5
        
    xmin = 0
    xmax = datas[0].shape[1]
    xticks = np.linspace(xmin,xmax,int(round((xmax-xmin)/10)+1))
    ymin = 0
    ymax = datas[0].shape[0]
    yticks = np.linspace(ymin,ymax,int(round((ymax-ymin)/10)+1))

    for i in range(0,len(names)):
        fig = plt.figure(figsize=(12,8))
        key = names[i]
        currentPlot = 1 #currentPlot+1

        ax = fig.add_subplot(1,1,currentPlot)
        ax.tick_params(axis='both',labelsize=fntsize)
        plt.xticks(xticks)
        plt.yticks(yticks)
        #plt.xlabel('Longitude',fontsize=fntsize)
        #plt.ylabel('Latitude',fontsize=fntsize)
        plt.title(key,fontsize=fntsize)

        if clims is None:
            clim = np.linspace(0,1,10)
            label = ''
        else:
            clim = clims[i]
            label = ''
        img = ax.imshow(datas[i],cmap='jet',vmin=clim[0],vmax=clim[-1])#,vmin=0,vmax=1)
        #img = ax.contourf(self.longitude,self.latitude,getattr(self,key),levels=clim,cmap=cmap)
        #img_cb = plt.colorbar(img,ax=ax,label=label)

        #img_cb.set_label(label=label,fontsize=fntsize)
        #img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        ax.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        print(saveName+'_'+key+'.png')
        for ln in ax.lines:
            ln.set_linewidth(lnwidth)
        if saveFig:
            fig.savefig(saveName+'_'+key+'.png')
        
    if closeFig:
        plt.clf()
        plt.close(fig)

def convolve_wildfire_train(data,labels,modelFnc,epochs=100,model_dir="../models/wf_model"):
    # Offset data from zero
    eps = 10**-12
    data = data+eps
  
    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=modelFnc, model_dir=model_dir)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            y=labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
    
    classifier.train(input_fn=train_input_fn,steps=epochs)
    
def convolve_wildfire_test(data,labels,modelFnc,model_dir="../models/wf_model"):
    # Offset data from zero
    eps = 10**-12
    data = data+eps
  
    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=modelFnc, model_dir=model_dir)
    
    # Test the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            y=labels,
            num_epochs=1,
            shuffle=False)
    '''
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    '''
    
    # Predict new measurements
    pred_results = classifier.predict(input_fn=eval_input_fn)
    pred_results_list = list(pred_results)
    
    prediction = []
    
    for pred in pred_results_list:
        prediction.append(pred['logits'])
        
    truth = []
    for label in labels:
        truth.append(label)
    
    prediction = np.array(prediction)
    truth = np.array(truth)
    
    return prediction, prediction, truth

def my_cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    
    dconv = True
    sz = 50
    #n_dimensions = 13
    case = 2
    n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    if case == 0:
        ks1 = [10,10]
        ks2 = [10,10]
        ks3 = [10,10]
        fs1 = 16
        fs2 = 32
        fs3 = 2
    elif case == 1:
        ks1 = [10,10]
        ks2 = [10,10]
        ks3 = [10,10]
        fs1 = 32
        fs2 = 64
        fs3 = 2
    elif case == 2:
        ks1 = [10,10]
        ks2 = [10,10]
        ks3 = [10,10]
        fs1 = 64
        fs2 = 128
        fs3 = 2
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=fs1,
            kernel_size=ks1,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=fs2,
            kernel_size=ks2,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    pool2flat = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
    
    if dconv:
        dense1 = tf.layers.dense(inputs=pool2flat, units=int(sz*sz*2), activation=tf.nn.leaky_relu)
        dense1_rs = tf.reshape(dense1,[-1,sz,sz,2])
        dconv1 = tf.layers.conv2d_transpose(
            inputs=dense1_rs,filters=fs3,
            kernel_size=ks3,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="dconv1")
        dconv1flat = tf.reshape(dconv1,[-1,dconv1.shape[1]*dconv1.shape[2]*dconv1.shape[3]])
        denseOut = tf.layers.dense(inputs=dconv1flat, units=int(sz*sz*2), activation=tf.nn.tanh)
        print("Input Layer Dimensions:\t",input_layer.shape)
        print("First Conv Layer Dim:\t",conv1.shape)
        print("First Pool Layer Dim:\t",pool1.shape)
        print("Second Conv Layer Dim:\t", conv2.shape)
        print("Second Pool Layer Dim:\t", pool2.shape)
        print("Classify Layer Dim:\t", dense1.shape)
        print("Deconv Layer Dim:\t", dconv1.shape)
        print("Output Layer Dim:\t",denseOut.shape)
    else:
        denseOut = tf.layers.dense(inputs=pool2flat, units=int(sz*sz*2), activation=tf.nn.tanh)
        
    logits = tf.reshape(denseOut,[-1,int(sz*sz*2)])
    predicted_classes = tf.argmax(input=tf.reshape(dense1,[-1,int(sz*sz),2]), axis=2)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5
    #loss = -1*tf.reduce_sum(tf.cast(labels,tf.float32)*tf.log(tf.cast(logits,tf.float32)))
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #print("Logits:",tf.shape(logits),logits.shape)
    #print("Labels:",tf.shape(labels),labels.shape)
    label_rs = tf.reshape(labels,[-1,int(sz*sz),2])
    label_classes = tf.argmax(input=label_rs,axis=2)
    accuracy = tf.metrics.accuracy(labels=label_classes,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    #with tf.variable_scope('conv1'):
        #tf.get_variable_scope().reuse_variables()
        #weights1 = tf.get_variable('kernel')
        #grid1 = put_kernels_on_grid (weights1)
        #tf.summary.image('conv1/kernels', grid1, max_outputs=1)
        
    #with tf.variable_scope('conv2'):
        #tf.get_variable_scope().reuse_variables()
        #weights2 = tf.get_variable('kernel')
        #grid2 = put_kernels_on_grid (weights2)
        #tf.summary.image('conv2/kernels', grid2, max_outputs=1)
        
    #with tf.variable_scope('dconv1'):
        #tf.get_variable_scope().reuse_variables()
        #weights = tf.get_variable('kernel')
        #grid = put_kernels_on_grid (weights)
        #tf.summary.image('dconv1/kernels', grid, max_outputs=1)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
      
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-4)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def put_kernels_on_grid (kernel, pad = 1):
  
    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))
  
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)
  
    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
  
    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad
  
    channels = kernel.get_shape()[2]
  
    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))
  
    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))
  
    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))
  
    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))
  
    # scaling to [0, 255] is not necessary for tensorboard
    return x

def my_cnn_model_fn2(features, labels, mode):
    """Model function for CNN."""
    newDimensions = True
    print("MODE=",mode)
    sz = 50
    n_dimensions = 4
    ks1 = [10,10]
    ks2 = [10,10]
    ks3 = [10,10]
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=64,
            kernel_size=ks1,
            padding="same",
            activation=tf.nn.leaky_relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=ks2,
            padding="same",
            activation=tf.nn.leaky_relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #print(pool2.shape,pool2.shape[0],pool2.shape[1],pool2.shape[2],pool2.shape[3])
    pool2flat = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
    if newDimensions:
        dense1 = tf.layers.dense(inputs=pool2flat, units=5000, activation=tf.nn.tanh)
        #dense1 = tf.layers.dense(inputs=pool2flat, units=7500, activation=tf.nn.tanh)
        #logits = tf.reshape(dense1,[-1,5000])
        logits = tf.reshape(dense1,[-1,5000])
        predicted_classes = tf.argmax(input=tf.reshape(dense1,[-1,2500,2]), axis=2)
    else:
        dense1 = tf.layers.dense(inputs=pool2flat, units=2500, activation=tf.nn.tanh)
        logits = tf.reshape(dense1,[-1,2500])
        
    
    """
    dconv1 = tf.layers.conv2d_transpose(
        inputs=dense1,filters=1,
        kernel_size=ks3,
        padding="same",
        activation=tf.nn.leaky_relu)
    dense2 = tf.layers.dense(inputs=dconv1, units=2500, activation=tf.nn.tanh)
    logits = tf.reshape(dense2,[-1, sz, sz, 1])
    """
    
    #predicted_classes = logits #tf.argmax(logits,1) #tf.argmax(logits, 1)
    #print("MODE==",mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
      
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5
    #loss = -1*tf.reduce_sum(tf.cast(labels,tf.float32)*tf.log(tf.cast(logits,tf.float32)))
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #print("Logits:",tf.shape(logits),logits.shape)
    #print("Labels:",tf.shape(labels),labels.shape)
    label_rs = tf.reshape(labels,[-1,2500,2])
    label_classes = tf.argmax(input=label_rs,axis=2)
    accuracy = tf.metrics.accuracy(labels=label_classes,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
      
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-4)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def datas2labels(datas,values=np.linspace(0,1,2,dtype=np.int32)):
    sz = datas.shape
    labels = np.zeros((sz[0],sz[1]*len(values)))
    for i in range(0,len(values)):
        (indr,indc) = np.where(datas == values[i])
        indc = indc+i*sz[1]
        #print(indc)
        labels[(indr,indc)] = 1
    return labels

def labels2datas(labels,width=2,fireThresh=0.05):
    sz = labels.shape
    noFireLabels = np.array(labels[:,0:2500],dtype=np.float32)
    fireLabels = np.array(labels[:,2500:5000],dtype=np.float32)#+np.array(labels[:,5000:],dtype=np.float32)
    inds = np.where(fireLabels-noFireLabels > 0)
    inds = np.where(fireLabels>fireThresh)
    print(np.max(fireLabels))
    datas = np.zeros((sz[0],int(sz[1]/width)))
    datas[inds] = 1
    return datas

def labels2probs(labels,width=2500,fireThresh=0.05):
    sz = labels.shape
    noFireLabels = np.array(labels[:,0:width],dtype=np.float32)
    fireLabels = np.zeros(noFireLabels.shape)
    for i in range(1,int(float(sz[1])/width)):
        fireLabels = fireLabels+np.array(labels[:,i*width:(i+1)*width],dtype=np.float32)
    datas = fireLabels/(noFireLabels+fireLabels)
    #fireLabels = np.array(labels[:,2500:5000],dtype=np.float32)#+np.array(labels[:,5000:],dtype=np.float32)
    #inds = np.where(fireLabels-noFireLabels > 0)
    #inds = np.where(fireLabels>fireThresh)
    #print(np.max(fireLabels))
    #datas = np.zeros((sz[0],int(sz[1]/width)))
    #datas = fireLabels/(noFireLabels+fireLabels)
    #datas[inds] = 1
    return datas

def arrayToImage(datas,outStyle=False):
    imgs = []
    if outStyle:
        for data in datas:
            imgs.append(np.reshape(data,(50,50)))
    else:
        for i in range(0,len(datas)):
            img = []
            for j in range(0,len(datas[i])):
                img.append(np.reshape(datas[i][j],(50,50)))
            imgs.append(img)
            
        #imgs = []
        #for data in datas:
        #    imgs.append(np.reshape(data,(50,50)))
    return imgs

def labels2labels(labels,width=3):
    sz = labels.shape
    noFireLabels = np.array(labels[:,0:2500],dtype=np.float32)
    fireLabels = np.array(labels[:,2500:5000],dtype=np.float32)+np.array(labels[:,5000:],dtype=np.float32)
    return noFireLabels, fireLabels

def inputs2labels(inputs,pixels=2500):
    newInputs = []
    n_dimensions = int(eval_data.shape[1]/pixels)
    for j in range(0,len(inputs)):
        new = []
        for i in range(0,n_dimensions):
            new.append(inputs[j,i*pixels:(i+1)*pixels])
        newInputs.append(new)
    return newInputs

def dropDataChannels(datas,pixels=2500,
                     channels=[True,True,True,True,True,False,True,False,False,True,False,False,True]):
    sz = datas.shape
    for i in range(int(sz[1]/pixels)-1,0,-1):
        print(i)
        if not channels[i]:
            datas = np.delete(datas,np.array(np.linspace((i-1)*pixels,i*pixels-1,pixels),dtype=np.int32),axis=1)
    return datas

def combineDataChannels(datas,pixels=2500,
                     channels=[True,True,True,True,True,False,True,False,False,True,True],
                     minFactor=[None,None,None,None,30,30,0,0,0,None,None],
                     maxFactor=[None,None,None,None,150,150,40,40,40,None,None],
                     combineType='max'):
    sz = datas.shape
    dataStored = []
    for i in range(int(sz[1]/pixels)-1,0,-1):
        if not channels[i]:
            tmp = datas[:,(i)*pixels:(i+1)*pixels].copy()
            if minFactor[i] is not None and maxFactor[i] is not None:
                tmp = (tmp-minFactor[i])/(maxFactor[i]-minFactor[i])
            else:
                tmp[tmp > 1] = 1
                tmp[tmp < 0] = 0
            dataStored.append(tmp.copy())
            datas = np.delete(datas,np.array(np.linspace((i)*pixels,(i+1)*pixels-1,pixels),dtype=np.int32),axis=1)
    dataStored = np.array(dataStored)
    
    if combineType == 'max':
        dataCombined = np.max(dataStored,axis=0)
    elif combineType == 'mean':
        dataCombined = np.mean(dataStored,axis=0)

    datas = np.append(datas,dataCombined,axis=1)
    return datas

def zeroDataChannels(datas,pixels=2500,
                     channels=[True,True,True,True,True,False,True,False,False,True,False,False,True]):
    sz = datas.shape
    for i in range(0,int(sz[1]/pixels)):
        if not channels[i]:
            datas[:,(i)*pixels:(i+1)*pixels] = 0.0
    return datas

def readPickledRawData(namespace):
    files = glob.glob(namespace+'*.pkl')
    allIn = []
    allOut = []
    for i in range(0,len(files)):
        [inData,outData] = uc.readPickle(files[i])
        allIn.extend(inData)
        allOut.extend(outData)
    inData = np.array(allIn)
    outData = np.array(allOut)
    
    return inData, outData

def findBestThreshold(predictionImgs,truthImgs,inputsImgs):
    thresh = -0.01
    threshes = []
    fMeasures = []
    confusionMatrixes = []
    while thresh < 1.0:
        thresh = thresh + 0.01
        confusionMatrix = []
        for i in range(0,len(truthImgs)):
            pImg = predictionImgs[i].copy()
            tImg = truthImgs[i].copy()
            confusionMatrix.append(findConfusionMatrix(pImg,tImg,thresh,inputsImgs[i][0]))
        confusionMatrix = np.array(confusionMatrix)
        threshes.append(thresh)
        fMeasures.append(np.mean(confusionMatrix[:,-2]))
        confusionMatrixes.append(confusionMatrix)
    bestThresh = threshes[np.argmax(fMeasures)]
    bestConfusionMatrix = np.mean(confusionMatrixes[np.argmax(fMeasures)],axis=0)
    return bestThresh, bestConfusionMatrix, threshes, fMeasures
    
def postProcessFirePerimiter(pImg,thresh):
    corners = [pImg[-1,-1].copy(),pImg[0,0].copy(),pImg[0,-1].copy(),pImg[-1,0].copy()]
    centers = pImg[24:26,24:26].copy()
    pImg = scsi.medfilt2d(pImg)
    (pImg[-1,-1],pImg[0,0],pImg[0,-1],pImg[-1,0]) = corners
    pImg[24:26,24:26] = centers
    pImg[pImg < thresh] = 0.0
    return pImg
    
def findConfusionMatrix(pImg,tImg,thresh,iImg):
    pImg = postProcessFirePerimiter(pImg,thresh)
    pImg[pImg>thresh] = 1.0

    TN = float(len(np.where(np.array(pImg.flatten() == 0) & np.array(tImg.flatten() == 0))[0]))
    FN = float(len(np.where(np.array(pImg.flatten() == 0) & np.array(tImg.flatten() == 1))[0]))
    FP = float(len(np.where(np.array(pImg.flatten() == 1) & np.array(tImg.flatten() == 0))[0]))
    TP = float(len(np.where(np.array(pImg.flatten() == 1) & np.array(tImg.flatten() == 1))[0]))
    
    totalFire = float(len(np.where(iImg.flatten()>1)[0]))
    
    try:
        accuracy = round((TP+TN)/(TP+TN+FP+FN),2)
    except ZeroDivisionError:
        accuracy = round((TP+TN)/(TP+TN+FP+FN+1),2)
    try:
        recall = round((TP)/(TP+FN),2)
    except ZeroDivisionError:
        recall = round((TP)/(TP+FN+1),2)
    try:
        precision = round((TP)/(TP+FP),2)
    except ZeroDivisionError:
        precision = round((TP)/(TP+FP+1),2)
    try:
        fMeasure = round((2*recall*precision)/(recall+precision),2)
    except ZeroDivisionError:
        fMeasure = round((2*recall*precision)/(recall+precision+1),2)
    
    confusionMatrix = [TN,FN,FP,TP,accuracy,recall,precision,fMeasure,totalFire]
    return confusionMatrix

def plotThresholdFMeasure(threshes,fMeasures):
    plt.figure(figsize=(12,12))
    plt.plot(threshes,fMeasures,'-k',linewidth=3)
    fs = 32
    plt.xlabel('Threshold',fontsize=fs)
    plt.ylabel('F-Measure',fontsize=fs)
    plt.ylim(0.0,1.0)
    plt.xlim(0.0,1.0)
    plt.tick_params(labelsize=fs)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    plt.grid()
    plt.tight_layout()
    plt.savefig('optimalThreshold.png',dpi=300)




def chanVeseFunctional(pImg, tImg):
    pimg2 = img_as_float(pImg*255)
    timg2 = img_as_float(tImg*255)
    cv = chan_vese(pimg2, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200, dt=0.5, init_level_set="checkerboard", extended_output=True)
    
    hv = np.zeros_like(cv[0],dtype=np.float32)+1
    hv[cv[1]>0] = 0
    
    C1 = np.sum(hv[pImg == 1]*timg2[pImg == 1])/float(timg2[pImg == 1].shape[0])
    C0 = np.sum((1-hv[pImg == 0])*timg2[pImg == 0])/float(timg2[pImg == 0].shape[0])
    
    Cmin = min([C0,C1])
    Cmax = max([C0,C1])
    
    cvff = np.sum(hv*(timg2-Cmax)**2+(1-hv)*(timg2-Cmin)**2)/(2500)
    return cvff

if __name__ == "__main__":
    args = sys.argv
    case = 2
    
    if case == 0: argsFile = '../config/rothermelFull.yaml'
    elif case == 1: argsFile = '../config/rothermelFull_cnnmodel3.yaml'
    elif case == 2: argsFile = '../config/rothermelFull_cnnmodel3_test.yaml'
        
    params = defaultdict(bool,yaml.load(open(argsFile,'r')))
    dataRawFile = params['dataRawFile']
    svdInputs = params['svdInputs']
    generatePlots= params['generatePlots']
    fakeRandom = params['fakeRandom']
    modelFnc = locals()[params['modelFnc']]
    model_dir = params['modelDir']
    model_dir = "../models/rothermelFull_cnnModel_3_bk"
    test_number = params['testNumber']
    
    zeroChannels = params['zeroChannels']
    dropChannels = params['dropChannels']
    combineChannels = params['combineChannels']
    ns = params['name']
    
    num = 10001
    train = False
    findBestThresh = False
    test = True
    testAll = True
    
    # Load data
    inData, outData = readPickledRawData(params['dataRawFile'])
    
    # Apply pre-processing
    #dataFile = dataRawFile+'.svd' if svdInputs else dataRawFile+'.out'
    if zeroChannels: inData = zeroDataChannels(inData,channels=params['zeroChannels']['channels'])
    if dropChannels: inData = dropDataChannels(inData,channels=params['dropChannels']['channels'])
    if combineChannels: inData = combineDataChannels(inData,channels=params['combineChannels']['channels'])
    
    # Organize data for tensorflow
    datas = (inData,outData)
    testing_data, training_data = splitdata_tf(datas,test_number=test_number,fakeRandom=fakeRandom)
    if testAll:
        testing_data = datas
    train_data = np.array(training_data[0],dtype=np.float32)
    train_labels = np.array(training_data[1]/255,dtype=np.int64)
    train_labels_exp = datas2labels(train_labels)
        
    eval_data = np.array(testing_data[0],dtype=np.float32)
    eval_labels = np.array(testing_data[1]/255,dtype=np.int64)
    eval_labels_exp = datas2labels(eval_labels)
    if train:
        convolve_wildfire_train(train_data,train_labels_exp,modelFnc,epochs=num,model_dir=model_dir)
    
    if findBestThresh:
        evalSummary, prediction_exp, truth_exp = convolve_wildfire_test(train_data,train_labels_exp,modelFnc,model_dir=model_dir)
        inputs = inputs2labels(eval_data)
        inputsImgs = arrayToImage(inputs)
        prediction = labels2probs(prediction_exp,fireThresh=0.75)
        truth = labels2datas(truth_exp,fireThresh=0.75)
        
        predictionImgs = arrayToImage(prediction,outStyle=True)
        truthImgs = arrayToImage(truth,outStyle=True)
        bestThresh, bestConfusionMatrix, threshes, fMeasures = findBestThreshold(predictionImgs,truthImgs,inputsImgs)
        plotThresholdFMeasure(threshes,fMeasures)
        print(bestThresh)
    else:
        #bestThresh = 0.28 # Test Data
        bestThresh = 0.41 # Training Data
        bestThresh = 0.0
        toUseThresh = 0.41
    
    if test:
        fs = 58
        lw = 6
        num = 0
        toPlot = True
        if num == 0: initialTime = 2997
        if num == 1: initialTime = 2994
        if num == 2: initialTime = 2982
        if num == 3: initialTime = 2967 #2970
        if num == 4: initialTime = 2955
        singleEval = np.reshape(eval_data[initialTime],(1,32500))
        singleEval_labels = np.reshape(eval_labels_exp[initialTime,:],(1,5000))

        #plt.subplot(1,2,1)
        simImgs = np.reshape(eval_labels_exp[initialTime+2,2500:].copy(),(50,50))
        simImgs[simImgs > 0] = 24
        
        simImg = np.reshape(eval_data[initialTime+2][:2500].copy(),(50,50)).copy()
        simImgs[simImg>0] = 18
        
        simImg = np.reshape(eval_data[initialTime+1][:2500].copy(),(50,50)).copy()
        simImgs[simImg>0] = 12
        simImg = np.reshape(eval_data[initialTime][:2500].copy(),(50,50)).copy()
        simImgs[simImg>0] = 6
        #simImgs[simImgs == 0] = np.nan
        
        if toPlot:
            plt.figure(figsize=(12,12))
            ax = plt.gca()
            im = ax.imshow(simImgs,cmap='hot_r',vmin=0,vmax=30)
            divider = make_axes_locatable(ax)
            plt.xticks([])
            plt.yticks([])
            #plt.tick_params(labelsize=fs)
            #plt.xlabel('km',fontsize=fs)
            #plt.ylabel('km',fontsize=fs)
            #cax = divider.append_axes("right",size="5%", pad=0.05)
            #c = plt.colorbar(im,ticks=[24,18,12,6,0],cax=cax)
            #plt.tick_params(labelsize=fs)
            #plt.ylabel('Hours',fontsize=fs)
            
            
            ax.annotate('Simulation',xy=(1,4),xycoords='data',textcoords='data',xytext=(1,4),fontsize=fs)
            
            plt.tight_layout()
            plt.savefig('..%soutputs%stimeAnalysis_simulation%.0f.eps'%(os.sep, os.sep, num))
            plt.savefig('..%soutputs%sFig9a%.0f.eps'%(os.sep, os.sep, num+1))
        
        netImgs = [simImg.copy()]
        errorImgs = []
        confusionMatrix = []
        cvsses = []
        
        t1 = uc.tic()
        evalSummary, prediction_exp, truth_exp = convolve_wildfire_test(singleEval,singleEval_labels,modelFnc,model_dir=model_dir)
        prediction = labels2probs(prediction_exp,fireThresh=1.0)
        predictionImgs = arrayToImage(prediction,outStyle=True)
        pImg = postProcessFirePerimiter(predictionImgs[0].copy(),bestThresh)
        truthImg = np.reshape(eval_labels_exp[initialTime,2500:],(50,50))
        iImg = np.reshape(singleEval[0,:2500],(50,50))
        confusionMatrix.append(findConfusionMatrix(pImg,truthImg,toUseThresh,iImg))
        pImg2 = pImg.copy()
        pImg2[pImg2 > toUseThresh] = 1.0
        pImg2[pImg2 <= toUseThresh] = 0.0
        errorImg = pImg2-truthImg
        errorImgs.append(errorImg)
        pImg[pImg>toUseThresh] = 1.0
        pImg = pImg*255        
        netImgs.append(pImg2)
        singleEval[0,0:2500] = np.reshape(pImg,(2500,))
        singleEval_labels = np.reshape(eval_labels_exp[initialTime+1,:],(1,5000))
        
        pImg2 = np.array(pImg2,dtype=np.uint8)
        tImg2 = np.array(truthImg,dtype=np.uint8)
        cvss = chanVeseFunctional(pImg2, tImg2)
        cvsses.append(cvss)
        
        evalSummary, prediction_exp, truth_exp = convolve_wildfire_test(singleEval,singleEval_labels,modelFnc,model_dir=model_dir)
        prediction = labels2probs(prediction_exp,fireThresh=1.0)
        predictionImgs = arrayToImage(prediction,outStyle=True)
        pImg = postProcessFirePerimiter(predictionImgs[0].copy(),bestThresh)
        truthImg = np.reshape(eval_labels_exp[initialTime+1,2500:],(50,50))
        iImg = np.reshape(singleEval[0,:2500],(50,50))
        confusionMatrix.append(findConfusionMatrix(pImg,truthImg,toUseThresh,iImg))
        pImg2 = pImg.copy()
        pImg2[pImg2 > toUseThresh] = 1.0
        pImg2[pImg2 <= toUseThresh] = 0.0
        errorImg = pImg2-truthImg
        errorImgs.append(errorImg)
        pImg[pImg>toUseThresh] = 1.0
        pImg = pImg*255        
        netImgs.append(pImg2)
        singleEval[0,0:2500] = np.reshape(pImg,(2500,))
        singleEval_labels = np.reshape(eval_labels_exp[initialTime+2,:],(1,5000))
        
        pImg2 = np.array(pImg2,dtype=np.uint8)
        tImg2 = np.array(truthImg,dtype=np.uint8)
        cvss = chanVeseFunctional(pImg2, tImg2)
        cvsses.append(cvss)

        evalSummary, prediction_exp, truth_exp = convolve_wildfire_test(singleEval,singleEval_labels,modelFnc,model_dir=model_dir)
        print(uc.toc(t1))
        if toPlot:
            prediction = labels2probs(prediction_exp,fireThresh=1.0)
            predictionImgs = arrayToImage(prediction,outStyle=True)
            pImg = postProcessFirePerimiter(predictionImgs[0].copy(),bestThresh)
            truthImg = np.reshape(eval_labels_exp[initialTime+2,2500:],(50,50))
            iImg = np.reshape(singleEval[0,:2500],(50,50))
            confusionMatrix.append(findConfusionMatrix(pImg,truthImg,toUseThresh,iImg))
            pImg2 = pImg.copy()
            pImg2[pImg2 > toUseThresh] = 1.0
            pImg2[pImg2 <= toUseThresh] = 0.0
            errorImg = pImg2-truthImg
            errorImgs.append(errorImg)
            pImg[pImg>toUseThresh] = 1.0
            pImg = pImg*255        
            netImgs.append(pImg2)
            
            pImg2 = np.array(pImg2,dtype=np.uint8)
            tImg2 = np.array(truthImg,dtype=np.uint8)
            cvss = chanVeseFunctional(pImg2, tImg2)
            cvsses.append(cvss)
            
            #singleEval[0,0:2500] = np.reshape(pImg,(2500,))
            #singleEval_labels = np.reshape(eval_labels_exp[2999,:],(1,5000))
            cnetImg  = netImgs[-1].copy()
            cnetImg[cnetImg > 0] = 24
            hourCounter = 24
            for i in range(len(netImgs)-2,-1,-1):
                hourCounter = hourCounter-6
                cnetImg[netImgs[i]>0] = hourCounter
            #cnetImg[cnetImg == 0] = np.nan
            plt.figure(figsize=(15,12))
            ax = plt.gca()
            im = ax.imshow(cnetImg,cmap='hot_r',vmin=0,vmax=30)
            divider = make_axes_locatable(ax)
            plt.xticks([])
            plt.yticks([])
            #plt.tick_params(labelsize=fs)
            #plt.xlabel('km',fontsize=fs)
            #plt.ylabel('km',fontsize=fs)
            cax = divider.append_axes("right",size="5%", pad=0.05)
            c = plt.colorbar(im,ticks=[24,18,12,6,0],cax=cax)
            plt.tick_params(labelsize=fs)
            plt.ylabel('Hours',fontsize=fs)
            ax.annotate('CNN',xy=(1,4),xycoords='data',textcoords='data',xytext=(1,4),fontsize=fs)
            plt.tight_layout()
            plt.savefig('..%soutputs%stimeAnalysis_network%.0f.eps'%(os.sep, os.sep, num))
            plt.savefig('..%soutputs%sFig9b%.0f.eps'%(os.sep, os.sep, num+1))
            
            confusionMatrix = np.array(confusionMatrix)
            cvsses = np.array(cvsses)
            
            plt.figure(figsize=(16,12))
            plt.plot([12,18,24],confusionMatrix[:,7],'-k',linewidth=lw,label='F-Measure');
            plt.plot([12,18,24],confusionMatrix[:,6],'-.r',linewidth=lw,label='Precision');
            plt.plot([12,18,24],confusionMatrix[:,5],'--b',linewidth=lw,label='Sensitivity');
            plt.plot([12,18,24],1-cvsses,':g',linewidth=lw+4,label='CVS');
            plt.ylim(0,1.1)
            plt.xlim(11,25)
            plt.xticks([12,18,24])
            plt.ylabel('Score',fontsize=fs)
            plt.xlabel('Hours',fontsize=fs)
            plt.legend(fontsize=fs,loc=4)
            plt.tick_params(labelsize=fs)
            plt.tight_layout()
            plt.savefig('..%soutputs%stimeAnalysis_fmeasure%.0f.eps'%(os.sep, os.sep, num))
            plt.savefig('..%soutputs%sFig9c%.0f.eps'%(os.sep, os.sep, num+1))
            
        