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
import util_common as uc
import tensorflow as tf
import pickle
import glob
import generate_dataset as gd
from generate_dataset import GriddedMeasurementPair
import psutil
import matplotlib.pyplot as plt


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
                     clim=None,closeFig=None,
                     saveFig=False,saveName=''):
    totalPlots = np.ceil(float(len(datas))**0.5)
    colPlots = totalPlots
    rowPlots = np.ceil((float(len(datas)))/colPlots)
    currentPlot = 0
    
    if saveFig:
        fntsize = 20
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
        #plt.xlabel('Longitude',fontsize=fntsize)
        #plt.ylabel('Latitude',fontsize=fntsize)
        plt.title(key,fontsize=fntsize)

        if clim is None:
            clim = np.linspace(0,1,10)
            label = ''
        else:
            label = ''
        img = ax.imshow(datas[i],cmap='jet')#,vmin=0,vmax=1)
        #img = ax.contourf(self.longitude,self.latitude,getattr(self,key),levels=clim,cmap=cmap)
        img_cb = plt.colorbar(img,ax=ax,label=label)

        img_cb.set_label(label=label,fontsize=fntsize)
        img_cb.ax.tick_params(axis='both',labelsize=fntsize)
        ax.grid(linewidth=lnwidth/4,linestyle='-.',color='k')
        for ln in ax.lines:
            ln.set_linewidth(lnwidth)
    if saveFig:
        fig.savefig(saveName)
        
    if closeFig:
        plt.clf()
        plt.close(fig)

if __name__ == "__main__":
    #import generate_dataset as gd
    #from generate_dataset import GriddedMeasurementPair
    #indir = ['C:/Users/JHodges/Documents/wildfire-research/output/GoodCandidateComparison/']
    #outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/NetworkTest/'
    
    indir = ['../networkData/20180213/']
    #outdir = '../networkData/output/'
    outdir = indir[0]+'output/'
    dataRawFile = indir[0]+'dataCluster.raw'
    
    # Define which parametric study to load
    study2use = 3
    neu = [100]
    num = 11
    lr = 10**-8
    af = 'custom'
    svdInputs = False
    basenumber = 5
    generatePlots=True
    k = 11
    
    #neu = [10000,5000,2500]
    #num = 11
    #lr = 10**-7
    ns = outdir+'custom'
    
    for n in neu:
        ns = ns+'_'+str(n)
    if svdInputs:
        ns = ns+'_svd_'+str(k)
    

    
    if svdInputs:
        dataFile = dataRawFile+'.svd' #indir[0]+'dataSvd.out'
    else:
        dataFile = dataRawFile+'.out'
    datas = gd.loadCandidates(indir,dataRawFile,forceRebuild=False)
    #datas = []
    #datas = gd.datasRemoveKey(datas,'In_WindX')
    #datas = gd.datasRemoveKey(datas,'In_WindY')
    #datas = gd.datasRemoveKey(datas,'In_VegetationIndexA')
    #datas = gd.datasRemoveKey(datas,'In_Elevation')
    newDatas, keys = gd.rearrangeDataset(datas,dataFile,svdInputs=svdInputs,k=k,forceRebuild=False)
    
    # Choose what networks to train
    type_one = True
    type_two = True
    
    test_data, train_data = network_wildfire_train(newDatas,ns,dataFile,af,tn=5,neu=neu,num=num,lr=lr)
    
    if generatePlots:
        test_prediction = network_wildfire_test(test_data,ns)
        train_data[0][0,0:2500]=0
        train_data[0][1,2500:5000]=0
        train_data[0][2,5000:7500]=0
        train_data[0][3,7500:10000]=0
        train_data[0][4,10000:12500]=0
        train_prediction = network_wildfire_test(train_data,ns)
        
        test_prediction_rm = []
        dataSize = int(test_prediction[0].shape[0]**0.5)
        
    
        
        if not svdInputs:
            datas, names = gd.datasKeyRemap(test_data, keys)
            names.append('Network Fire Mask')
        else:
            datas = []
            names = []
            names.extend(['Input Data','Validation Data','Network Fire Mask'])
        for toPlot in range(0,test_data[0].shape[0]):
            if svdInputs:
                svdData = test_data[0][toPlot].copy()
                svdData = svdData[0:int(len(svdData)/basenumber)]
                inData = gd.reconstructImg(svdData,k=k)
                outData = gd.reconstructImg(test_prediction[toPlot],k=k)
                valData = gd.reconstructImg(test_data[1][toPlot].copy(),k=k)
                datas.append([inData,valData])
            else:
                
                #inData = np.reshape(test_data[0][toPlot][0:2500],(dataSize,dataSize))
                outData = np.reshape(test_prediction[toPlot],(dataSize,dataSize))
                #valData = np.reshape(test_data[1][toPlot][0:2500],(dataSize,dataSize))
            #outData[outData<0]=0
            data = datas[toPlot]
            data.extend([outData])
            
            plotWildfireTest(data,names,
                             saveFig=True,saveName=ns+'testData_'+str(toPlot)+'.png')
        if not svdInputs:
            datas, names = gd.datasKeyRemap(train_data, keys)
            names.append('Network Fire Mask')
        else:
            datas = []
            names = []
            names.extend(['Input Data','Network Fire Mask','Validation Data'])
        for toPlot in range(0,test_data[0].shape[0]):
            if svdInputs:
                svdData = train_data[0][toPlot].copy()
                svdData = svdData[0:int(len(svdData)/basenumber)]
                inData = gd.reconstructImg(svdData,k=k)
                outData = gd.reconstructImg(train_prediction[toPlot],k=k)
                valData = gd.reconstructImg(train_data[1][toPlot].copy(),k=k)
                datas.append([inData,valData])
            else:
                outData = np.reshape(train_prediction[toPlot],(dataSize,dataSize))
            data = datas[toPlot]
            data.extend([outData])
            plotWildfireTest(data,names,
                             saveFig=True,saveName=ns+'trainData_'+str(toPlot)+'.png')