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
import pandas as pd
import yaml
#from util_common import tic, toc
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os

def tic():
    ''' tic: This function stores the current time to the microsecond level
    
        OUTPUTS:
          old_time: float containing the current time to the microsecond level
    '''
    import datetime
    old_time = datetime.datetime.now().time().hour*3600+datetime.datetime.now().time().minute*60+datetime.datetime.now().time().second+datetime.datetime.now().time().microsecond/1000000
    return old_time

def toc(old_time):
    ''' toc: This function returns the amount of time since the supplied time
    
        INPUTS:
          old_time: float containing previous time
        
        OUTPUTS:
          dur: float containing time since previous time
    '''
    new_time = tic()
    dur = new_time-old_time
    return dur

def read_scored_data(input_file,first_header='Output'):
    ''' read_scored_data: This function reads scored data
    
        Inputs:
          input_file: string containing path to score data.
          first_header: header of score for first reference shape.
        Outputs:
          data: list containing scores for each reference shape.
    '''
    raw_data = pd.read_csv(input_file)
    param_end = min([i for i, s in enumerate(raw_data.columns) if first_header in s])
    loaded_params = raw_data.values[:,1:param_end]
    loaded_scores = raw_data.values[:,param_end:]
    params = []
    scores = []
    data = []
    for i in range(0,len(loaded_params)):
        params.append(np.reshape(loaded_params[i].T,(loaded_params.shape[1],1)))
        scores.append(np.reshape(loaded_scores[i].T,(loaded_scores.shape[1],1)))
        data.append([params[i],scores[i]])
    return data

def rearrange_data_tf(data):
    ''' rearrange_data_tf: This function formats the input data into that
        required as input to tensorflow.
        
        Inputs:
          data: list of data from raw format
        Outputs:
          data_tf: tuple containing data in format for tensorflow
    '''
    xdim = len(data[0][0])
    ydim = len(data[0][1])
    
    data_x = np.zeros((len(data),xdim))
    data_y = np.zeros((len(data),ydim))
    
    for d in range(0,len(data)):
        data_x[d,:] = np.reshape(data[d][0],(xdim,))
        data_y[d,:] = np.reshape(data[d][1],(ydim,))
    
    data_tf = (data_x,data_y)
    return data_tf

def splitdata(data,test_number=None):
    ''' splitdata: This function will split the data into test and training
        sets.
        
        Inputs:
          data: list of data from raw format
          test_number: number of samples to withold for testing. If none, half
            the data is used.
        Outputs:
          test_data: portion of input data for testing
          training_data: portion of input data for training
    '''
    total_len = len(data)
    
    if test_number is None:
        random_inds = np.array(np.round(np.random.rand(int(total_len/2),1)*total_len,decimals=0)-1,dtype=np.int64)
    else:
        random_inds = np.array(np.round(np.random.rand(test_number,1)*total_len,decimals=0)-1,dtype=np.int64)
    random_inds = np.array(np.unique(random_inds),dtype=np.int64)
    
    test_data = []
    for i in range(0,len(random_inds)):
        test_data.append(data[random_inds[i]])
    training_data = []
    for i in range(0,total_len):
        if i not in random_inds:
            training_data.append(data[i])
    return test_data, training_data

def splitdata_tf(data,test_number=None):
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
    w2, b2, af = pickle.load(f)
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
    return w,b,af,dims,ydim

def tensorflow_network(data,num=1001,neurons=None,test_number=None,ns='',
                       train=True,learning_rate=0.00001,
                       activation_function='relu'):
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
    else:
        assert False, 'Did not recognize input format. See documentation.'
        
    # Start tensorflow session
    sess = tf.Session()
    
    if train:
        # Determine input and output layer dimensions
        dims = data[0].shape[1]
        ydim = data[1].shape[1]
        
        # Split and arrange data
        test_data, training_data = splitdata_tf(data,test_number=test_number)
        
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
    else:
        # Import saved network parameters
        w1,b1,activation_function,dims,ydim = import_wb("./"+ns+"model.pkl",sess)
        w2,b2 = extract_wb(w1,b1,sess)
        if type(data) is tuple:
            test_data = data[0]
        else:
            test_data = data
    
    X = tf.placeholder("float", shape=[None, dims])
    y = tf.placeholder("float", shape=[None, ydim])
    
    # Forward propagation
    
    if activation_function == 'sigmoid':
        yhat = forwardprop(X, w1, b1)
    elif activation_function == 'relu':
        yhat = forwardprop_relu(X, w1, b1)
    else:
        yhat = forwardprop_relu(X, w1, b1)
    
    if train:
        # Backward propagation
        cost = (y-yhat)**2
        updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        # Initialize everything else
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Perform num training epochs
        for epoch in range(num):
            sess.run(updates, feed_dict={X: training_data[0], y: training_data[1]})
            if epoch % int(num/10) == 0:
                #train_accuracy = np.mean(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]}))**2)**0.5
                #test_accuracy  = np.mean(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]}))**2)**0.5
                train_accuracy = np.mean(abs(training_data[1]-sess.run(yhat, feed_dict={X: training_data[0]})))
                test_accuracy  = np.mean(abs(test_data[1]-sess.run(yhat, feed_dict={X: test_data[0]})))
                #print(test_data[1][0])
                #print(sess.run(yhat, feed_dict={X: test_data[0]})[0])
                print("Epoch = %d, train rmse = %.2f, test rmse = %.2f"
                      % (epoch + 1, train_accuracy, test_accuracy))
        
        # Save network parameters using pickle
        save_path = "./"+ns+"model.pkl"
        with open(save_path,'wb') as f:
            w2,b2 = extract_wb(w1,b1,sess)
            pickle.dump([w2,b2,activation_function],f)
            
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
    
def apply_weights(data,weights):
    ''' apply_weights: This function will apply weights to scored data
        
        Inputs:
          data: list of data from raw format
          weights: numpy array of weights
          
        Outputs:
          w_scores: numpy array of weighted scores
    '''
    w_scores = []
    for i in range(0,len(data)):
        score = np.reshape(data[i][1],(3,int(data[i][1].shape[0]/3)))
        w_scores.append(np.reshape(np.multiply(score,weights),(score.shape[1]*3,)))
    w_scores = np.array(w_scores)
    return w_scores

def scaled_scores(data,args=defaultdict(bool)):
    ''' scaled_scores: This function converts raw scores to percentiles
        
        Inputs:
          data: list of data from raw format
          args: defaultdict() with criteria and datarefs
          
        Outputs:
          data_scaled: percentile scores
          data_for_weights: separate data format for weight estimation
          scalefactors: minimums and maximums to convert percentiles to raw
            scores
    '''
    
    # Extract criteria and datarefs from arguments
    if args['criteria'] == False:
        criteria = ['DoG','Bif','L2','Li','L2-*','Li-*']
    else:
        criteria = args['criteria']
    if args['datarefs'] == False:
        datarefs = ['P0','P1','P2']
    else:
        datarefs = args['datarefs']
    total_criteria = len(criteria)
    total_datarefs = len(datarefs)
    
    # Scores to scoretable
    raw_scores = []
    for i in range(0,len(data)):
        raw_score = np.reshape(data[i][1],(total_datarefs,int(data[i][1].shape[0]/total_datarefs)))
        raw_scores.append(raw_score)
    
    # Arrange scores
    arranged_scores = []
    for i in range(0,len(raw_scores)):
        for j in range(0,raw_scores[i].shape[0]):
            arranged_scores.append(raw_scores[i][j,:])
    arranged_scores = np.array(arranged_scores).T
    scaled_scores = arranged_scores.copy()

    # Determine scalefactors    
    scalefactors = []
    for i in range(0,total_criteria):
        mn = min(scaled_scores[i,:])
        mx = max(scaled_scores[i,:])
        scaled_scores[i,:]= (scaled_scores[i,:]-mn)/(mx-mn)
        scalefactors.append([mn,mx])
    
    # Rearrange data to format for weight estimation
    data_for_weights = []
    for i in range(0,arranged_scores.shape[1]):
        scaled_tmp = np.reshape(scaled_scores[:,i],(scaled_scores[:,i].shape[0],1))
        raw_tmp = np.reshape(arranged_scores[:,i],(arranged_scores[:,i].shape[0],1))
        data_for_weights.append([raw_tmp,scaled_tmp])
    
    # Calculate data percentiles
    data_scaled = []
    for i in range(0,len(data)):
        scaled_tmp = np.concatenate((scaled_scores[:,3*i],scaled_scores[:,3*i+1],scaled_scores[:,3*i+2]))
        scaled_tmp = np.reshape(scaled_tmp,(len(scaled_tmp),1))
        data_scaled.append([data[i][0],scaled_tmp])
    
    return data_scaled, data_for_weights, scalefactors

def linear_weights(data_scaled):
    ''' linear_weights: This function calculates optimal weights using linear
        regression.
        
        Inputs:
          data_scaled: list of data from raw format
          
        Outputs:
          weights: numpy array containing weights to convert raw scores to
            scaled scores
    '''
    
    # Reorganize data
    scores = []
    for i in range(0,len(data_scaled)):
        scores.append(data_scaled[i][1])
    scores = np.reshape(np.array(scores),(len(data_scaled),data_scaled[0][1].shape[0])).T
    raw_scores = []
    for i in range(0,len(data_scaled)):
        raw_scores.append(data_scaled[i][0])
    raw_scores = np.reshape(np.array(raw_scores),(len(data_scaled),data_scaled[0][0].shape[0])).T
    
    # Specify design scores
    design_scores = np.mean(scores,axis=0)*1.0
    
    # Perform linear regression
    weights,residuals,rank,s = np.linalg.lstsq(raw_scores.T,design_scores)
    
    return weights

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
    h = tf.nn.sigmoid(tf.add(tf.matmul(X,w[0]),b[0]))
    for i in range(1,len(w)-1):
        h = tf.nn.sigmoid(tf.add(tf.matmul(h,w[i]),b[i]))
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

def percentile2scoretable(percentile, args=defaultdict(bool)):
    ''' percentile2scoretable: This function rearranges a list of percentiles
        into a scoretable
        
        Inputs:
          percentile: list of percentiles
          args: defaultdict() with datarefs
        
        Outputs:
          scoretable: list of scoretables
    '''
    
    # Extract arguments
    if args['datarefs'] == False:
        datarefs = ['P0','P1','P2']
    else:
        datarefs = args['datarefs']
    total_datarefs = len(datarefs)
    
    # Reshape data
    sz = np.shape(percentile)
    scoretable = []
    for i in range(0,sz[0]):
        scoretable.append(np.reshape(percentile[i],(total_datarefs,int(sz[1]/total_datarefs))))
    
    return scoretable

def scoretable2score(scoretable, args=defaultdict(bool)):
    ''' scoretable2score: This function calculates aggregate from scoretable
        
        Inputs:
          scoretable: list of scoretables
          args: defaultdict() with agg_type
        
        Outputs:
          score: list of scoretables with aggregate
    '''
    # Extract arguments
    if args['agg_type'] == False:
        agg = 'Sum'
    else:
        agg = args['agg_type']
    
    # Loop through scoretables
    score = []
    sz = np.shape(scoretable[0])
    for i in range(0,len(scoretable)):
        score_tmp = np.zeros((sz[0]+1,sz[1]+1))
        score_tmp[0:-1,1:] = scoretable[i]
        if agg == 'Mean':
            score_tmp[0:-1,0] = np.mean(scoretable[i],axis=1)
            score_tmp[-1,1:] = np.mean(scoretable[i],axis=0)
            score_tmp[-1,0] = np.mean(scoretable[i])
        if agg == 'Sum':
            score_tmp[0:-1,0] = np.sum(scoretable[i],axis=1)
            score_tmp[-1,1:] = np.sum(scoretable[i],axis=0)
            score_tmp[-1,0] = np.sum(scoretable[i])
        score.append(score_tmp)
    return score

def scoretable2pressure(scoretable,col=0,args=defaultdict(bool)):
    ''' scoretable2pressure: This function calculates aggregate for each
        reference shape
        
        Inputs:
          scoretable: list of scoretables
          col: column to use for aggregate
          args: defaultdict() with agg_type
        
        Outputs:
          score: list of scoretables with aggregate
    '''
    # Extract arguments
    if args['agg_type'] == False:
        agg = 'Sum'
    else:
        agg = args['agg_type']
    
    # Loop through scoretables
    pressure_scores = []
    sz = np.shape(scoretable[0])
    for i in range(0,len(scoretable)):
        score_tmp = np.zeros((sz[0]+1,sz[1]+1))
        score_tmp[0:-1,1:] = scoretable[i]
        if agg == 'Mean':
            score_tmp[0:-1,0] = np.mean(scoretable[i],axis=1)
        if agg == 'Sum':
            score_tmp[0:-1,0] = np.sum(scoretable[i],axis=1)
        pressure_scores.append([score_tmp[0,col],score_tmp[1,col],score_tmp[2,col]])
    return pressure_scores

def allscores2overall(allscore,col=0):
    ''' allscores2overall: This function calculates the overall score
        
        Inputs:
          allscore: list of scoretables
          col: column to use for aggregate
        
        Outputs:
          score: list of overall scores
    '''
    score = []
    for s in allscore:
        score.append(s[-1,col])
    return score

def rescale_scores(allscores,scalefactors,weights,args=defaultdict(bool)):
    ''' rescale_scores: This function re-scales scores
        
        Inputs:
          allscores: list of scoretables
          scalefactors: list with scale factors
          weights: numpy array of weights
          args: defaultdict() with datarefs
          
        Outputs:
          rescale_scores: list of rescaled scoretables
    '''
    
    # Extract arguments
    if args['datarefs'] == False:
        datarefs = ['P0','P1','P2']
    else:
        datarefs = args['datarefs']
    total_datarefs = len(datarefs)
    
    # Loop through scoretables
    rescale_scores = []
    for j in range(0,len(allscores)):
        scores = allscores[j]
        scores_tmp = np.reshape(scores,(total_datarefs,int(len(scores)/total_datarefs)))
        for i in range(0,len(scalefactors)):
            scores_tmp[:,i] = scores_tmp[:,i]*(scalefactors[i][1]-scalefactors[i][0])+scalefactors[i][0]
            scores_tmp[:,i] = np.multiply(scores_tmp[:,i],weights[i])
        scores_tmp = np.reshape(scores_tmp,((len(scores),)))
        rescale_scores.append(scores_tmp)
    return rescale_scores

def network_weight_estimate(data,ns,tn=10,pf=True):
    neu = [30]
    lr = 0.000001
    num = 10001
    t1 = tic()
    train_data, test_data,_,tp = tensorflow_network(data,ns=ns,neurons=neu,num=num,test_number=tn,learning_rate=lr)
    print("network_weight_estimate time:",toc(t1))
    if pf:
        plt.figure()
        plt.scatter(test_data[1][0],tp[0])
        plt.plot(test_data[1][0],test_data[1][0])
        plt.title('Weight Estimate (TensorFlow)')
    
def network_score_from_param_train(data,ns,tn=10):
    neu = [30,30,30]
    lr = 0.00001
    num = 10001
    af='relu'
    t1 = tic()
    train_data, test_data, save_path, tp2 = tensorflow_network(data,ns=ns,neurons=neu,num=num,test_number=tn,learning_rate=lr,activation_function=af)
    print("Tensor flow param->score time:",toc(t1))
    plt.figure()
    plt.plot(test_data[1][0],test_data[1][0])
    plt.xlabel('True Scaled Score')
    plt.ylabel('Pred Scaled Score')
    plt.title('Score Estimate (TensorFlow)')
    return test_data
    
def network_score_from_param_test(data,ns):
    t1 = tic()
    test_prediction, test_data2 = tensorflow_network(data,train=False,ns=ns)
    print("Tensor flow retest param->score time:",toc(t1))
    plt.figure(figsize=(12,8))
    plt.plot(data[1][0],data[1][0])
    plt.xlabel('True Scaled Score')
    plt.ylabel('Pred Scaled Score')
    plt.title('Score Estimate (TensorFlow)')
    for i in range(0,len(test_prediction)):
        plt.scatter(data[1][i],test_prediction[i])

if __name__ == "__main__":
  
    # Define which parametric study to load
    study2use = 2
    
    # Choose what networks to train
    type_one = False
    type_two = True
    
    # Load parametric study information
    in_params = yaml.load(open('default.yaml','r'))
    if study2use == 0:
        logname = 'network1'
        input_file = 'network1'
        plot_name = 'network1_'
    if study2use == 1:
        logname = 'network2'
        input_file = 'network2'
        plot_name = 'network2_'
    if study2use == 2:
        logname = 'network3'
        input_file = 'network3'
        plot_name = 'network3_'
        
    args = defaultdict(bool,in_params['args'])
    
    # Read scored data
    data = read_scored_data(input_file+'.csv')
    
    # Scale data based on set of simulations
    data_scaled, data_for_weights, scalefactors = scaled_scores(data,args=args)
    data_scaled = rearrange_data_tf(data_scaled.copy())
    
    # Determine weights solving linear system
    weights = linear_weights(data_for_weights)
    data_for_weights = rearrange_data_tf(data_for_weights.copy())
    
    # Calculate weighted scores
    data_weighted = apply_weights(data,weights)
    
    # Use network to learn how to scale scores

    if type_one:
        network_weight_estimate(data_for_weights,input_file+'weights',tn=5)
        
    if type_two:
        test_data = network_score_from_param_train(data_scaled,input_file+'scores',tn=5)
        network_score_from_param_test(test_data,input_file+'scores')
        
        