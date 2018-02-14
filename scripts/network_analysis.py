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
        print("Loading data from file %s"%(ds+'data.out'))
    else:
        #print("Did not recognize input format. See documentation.")
        assert False, 'Did not recognize input format. See documentation.'
    
    if glob.glob(ns+'model.pkl') and glob.glob(ds+'data.out') and continue_train and train:
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
        if not continue_train:
            init = tf.global_variables_initializer()
            sess.run(init)
            #with open("./"+ds+"data.out",'wb') as f:
            #    pickle.dump([test_data,training_data],f)
        # Perform num training epochs
        for epoch in range(num):
            sess.run(updates, feed_dict={X: training_data[0], y: training_data[1]})
            if epoch % int(num/10) == 0:
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
    
def rearrangeDataset(datas,debugPrint=False,svdInputs=False,k=25):
    
    
    allInputs = []
    allOutputs = []
    for i in range(0,len(datas)):
        if psutil.virtual_memory()[2] < 80.0:
            data = datas[i]
            inputs = []
            outputs = []
            nanError = False
            nanKey = ''
            for key in data.__dict__.keys():
                if "In_" in key:
                    d = getattr(data,key)
                    sz = np.shape(d)
                    d = np.reshape(d,(sz[0]*sz[1],))
                    if len(np.where(np.isnan(d))[0]) > 0:
                        #print(key)
                        #nanError = True
                        if np.isnan(np.nanmin(d)):
                            nanError = True
                            nanKey = nanKey+key+", "
                        d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                    if svdInputs:
                        d = np.reshape(d,(sz[0],sz[1]))
                        d = im2vector(d,k=k)
                    if d is not None:
                        inputs.extend(d)
                    else:
                        nanError = True
                if "Out_" in key:
                    d = getattr(data,key)
                    sz = np.shape(d)
                    d = np.reshape(d,(sz[0]*sz[1],))
                    if len(np.where(np.isnan(d))[0]) > 0:
                        #print(key)
                        #nanError = True
                        if np.isnan(np.nanmin(d)):
                            nanError = True
                            nanKey = nanKey+key+", "
                        d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                    if svdInputs:
                        d = np.reshape(d,(sz[0],sz[1]))
                        d = im2vector(d,k=k)
                    outputs.extend(d)
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            if not nanError:
                allInputs.append(inputs)#[0:5000])
                allOutputs.append(outputs)
            else:
                dataTime = data.strTime()[0]
                if debugPrint:
                    print("nanError at: %s for keys: %s"%(dataTime,nanKey))
        else:
            print("Not enough memory to reshape.")
    print(np.shape(allInputs),np.shape(allOutputs))
    newDatas = (np.array(allInputs),np.array(allOutputs))
    return newDatas

def rearrangeDatasetAF(datas,debugPrint=False,svdInputs=False,k=25):
    
    
    allInputs = []
    allOutputs = []
    for i in range(0,len(datas)):
        if psutil.virtual_memory()[2] < 90.0:
            data = datas[i]
            inputs = []
            outputs = []
            nanError = False
            nanKey = ''
            for key in data.__dict__.keys():
                if "In_FireMask" in key:
                    d = getattr(data,key)
                    sz = np.shape(d)
                    d = np.reshape(d,(sz[0]*sz[1],))
                    if len(np.where(np.isnan(d))[0]) > 0:
                        #print(key)
                        #nanError = True
                        if np.isnan(np.nanmin(d)):
                            nanError = True
                            nanKey = nanKey+key+", "
                        d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                    if svdInputs:
                        d = np.reshape(d,(sz[0],sz[1]))
                        d = im2vector(d,k=k)
                    if d is not None:
                        inputs.extend(d)
                    else:
                        nanError = True
                if "Out_FireMask" in key:
                    d = getattr(data,key)
                    sz = np.shape(d)
                    d = np.reshape(d,(sz[0]*sz[1],))
                    if len(np.where(np.isnan(d))[0]) > 0:
                        #print(key)
                        #nanError = True
                        if np.isnan(np.nanmin(d)):
                            nanError = True
                            nanKey = nanKey+key+", "
                        d[np.where(np.isnan(d))[0]] = np.nanmin(d)
                    if svdInputs:
                        d = np.reshape(d,(sz[0],sz[1]))
                        d = im2vector(d,k=k)
                    outputs.extend(d)
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            if not nanError:
                allInputs.append(inputs)#[0:5000])
                allOutputs.append(outputs)
            else:
                dataTime = data.strTime()[0]
                if debugPrint:
                    print("nanError at: %s for keys: %s"%(dataTime,nanKey))
        else:
            print("Not enough memory to reshape.")
    
    print(np.shape(allInputs),np.shape(allOutputs))
    newDatas = (np.array(allInputs),np.array(allOutputs))
    return newDatas

def im2vector(img,k=10):
    data = []
    try:
        u,s,v = np.linalg.svd(img)
        u = np.reshape(u[:,:k],(u.shape[0]*k,))
        v = np.reshape(v[:k,:],(v.shape[0]*k,))
        s = s[:k]
        data.extend(u)
        data.extend(v)
        data.extend(s)
        return np.array(data)
    except np.linalg.LinAlgError:
        return None

def reconstructImg(img,k=10):
    sz = int((np.shape(img)[0]-k)/(2*k))
    u = np.reshape(img[0:sz*k],(sz,k))
    v = np.reshape(img[sz*k:2*sz*k],(k,sz))
    s = img[2*sz*k:]
    data = np.dot(u,np.dot(np.diag(s),v))
    return data

def network_wildfire_train(data,ns,ds,neu=[100,100,100],tn=10,num=11,lr=10**-7):
    af='sigmoid'
    cf='sae'
    #for n in neu:
    #    ns = ns+'_'+str(n)
    t1 = uc.tic()
    train_data, test_data, save_path, tp2 = tensorflow_network(
            data,ns=ns,neurons=neu,num=num,test_number=tn,learning_rate=lr,
            activation_function=af,comparison_function=cf,
            fakeRandom=True,ds=ds)
    print("Tensor flow param->score time:",uc.toc(t1))
    plt.figure()
    plt.plot(test_data[1][0],test_data[1][0])
    plt.xlabel('True Scaled Score')
    plt.ylabel('Pred Scaled Score')
    plt.title('Score Estimate (TensorFlow)')
    return test_data

def network_wildfire_test(data,ns):
    t1 = uc.tic()
    test_prediction, test_data2 = tensorflow_network(data,train=False,ns=ns)
    print("Tensor flow retest param->score time:",uc.toc(t1))
    plt.figure(figsize=(12,8))
    d = data[1][0].copy()
    d[d<7] = 0
    d[d>=7] = 1
    plt.plot(d,test_prediction[0])
    plt.xlabel('Measured Active Fire Index')
    plt.ylabel('Predicted Active Fire Index')
    #plt.title('Score Estimate (TensorFlow)')
    #for i in range(0,len(test_prediction)):
    #    plt.scatter(d[i],test_prediction[i])
    return test_prediction

if __name__ == "__main__":
    #import generate_dataset as gd
    #from generate_dataset import GriddedMeasurementPair
    #indir = ['C:/Users/JHodges/Documents/wildfire-research/output/GoodCandidateComparison/']
    #outdir = 'C:/Users/JHodges/Documents/wildfire-research/output/NetworkTest/'
    
    indir = ['../networkData/20180213/']
    outdir = '../networkData/output/'
    
    # Define which parametric study to load
    study2use = 3
    neu = [100]
    num = 11
    lr = 10**-7
    ns = outdir+'test'
    ds = indir[0]
    for n in neu:
        ns = ns+'_'+str(n)
    
    svdInputs = False
    k = 25
    
    if svdInputs:
        file = ds+'dataSvd.out'
    else:
        file = ds+'data.out'
    if not glob.glob(file):
        datas = gd.loadCandidates(indir)
        if psutil.virtual_memory()[2] < 60:
            with open(ds+'dataraw.out','wb') as f:
                pickle.dump(datas,f)
        else:
            print("Not enough memory available.")
        newDatas = rearrangeDataset(datas,svdInputs=svdInputs,k=k,debugPrint=True)
        with open(file,'wb') as f:
            pickle.dump(newDatas,f)
    else:
        with open(file,'rb') as f:
            print("Loading data from %s"%(file))
            newDatas = pickle.load(f)
    
    # Choose what networks to train
    type_one = True
    type_two = True
    
    test_data = network_wildfire_train(newDatas,ns,ds,tn=5,neu=neu,num=num,lr=lr)
    test_prediction = network_wildfire_test(test_data,ns)
    
    test_prediction_rm = []
    dataSize = int(test_prediction[0].shape[0]**0.5)
    

    
    
    toPlot = 2
    
    basenumber = 5
    
    if svdInputs:
        svdData = test_data[0][toPlot].copy()
        svdData = svdData[0:int(len(svdData)/basenumber)]
        inData = reconstructImg(svdData,k=k)
        outData = reconstructImg(test_prediction[toPlot],k=k)
        valData = reconstructImg(test_data[1][toPlot].copy(),k=k)
    else:
        inData = np.reshape(test_data[0][toPlot][0:2500],(dataSize,dataSize))
        outData = np.reshape(test_prediction[toPlot],(dataSize,dataSize))
        valData = np.reshape(test_data[1][toPlot][0:2500],(dataSize,dataSize))
    plt.figure(figsize=(12,8))
    plt.imshow(inData,cmap='jet')
    #plt.contourf(theData,cmap='jet',levels=np.linspace(0,1,10))
    plt.colorbar()
    
    plt.figure(figsize=(12,8))
    plt.imshow(outData,cmap='jet')
    #plt.contourf(test_prediction_rm[toPlot],cmap='jet',levels=np.linspace(0,1,10))
    plt.colorbar()
    
    #theData = np.reshape(test_data[1][toPlot][0:2500],(dataSize,dataSize))
    plt.figure(figsize=(12,8))
    plt.imshow(valData,cmap='jet')
    #plt.contourf(theData,cmap='jet',levels=np.linspace(0,1,10))
    plt.colorbar()
    