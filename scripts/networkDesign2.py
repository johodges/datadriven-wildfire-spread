# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:05:20 2018

@author: JHodges
"""

import tensorflow as tf

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