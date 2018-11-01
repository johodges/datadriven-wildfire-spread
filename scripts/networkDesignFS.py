# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:05:20 2018

@author: JHodges
"""

import tensorflow as tf

def cnnModelFS2(features, labels, mode):
    """Model function for CNN."""
    
    dconv = True
    #n_dimensions = 13
    sz = features["x"].get_shape()
    n_dimensions = sz[3]
    
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [10,10]
    ks2 = [10,10]
    ks3 = [90,90]
    #ks4 = [137,137]
    fs1 = 16
    fs2 = 16
    fs3 = 1
    #fs4 = 1
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz[1], sz[2], n_dimensions])
    print("Input Layer Dimensions:\t",input_layer.shape)
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=dropOut_layer,
            filters=fs1,
            strides=(3,3),
            kernel_size=ks1,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv1")
    print("First Conv Layer Dim:\t",conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=3)
    print("First Pool Layer Dim:\t",pool1.shape)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=fs2,
            strides=(3,3),
            kernel_size=ks2,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv2")
    print("Second Conv Layer Dim:\t", conv2.shape)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=3)
    print("Second Pool Layer Dim:\t", pool2.shape)
    pool2flat = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
    
    if dconv:
        pool2flatL = int(pool2flat.shape[1])
        pool2flatSR = int(pool2flatL**0.5)
        pool2flatL = int(pool2flatSR**2)
        dense1 = tf.layers.dense(inputs=pool2flat, units=pool2flatL, activation=tf.nn.leaky_relu)
        print("Classify Layer Dim:\t", dense1.shape)
        dense1_rs = tf.reshape(dense1,[-1,pool2flatSR,pool2flatSR,1])
        dconv1 = tf.layers.conv2d_transpose(
            inputs=dense1_rs,filters=fs3,
            kernel_size=ks3,
            strides=(19,19),
            padding="valid",
            activation=tf.nn.leaky_relu,
            name="dconv1")
        print("Deconv Layer 1 Dim:\t", dconv1.shape)
        #dconv2 = tf.layers.conv2d_transpose(
        #    inputs=dconv1,filters=fs4,
        #    kernel_size=ks4,
        #    strides=(3,3),
        #    padding="valid",
        #    activation=tf.nn.leaky_relu,
        #    name="dconv2")
        #print("Deconv Layer 2 Dim:\t", dconv2.shape)
        denseOut = tf.reshape(dconv1,[-1,dconv1.shape[1],dconv1.shape[2],dconv1.shape[3]])
        #dconv1flat = tf.reshape(dconv1,[-1,dconv1.shape[1]*dconv1.shape[2]*dconv1.shape[3]])
        #denseOut = tf.layers.dense(inputs=dconv1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
        print("Output Layer Dim:\t",denseOut.shape)
        #print("Input Layer Dimensions:\t",input_layer.shape)
        #print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
        #print("First Conv Layer Dim:\t",conv1.shape)
        #print("First Pool Layer Dim:\t",pool1.shape)
        #print("Second Conv Layer Dim:\t", conv2.shape)
        #print("Second Pool Layer Dim:\t", pool2.shape)
        #print("Classify Layer Dim:\t", dense1.shape)
        #print("Deconv Layer Dim:\t", dconv1.shape)
        #print("Output Layer Dim:\t",denseOut.shape)
    else:
        denseOut = tf.layers.dense(inputs=pool2flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
    
    logits = tf.reshape(denseOut,[-1,sz[1],sz[2]])
    predicted_classes = tf.round(logits)
            
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    print("Predicted Classes:\t",predicted_classes.shape)
    print("Labels:\t",labels.shape)
    loss = tf.reduce_sum((labels-logits)**2)**0.5
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(predicted_classes_rs,tf.float32))**2)**0.5
    
    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
      
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-4)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
def cnnModelFS(features, labels, mode):
    """Model function for CNN."""
    dconv = True
    #n_dimensions = 13
    try:
        sz = features["x"].get_shape()
        n_dimensions = sz[3]
        input_layer = tf.reshape(features["x"], [-1, sz[1], sz[2], n_dimensions])
    except:
        sz = features.shape
        n_dimensions = sz[3]
        input_layer = tf.reshape(features, [-1, sz[1], sz[2], n_dimensions])
    
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [30,30]
    ks2 = [10,10]
    ks3 = [50,50]
    dcs = (16,16)
    #ks3 = [99,99]
    #dcs = (32,32)
    #ks4 = [137,137]
    fs1 = 16
    fs2 = 16
    fs3 = 1
    #fs4 = 1
    
    # Input Layer
    
    print("Input Layer Dimensions:\t",input_layer.shape)
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=dropOut_layer,
            filters=fs1,
            strides=(10,10),
            kernel_size=ks1,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv1")
    print("First Conv Layer Dim:\t",conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[10, 10], strides=5)
    print("First Pool Layer Dim:\t",pool1.shape)
    pool1flat = tf.reshape(pool1,[-1,pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])
    
    if dconv:
        pool2flatL = int(pool1flat.shape[1])
        pool2flatSR = int(pool2flatL**0.5)
        pool2flatL = int(pool2flatSR**2)
        dense1 = tf.layers.dense(inputs=pool1flat, units=2500, activation=tf.nn.leaky_relu)
        print("Classify Layer Dim:\t", dense1.shape)
        dense1_rs = tf.reshape(dense1,[-1,50,50,1])
        dconv1 = tf.layers.conv2d_transpose(
            inputs=dense1_rs,filters=fs3,
            kernel_size=ks3,
            strides=dcs,
            padding="valid",
            activation=tf.nn.leaky_relu,
            name="dconv1")
        print("Deconv Layer 1 Dim:\t", dconv1.shape)
        #dconv2 = tf.layers.conv2d_transpose(
        #    inputs=dconv1,filters=fs4,
        #    kernel_size=ks4,
        #    strides=(3,3),
        #    padding="valid",
        #    activation=tf.nn.leaky_relu,
        #    name="dconv2")
        #print("Deconv Layer 2 Dim:\t", dconv2.shape)
        denseOut = tf.reshape(dconv1,[-1,dconv1.shape[1],dconv1.shape[2],dconv1.shape[3]])
        #dconv1flat = tf.reshape(dconv1,[-1,dconv1.shape[1]*dconv1.shape[2]*dconv1.shape[3]])
        #denseOut = tf.layers.dense(inputs=dconv1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
        print("Output Layer Dim:\t",denseOut.shape)
        #print("Input Layer Dimensions:\t",input_layer.shape)
        #print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
        #print("First Conv Layer Dim:\t",conv1.shape)
        #print("First Pool Layer Dim:\t",pool1.shape)
        #print("Second Conv Layer Dim:\t", conv2.shape)
        #print("Second Pool Layer Dim:\t", pool2.shape)
        #print("Classify Layer Dim:\t", dense1.shape)
        #print("Deconv Layer Dim:\t", dconv1.shape)
        #print("Output Layer Dim:\t",denseOut.shape)
    else:
        denseOut = tf.layers.dense(inputs=pool1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
    
    logits = tf.reshape(denseOut,[-1,sz[1],sz[2]])
    predicted_classes = tf.round(logits)
            
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    print("Predicted Classes:\t",predicted_classes.shape)
    print("Labels:\t",labels.shape)
    loss = tf.reduce_sum((labels-logits)**2)**0.5
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(predicted_classes_rs,tf.float32))**2)**0.5
    
    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    #tf.get_variable_scope().reuse_variables()
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
      
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-7)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def cnnModelFS_iter(features, labels, mode):
    """Model function for CNN."""
    
    print("GOTHERE1")
    
    files = features["files"]
    batchSize = features["batchSize"]
    
    # Offset data from zero
    eps = 10**-12
  
    print("GOTHERE")

    dataset = tf.data.Dataset.from_tensor_slices((files)).shuffle(10500).repeat()
    dataset = dataset.map(
            lambda filename: tuple(tf.py_func(readSpecH5, [filename], (tf.float64,tf.float64))))
    batched_dataset = dataset.batch(batchSize)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    input_layer = next_element[0][::2,::2,:]
    labels = next_element[1][::2,::2]
    
    input_layer = input_layer+eps
    
    sz = input_layer.get_shape()
    n_dimensions = sz[3]
    
    dconv = True
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [30,30]
    ks2 = [10,10]
    ks3 = [50,50]
    dcs = (16,16)
    #ks3 = [99,99]
    #dcs = (32,32)
    #ks4 = [137,137]
    fs1 = 16
    fs2 = 16
    fs3 = 1
    #fs4 = 1
    
    # Input Layer
    
    print("Input Layer Dimensions:\t",input_layer.shape)
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=dropOut_layer,
            filters=fs1,
            strides=(10,10),
            kernel_size=ks1,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv1")
    print("First Conv Layer Dim:\t",conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[10, 10], strides=5)
    print("First Pool Layer Dim:\t",pool1.shape)
    pool1flat = tf.reshape(pool1,[-1,pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])
    
    if dconv:
        pool2flatL = int(pool1flat.shape[1])
        pool2flatSR = int(pool2flatL**0.5)
        pool2flatL = int(pool2flatSR**2)
        dense1 = tf.layers.dense(inputs=pool1flat, units=2500, activation=tf.nn.leaky_relu)
        print("Classify Layer Dim:\t", dense1.shape)
        dense1_rs = tf.reshape(dense1,[-1,50,50,1])
        dconv1 = tf.layers.conv2d_transpose(
            inputs=dense1_rs,filters=fs3,
            kernel_size=ks3,
            strides=dcs,
            padding="valid",
            activation=tf.nn.leaky_relu,
            name="dconv1")
        print("Deconv Layer 1 Dim:\t", dconv1.shape)
        #dconv2 = tf.layers.conv2d_transpose(
        #    inputs=dconv1,filters=fs4,
        #    kernel_size=ks4,
        #    strides=(3,3),
        #    padding="valid",
        #    activation=tf.nn.leaky_relu,
        #    name="dconv2")
        #print("Deconv Layer 2 Dim:\t", dconv2.shape)
        denseOut = tf.reshape(dconv1,[-1,dconv1.shape[1],dconv1.shape[2],dconv1.shape[3]])
        #dconv1flat = tf.reshape(dconv1,[-1,dconv1.shape[1]*dconv1.shape[2]*dconv1.shape[3]])
        #denseOut = tf.layers.dense(inputs=dconv1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
        print("Output Layer Dim:\t",denseOut.shape)
        #print("Input Layer Dimensions:\t",input_layer.shape)
        #print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
        #print("First Conv Layer Dim:\t",conv1.shape)
        #print("First Pool Layer Dim:\t",pool1.shape)
        #print("Second Conv Layer Dim:\t", conv2.shape)
        #print("Second Pool Layer Dim:\t", pool2.shape)
        #print("Classify Layer Dim:\t", dense1.shape)
        #print("Deconv Layer Dim:\t", dconv1.shape)
        #print("Output Layer Dim:\t",denseOut.shape)
    else:
        denseOut = tf.layers.dense(inputs=pool1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
    
    logits = tf.reshape(denseOut,[-1,sz[1],sz[2]])
    predicted_classes = tf.round(logits)
            
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    print("Predicted Classes:\t",predicted_classes.shape)
    print("Labels:\t",labels.shape)
    loss = tf.reduce_sum((labels-logits)**2)**0.5
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(predicted_classes_rs,tf.float32))**2)**0.5
    
    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    #tf.get_variable_scope().reuse_variables()
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
      
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-7)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



def cnnModelFS3(features, labels, mode):
    """Model function for CNN."""
    
    dconv = True
    #n_dimensions = 13
    try:
        sz = features["x"].get_shape()
        n_dimensions = sz[3]
        input_layer = tf.reshape(features["x"], [-1, sz[1], sz[2], n_dimensions])
    except:
        sz = features.shape
        n_dimensions = sz[3]
        input_layer = tf.reshape(features, [-1, sz[1], sz[2], n_dimensions])
    
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [30,30]
    ks2 = [10,10]
    ks3 = [25,25]
    dcs = (8,8)
    #ks3 = [99,99]
    #dcs = (32,32)
    #ks4 = [137,137]
    fs1 = 16
    fs2 = 16
    fs3 = 1
    #fs4 = 1
    
    # Input Layer
    
    print("Input Layer Dimensions:\t",input_layer.shape)
    #dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    #print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=fs1,
            strides=(5,5),
            kernel_size=ks1,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv1")
    print("First Conv Layer Dim:\t",conv1.shape)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[5, 5], strides=3)
    print("First Pool Layer Dim:\t",pool1.shape)
    pool1flat = tf.reshape(pool1,[-1,pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])
    
    if dconv:
        pool2flatL = int(pool1flat.shape[1])
        pool2flatSR = int(pool2flatL**0.5)
        pool2flatL = int(pool2flatSR**2)
        dense1 = tf.layers.dense(inputs=pool1flat, units=2500, activation=tf.nn.leaky_relu)
        print("Classify Layer Dim:\t", dense1.shape)
        dense1_rs = tf.reshape(dense1,[-1,50,50,1])
        dconv1 = tf.layers.conv2d_transpose(
            inputs=dense1_rs,filters=fs3,
            kernel_size=ks3,
            strides=dcs,
            padding="valid",
            activation=tf.nn.leaky_relu,
            name="dconv1")
        print("Deconv Layer 1 Dim:\t", dconv1.shape)
        #dconv2 = tf.layers.conv2d_transpose(
        #    inputs=dconv1,filters=fs4,
        #    kernel_size=ks4,
        #    strides=(3,3),
        #    padding="valid",
        #    activation=tf.nn.leaky_relu,
        #    name="dconv2")
        #print("Deconv Layer 2 Dim:\t", dconv2.shape)
        denseOut = tf.reshape(dconv1,[-1,dconv1.shape[1],dconv1.shape[2],dconv1.shape[3]])
        #dconv1flat = tf.reshape(dconv1,[-1,dconv1.shape[1]*dconv1.shape[2]*dconv1.shape[3]])
        #denseOut = tf.layers.dense(inputs=dconv1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
        print("Output Layer Dim:\t",denseOut.shape)
        #print("Input Layer Dimensions:\t",input_layer.shape)
        #print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
        #print("First Conv Layer Dim:\t",conv1.shape)
        #print("First Pool Layer Dim:\t",pool1.shape)
        #print("Second Conv Layer Dim:\t", conv2.shape)
        #print("Second Pool Layer Dim:\t", pool2.shape)
        #print("Classify Layer Dim:\t", dense1.shape)
        #print("Deconv Layer Dim:\t", dconv1.shape)
        #print("Output Layer Dim:\t",denseOut.shape)
    else:
        denseOut = tf.layers.dense(inputs=pool1flat, units=int(sz[1]*sz[2]), activation=tf.nn.tanh)
    
    logits = tf.reshape(denseOut,[-1,sz[1],sz[2]])
    predicted_classes = tf.round(logits)
            
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    print("Predicted Classes:\t",predicted_classes.shape)
    print("Labels:\t",labels.shape)
    loss = tf.reduce_sum((labels-logits)**2)**0.5
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(predicted_classes_rs,tf.float32))**2)**0.5
    
    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    #tf.get_variable_scope().reuse_variables()
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
      
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-5)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)