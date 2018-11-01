# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:05:20 2018

@author: JHodges
"""

import tensorflow as tf

def cnnModel3(features, labels, mode):
    """Model function for CNN."""
    
    dconv = True
    sz = 50
    n_dimensions = 13
    #n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [10,10]
    ks2 = [10,10]
    ks3 = [10,10]
    fs1 = 32
    fs2 = 64
    fs3 = 2
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    
    #print(input_layer.shape)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=dropOut_layer,
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
        print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
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
    
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5

    label_rs = tf.reshape(labels,[-1,int(sz*sz),2])
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
    
    
    
def cnnModel4(features, labels, mode):
    """Model function for CNN."""
    
    dropoutRate = 0.25 if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    (sz,n_dimensions) = (50,13)
    (ks1,fs1,ks2,fs2,ks3,fs3) = ([10,10],32,[10,10],64,[10,10],2)
    lrelu = tf.nn.leaky_relu
    
    #n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer,filters=fs1,kernel_size=ks1,padding="same",activation=lrelu,name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,filters=fs2,kernel_size=ks2,padding="same",activation=lrelu,name="conv2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2flat = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
    # Dense classification layer    
    dense1 = tf.layers.dense(inputs=pool2flat, units=int(sz*sz*2), activation=lrelu)
    # Dropout layer
    dropOut_layer = tf.layers.dropout(dense1,rate=dropoutRate)
    dense1_rs = tf.reshape(dropOut_layer,[-1,sz,sz,2])
    dconv1 = tf.layers.conv2d_transpose(inputs=dense1_rs,filters=fs3,kernel_size=ks3,padding="same",activation=lrelu,name="dconv1")
    dconv1flat = tf.reshape(dconv1,[-1,dconv1.shape[1]*dconv1.shape[2]*dconv1.shape[3]])
    # Output layer
    denseOut = tf.layers.dense(inputs=dconv1flat, units=int(sz*sz*2), activation=tf.nn.tanh)
    logits = tf.reshape(denseOut,[-1,int(sz*sz*2)])
    predicted_classes = tf.argmax(input=tf.reshape(dense1,[-1,int(sz*sz),2]), axis=2)
    # Print sizes for debugging
    print("Input Layer Dimensions:\t",input_layer.shape)
    print("First Conv Layer Dim:\t",conv1.shape)
    print("First Pool Layer Dim:\t",pool1.shape)
    print("Second Conv Layer Dim:\t", conv2.shape)
    print("Second Pool Layer Dim:\t", pool2.shape)
    print("Classify Layer Dim:\t", dense1.shape)
    print("Dropout Layer Dimensions:\t",dropOut_layer.shape)
    print("Deconv Layer Dim:\t", dconv1.shape)
    print("Output Layer Dim:\t",denseOut.shape)    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'class_ids': predicted_classes,'probabilities': tf.nn.softmax(logits),'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    #loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5
    loss = tf.losses.sigmoid_cross_entropy(labels,logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=10**-4)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
def cnnModel_e64a128d(features, labels, mode):
    """Model function for CNN."""
    
    sz = 50
    #n_dimensions = 13
    n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [10,10]
    ks2 = [10,10]
    fs1 = 8
    fs2 = 128
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    print("Input Layer Dimensions:\t",input_layer.shape)
    
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("DropOut Layer Dimensions:\t",dropOut_layer.shape)
    
    i0 = tf.reshape(dropOut_layer[:,:,:,0], [-1, sz, sz, 1])
    print("i0 Layer Dimensions:\t",i0.shape)
    i1 = tf.reshape(dropOut_layer[:,:,:,1], [-1, sz, sz, 1])
    i2 = tf.reshape(dropOut_layer[:,:,:,2], [-1, sz, sz, 1])
    i3 = tf.reshape(dropOut_layer[:,:,:,3], [-1, sz, sz, 1])
    i4 = tf.reshape(dropOut_layer[:,:,:,4], [-1, sz, sz, 1])
    i5 = tf.reshape(dropOut_layer[:,:,:,5], [-1, sz, sz, 1])
    i6 = tf.reshape(dropOut_layer[:,:,:,6], [-1, sz, sz, 1])
    
    c0 = tf.layers.conv2d(inputs=i0,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="0")
    print("c0 Layer Dimensions:\t",c0.shape)
    c1 = tf.layers.conv2d(inputs=i1,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="1")
    c2 = tf.layers.conv2d(inputs=i2,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="2")
    c3 = tf.layers.conv2d(inputs=i3,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="3")
    c4 = tf.layers.conv2d(inputs=i4,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="4")
    c5 = tf.layers.conv2d(inputs=i5,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="5")
    c6 = tf.layers.conv2d(inputs=i6,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="6")
    
    p0 = tf.layers.max_pooling2d(inputs=c0, pool_size=[2, 2], strides=2)
    print("p0 Layer Dimensions:\t",p0.shape)
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=2)
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], strides=2)
    p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=2)
    p4 = tf.layers.max_pooling2d(inputs=c4, pool_size=[2, 2], strides=2)
    p5 = tf.layers.max_pooling2d(inputs=c5, pool_size=[2, 2], strides=2)
    p6 = tf.layers.max_pooling2d(inputs=c6, pool_size=[2, 2], strides=2)
    
    pool1 = tf.concat([p0,p1,p2,p3,p4,p5,p6],axis=3)
    #pool1 = tf.Variable(tf.zeros([p0.shape[0],int(sz/2),int(sz/2),int(fs1*n_dimensions)]))
    print("pool1 Layer Dimensions:\t",pool1.shape)
    #pool1
    #pool1[:,:,:,int(0*fs1):int(1*fs1)] = p0
    #pool1[:,:,:,int(1*fs1):int(2*fs1)] = p1
    #pool1[:,:,:,int(2*fs1):int(3*fs1)] = p2
    #pool1[:,:,:,int(3*fs1):int(4*fs1)] = p3
    #pool1[:,:,:,int(4*fs1):int(5*fs1)] = p4
    #pool1[:,:,:,int(5*fs1):int(6*fs1)] = p5
    #pool1[:,:,:,int(6*fs1):int(7*fs1)] = p6
    

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

    denseOut = tf.layers.dense(inputs=pool2flat, units=int(sz*sz*2), activation=tf.nn.tanh)
        
    logits = tf.reshape(denseOut,[-1,int(sz*sz*2)])
    predicted_classes = tf.argmax(input=tf.reshape(logits,[-1,int(sz*sz),2]), axis=2)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5

    label_rs = tf.reshape(labels,[-1,int(sz*sz),2])
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
    
def cnnModel_e56a28d(features, labels, mode):
    """Model function for CNN."""
    
    sz = 50
    #n_dimensions = 13
    n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [10,10]
    ks2 = [10,10]
    fs1 = 8
    fs2 = 28
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    print("Input Layer Dimensions:\t",input_layer.shape)
    
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("DropOut Layer Dimensions:\t",dropOut_layer.shape)
    
    i0 = tf.reshape(dropOut_layer[:,:,:,0], [-1, sz, sz, 1])
    print("i0 Layer Dimensions:\t",i0.shape)
    i1 = tf.reshape(dropOut_layer[:,:,:,1], [-1, sz, sz, 1])
    i2 = tf.reshape(dropOut_layer[:,:,:,2], [-1, sz, sz, 1])
    i3 = tf.reshape(dropOut_layer[:,:,:,3], [-1, sz, sz, 1])
    i4 = tf.reshape(dropOut_layer[:,:,:,4], [-1, sz, sz, 1])
    i5 = tf.reshape(dropOut_layer[:,:,:,5], [-1, sz, sz, 1])
    i6 = tf.reshape(dropOut_layer[:,:,:,6], [-1, sz, sz, 1])
    
    c0 = tf.layers.conv2d(inputs=i0,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="0")
    print("c0 Layer Dimensions:\t",c0.shape)
    c1 = tf.layers.conv2d(inputs=i1,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="1")
    c2 = tf.layers.conv2d(inputs=i2,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="2")
    c3 = tf.layers.conv2d(inputs=i3,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="3")
    c4 = tf.layers.conv2d(inputs=i4,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="4")
    c5 = tf.layers.conv2d(inputs=i5,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="5")
    c6 = tf.layers.conv2d(inputs=i6,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="6")
    
    p0 = tf.layers.max_pooling2d(inputs=c0, pool_size=[2, 2], strides=2)
    print("p0 Layer Dimensions:\t",p0.shape)
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=2)
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], strides=2)
    p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=2)
    p4 = tf.layers.max_pooling2d(inputs=c4, pool_size=[2, 2], strides=2)
    p5 = tf.layers.max_pooling2d(inputs=c5, pool_size=[2, 2], strides=2)
    p6 = tf.layers.max_pooling2d(inputs=c6, pool_size=[2, 2], strides=2)
    
    pool1 = tf.concat([p0,p1,p2,p3,p4,p5,p6],axis=3)
    #pool1 = tf.Variable(tf.zeros([p0.shape[0],int(sz/2),int(sz/2),int(fs1*n_dimensions)]))
    print("pool1 Layer Dimensions:\t",pool1.shape)
    #pool1
    #pool1[:,:,:,int(0*fs1):int(1*fs1)] = p0
    #pool1[:,:,:,int(1*fs1):int(2*fs1)] = p1
    #pool1[:,:,:,int(2*fs1):int(3*fs1)] = p2
    #pool1[:,:,:,int(3*fs1):int(4*fs1)] = p3
    #pool1[:,:,:,int(4*fs1):int(5*fs1)] = p4
    #pool1[:,:,:,int(5*fs1):int(6*fs1)] = p5
    #pool1[:,:,:,int(6*fs1):int(7*fs1)] = p6
    

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=fs2,
            kernel_size=ks2,
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv2")
    print("conv2 Layer Dimensions:\t",conv2.shape)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print("pool2 Layer Dimensions:\t",pool2.shape)
    pool2flat = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])

    denseOut = tf.layers.dense(inputs=pool2flat, units=int(sz*sz*2), activation=tf.nn.tanh)
        
    logits = tf.reshape(denseOut,[-1,int(sz*sz*2)])
    predicted_classes = tf.argmax(input=tf.reshape(logits,[-1,int(sz*sz),2]), axis=2)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5

    label_rs = tf.reshape(labels,[-1,int(sz*sz),2])
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
    
def cnnModel_unitTest1(features, labels, mode):
    """Model function for CNN."""
    
    sz = 50
    #n_dimensions = 13
    n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [10,10]
    ks2 = [10,10]
    fs1 = 8
    fs2 = 28
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    print("Input Layer Dimensions:\t",input_layer.shape)
    
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("DropOut Layer Dimensions:\t",dropOut_layer.shape)
    
    i0 = tf.reshape(dropOut_layer[:,:,:,0], [-1, sz, sz, 1])
    print("i0 Layer Dimensions:\t",i0.shape)
    
    c0 = tf.layers.conv2d(inputs=i0,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="0")
    print("c0 Layer Dimensions:\t",c0.shape)

    p0 = tf.layers.max_pooling2d(inputs=c0, pool_size=[2, 2], strides=2)
    print("p0 Layer Dimensions:\t",p0.shape)
    
    pool1 = tf.concat([p0],axis=3)
    #pool1 = tf.Variable(tf.zeros([p0.shape[0],int(sz/2),int(sz/2),int(fs1*n_dimensions)]))
    print("pool1 Layer Dimensions:\t",pool1.shape)

    pool1flat = tf.reshape(pool1,[-1,pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])

    denseOut = tf.layers.dense(inputs=pool1flat, units=int(sz*sz*2), activation=tf.nn.tanh)
        
    logits = tf.reshape(denseOut,[-1,int(sz*sz*2)])
    predicted_classes = tf.argmax(input=tf.reshape(logits,[-1,int(sz*sz),2]), axis=2)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5

    label_rs = tf.reshape(labels,[-1,int(sz*sz),2])
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
    
def cnnModel_e56(features, labels, mode):
    """Model function for CNN."""
    
    sz = 50
    #n_dimensions = 13
    n_dimensions = int(features["x"].get_shape().as_list()[1]/(sz**2))
    print("MODE=%s\nInput Dimensions=%s"%(mode,n_dimensions))
    ks1 = [10,10]
    ks2 = [10,10]
    fs1 = 8
    fs2 = 28
    strides = 4
    
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, sz, sz, n_dimensions])
    print("Input Layer Dimensions:\t",input_layer.shape)
    
    dropOut_layer = tf.layers.dropout(input_layer,rate=0.5)
    print("DropOut Layer Dimensions:\t",dropOut_layer.shape)
    
    i0 = tf.reshape(dropOut_layer[:,:,:,0], [-1, sz, sz, 1])
    print("i0 Layer Dimensions:\t",i0.shape)
    i1 = tf.reshape(dropOut_layer[:,:,:,1], [-1, sz, sz, 1])
    i2 = tf.reshape(dropOut_layer[:,:,:,2], [-1, sz, sz, 1])
    i3 = tf.reshape(dropOut_layer[:,:,:,3], [-1, sz, sz, 1])
    i4 = tf.reshape(dropOut_layer[:,:,:,4], [-1, sz, sz, 1])
    i5 = tf.reshape(dropOut_layer[:,:,:,5], [-1, sz, sz, 1])
    i6 = tf.reshape(dropOut_layer[:,:,:,6], [-1, sz, sz, 1])
    
    c0 = tf.layers.conv2d(inputs=i0,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="0")
    print("c0 Layer Dimensions:\t",c0.shape)
    c1 = tf.layers.conv2d(inputs=i1,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="1")
    c2 = tf.layers.conv2d(inputs=i2,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="2")
    c3 = tf.layers.conv2d(inputs=i3,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="3")
    c4 = tf.layers.conv2d(inputs=i4,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="4")
    c5 = tf.layers.conv2d(inputs=i5,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="5")
    c6 = tf.layers.conv2d(inputs=i6,filters=fs1,kernel_size=ks1,padding="same",activation=tf.nn.leaky_relu,name="6")
    
    p0 = tf.layers.max_pooling2d(inputs=c0, pool_size=[2, 2], strides=strides)
    print("p0 Layer Dimensions:\t",p0.shape)
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=strides)
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], strides=strides)
    p3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2], strides=strides)
    p4 = tf.layers.max_pooling2d(inputs=c4, pool_size=[2, 2], strides=strides)
    p5 = tf.layers.max_pooling2d(inputs=c5, pool_size=[2, 2], strides=strides)
    p6 = tf.layers.max_pooling2d(inputs=c6, pool_size=[2, 2], strides=strides)
    
    pool1 = tf.concat([p0,p1,p2,p3,p4,p5,p6],axis=3)
    #pool1 = tf.Variable(tf.zeros([p0.shape[0],int(sz/2),int(sz/2),int(fs1*n_dimensions)]))
    print("pool1 Layer Dimensions:\t",pool1.shape)
    #pool1
    #pool1[:,:,:,int(0*fs1):int(1*fs1)] = p0
    #pool1[:,:,:,int(1*fs1):int(2*fs1)] = p1
    #pool1[:,:,:,int(2*fs1):int(3*fs1)] = p2
    #pool1[:,:,:,int(3*fs1):int(4*fs1)] = p3
    #pool1[:,:,:,int(4*fs1):int(5*fs1)] = p4
    #pool1[:,:,:,int(5*fs1):int(6*fs1)] = p5
    #pool1[:,:,:,int(6*fs1):int(7*fs1)] = p6
    
    pool1flat = tf.reshape(pool1,[-1,pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])

    denseOut = tf.layers.dense(inputs=pool1flat, units=int(sz*sz*2), activation=tf.nn.tanh)
        
    logits = tf.reshape(denseOut,[-1,int(sz*sz*2)])
    predicted_classes = tf.argmax(input=tf.reshape(logits,[-1,int(sz*sz),2]), axis=2)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,#[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
          }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.reduce_sum(abs(tf.cast(labels,tf.float32)-tf.cast(logits,tf.float32))**2)**0.5

    label_rs = tf.reshape(labels,[-1,int(sz*sz),2])
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