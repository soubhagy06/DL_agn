'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
           'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
           'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
           'out':tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
           'b_conv2':tf.Variable(tf.random_normal([64])),
           'b_fc':tf.Variable(tf.random_normal([1024])),
           'out':tf.Variable(tf.random_normal([n_classes]))}

x_im = tf.reshape(x, shape=[-1, 28, 28, 1])

conv1 = tf.nn.relu(conv2d(x_im, weights['W_conv1']) + biases['b_conv1'])
conv1 = maxpool2d(conv1)
    
conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2,[-1, 7*7*64])
fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
fc = tf.nn.dropout(fc, keep_rate)

prediction = tf.matmul(fc, weights['out'])+biases['out']


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    trainX = trainX.reshape(len(trainX),-1)
    trainX = trainX/trainX.max()

    onehot = np.zeros((trainY.shape[0], 10))
    onehot[np.arange(trainY.shape[0]), trainY] = 1
    trainY = onehot

    
    # prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    saver = tf.train.Saver() 

    hm_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(trainX.shape[0]/batch_size)):
                epoch_x = trainX[i*batch_size:(i+1)*batch_size]
                epoch_y = trainY[i*batch_size:(i+1)*batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                print ('Training Batch Accuracy:',accuracy.eval({x:epoch_x, y:epoch_y}))
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        saver.save(sess, './cnn/cnn-model')


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    testX = testX.reshape(len(testX),-1)
    testX = testX/testX.max()

    sess = tf.Session()
    new_saver = tf.train.Saver()
    #new_saver = tf.train.import_meta_graph('./cnn/cnn-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./cnn'))
    sess.run(tf.global_variables())
    
    predict = sess.run(prediction, feed_dict={x: testX[0:testX.shape[0]/4]})
    y_label = np.argmax(predict, 1)

    predict = sess.run(prediction, feed_dict={x: testX[testX.shape[0]/4:testX.shape[0]/2]})
    y_label = np.append(y_label, np.argmax(predict, 1))

    predict = sess.run(prediction, feed_dict={x: testX[testX.shape[0]/2:testX.shape[0]*3/4]})
    y_label = np.append(y_label, np.argmax(predict, 1))

    predict = sess.run(prediction, feed_dict={x: testX[testX.shape[0]*3/4:]})
    y_label = np.append(y_label, np.argmax(predict, 1))


    return y_label
