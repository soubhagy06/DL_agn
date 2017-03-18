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

n_nodes_hl1 = 500
n_nodes_hl2 = 300

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')



hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}


l1 = tf.add(tf.matmul(x,hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
l2 = tf.nn.relu(l2)

prediction = tf.matmul(l2,output_layer['weights']) + output_layer['biases']


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    trainX = trainX.reshape(len(trainX),-1)
    trainX = trainX/trainX.max()

    onehot = np.zeros((trainY.shape[0], 10))
    onehot[np.arange(trainY.shape[0]), trainY] = 1
    trainY = onehot
    
    #prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    #sess = tf.InteractiveSession()
    saver = tf.train.Saver() 

    # Train the model and save it in the end
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

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss, 'Accuracy:',accuracy.eval({x:trainX, y:trainY}))

        saver.save(sess, './densenn/dense-model')

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
    #new_saver = tf.train.import_meta_graph('./densenn/dense-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./densenn'))
    sess.run(tf.global_variables())
    predict = sess.run(prediction, feed_dict={x: testX})
    y_label = np.argmax(predict, 1)

    return y_label