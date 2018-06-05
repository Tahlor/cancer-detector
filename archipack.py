import numpy as np
import pickle
import os
try:
    import tensorflow as tf
except:
    pass

import fnmatch

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    #matching_files = [n for n in fnmatch.filter(os.listdir(base), pattern) if os.path.isfile(os.path.join(base, n))]
    matching_files_and_folders = fnmatch.filter(os.listdir(base), pattern)
    return len(matching_files_and_folders)>0

def createLogDir(name = "", force_numerical_ordering = True):
    n = 1

    # Add padding
    if name != "" and name[0] != " ":
        name = " " + name

    # Check for existence
    basepath = "./tf_logs"
    if not os.path.exists(basepath):
        os.mkdir(basepath)

    if force_numerical_ordering:
        while find_files(basepath, str(n) + " *") or os.path.exists(os.path.join(basepath, str(n)    )) :
            n += 1
    else:
        while os.path.exists(os.path.join(basepath, str(n) + name )):
            n += 1

    # Create
    logdir = os.path.join(basepath, str(n) + name)
    os.mkdir(logdir)
    training_accuracy_list = []
    print(logdir)
    return logdir

def shuffleDataAndLabelsInPlace ( arr1, arr2):
    from numpy.random import RandomState
    import sys
    seed = np.random.randint(0, sys.maxsize/10**10)
    prng = RandomState(seed)
    prng.shuffle(arr1)
    prng = RandomState(seed)
    prng.shuffle(arr2)
    
def unpickleCIFAR( file ):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding = 'latin-1')
    fo.close()
    return dict

def importCIFAR(filename = 'cifar-10-batches-py/data_batch_1'): 
    data = unpickle( filename )
    features = data['data']
    labels = data['labels']
    labels = np.atleast_2d( labels ).T
    return labels, features

def importCIFARall():
    filename = 'cifar-10-batches-py/data_batch_1'
    data = unpickle(filename)
    features = data['data']
    labels = data['labels']
    for i in range(2,5):
        filename = 'cifar-10-batches-py/data_batch_' + str(i)
        data = unpickle(filename)
        labels = np.append(labels, data['labels'])
        features = np.vstack([features, data['data']])

    labels = np.atleast_2d( labels ).T
    print(features.shape)
    print(labels.shape)
    return labels, features

def splitTrainingTest(percent_training, features, labels):
    assert features.shape[0] == labels.shape[0]
    count = features.shape[0]
    percent_training = percent_training/100 if percent_training > 1 else percent_training
    training_size = int(count*percent_training)
    training_features  = features[:training_size]
    test_features      = features[training_size:]
    training_labels    = labels[:training_size]
    test_labels        = labels[training_size:]
    return training_features, training_labels, test_features, test_labels

def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv" ):
    x_depth = np.shape(x)[3]
    with tf.name_scope(name) as scope:
        W = tf.Variable( 1e-3*np.random.randn( filter_size, filter_size, x_depth, num_filters ).astype(np.float32), name="W" )
        b = tf.Variable( 1e-3*np.random.randn( num_filters).astype(np.float32), name="b" )
        op = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        op = tf.nn.bias_add(op, b)
       
    if not is_output:
        op = tf.nn.relu(op)
    return op

#shape of input = [batch, in_height, in_width, in_channels]
#shape of filter = [filter_height, filter_width, in_channels, out_channels]

def fc( x, out_size=50, name="fc" , is_output=False, batch_size = 1):    
    x_dim = x.get_shape().as_list()[1]
    
    with tf.name_scope(name) as scope:
        W = tf.Variable( 1e-3*np.random.randn(out_size, x_dim).astype(np.float32), name="W")
        b = tf.Variable( 1e-3*np.random.randn(out_size, 1).astype(np.float32), name="b" )
        
        # should automagically take care of batching 
        # op = tf.matmul(W, x) + b
        op = tf.einsum('ij,bjk->bik', W, x) + b
        
        #op = tf.nn.bias_add(op, b)
        
    if not is_output:
        op = tf.nn.relu(op)
        
    return op

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def deconv( x, output_shape = [1,32,32,3], filter_size=3, stride=2, num_filters=64, is_output=False, name="deconv" ):
    # tf.layers.conv2d_transpose is better for batching
    x_depth = np.shape(x)[3]
    with tf.name_scope(name) as scope:
        W = tf.Variable( 1e-3*np.random.randn( filter_size, filter_size, x_depth, num_filters ).astype(np.float32), name="W")
        b = tf.Variable( 1e-3*np.random.randn( num_filters).astype(np.float32), name="b")
        op = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME', data_format='NHWC', name="deconv")
        op = tf.nn.bias_add(op, b)

    if not is_output:
        op = tf.nn.relu(op)
    return op



### UNFINISHED
def max_pool( x, out_size=50, name="max_pool" , is_output=False):


    x_dim = np.shape(x)[0]
    x_dim = x.get_shape().as_list()[0]

    with tf.name_scope(name) as scope:
        W = tf.Variable( 1e-3*np.random.randn(out_size, x_dim).astype(np.float32), name="W" )
        b = tf.Variable( 1e-3*np.random.randn(out_size, 1).astype(np.float32), name="b" )
        op = tf.matmul(W, x) + b
        #op = tf.nn.bias_add(op, b)
        
    if not is_output:
        op = tf.nn.relu(op)
        
    return op


def conv2( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv" ):

    '''
    x is an input tensor
    Declare a name scope using the "name" parameter
    Within that scope:
      Create a W filter variable with the proper size
      Create a B bias    with the proper size
      Convolve x with W by calling the tf.nn.conv2d function
      Add the bias
      If is_output is False,
        Call the tf.nn.relu function
      Return the final op
    '''
    x_depth = np.shape(x)[3]
    with tf.name_scope(name) as scope:
        W = tf.Variable( 1e-3*np.random.randn( filter_size, filter_size, x_depth, num_filters ).astype(np.float32), name="W" )
        #b = tf.Variable( 1e-3*np.random.randn( 1, 1, 1, num_filters ).astype(np.float32), name="b" )
        b = tf.Variable( 1e-3*np.random.randn( num_filters).astype(np.float32), name="b" )
        op = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        op = tf.nn.bias_add(op, b)
       
    if not is_output:
        op = tf.nn.relu(op)
    return op
#shape of input = [batch, in_height, in_width, in_channels]
#shape of filter = [filter_height, filter_width, in_channels, out_channels]

def fc2( x, out_size=50, name="fc" , is_output=False):

    '''
    x is an input tensor - we expect a vector
    Declare a name scope using the "name" parameter
    Within that scope:
      Create a W filter variable with the proper size
      Create a B bias variable with the proper size
      Multiply x by W and add b
      If is_output is False,
        Call the tf.nn.relu function
      Return the final op
    '''
    x_dim = np.shape(x)[0]
    x_dim = x.get_shape().as_list()[0]

    with tf.name_scope(name) as scope:
        W = tf.Variable( 1e-3*np.random.randn(out_size, x_dim).astype(np.float32), name="W" )
        b = tf.Variable( 1e-3*np.random.randn(out_size, 1).astype(np.float32), name="b" )
        op = tf.matmul(W, x) + b
        #op = tf.nn.bias_add(op, b)
        
    if not is_output:
        op = tf.nn.relu(op)
        
    return op


    
if __name__ == '__main__':
    #importCIFARall()
    logdir = createLogDir()
    pass
    
