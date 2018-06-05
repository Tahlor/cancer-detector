import tensorflow as tf
import numpy as np
import math
from cancer_data import *
from archipack import *
from tensorflow.python import debug as tf_debug

# Setup
image_size = 512
learning_rate = 0.001
variant = "Dropout" # "Normal" "Dropout" "L2"
variant = "Normal"
variant = "L2"

batch_size = 1
sub_variant = ""

L2_lambda = 0
static_keep_prob = 1

if variant == "Dropout":
    static_keep_prob = .7
    sub_variant = " - " + str(static_keep_prob)
elif variant == "L2":
    L2_lambda = 1
    sub_variant = " - " + str(L2_lambda)
    
variant_name = variant + sub_variant + ", image size - " + str(image_size) + ", batch size - " + str(batch_size)
logdir = createLogDir(variant_name)

tf.reset_default_graph()

# Generate out image
# Implement pooling and deconv
# Super computer...? GPU?
# Fix precision, recall
# Regularization

precision = tf.Variable(0)
recall = tf.Variable(0)

# Placeholders
x  = tf.placeholder( tf.float32, [None, image_size, image_size, 3] , name="input_data")
y_true = tf.placeholder( tf.int64,   [None, image_size, image_size] , name="output_data")
keep_prob_pl = tf.placeholder(tf.float32)
keep_prob = static_keep_prob

with tf.name_scope( "Conv" ) as scope:
    # NET
    h0 = conv(x, stride = 1, num_filters = 64, is_output = False)
    h1 = conv(h0, stride = 1, num_filters = 64, is_output = False)

with tf.name_scope( "downSampleConv" ) as scope:    
    pool_2 = max_pool_2x2(h1)
    drop_3 = tf.nn.dropout(pool_2, keep_prob)
    h4 = conv(drop_3, stride = 1, num_filters = 64, is_output = False)
    h41 = conv(h4, stride = 1, num_filters = 64, is_output = False)
    
with tf.name_scope( "upSampleConv" ) as scope:    
    up_sample5 = deconv(h41, output_shape = [1, image_size, image_size, 64], filter_size=3, stride=2, num_filters=64, is_output=False)
    drop_6 = tf.nn.dropout(up_sample5, keep_prob)

    # Concatenate h1 at this point?

    h7 = conv(drop_6, stride = 1, num_filters = 64, is_output = False)
    h8 = conv(h7, stride = 1, num_filters = 64, is_output = False)
    y_hat = conv(h8, stride = 1, num_filters = 2, is_output = True)

# flatten h1 to 1, 512, 512
# end up with 512, 512, 2 => ~probability of cancer and not

# =============================

with tf.name_scope( "loss_function" ) as scope:
    xent  = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true)
    cross_entropy = tf.reduce_mean( xent )

    #Basic
    #loss_L2_by_layer = L2_lambda * (tf.nn.l2_loss(y_hat) )
    loss_L2_by_layer = L2_lambda * (tf.nn.l2_loss(y_hat) +  tf.nn.l2_loss(h0) + tf.nn.l2_loss(h1) +
                                    tf.nn.l2_loss(pool_2) +  tf.nn.l2_loss(drop_3) + tf.nn.l2_loss(h4) +
                                    tf.nn.l2_loss(h41) + tf.nn.l2_loss(up_sample5) + tf.nn.l2_loss(drop_6)
                                    + tf.nn.l2_loss(h7) + tf.nn.l2_loss(h8))
    loss_L2 = tf.add(cross_entropy, loss_L2_by_layer, name='loss_L2')
        

with tf.name_scope( "accuracy" ) as scope:
    # fix y_hat - needs to be binary 0/1; y_hat is a 2-layer guy, 1 for cancer, 0 benign
    labels_hat = tf.argmax(y_hat, axis = 3) 
    correct_prediction = tf.equal( y_true, labels_hat)
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

    TP = tf.count_nonzero(labels_hat * y_true, dtype=tf.float32)
    TN = tf.count_nonzero((labels_hat - 1) * (y_true - 1), dtype=tf.float32)
    FP = tf.count_nonzero(labels_hat * (y_true - 1), dtype=tf.float32)
    FN = tf.count_nonzero((labels_hat - 1) * y_true, dtype=tf.float32)

    recall =    (TP) / (TP + FN)
    precision = (TP) / (TP + FP)

    # Assume 100% if division by 0
    recall = tf.where(tf.is_nan(recall), 0., recall)
    precision = tf.where(tf.is_nan(precision), 0., precision)
    #recall = tf.metrics.recall(y_true, labels_hat)
    #precision = tf.metrics.precision(y_true, labels_hat)

# =============================

if variant != "L2":
    train_step =    tf.train.AdamOptimizer( learning_rate ).minimize( cross_entropy )
else:
    train_step = tf.train.AdamOptimizer( learning_rate ).minimize( loss_L2 )


#saver = tf.train.Saver()

#Preprocess Data()
s = "" if image_size == 512 else str(image_size)
test_labels, test_data, train_data, train_labels = load_data(image_size)
THE_test_label = unpickle("testImageLabels"+s)
THE_test_data = unpickle("testImageData"+s)

# Shuffle
#shuffleDataAndLabelsInPlace(train_data, train_labels)
#shuffleDataAndLabelsInPlace(test_data, test_labels)

test_size = len(test_data)
#io_batch = 1750
epoch = 0

# Start session
    
#server = tf.train.Server.create_local_server()
#sess = tf.Session(server.target)

init = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()
sess = tf.Session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

sess.run( init   )
sess.run(init2)



train_writer = tf.summary.FileWriter( logdir , sess.graph )

# OUTPUT IMAGE
#true_image = tf.cast(y_true, tf.float32)
recast_pred = tf.cast(labels_hat, tf.float32)

# create 3-pixel levels
add_layers_pred = tf.tile(tf.reshape(recast_pred, (image_size,image_size,1)), (1,1,3))[None, :]
image_pred = tf.summary.image('myClassification', add_layers_pred, 3.0)

# SUMMARY
ce_summ = tf.summary.scalar('cross_entropy', cross_entropy)
acc_summ = tf.summary.scalar('testing_accuracy', accuracy)
train_acc_summ = tf.summary.scalar('training_accuracy', accuracy)
rec_summ = tf.summary.scalar('recall', recall)
prec_summ = tf.summary.scalar('precision', precision)
L2_summ = tf.summary.scalar('l2_loss', loss_L2)

#merged = tf.summary.merge_all()
merged = tf.summary.merge([ce_summ, acc_summ, rec_summ, prec_summ, L2_summ] )

#deconv_shape = tf.stack([batch_size, upscale*x_shape[1].value, upscale*x_shape[2].value, num_filters])
#tf.layers.conv2d_transpose

"""
train_data =   [n[None,:] for n in train_data]
train_labels = [n[None,:] for n in train_labels]
test_data =    [n[None,:] for n in test_data]
test_labels =  [n[None,:] for n in test_labels]
"""
# 
for epoch in range(0,3):
    #train_data, train_labels = read_data(io_batch, epoch, "train")
    shuffleDataAndLabelsInPlace(train_data, train_labels)
    mx = len(train_data)
    #for i in range(0, 1):
    for i in range(0,  int(math.ceil(mx/batch_size))):
        batch = min(mx-i, batch_size)
        step = epoch * len(train_data) + i * batch_size
        j = step % test_size

        if j == 0:
            shuffleDataAndLabelsInPlace(test_data, test_labels)

        in_x = train_data[i:i+batch_size]
        in_y = train_labels[i:i+batch_size]

        test_x = [test_data[j]]
        test_y = [test_labels[j]]

        ## TRAIN    
        train_acc, _ = sess.run( [train_acc_summ, train_step],
                                     feed_dict={x: in_x, y_true: in_y , keep_prob_pl : 1.0})

        train_writer.add_summary(train_acc, step )

        if (step % 20 == 0):
            print(step)
        
        #ss_test = sess.run( merged, feed_dict={x: test_x, y_true: test_y})
        ss_test = sess.run( merged, feed_dict={x: test_x, y_true: test_y, keep_prob_pl: 1})
        train_writer.add_summary( ss_test, step )
        #print( "%d %.2f" % ( i, acc ) )

    # Test the image
    image = sess.run( image_pred, feed_dict={x: THE_test_data[None,:], y_true: THE_test_label[None,:], keep_prob_pl : 1.0})
    train_writer.add_summary( image )

# Final test set
accuracy_list = []
for step in range(0, test_size):
    j = step % test_size

    test_x = [test_data[j]]
    test_y = [test_labels[j]]

    if (step % 20 == 0):
        print(step)
    
    acc, pre, rec = sess.run( [accuracy, precision, recall],
                              feed_dict={x: test_x, y_true: test_y, keep_prob_pl : 1.0})

    accuracy_list.append([acc, pre, rec])

print(np.mean(accuracy_list, axis =0))
train_writer.close()
