from __future__ import absolute_import
from __future__ import print_function

import collections, math, os, random, zipfile, argparse, sys, logging, random, itertools

from itertools import islice
from six.moves import urllib
from six.moves import xrange #pylint: disable=redefine-buildin
import string
import time
from cu import *
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("save_path", "cache", "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_integer("embedding_width", 64, "The embedding dimension size.")
flags.DEFINE_float("learning_rate", 0.128, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 2048,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 2048,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_string("kernel_height","2345678","the height of each kenel")
flags.DEFINE_integer("num_skips",8,"")
flags.DEFINE_integer("skip_window",4,"")
flags.DEFINE_integer("vocabulary_size",50000,"")
flags.DEFINE_string("kenel_depth","16,16,16,16,16,16,16","the depth of each kenel")
flags.DEFINE_string("filename","/cse/research/asr/HaoLi/train.txt","filename")
flags.DEFINE_integer("char_vocab_size",128,"")
FLAGS = flags.FLAGS
start_time = time.time()

def shuffle(words):
    #random.shuffle(words)
    data = np.array([w.split() for w in words])
    return list(itertools.chain.from_iterable(data))

################################# GLOBAL VARIABLES ################################
batch_size = FLAGS.batch_size
embedding_width = FLAGS.embedding_width
kenel_height = map(int,list(FLAGS.kernel_height))
kenel_width = embedding_width
kenel_depth = [ int(x) for x in FLAGS.kenel_depth.split(',') ]
in_channel = 1
vocabulary_size = FLAGS.vocabulary_size
feature_size = ((kenel_depth[0]+kenel_depth[-1])*len(kenel_depth))/2
num_skips = FLAGS.num_skips
skip_window = FLAGS.skip_window
filename = FLAGS.filename
lines = read_data(filename)
words = shuffle(lines)
data, count, dictionary,reverse_dictionary = build_data(vocabulary_size,words)
num_sampled = 128   # Number of negative examples to sample.
lr = FLAGS.learning_rate
char_vocab_size = FLAGS.char_vocab_size

def convolution(embeddings,kenel,paded_ecb,biases):
    embed_cube = tf.expand_dims( tf.gather( embeddings, paded_ecb),3) 
    conv = tf.nn.conv2d(embed_cube,kenel,[1,1,1,1],padding='VALID')
    bias = tf.nn.bias_add(conv,biases)
    conv = tf.nn.relu(bias)
    pool = tf.reduce_max(conv,reduction_indices=[1,2])
    return pool

def test_cos_wb():
    sample = np.random.choice(vocabulary_size-1, 500-batch_size/num_skips, replace=True)
    sample = map(lambda x: reverse_dictionary[x], sample)
    return sample
    
############################Define graph########################################
#######################variables##########################################
train_inputs = tf.placeholder(tf.int64, shape=[None,None])
train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
test_cos_inputs = tf.placeholder(tf.int64, shape=[None,None])
learning_rate = tf.placeholder(tf.float64)

######################Graph##############################################
embeddings = tf.Variable(tf.random_uniform([char_vocab_size, embedding_width], -0.32, 0.32))

kenels = [tf.Variable(tf.random_uniform([kenel_height[i],kenel_width,in_channel,kenel_depth[i]],-0.32,0.32))
				for i in range(len(kenel_height))]
biases = [tf.Variable(tf.zeros([kenel_depth[k]])) 
				for k in range(len(kenel_height))] 
features = tf.concat(1,[convolution(embeddings,kenels[k],train_inputs,biases[k]) 
								for k in range(len(kenel_height))])


##---------------------- cosine similarity --------------------------------#
norm = tf.sqrt(tf.reduce_sum(tf.square(features),1,keep_dims=True))
norm_fs = features/norm

test_ci_features = tf.concat(1,[convolution(embeddings,kenels[k],test_cos_inputs,biases[k]) 
								for k in range(len(kenel_height))])
t_norm = tf.sqrt(tf.reduce_sum(tf.square(test_ci_features),1,keep_dims=True))
norm_tci = test_ci_features / t_norm

all_f = tf.concat(0,[features,test_ci_features])
all_norm = tf.sqrt(tf.reduce_sum(tf.square(all_f),1,keep_dims=True))
norm_all = all_f / all_norm

similairty = tf.matmul(norm_fs,norm_all,transpose_b=True)
#--------------------------------------------------------------------------#
for i in range(int(math.log(num_skips)/math.log(2))):
    features = tf.concat(1,[features,features])
features = tf.reshape(features,[batch_size,-1])


#---------------- NCE -----------------------------------------------------#
# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size+1, feature_size],stddev=1.0 / math.sqrt(feature_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size+1]))

#loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, features, train_labels,
#num_sampled, vocabulary_size))
#
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#----------------NEGATIVAE SAMPLING---------------------------------------#
sample,_,_ = tf.nn.fixed_unigram_candidate_sampler(
                true_classes = tf.reshape(train_labels,[-1,skip_window * 2]),
                num_true = skip_window * 2,
                num_sampled = num_sampled,
                unique = True,
                range_max = vocabulary_size + 1,
                distortion = 0.75,
                unigrams = dictionary.values()
                )

## 122 x 128 pw
## 128 x 1 pb
pw = tf.reshape(tf.gather(nce_weights,train_labels),[-1,tf.shape(features)[1]])
pb = tf.reshape(tf.gather(nce_biases,train_labels),[-1])

## 128 x 122 nw
## 128 x 1 nb
nw = tf.gather(nce_weights,sample)
nb = tf.reshape(tf.gather(nce_biases,sample),[-1])

## 128 x 122 mul 122 x 128 f x pw + 128 x 1 pb
## 128 x 122 mul 122 x 128 + 128 x 1 nb
p_logits = tf.reduce_sum(tf.mul(features,pw),1) + pb
n_logits = tf.matmul(features,nw,transpose_b = True) + nb

xent = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(
                p_logits, tf.ones_like( p_logits ) ) ) + \
           tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(
                n_logits, tf.zeros_like( n_logits ) ) )

xent = xent / tf.cast( tf.shape(train_labels)[0], dtype = tf.float32 ) / (2 * skip_window)

optimizer = tf.train.GradientDescentOptimizer( lr )
train_step = optimizer.minimize( xent, gate_gradients = optimizer.GATE_NONE )

saver = tf.train.Saver()
#-------------------------------------------------------------------------#
###################### End of the Graph ##################################


####################################### RUN COMPUTATION GRAPH##################################
logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
			 level= logging.INFO)
logger = logging.getLogger()
#num_steps = 300000
num_steps = len(words)/(batch_size/num_skips)
#num_steps = 20000
config = tf.ConfigProto(  gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.72 ) )
sess = tf.Session( config = config )

with sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(10):
        words = shuffle(lines)
        average_loss = 0
        last_loss = 200
    
        for step in xrange(num_steps):
            ## ewb: encoded word batch
            ## bwsi: batch_word_start_index
            ewb,label,bwsi = generate_batch(batch_size,num_skips,skip_window,words,data)
            wb = np.asarray(words[bwsi:bwsi+(batch_size/num_skips)])
            label = np.reshape(np.asarray(label),(batch_size,-1))
            feed_dict = {train_inputs : ewb, train_labels : label,learning_rate :lr}
            _, loss_val= sess.run([train_step, xent], feed_dict=feed_dict)
            average_loss += loss_val
            
            if step % 2000 == 0:
                average_loss = average_loss /2000
                if average_loss >= last_loss:
                    lr *= 0.5 ** (1./512)
                last_loss = average_loss
                print(average_loss, " ", step)
                logger.info(("ave_loss :", average_loss, "with learning rate :", lr, "and bwsi :", bwsi, "in epoch :", epoch, " at step :", step, "out of :", num_steps))
                average_loss = 0
            
    #            f = sess.run(features,feed_dict=feed_dict)
    #            twb = np.asarray(test_cos_wb()) ## noise words
    #            ctwb = np.append(wb,twb) ## sample words
    #            tb = encode(twb) ## encoded sample words
    #
    #            feed_dict = {train_inputs:ewb,test_cos_inputs:tb}
    #            top_k = 8
    #            sim= sess.run(similairty,feed_dict = feed_dict)
    #
    #            for i in range(batch_size/num_skips):
    #                nearest = (-sim[i, :]).argsort()[0:top_k+1]
    #                near_words = [ctwb[w] for w in nearest]
    #                print("near " + str(words[bwsi+i]) + ":" + str(near_words))
    #            print()
        
        saver.save(sess, 'parameters'+str(epoch))
    print("--- %s seconds ---" % (time.time() - start_time))
