from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

from six.moves import urllib
from six.moves import xrange #pylint: disable=redefine-buildin
import string
import numpy as np
import tensorflow as tf

word_height = 24
embedding_width = 24
char_vocab_size = 100

scale = 4
kenel_height = [2,3,4,5,6,7,8]
kenel_width = 24 #kenel_width == char embedding width
in_channel = 1
out_channel = 1

# make character dictionary based on ascii 
char_dict = dict(zip(list(string.printable),range(1,char_vocab_size+1)))
char_dict['empty'] = 0
reverse_char_dict = dict(zip(range(char_vocab_size),list(string.printable)))
reverse_char_dict['0'] = 'empty'

#step 1: Download the data.
def maybe_download(filename,expected_bytes):
	'''Download a file if not present, and make sure it's the right size.'''
	url = 'http://mattmahoney.net/dc/'
	if not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified',filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
	return filename

#Read the data into a list of strings
def read_data(filename):
	'''Extract the first file enclosed in a zip file as a list of words'''
	with zipfile.ZipFile(filename) as f:
		data = f.read(f.namelist()[0]).split()
	return data

filename = maybe_download('text8.zip',31344016)
words = read_data(filename)
vocabulary_size = 50000

def build_dataset(words):
	count = [['UNK',-1]]
	## count the words in `words' and select the most common 50000 words in `words' and put them into count
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary) ## index each word
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary: ## if the word in the 50000 most common words dictionary
			index = dictionary[word]
		else:
			index = 0 ## if not in the dictionary set the index do 0
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count ##count[0][1] is -1 in ['UNK', -1] at first, now we put the count for unkown words at count[0][1]
	reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  batch = [reverse_dictionary[index] for index in batch]
  return batch, labels

# encode word in batch
def encode_word_batch(word_batch):
	char_batch = np.array([np.array(list(w)) for w in word_batch])
	encoded_cb = np.array(map(lambda x: map(lambda y: char_dict[y], x), char_batch))
	return encoded_cb

# pad zeros in each ecoded_char_batch to get a fixed size encoded_cb
def pad_encode_char_batch(encoded_cb):
	np.array([map(lambda x: x.extend([0]*(embedding_width-len(x))),encoded_cb)])
	return np.array(encoded_cb)

def generate_kenel():
	kenel = []
	for i in range(len(kenel_height)):
		kenel.append(tf.Variable(tf.random_uniform([kenel_height[i],kenel_width,in_channel,kenel_height[i]*scale],-1.0,1.0)))
	return tf.convert_to_tensor(kenel)

def convolution(embeddings,kenel,paded_ecb,kenel_height):
	# embed_cube = []
	with tf.device('/cpu:0'):
		# for w in paded_ecb:
		# 	embed = tf.nn.embedding_lookup(embeddings,w)
		# 	embed_cube.append(embed)
		embed_cube = [tf.map_fn(lambda w: tf.nn.embedding_lookup(embeddings,w),paded_ecb)]
		embed_cube = tf.expand_dims(embed_cube,3)
		conv = tf.nn.conv2d(embed_cube,kenel,[1,1,1,1],padding='VALID')
		biases = tf.random_uniform([kenel_height*scale],-1.0,1.0)
		bias = tf.nn.bias_add(conv,biases)
		conv = tf.tanh(bias)
		pool = tf.reduce_max(conv,reduction_indices=[1,2])
	return tf.convert_to_tensor(pool)

def concatenate(a,b):
	if a != None and b != None:
		return tf.concat(1,[a,b])
	elif a == None and b == None:
		raise ValueError('can not cancatenate two none tensors')
	elif a == None:
		return b
	elif b == None:
		return a

def generate_feature_map(embeddings,kenel,paded_ecb):
	feature = None
	for i in range(len(kenel_height)):
		feature = concatenate(feature,convolution(embeddings,kenel[i],paded_ecb,kenel_height[i]))
	tf.convert_to_tensor(feature)
	return feature

def main():
	'''
	# Generate word batch and label
	word_batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=3)
	encoded_wb = encode_word_batch(word_batch) #encode word to char index array
	paded_ecb = pad_encode_char_batch(encoded_wb) # for each char index array pad zeros to width embedding_width 

    ##for each word batch, find the char embedding 2d surface of each word
	embeddings = tf.random_uniform([char_vocab_size,embedding_width],-1.0,1.0)
	kenel = generate_kenel()
	feature_tensor = generate_feature_map(embeddings,kenel,paded_ecb)


	with tf.Session() as sess:
		print(sess.run(tf.shape(feature_tensor)))
	'''
	batch_size = 128
	skip_window = 1 
	num_skips = 2 
	# We pick a random validation set to sample nearest neighbors. Here we limit the
	# validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent.
	valid_size = 16     # Random set of words to evaluate similarity on.
	valid_window = 100  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)
	num_sampled = 64    # Number of negative examples to sample.
	# Define graph
	graph = tf.Graph()
	with graph.as_default():
		# Input data.
		train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
		
		with tf.device('/cpu:0'):
			# Look up embeddings for input chars.
			embeddings = tf.Variable(
			    tf.random_uniform([char_vocab_size, embedding_width], -1.0, 1.0))
			############### convolution on char embeddings #################
			# generate kenels
			features = generate_feature_map(embeddings,kenel_height,train_inputs)
			# Construct the variables for the NCE loss
			nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_width],stddev=1.0 / math.sqrt(embedding_size)))
			nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		loss = tf.reduce_mean(
		  tf.nn.nce_loss(nce_weights, nce_biases, features, train_labels,
		                 num_sampled, vocabulary_size))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		# Compute the cosine similarity between minibatch examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(features), 1, keep_dims=True))
		normalized_features = embeddings / norm
		valid_features = tf.nn.embedding_lookup(normalized_features, valid_dataset)
		similarity = tf.matmul(valid_features, normalized_embeddings, transpose_b=True)


	# Step 5: Begin training.
	num_steps = 100001

	with tf.Session(graph=graph) as session:
	  # We must initialize all variables before we use them.
	  tf.initialize_all_variables().run()
	  print("Initialized")

	  average_loss = 0
	  for step in xrange(num_steps):
	    word_batch, batch_labels = generate_batch(batch_size, num_skips, skip_window)
	    encoded_wb = encode_word_batch(word_batch)
	    batch_inputs = pad_encode_char_batch(encoded_wb)
	    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

	    # We perform one update step by evaluating the optimizer op (including it
	    # in the list of returned values for session.run()
	    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
	    average_loss += loss_val

	    if step % 2000 == 0:
	      if step > 0:
	        average_loss /= 2000
	      # The average loss is an estimate of the loss over the last 2000 batches.
	      print("Average loss at step ", step, ": ", average_loss)
	      average_loss = 0

	    # Note that this is expensive (~20% slowdown if computed every 500 steps)
	    if step % 10000 == 0:
	      sim = similarity.eval()
	      for i in xrange(valid_size):
	        valid_word = reverse_dictionary[valid_examples[i]]
	        top_k = 8 # number of nearest neighbors
	        nearest = (-sim[i, :]).argsort()[1:top_k+1]
	        log_str = "Nearest to %s:" % valid_word
	        for k in xrange(top_k):
	          close_word = reverse_dictionary[nearest[k]]
	          log_str = "%s %s," % (log_str, close_word)
	        print(log_str)
	  final_embeddings = normalized_embeddings.eval()

if __name__ == "__main__":
	main()