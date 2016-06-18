from __future__ import absolute_import
from __future__ import print_function

import collections, math, os, random, zipfile, argparse, sys, logging

from itertools import islice
from six.moves import urllib
from six.moves import xrange #pylint: disable=redefine-buildin
import string
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("save_path", "/catch", "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data", "text8.zip", "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_integer("embedding_width", 24, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("char_vocab_size",101,"the size of ascii")
flags.DEFINE_string("kenel_height","2345678","the height of each kenel")
flags.DEFINE_integer("kenel_width",24,"the width of each kenel which is equal to the width of embedding_width")
flags.DEFINE_integer("in_channel",1,"the 3rd dimention of input batch")
flags.DEFINE_integer("num_skips",8,"")
flags.DEFINE_integer("skip_window",4,"")
flags.DEFINE_integer("vocabulary_size",50000,"")
FLAGS = flags.FLAGS

class Options(object):
	"""Options used by char2vec model"""

	def __init__(self):
	# Model options

		#Embedding dimesion
		self.emb_dim = FLAGS.embedding_width
		# Training options.
		# The training text file.
		self.train_data = FLAGS.train_data
		# Number of negative samples per example.
		self.num_samples = FLAGS.num_neg_samples
		# The initial learning rate.
		self.learning_rate = FLAGS.learning_rate
		# Number of epochs to train. After these many epochs, the learning
		# rate decays linearly to zero and the training stops.
		self.epochs_to_train = FLAGS.epochs_to_train
		# Number of examples for one training step.
		self.batch_size = FLAGS.batch_size
		# Where to write out summaries.
		self.save_path = FLAGS.save_path
		# aci
		self.char_vocab_size = FLAGS.char_vocab_size
		#
		self.num_skips = FLAGS.num_skips
		#
		self.skip_window = FLAGS.skip_window
		#
		self.vocabulary_size = FLAGS.vocabulary_size
		#
		self.kenel_height = FLAGS.kenel_height
		#
		self.kenel_width = FLAGS.kenel_width
		#
		self.in_channel = FLAGS.in_channel
		# make character dictionary based on ascii
		self.char_dict = dict(zip(list(string.printable),range(1,self.char_vocab_size+1)))
		self.reverse_char_dict = dict(zip(range(self.char_vocab_size),list(string.printable)))

	def custom_init(self,emb_dim,num_samples,learning_rate,epochs_to_train,batch_size
			,num_skips,skip_window,kenel_height,vocabulary_size):
		__init__()
		set_emb_dim(emb_dim)
		set_num_samples(num_samples)
		set_learning_rate(learning_rate)
		set_epochs_to_train(epochs_to_train)
		set_batch_size(batch_size)
		set_num_skips(num_skips)
		set_skip_window(skip_window)
		set_kenel_height(kenel_height)
		set_vocabulary_size(vocabulary_size)

	################### SETTING METHOD #################
	def set_emb_dim(self,dims):
		self.emb_dim = dims
		self.kenel_width = dims
	def set_train_data(self,data):
		self.train_data = data
	def set_num_samples(self,num):
		self.num_samples = num
	def set_learning_rate(self,r):
		self.learning_rate = r
	def set_epochs_to_train(self,epochs):
		self.epochs_to_train = epochs
	def set_batch_size(self,batch_size):
		self.batch_size = batch_size
	def set_save_path(self,path):
		self.save_path = path
	def set_num_skips(self,num_skips):
		self.num_skips = num_skips
	def set_skip_window(self,skip_window):
		self.skip_window = skip_window
	def set_kenel_height(self,kenel_height):
		self.kenel_height = kenel_height
	def set_vocabulary_size(self,size):
		self.vocabulary_size = size



class Char2Vec(object):
	"""docstring for model"""
	def __init__(self, options,session):
		self.op = options
		self._session = session
		self.data_index = 0
		self.word_height = None
		self.data = None
		self.count = None
		self.dictionary = None
		self.reverse_dictionary = None
	
	#step 1: Download the data.
	def maybe_download(self,filename,expected_bytes):
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

	def read_data(self,filename):
		'''Extract the first file enclosed in a zip file as a list of words'''
		with zipfile.ZipFile(filename) as f:
			data = f.read(f.namelist()[0]).split()
		return data

	def calculate_longest_word(self,words):
		maxlen = 0
		for word in words:
			maxlen = max(maxlen,len(word))
		return maxlen

	def build_dataset(self,words):
		count = [['UNK',-1]]
		## count the words in `words' and select the most common 50000 words in `words' and put them into count
		count.extend(collections.Counter(words).most_common(self.op.vocabulary_size))
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

	# Step 3: Function to generate a training batch for the skip-gram model.
	def generate_batch(self,batch_size, num_skips, skip_window,words):
	  assert batch_size % num_skips == 0
	  assert num_skips <= 2 * skip_window
	  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
	  buffer = collections.deque(maxlen=span)
	  for _ in range(span):
	    buffer.append(self.data[self.data_index])
	    self.data_index = (self.data_index + 1) % len(self.data)
	  for i in range(batch_size // num_skips):
	    target = skip_window  # target label at the center of the buffer
	    targets_to_avoid = [ skip_window ]
	    for j in range(num_skips):
	      while target in targets_to_avoid:
	        target = random.randint(0, span - 1)
	      targets_to_avoid.append(target)
	      batch[i * num_skips + j] = buffer[skip_window]
	      labels[i * num_skips + j, 0] = buffer[target]
	    buffer.append(self.data[self.data_index])
	    self.data_index = (self.data_index + 1) % len(self.data)

	  batch = [self.reverse_dictionary[index] for index in batch]
	  return batch, labels

	# encode word in batch
	def encode_word_batch(self,word_batch):
		char_batch = [list(w) for w in word_batch]
		encoded_cb = map(lambda x: map(lambda y:self.op.char_dict[y], x), char_batch)
		return encoded_cb

	# pad zeros in each ecoded_char_batch to get a fixed size encoded_cb
	def pad_encode_char_batch(self,encoded_cb):
		map(lambda x: x.extend([0]*(self.word_height-len(x))),encoded_cb)
		return encoded_cb

	def convolution(self,embeddings,kenel,paded_ecb,kenel_height):
		with tf.device('/cpu:0'):
			embed_cube = tf.map_fn(lambda x:tf.nn.embedding_lookup(embeddings,x),paded_ecb,dtype=tf.float32)
			embed_cube = tf.expand_dims(embed_cube,3)
			conv = tf.nn.conv2d(embed_cube,kenel,[1,1,1,1],padding='VALID')
			biases = tf.Variable(tf.random_uniform([kenel_height],-1.0,1.0))
			bias = tf.nn.bias_add(conv,biases)
			conv = tf.tanh(bias)
			pool = tf.reduce_max(conv,reduction_indices=[1,2])
		return pool

	def generate_valid_set(self):
		rm = random.randint(0,len(words)-(valid_size+1))
		valid_set = words[rm:rm+valid_size]
		valid_cb = pad_encode_char_batch(encode_word_batch(valid_set))
		return rm, valid_cb

	def train(self):
		opts = self.op
		batch_size = self.op.batch_size
		word_height = self.word_height
		char_vocab_size = self.op.char_vocab_size
		embedding_width = self.op.emb_dim
		kenel_height = map(int,list(self.op.kenel_height))
		kenel_width = self.op.kenel_width
		in_channel = self.op.in_channel
		filename = self.maybe_download(opts.train_data,31344016)
		words = self.read_data(filename)
		vocabulary_size = len(collections.Counter(words))
		self.op.set_vocabulary_size(vocabulary_size)
		feature_size = ((kenel_height[0]+kenel_height[-1])*len(kenel_height))/2
		kenel_height = map(int,list(self.op.kenel_height))
		num_skips = opts.num_skips
		skip_window = opts.skip_window
		self.word_height = self.calculate_longest_word(words)
		self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset(words)
		print(len(self.reverse_dictionary))

		'''
		############################ negative sampling###################################
		num_sampled = 64    # Number of negative examples to sample.
		############################Define graph########################################
		# graph = tf.Graph()
		# with graph.as_default():
		#######################variables##########################################
		train_inputs = tf.placeholder(tf.int32, shape=[batch_size,word_height])
		train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
		learning_rate = tf.placeholder(tf.float32)
		######################Graph##############################################
		# with tf.device('/cpu:0'):
		embeddings = tf.Variable(tf.random_uniform([char_vocab_size, embedding_width], -1.0, 1.0))
		kenels = [tf.Variable(tf.random_uniform([kenel_height[i],kenel_width,in_channel,kenel_height[i]],-1.0,1.0))
					for i in range(len(kenel_height))]
		features = tf.concat(1,[self.convolution(embeddings,kenels[k],train_inputs,kenel_height[k]) for k in range(len(kenels))])
		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, feature_size],stddev=1.0 / math.sqrt(feature_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

		# Compute the average NCE loss for the batch.
		# tf.nce_loss automatically draws a new sample of the negative labels each
		# time we evaluate the loss.
		loss = tf.reduce_mean(
			tf.nn.nce_loss(nce_weights, nce_biases, features, train_labels,num_sampled, vocabulary_size))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		norm = tf.sqrt(tf.reduce_sum(tf.square(features), 1, keep_dims=True))
		normalized_features = features / norm
		saver = tf.train.Saver()

		####################################### RUN COMPUTATION GRAPH##################################
		logging.basicConfig( format = '%(asctime)s : %(levelname)s : %(message)s', 
						 level= logging.INFO)
		logger = logging.getLogger()
		# num_steps = 100001
		num_steps = 150000
		# with self._session as session:
		self._session.run(tf.initialize_all_variables())
		average_loss = 0
		for step in xrange(num_steps):
			word_batch, batch_labels = self.generate_batch(batch_size, num_skips, skip_window, words)
			# batch, labels = self.generate_batch(batch_size=8, num_skips=2, skip_window=3,words=words)
			# for i in range(8):
			#   print(batch[i], '->', labels[i, 0])
			#   print(self.dictionary[batch[i]], '->', self.reverse_dictionary[labels[i][0]])
			encoded_wb = self.encode_word_batch(word_batch)
			batch_inputs = np.asarray(self.pad_encode_char_batch(encoded_wb))
			lr = self.op.learning_rate
			feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels,learning_rate :lr}
			_, loss_val = self._session.run([optimizer, loss], feed_dict=feed_dict)
			average_loss += loss_val
			last_loss = 1000
			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
				# logger.info(("Average loss at step ", step, ": ", average_loss))
				# logger.info(("learning at step ", step, ": ", self.op.learning_rate))
				# logger.info(("Average_loss at last step ", (step-2000), ": ", last_loss))
				if average_loss >= last_loss:
					self.op.learning_rate *= 0.95
				last_loss = average_loss
			logger.info(("loss at last step ", (step), ": ", loss_val, "learning rate is :", self.op.learning_rate))
			# print("loss at last step ", (step), ": ", loss_val, "learning rate is :", self.op.learning_rate)
			# logger.info(("learning at step ", step, ": ", self.op.learning_rate))
			print(step,',',loss_val)
			average_loss = 0

		# # final_embeddings = normalized_features.eval()
		# # final_embeddings = embeddings.eval()
		# # w_kenels = kenels.eval()
		save_path = saver.save(self._session, "cache")'''


def main():
	opts = Options()
	opts.set_learning_rate(1.0)
	opts.set_skip_window(4)
	opts.set_num_skips(8)
	with tf.Graph().as_default(), tf.Session() as session:
	    with tf.device("/cpu:0"):
	      model = Char2Vec(opts, session)
	      # for _ in xrange(opts.epochs_to_train):
	      model.train()
	# opts = Options()
	# opts.set_learning_rate(.02)
	# with tf.Graph().as_default(), tf.Session() as session:
	# 	model = Char2Vec(opts,session)
	# 	filename = model.maybe_download(opts.train_data,31344016)
	# 	words = model.read_data(filename)
	# 	dictionary = dict()
	# 	for w in collections.Counter(words).keys():
	# 		dictionary[w] = len(dictionary)
	# 	print(dictionary)


if __name__ == "__main__":
	main()
