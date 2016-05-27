#NGramCounter builds a dictionary relating ngrm tuples to the number
#of times that ngram occurs in a text (as integers)
class NGramCounter(object):
	#paramter n is the 'order' (length) of the desired n-gram
	def __init__(self,n,max):
		self.n = n
		self.ngrams = dict()
		self.beginnings = list()
		self.max = max

	#feed method calls tokenize to break the given string up into units
	def tokenizes(self,text):
		return text.split(" ")

	#feed method takes text, tokenizes it, and visits every group of n tokens
	#in turn, adding the group to self.ngrams or 
	#incrementing count in same

	def feed(self,text):
		tokens  = self.tokenizes(text)
		#e.g, for a list of length 10, and n of 4, 10 - 4 + 1 = 7
		#tokens[7:11] will give last three elements of list
		'''
		for i in range(len(tokens) - self.n + 1):
			gram = tuple(tokens[i:i+self.n])
			if gram in self.ngrams:
				self.ngrams[gram] += 1
			else:
				self.ngrams[gram] = 1
		'''
		#store the first ngram of this line
		begining = tuple(tokens[:self.n])
		self.beginnings.append(begining)

		for i in range(len(tokens) - self.n):#get the element after the gram
			gram = tuple(tokens[i:i+self.n])
			next = tokens[i+self.n]
			#if we've already seen this ngram, append; otherwise, set the
			#value for this key as a new list
			if gram in self.ngrams:
				self.ngrams[gram].append(next)	
			else:
				self.ngrams[gram] = [next]
	def get_ngrams(self):
		return self.ngrams

	def generate(self):
		from random import choice

		#get a ramdom line begining; conver to a list.
		current = choice(self.beginnings)
		output = list(current)

		for i in range(self.max):
			if current in self.ngrams:
				possible_next = self.ngrams[current]
				next = choice(possible_next)
				output.append(next)
				#get the last n entries of the output; we will use
				#this to look up
				#an ngram in the next iteration of the loop
				current = tuple(output[-self.n:])
			else:
				break

		output_str = self.concatenate(output)
		return output_str

	def concatenate(self,str):
		return ' '.join(str)
# if __name__ == '__main__':

# 	import sys

# 	#create an NGramCounter object and feed data to it
# 	ngram_counter = NGramCounter(4)
# 	for line in sys.stdin:
# 		line = line.strip()
# 		ngram_counter.feed(line)

# 	#get ngrams from ngram counter; iterate over keys, printing out keys
# 	#with a count greater than one
# 	ngrams = ngram_counter.get_ngrams()
# 	for ngram in ngrams.keys():
# 		count = ngrams[ngram]
# 		if count > 1:
# 			print ' '.join(ngram) + ": " + str(count)