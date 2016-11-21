import collections
import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef int data_index = 0
cdef int length = 0

cdef extern from "<regex>" namespace "std" nogil:
    cdef cppclass regex:
        regex ( string& s) except +
    bint regex_match(string& s, regex& r)

cdef regex* number = new regex(r"^(?=[^A-Za-z]+$).*[0-9].*$")

def read_data(filename):
    '''Extract the first file enclosed in a zip file as a list of words'''
    with open(filename) as f:
        words = [line for line in f]
    return words

def build_data(vocabulary_size,words):
    cdef int i=0, unk_count
    cdef vector[int] data
    global length
    length = len(words)
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size))
    dictionary = dict([(w[0], i) for i, w in enumerate(count)])
    for i in range(length):
        if words[i] in dictionary:
            index = dictionary[words[i]]
        else:
            index = 0
            unk_count += 1
        data.push_back(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary,reverse_dictionary

def generate_batch(batch_size,num_skips,skip_window,words,data):
    global number
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    cdef int ubs = batch_size / num_skips
    cdef int span = 2*skip_window + 1
    cdef vector[int] cbf
    cdef vector[vector[int]] wbf
    cdef vector[int] label
    cdef int i,j,k
    global length

    global data_index
    if data_index == 0:
        data_index = skip_window 
    
    
    cdef left, right
    maxl = 10
    for k in range(ubs):
        left = data_index - skip_window 
        right = data_index + skip_window
        ## encode word ###
        cwl = <int>len(words[data_index])
        if cwl > maxl:
                maxl = cwl
        for j in range(cwl):
            cbf.push_back(ord(words[data_index][j]))
        wbf.push_back(cbf)
        cbf.clear()
        
        ## record labels ##
        for i in range(left,right+1):
            if i == data_index:
                continue
            i = i % length
            label.push_back(data[i])
            
        data_index = (data_index+1) % (length)
        
    ## padding zeros
    for i in range(ubs):
        wbf[i].resize(maxl)
            
    return wbf,label,(data_index-ubs)

def encode(words_batch):
    cdef int i,j,wbl,wl
    cdef vector[int] cbf
    cdef vector[vector[int]] wbf
    wbl = len(words_batch)
    maxl = 10
    for i in range(wbl):
        w = words_batch[i]
        wl = len(w)
        if wl > maxl:
            maxl = wl
        for j in range(wl):
            cbf.push_back(ord(w[j]))
        wbf.push_back(cbf)
        cbf.clear()
    
    for i in range(wbl):
        wbf[i].resize(maxl)
    return wbf