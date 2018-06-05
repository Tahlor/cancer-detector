import numpy as np
from archipack import *
import os
import sys

try:
    from skimage import io, transform
except:
    pass

def count_files(pattern = "train", path = 'cancer_data/inputs'):
    files = os.listdir(path)
    total = 0
    # Count number of files
    for f in files:
        if pattern in f:
            total += 1
    return total
    
# Read and transform
def read_data(batch_size = 2000, batch_number = 0, datatype = "train", size = 512):
    files = os.listdir('cancer_data/inputs')
    trd, trl  = [], []
    count = -1 # make sure 0 indexed
    total = count_files(pattern = datatype)            
    batch_size = min(batch_size, total)

    begin_batch = (batch_number*batch_size) % total
    end_batch   = (batch_number+1) * batch_size % total
    if end_batch == 0:
        end_batch = (batch_number+1) * batch_size
    print(total, begin_batch, end_batch)
    #for f in tqdm(files):
    for f in files:
       if datatype in f:
            count += 1
            #print(count, begin_batch, end_batch)           
            if begin_batch >  end_batch:
                if count >= end_batch and count < begin_batch:
                   continue 
            elif count not in range(begin_batch, end_batch):
                continue
            m = transform.resize(io.imread('cancer_data/inputs/' + f), (size,size,3), mode='constant')
            n = transform.resize(io.imread('cancer_data/outputs/' + f), (size,size,3), mode='constant')
            trd.append(whiten_data(m))
            # Don't whiten labels, only keep 1 layer
            trl.append(n[:,:,1])
            #print(f)
            #print(n[:,:,1])
            if f == "pos_test_000072.png":
                s = "" if size == 512 else str(size)
                pickle(trd[-1], "testImageData"+s)
                pickle(trl[-1], "testImageLabels"+s)
    return trd, trl


def whiten_data(I):
    I = I -  I.mean()
    I = I / np.std(I)
    return I

def whiten_data_list(inputList):
    for i, I in enumerate(inputList):
        I = I -  I.mean()
        #print(I.shape)
        #print(np.std(I))
        I = I / np.std(I)
        inputList[i] = I
    return inputList

def pickle(myList, destination):
    import pickle
    with open(destination, 'wb') as f:
        pickle.dump(myList, f, protocol=2)

def unpickle(destination):
    import sys
    if sys.version_info[0] >= 0:
        import pickle
        with open(destination, 'rb') as f:
            myList = pickle.load(f)
        return myList

            
def preprocess_data(datatype = 'test', io_batch = 2000, size= 512):
    s = "" if size == 512 else str(size)
    if datatype in ['test','both']:
        test_data, test_labels = read_data(io_batch, 0, "test", size = size)
        shuffleDataAndLabelsInPlace(test_data, test_labels)
        pickle(test_data, "test_data"+s)
        del test_data
        pickle(test_labels, "test_labels"+s)
        del test_labels
    if datatype in ['train','both']:
        train_data, train_labels = read_data(io_batch, 0, "train", size = size)
        shuffleDataAndLabelsInPlace(train_data, train_labels)
        pickle(train_labels, "train_labels"+s)
        del train_labels
        pickle(train_data, "train_data"+s)
        del train_data

def load_data(size = 512):
    s = "" if size == 512 else str(size)
    test_labels = unpickle("test_labels"+s)
    test_data = unpickle("test_data"+s)
    train_data = unpickle("train_data"+s)
    train_labels = unpickle("train_labels"+s)
    return test_labels, test_data, train_data, train_labels
    
if __name__ == '__main__':
    #trd, trl = read_data(100,50, "train")
    #tstd, tstl = read_data(100,50, "test")
    #trd = whiten_data(trd)
    #train_data, train_labels = read_data(1, 0, "train")
    #print(train_labels[0].shape)
    preprocess_data('both', io_batch = 2000, size = 512)
    #read_data(2000, 0, "test", size = 32)
    #load_data(size = 32)
    pass
    
