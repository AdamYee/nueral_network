'''
Created on Nov 21, 2011

@author: adam
'''

from neuralnetwork import getExamples, NeuralNetwork
from time import time
import math
import sys

# Partition data set into parts based on chunk size
def chunks(lst, n):
    """ Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def partition(dataset, K_size):
    chunksize = int(math.ceil(len(dataset)/float(K_size)))
    partitioned = []
    for k in chunks(dataset, chunksize): # converting generator to list
        partitioned.append(k)
    return partitioned

def makeChunks(filename):
    """ Partitions dataset into K chunks. 
    """
    dataset = getExamples(filename) # shuffled dataset
    return partition(dataset, 10)

def findOptimaEpoch():
    chunks = makeChunks("car.data")
    tuning_set = chunks[0]
    grow_set = []
    for c in chunks[1:]:
        grow_set += c
    epochs = 80
    g = []
    r = 0.1
    while r < 0.9:
        g.append(r)
        r += 0.1
#    for lr in g: # vary the learning rate
#    for blah in range(0,5): # repeat five times
#    for n in range(1,16):
    t0 = time()
    error_rates = []
    NN = NeuralNetwork(13, 3) # layers fixed at 3
    NN.buildNetwork(0.65,0.9,0.1) # pass the learning rate
    for i in range(epochs):
        NN.trainTheNetwork(grow_set)
        error_rate, tp, fp, tn, fn = NN.testExampleData(tuning_set)
        error_rates.append(error_rate)
    fp = open('finding-optima-epoch.txt','a')
    # layers: {0}, neurons: {1}
    fp.write('{0} {1} {2}\n'.format(*(NN.n_layers, NN.m_neurons, 0.65)))
    # Number of run epochs: {0}\n
    fp.write('{0}\n'.format(epochs))
    # error rates: {0}\n
    fp.write('{0}\n'.format(str(error_rates)))
    # epoch with lowest error rate: {0} -> {1} (zero indexing)\n
    fp.write('optima epoch: {0} @ {1}\n'.format(error_rates.index(min(error_rates)), min(error_rates)))
    # Total training time: {0}\n\n
    fp.write('{0}\n\n'.format(time()-t0))
    fp.close()
    print(error_rates.index(min(error_rates)))
    del(NN)
   
def kfoldCrossValidation(epochs):
    chunks = makeChunks('car.data')
    err_list = []
    tp_sum = 0
    falsep_sum = 0
    tn_sum = 0
    fn_sum = 0
    for fold in range(10):
        t0 = time()
        # defaulting to 10 chunks
        validation_set = chunks[fold]
        training_set = []
        for chunk in chunks[:fold] + chunks[fold+1:]:
            training_set += chunk
        # best determined network size
        NN = NeuralNetwork(13, 3)
        # best determined learning rate, upper and lower classification limits
        NN.buildNetwork(0.65,0.9,0.1)
        for i in range(epochs): # train the network
            NN.trainTheNetwork(training_set)
        ttime = time()-t0
        er, tp, falsep, tn, fn = NN.testExampleData(validation_set)
        err_list.append( er )
        tp_sum += tp
        falsep_sum += falsep
        tn_sum += tn
        fn_sum += fn
        fp = open('NN-kfoldcrossvalidation.txt','a')
        fp.write('{0} {1} {2}\n'. # layers: {0}, neurons: {1}
                 format(*(NN.n_layers, NN.m_neurons, 0.65)))
        fp.write('{0}\n'.format(epochs)) # Number of run epochs: {0}
        fp.write('error rate: {0}\n'.format(er)) # error rate: {0}
        fp.write('T/F analysis: TP[{0}], FP[{1}], TN[{2}], FN[{3}]\n'.
                 format(tp, falsep, tn, fn))
        fp.write('{0}\n\n'.format(ttime)) # Total training time: {0}
        fp.close()
        del(NN)
    return err_list, tp_sum, falsep_sum, tn_sum, fn_sum

if __name__ == '__main__':
    choice = raw_input("1) find optima epoch\n2) perform k-fold cross validation\nchoice > ")
    if choice == '1':
        print('Network size fixed (3 layer, 13 neurons per layer).')
        print('Neuron learning rate set to 0.65')
        print('Number of epochs set to 80. The lowest epoch error rate is recorded.')
        print('analysis output appended to finding-optima-epoch.txt.')
        print('Calculating, please wait...')
        findOptimaEpoch()
        print('Analysis complete, please see finding-optima-epoch.txt for results.')
    elif choice == '2':
        try:
            epochs = input('Number of epochs > ')
        except Exception as error:
            print(error),
            print('- re-run NNanalysis.py')
            sys.exit(1)
        print('Network size fixed (3 layer, 13 neurons per layer).')
        print('Neuron learning rate set to 0.65')
        print('analysis output appended to NN-kfoldcrossvalidation.txt.')
        print('Calculating, please wait...')
        errorlist,tp,falsep,tn,fn = kfoldCrossValidation(epochs)
        fp = open('NN-kfoldcrossvalidation.txt','a')
        fp.write('classification limits: 0.1 < output > 0.9\n')
        fp.write('error rate list: {0}\n'.format(str(errorlist)))
        fp.write('average error rate: {0}\n'.
                 format( str(sum(errorlist)/float(len(errorlist))) )
            )
        fp.write('total sum of true/false: TP[{0}], FP[{1}], TN[{2}], FN[{3}]\n'.
                 format(tp, falsep, tn, fn))
        fp.write('----------------------------------------------------------------\n\n')
        fp.close()
        print('Analysis complete, please see NN-kfoldcrossvalidation.txt for results.')
    else:
        print('not an option, re-run NNanalysis.py')







#