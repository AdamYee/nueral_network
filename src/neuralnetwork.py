'''
Created on Sep 29, 2011

@author: adam
'''

import math
import random
import time

def getExamples(filename):
    """
    @filename: filename of dataset to use
    returns: shuffled list of each record in the dataset
    """
    examples = []
    f = open(filename)
    for line in f:
        line = line.rstrip().split(',')
        examples.append(line)
    random.shuffle(examples)
#    print(examples)
    return examples

class Neuron:
    def __init__(self, name, lr=0.5):
        self.input_names_and_weights = {}
        self.input_names_and_values = {}
        self.netinput = 0
        self.output = 0
        self.error = 0
        self.name = name
        self.learning_rate = lr
        
    def calcOutput(self, inputs):
#        print(self.name+': input_names_and_weights: '+str(self.input_names_and_weights))
        self.netinput = 0
        for input_name, weight in self.input_names_and_weights.items():
            self.netinput += inputs[input_name]*weight
#        print(self.netinput)
        self.output = 1 / (1 + math.exp(-self.netinput))
        return self.output
    
    def calcHiddenError(self, feeding_layer):
        self.error = self.output * (1 - self.output)
        ksum = 0
        for neuron in feeding_layer:
            ksum += neuron.error * neuron.input_names_and_weights[self.name]
        self.error *= ksum
    
    def updateWeights(self):
        for in_name, weight in self.input_names_and_weights.items():
            self.input_names_and_weights[in_name] = weight + \
                self.learning_rate * self.error * self.input_names_and_values[in_name]
    
    def __str__(self):
        return self.name
    
    def __unicode__(self):
        return self.name

class NeuralNetwork:
    def __init__(self, m=2, n=2): # neurons by hidden layers
        self.m_neurons = m
        self.n_layers = n
        self.network = [] # list of layers (lists)
        self.inputs = {'buying':['vhigh','high','med','low'],
                       'maint':['vhigh','high','med','low'],
                       'doors':['2','3','4','5more'],
                       'persons':['2','4','more'],
                       'lug_boot':['small','med','big'],
                       'safety':['low','med','high']}
        self.outputs = ['unacc','acc','good','vgood']
        self.output_layer = None
        self.upper_limit = 0.9
        self.lower_limit = 0.1
    
    def buildNetwork(self, lr, upper, lower):
        """
        """
        self.upper_limit = upper
        self.lower_limit = lower
        # Input Layer
        # Does not have inputs. It is a long list of feature-attribute input nodes
        inlayer = []
        for feature, attributes in self.inputs.items():
            for attr in attributes:
                name = feature+'-'+attr
                neuron = Neuron(name, lr) # i.e. safety-low
                inlayer.append(neuron)
        self.network.append(inlayer)
        
        if self.n_layers > 0: # if hidden layers are to be created
            # Hidden Layers
            for n in range(self.n_layers):
                layer = []
                for m in range(self.m_neurons):
                    if n == 0:
                        # input layer -> hidden layer
                        neuron = Neuron('N'+str(m)+'|'+str(n), lr)
                        # for each node in the first hidden layer
                        for _input in self.network[0]:
                            # calculate a weight
                            weight = random.randrange(-100,100,1)/100.0
                            # and assign in to an input from the 'input layer'
                            neuron.input_names_and_weights[_input.name] = weight # dict
                    else:
                        # hidden layer -> hidden layer
                        # The same as above, but inputs comes from other hidden layer neurons
                        # instead of from the 'input layer'
                        neuron = Neuron('N'+str(m)+'|'+str(n), lr)
                        for i in range(self.m_neurons):
                            weight = random.randrange(-100,100,1)/100.0
                            # assign the weight to the inputs from the layer before (n-1)
                            neuron.input_names_and_weights['N'+str(i)+'|'+str(n-1)] = weight # dict
                    # add Neuron to current layer
                    layer.append(neuron)
                # add the layer to the network
                self.network.append(layer)
            
            # Output Layer
            output_layer = []
            for output in self.outputs:
                neuron = Neuron(output, lr)
                for i in range(self.m_neurons):
                    weight = random.randrange(-100,100,1)/100.0
                    # using n (last hidden layer)
                    neuron.input_names_and_weights['N'+str(i)+'|'+str(n)] = weight
                output_layer.append(neuron) 
            self.network.append(output_layer)
            
        else: # since no hidden layers, just append the output layer
            output_layer = []
            for output in self.outputs:
                neuron = Neuron(output, lr)
                for in_neuron in self.network[0]:
                    weight = random.randrange(-100,100,1)/100.0
                    # using n (last hidden layer)
                    neuron.input_names_and_weights[in_neuron.name] = weight
                output_layer.append(neuron) 
            self.network.append(output_layer)
        self.output_layer = self.network[-1]
            
    def dumpNetwork(self,option):
        if option == "toScreen":
            print('\nFirst shown is the basic structure of the network')
            print('Hidden layers follow this notation: N(neuron)|(layer)')
            print('Inputs:\t\t'),
            for _input in self.network[0]:
                print(_input),
            print
            i = 0
            for layer in self.network[1:]:
                if layer is self.output_layer:
                    print('Output Layer:\t'),
                else:
                    print('Layer {0}:\t'.format(i)),
                for neuron in layer:
                    print(neuron.name),
                i += 1
                print
            print("\nSecond shown is the network with weights listed for each neuron's inputs")
            print('Neuron\tWeights of inputs')
            for layer in self.network[1:]:
                for neuron in layer:
                    print(neuron.name+'\t'),
                    keyList = neuron.input_names_and_weights.keys()
                    keyList.sort()
                    print('{'),
                    for key in keyList:
                        print('%s (%.2f),' % (key, neuron.input_names_and_weights[key])),
                    print('}')
                print
            print
        elif option == "toFile":
            fp = open('weights.txt','w')
            fp.write('First shown is the basic structure of the network\n')
            fp.write('Hidden layers are follow this notation: N(neuron)|(layer)\n')
            fp.write('Inputs:\t\t')
            for _input in self.network[0]:
                fp.write(_input.name+', ')
            fp.write('\n')
            i = 0
            for layer in self.network[1:]:
                if layer is self.output_layer:
                    fp.write('Output Layer:\t')
                else:
                    fp.write('Layer {0}:\t'.format(i))
                for neuron in layer:
                    fp.write(neuron.name+', ')
                i += 1
                fp.write('\n')
            fp.write("\nSecond shown is the network with weights listed for each neuron's inputs\n")
            fp.write('Neuron\tWeights\n')
            for layer in self.network[1:]:
                for neuron in layer:
                    fp.write(neuron.name+'\t')
                    keyList = neuron.input_names_and_weights.keys()
                    keyList.sort()
                    fp.write('{ ')
                    for key in keyList:
                        fp.write('%s: %.2f, ' % (key, neuron.input_names_and_weights[key]))
                    fp.write(' }\n')
                fp.write('\n')
            fp.close()
    
    def parseExampleToInputs(self, example):
        # Feature Values:
        # buying       vhigh, high, med, low
        # maint        vhigh, high, med, low
        # doors        2, 3, 4, 5more
        # persons      2, 4, more
        # lug_boot     small, med, big
        # safety       low, med, high        
        return ['buying-'+example[0],
                  'maint-'+example[1],
                  'doors-'+example[2],
                  'persons-'+example[3],
                  'lug_boot-'+example[4],
                  'safety-'+example[5]]
        
    def feedForward(self, example):
        """
        @param: example
        """
        flagged_inputs = self.parseExampleToInputs(example)
        inputs = {}
        # assign input values [1|0] to each Input Neuron
        for neuron in self.network[0]: # input layer
            neuron.output = 0 # reset!
            for _input in flagged_inputs:
                if neuron.name == _input:
                    neuron.output = 1; break
            inputs[neuron.name] = neuron.output
        outputs = {}
        for layer in self.network[1:]: # hidden layers + output layer
            outputs.clear()
            for neuron in layer:
                neuron.input_names_and_values = inputs
                outputs[neuron.name] = neuron.calcOutput(inputs)
            # give outputs of each neuron to next layer
            inputs = outputs.copy()
    
    def calcOutputError(self, correct_category):
        """
        @param: correct_category - the classification we're looking for
        
        For each output layer neuron, calculate the error.
        """
        correct_output = {'unacc':0.0,'acc':0.0,'good':0.0,'vgood':0.0}
        correct_output[correct_category] = 1.0
        for o_neuron, category in zip(self.output_layer, self.outputs):
            o_neuron.error = o_neuron.output * \
                            (1 - o_neuron.output) * \
                            (correct_output[category] - o_neuron.output)
                            
    def trainTheNetwork(self, training_set):
        for example in training_set:
            # step 1: feed inputs for a single example
            self.feedForward(example)
            # step 2: calculate the error for each output
            self.calcOutputError(example[6])
            
            # step 3: update weights of each output neuron's input
            for neuron in self.output_layer:
                neuron.updateWeights()
            
            # step 4: calculate error of hidden layer neurons
            # step 5: update hidden layer neuron weights
            # hidden layers: [1:-1] ignore the inputs
            # feeding layers: [2:] list shifted right 1 (ahead of the hidden layers)
            for hidden_layer, feeding_layer in zip(reversed(self.network[1:-1]), reversed(self.network[2:])):
                for hidden_neuron in hidden_layer:
                    hidden_neuron.calcHiddenError(feeding_layer) # pass feeding layer
                    hidden_neuron.updateWeights()

    def classifyExample(self, example):
        self.feedForward(example)
        # a dictionary with 0/1 flags
        d = {'TP':1,'FP':0,'TN':1,'FN':0,'match':1}
        for neuron in self.output_layer:
            if neuron.name == example[6]:
                # the category that should match
                if neuron.output < self.upper_limit: # FN 
                    d['match'] = 0
                    d['FN'] = 1
                    d['TP'] = 0
                else: # TP
                    pass
                break
        for neuron in self.output_layer:
            # all other category outputs that should not match
            if neuron.name != example[6]:
                if neuron.output > self.lower_limit: # FP
                    d['match'] = 0
                    d['FP'] = 1
                    d['TN'] = 0
                    # if there is a false positive, the match is unsuccessful
                    # and we don't care about the rest of the output layer neurons
                    break
                else: # TN
                    pass
        return d
    
    def testExampleData(self, test_set):
        # returns error rate and true/false outputs
        total_correct = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for example in test_set:
            d = self.classifyExample(example)
            total_correct += d['match']
            tp += d['TP']
            fp += d['FP']
            tn += d['TN']
            fn += d['FN']
        return (len(test_set)-total_correct) / float(len(test_set)), tp, fp, tn, fn
    
# Utility functions

def getIntInputInRange(start,end):
    end += 1
    try:
        x = int(raw_input("Range [" + str(start) + "-" + str(end-1) + "]: "))
        if x not in range(start,end):
            raise Exception("Invalid range")
        return x
    except ValueError:
        print("That was no valid number. Try again...")

def getInput(param,start,end):
    while 1:
        try:
            print("("+param+")"),
            theinput = getIntInputInRange(start, end)
            if theinput is None:
                print("Enter a number")
                continue
            else:
                return theinput
        except Exception as error:
            print(error)

def createNetwork():
    # Create n-layer network with H hidden units per layer
    # with full connectivity between layers.
    layers = getInput('Layers',0,3)
    if layers > 0:
        neurons = getInput('Neurons',0,50)
        NN = NeuralNetwork(neurons, layers)
    else:
        NN = NeuralNetwork(0, 0)
    
    while 1:
        try:
            print('(Learning Rate)'),
            lr = float(raw_input('0.0 > lr < 1.0: '))
            if lr > 0.0 and lr < 1.0: break
            else: print('invalid range'); continue
        except Exception as error:
            print(error)
    
    while 1:
        try:
            print('(Upper Classification Limit)'),
            upper = float(raw_input('0.0 > upper < 1.0: '))
            if lr > 0.0 and lr < 1.0: break
            else: print('invalid range'); continue
        except Exception as error:
            print(error)
    while 1:
        try:
            print('(Lower Classification Limit)'),
            lower = float(raw_input('0.0 > lower < 1.0: '))
            if lr > 0.0 and lr < 1.0: break
            else: print('invalid range'); continue
        except Exception as error:
            print(error)
    
    NN.buildNetwork(lr, upper, lower)
    return NN

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

if __name__ == '__main__':
    print("Comp251Project2_NN - Adam Yee")
    print('Create the Neural Network by specifying the following...')
    logging_enabled = ''
    while logging_enabled != 'y' and logging_enabled != 'n':
        logging_enabled = raw_input('Enable logging for testing? (y or n) ')
    NN = createNetwork()
    print('Train the network with how many epochs?')
    while 1:
        try:
            print("(Epochs)"),
            epochs = getIntInputInRange(1, 1000)
            if epochs is None:
                print("Enter a number")
                continue
            else:
                break
        except Exception as error:
            print(error)
    print('Training the network, please wait...')
    chunks = makeChunks("car.data")
    test_set = chunks[0]
    training_set = []
    for c in chunks[1:]:
        training_set += c
    t0 = time.time()
    error_rates = []
    if logging_enabled == 'y':
        for i in range(epochs):
            NN.trainTheNetwork(training_set)
            error_rate, tp, fp, tn, fn = NN.testExampleData(test_set)
            error_rates.append(error_rate)
    else:
        for i in range(epochs):
            NN.trainTheNetwork(training_set)
    total_time = time.time()-t0
    print('Total training time: {0}'.format(total_time))
    print('The network has been trained to the '+str(epochs)+'th epoch and is now ready for use\n')
    
    choice = ''
    while choice != '9':
        print('1) print\n2) dump to "weights.txt"\n3) Run test set\n9) exit')
        choice = raw_input('choice > ')
        if choice == '1':
            NN.dumpNetwork('toScreen')
        elif choice == '2':
            NN.dumpNetwork('toFile')
        elif choice == '3':
            er, tp, fp, tn, fn = NN.testExampleData(test_set)
            print('Error rate: {0}, TP[{1}], FP[{2}], TN[{3}], FN[{4}]'.
                 format(er, tp, fp, tn, fn))
            if logging_enabled == 'y':
                print('All test output written to NN-single-test.txt.')
                f = open('NN-single-test.txt','a')
                f.write('Single test\n')
                f.write('classification limits: {0} < output > {1}\n'.
                         format(NN.lower_limit, NN.upper_limit))
                f.write('hidden layers: {0}\n'.format(NN.n_layers))
                f.write('neurons per layer: {0}\n'.format(NN.m_neurons))
                f.write('epochs: {0}\n'.format(epochs))
                f.write('total training time: {0}\n'.format(total_time))
                f.write('List of error rates per epoch:\n')
                f.write(str(error_rates))
                f.write('\noptima epoch: {0} @ {1}\n'.
                        format(error_rates.index(min(error_rates)), min(error_rates)))
                f.write('Error rate at {5} epoch: {0}, TP[{1}], FP[{2}], TN[{3}], FN[{4}]\n\n'.
                     format(er, tp, fp, tn, fn, epochs))
                f.close()
            



#