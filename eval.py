from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
#from utils import load_batched_data
import os
import sys

INPUT_PATH = './data_root/mfcc' #directory of directories of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './data_root/char_y/' #directory of directories of nCharacters 1-D array .npy files

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300
batchSize = 1 

####Network Parameters
nFeatures = 13 #13 MFCC coefficients
nHidden = 256 # 128
nClasses = 29 #28 characters, plus the "blank" for CTC

# we know this in advance
maxTimeSteps = 250 

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow
        
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
#    targetIxs = tf.placeholder(tf.int64)
#    targetVals = tf.placeholder(tf.int32)
#    targetShape = tf.placeholder(tf.int64)
#    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    ####Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    ####Network
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')
    fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
    outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    ####Optimizing
    logits3d = tf.pack(logits)
#    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
#    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    ####Evaluating
#    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
#    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
#                tf.to_float(tf.size(targetY.values))

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    assert len(sys.argv) == 4

    restore_path = sys.argv[1]
    saver = tf.train.Saver()
    saver.restore(session, restore_path)
    print("Restored from '%s'" % restore_path)

    input_path = sys.argv[2]
    print('Loading input from ', input_path)
    in_data = np.load(input_path)
    time_steps = in_data.shape[1] 
    print('in_data.shape', in_data.shape )
    padded_data = np.pad(in_data, pad_width=((0,0),(0,maxTimeSteps-time_steps)), mode='constant', constant_values=0)
    print('padded_data.shape', padded_data.shape )

    input_data = np.reshape(np.transpose(padded_data), (maxTimeSteps, batchSize, nFeatures))
    print('input shape ', input_data.shape)

    seq_len = np.array([time_steps])
    print('sequence len', seq_len)

    feedDict = {inputX: input_data, seqLengths: seq_len}
    pred, logits3d = session.run([predictions, logits3d], feed_dict=feedDict)
    print('predictions:', pred)
    print('logits:', logits3d.shape)

    output_path =  sys.argv[3]
    print("saved to '%s'" % output_path)
    np.savetxt(output_path, logits3d[:,0,:])
