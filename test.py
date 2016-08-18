from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import load_batched_data
from model_graph import *
import os
import sys

INPUT_PATH = './data_root/mfcc' #directory of directories of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './data_root/char_y/' #directory of directories of nCharacters 1-D array .npy files

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
    output_path = sys.argv[3]
    print('Loading input from ', input_path)
    print('Loading output from ', output_path)
    batchedData, subMax, subdirN = load_batched_data(input_path, output_path, batchSize, maxTimeSteps)
    totalErr = 0
    batches = 0
    for batch in batchedData:
        batchInputs, batchTargetSparse, batchSeqLengths = batch
        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse       
        feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                    targetShape: batchTargetShape, seqLengths: batchSeqLengths}
        er, pred = session.run([errorRate, predictions], feed_dict=feedDict)
        print('error rate:', er)
        totalErr += er
        batches += 1
        print('inputs: ', batchInputs.shape)
        print('seqlen: ', batchSeqLengths )
        #print('targetIxs:', batchTargetIxs)
        #print('targetVals:', batchTargetVals)
        #print('predictions:', pred)

    print('average error rate:', totalErr / batches)
