'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict character sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is test code to run on the
8-item data set in the "sample_data" directory, for those without access to TIMIT.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from utils import load_batched_data
from model_graph import *
import os

INPUT_PATH = './data_root/mfcc' #directory of directories of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './data_root/char_y/' #directory of directories of nCharacters 1-D array .npy files

print('maxTimeSteps=', maxTimeSteps)

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = 0.0
        totalN = 0

        subdirs_list = np.random.permutation(os.listdir(INPUT_PATH))
        for subdir in subdirs_list:
            input_path = os.path.join(INPUT_PATH, subdir)
            output_path = os.path.join(TARGET_PATH, subdir) # assuming subdirs are same in both INPUT_PATH and TARGET_PATH roots
            print('Loading input from ', input_path)
            #print('Loading output from ', output_path)
            batchedData, subMax, subdirN = load_batched_data(input_path, output_path, batchSize, maxTimeSteps)
            totalN += subdirN

            batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
            for batch, batchOrigI in enumerate(batchRandIxs):
                print('file ', batchOrigI)
                batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
                _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
#                print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
#                if (batch % 1) == 0:
#                    print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
#                    print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
                batchErrors += er*len(batchSeqLengths)
            print('error rate so far:', batchErrors / totalN)

        # Epoch finished
        print('Epoch', epoch+1, 'error rate:', batchErrors / totalN)
        save_path = saver.save(session, "./model.ckpt", global_step=epoch)
        print("Model saved in file: %s" % save_path)
