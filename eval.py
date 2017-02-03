from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np

import params
params.batchSize = 1

assert len(sys.argv) == 4
input_path = sys.argv[2]
print('Loading input from ', input_path)
in_data = np.transpose(np.loadtxt(input_path))
assert in_data.shape[0] == params.nFeatures
time_steps = in_data.shape[1]
params.maxTimeSteps = time_steps

from model_graph import *

classes = " abcdefghijklmnopqrstuvwxyz'"
i = 0
classmap = {}
for c in classes:
    classmap[i] = c
    i += 1

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()

    restore_path = sys.argv[1]
    saver = tf.train.Saver()
    saver.restore(session, restore_path)
    print("Restored from '%s'" % restore_path)

    print('in_data.shape ', in_data.shape )
    print('maxTimeSteps ', maxTimeSteps)
    assert time_steps <= maxTimeSteps

    input_data = np.reshape(np.transpose(in_data), (maxTimeSteps, batchSize, nFeatures))
    print('input shape ', input_data.shape)

    seq_len = np.array([time_steps])
    print('sequence len', seq_len)

    feedDict = {inputX: input_data, seqLengths: seq_len}
    pred, logits3d = session.run([predictions, logits3d], feed_dict=feedDict)
    print('predictions:', pred.values)
    pred_chars = [ classmap[c] for c in pred.values]
    print('prediction chars:', pred_chars)

    print('logits:', logits3d.shape)

    output_path =  sys.argv[3]
    print("saved to '%s'" % output_path)
    np.savetxt(output_path, np.exp(logits3d[:,0,:]))
