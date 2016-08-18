####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300

####Network Parameters
nFeatures = 13 #13 MFCC coefficients
nHidden = 256 # 128
nClasses = 29 #28 characters, plus the "blank" for CTC
batchSize = 4

# we know this in advance
maxTimeSteps = 250

