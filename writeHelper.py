__author__ = 'Ariel'


def writePredMemory(file, pred, tuples):
    with open(file, 'w') as f:
        for x in tuples:
            f.write('%s\n' % pred[x])
    return

def writePredModel(file, pred):
    with open(file, 'w') as f:
        for x in pred:
            f.write('%s\n' % x)
    return