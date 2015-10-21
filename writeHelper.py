__author__ = 'Ariel'


def writePred(file, pred, tuples):
    with open(file, 'w') as f:
        for x in tuples:
            f.write('%d\n' % pred[x])
    return
