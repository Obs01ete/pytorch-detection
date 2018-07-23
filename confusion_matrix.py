# Create a plot of the confusion matrix

import numpy as np
import matplotlib.pyplot as plt

def save_confusion_matrix(confusion_matrix, labelmap, path):
    plt.interactive(False)

    norm_conf = []
    for i in confusion_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(
        np.array(norm_conf), cmap=plt.cm.jet,
        interpolation='nearest')

    width, height = confusion_matrix.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(
                str(confusion_matrix[x, y]), xy=(y, x),
                horizontalalignment='center',
                verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), labelmap[:width], rotation='vertical')
    plt.yticks(range(height), labelmap[:height])
    plt.xlabel('Detected class')
    plt.ylabel('Annotation class')
    plt.subplots_adjust(bottom=0.32, left=0.15)
    plt.savefig(path, format='png', dpi=300)

    pass