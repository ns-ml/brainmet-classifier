from sklearn.metrics import confusion_matrix, classification_report
from time import time
import matplotlib.pyplot as plt
import numpy as np


def benchmark(clf, x_test, y_test, name="Classifier"):
    """Provide benchmark metrics for a classifier

    Arguments:
        clf {[fitted classifier]} -- [result of a clf.fit() call]
        name {[string]} -- [classifier name]
        x_test {[numpy.ndarray]} -- [held out test data]
        y_test {[numpy.ndarray]} -- [held out test labels]
    """
    print('Predicting the outcomes of the testing set')
    t0 = time()
    pred = clf.predict(x_test)
    print('Done in %fs' % (time() - t0))

    print('Classification report on test set for classifier:')
    print(clf)
    print()
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    print("Confusion Matrix:")
    print(cm)


def plot_confusion_matrix(clf, x_test, y_test, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot a confusion matrix for a clf based on held out test data

    Give a binary trained classifier, this function will return plotted
    confusion matrix

    Arguments:
        clf {[fitted classifier]} -- [result of a clf.fit() call]
        x_test {[numpy.ndarray]} -- [held out test data]
        y_test {[numpy.ndarray]} -- [held out test labels]

    Keyword Arguments:
        title {str} -- [Plot title] (default: {'Confusion Matrix'})
        cmap {[type]} -- [color scheme] (default: {plt.cm.Blues})
    """
    pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(pred))
    plt.yticks(tick_marks, np.unique(y_test))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
