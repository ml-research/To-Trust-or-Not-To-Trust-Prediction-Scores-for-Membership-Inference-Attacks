from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def get_roc(labels, prediction_scores):
    """
    Takes the labels and the scores and returns the receiver operating characteristic (ROC).
    :param labels: The labels with values of 0 and 1.
    :param prediction_scores: The prediction scores.
    :return: Returns a quadruple containing the false positive rate, the true positive rate, the thresholds the AUROC
    """
    fpr, tpr, thresholds = roc_curve(labels, prediction_scores)
    auroc = auc(fpr, tpr)

    return (fpr, tpr, thresholds, auroc)


def get_roc_plot(fpr, tpr, plot_title=None, figure_kwargs={}):
    # large parts taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    f, ax = plt.subplots(**figure_kwargs)
    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area={auroc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.axis(xmin=0.0, xmax=1.0)
    ax.axis(xmin=0.0, xmax=1.05)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if plot_title is not None:
        ax.title.set_text(plot_title)
    else:
        ax.title.set_text('Receiver Operating Characteristic')
    ax.legend(loc='lower right')

    return f, ax
