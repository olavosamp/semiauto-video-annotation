import os
import matplotlib as mpl
# Use non interactive backend if not running on Windows, meaning it's on the remote server
if os.name != "nt":
    mpl.use("Agg")

import numpy                    as np
import matplotlib.pyplot        as plt
import seaborn                  as sns
from pathlib                    import Path

import libs.dirs                as dirs
import libs.commons             as commons

def set_mpl_fig_options(figsize=commons.MPL_FIG_SIZE_SMALL):
    return plt.figure(figsize=figsize, tight_layout=True)


def plot_confusion_matrix(conf_mat, labels=[], title=None, normalize=True,
                         show=True, save_path="./confusion_matrix.jpg"):
    '''
        conf_mat: array of floats or ints
        Square array that configures a confusion matrix. The true labels are assumed to be on the lines axis
        and the predicted labels, on the columns axis.

        labels: list
        List of class labels. Label list must be of lenght equal to the number of classes of the confusion
        matrix. Element i of list is the label of class in line i of the confusion matrix.

    '''
    fig = set_mpl_fig_options(commons.MPL_FIG_SIZE_SMALL)

    numClasses = np.shape(conf_mat)[0]
    conf_mat = np.array(conf_mat, dtype=np.float32)

    if normalize:
        # Normalize confusion matrix line-wise
        for line in range(numClasses):
            classSum = np.sum(conf_mat[line, :])
            conf_mat[line, :] = np.divide( conf_mat[line, :], classSum)

    # If labels list match number of classes, use it as class labels
    if len(labels) == numClasses:
        xLabels = labels
        yLabels = labels
    else:
        xLabels = False
        yLabels = False

    sns.heatmap(conf_mat, annot=True, cbar=True, square=True, vmin=0., vmax=1., fmt='.2f',
                xticklabels=xLabels, yticklabels=yLabels, cmap='cividis')

    ax = plt.gca()
    plt.setp(ax.get_yticklabels(), va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if title is not None:
        plt.title(title)
    else:
        plt.title("Confusion Matrix")

    if save_path is not None:
        # Save figure to given path
        save_path = Path(save_path)
        dirs.create_folder(save_path.parent)

        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()


def plot_model_history(data, data_labels=[], xlabel="", ylabel="", title="Model History",
                       save_path=None, show=False):
    '''
        Arguments:
            data: list
            One or more datasets that are lists of y values. If more than one, all datasets
            must be of same length.

            data_labels: string or list of strings
            Data label for each data set given.
    '''
    assert isinstance(data, list), "data argument must be a list of values or a list of datasets."
    fig = plt.figure(figsize=commons.MPL_FIG_SIZE)
    
    # User passed several datasets
    if hasattr(data[0], "__len__"):
        dataLen   = len(data)
        labelsLen = len(data_labels)

        assert dataLen == labelsLen, "You must pass one label for each dataset"
        if labelsLen < dataLen:
            # Pad labels
            data_labels += [""]*(dataLen - labelsLen)
        else:
            # Trim labels
            data_labels = data_labels[:dataLen]

        x = range(len(data[0]))
        for y, label in zip(data, data_labels):
            # Plot dataset with its corresponding label
            plt.plot(x, y, '.-', label=label)
    # User passed only one dataset
    else:
        x = range(len(data))
        plt.plot(x, data, '.-', label=data_labels)

    plt.legend()
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    if save_path is not None:
        fig.savefig(save_path, orientation='portrait', bbox_inches='tight')
    if show and mpl.get_backend() != "agg":
        plt.show()
    return fig


def plot_outputs_histogram(normalized_outputs,
                           labels=None,
                           lower_thresh=None,
                           upper_thresh=None,
                           title="Outputs Histogram",
                           show=True,
                           log=False,
                           save_path=None,
                           save_formats=[".png", ".pdf"]):
    fig = plt.figure(figsize=commons.MPL_FIG_SIZE_SMALL)
    # plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    #                     wspace=None, hspace=None)

    if labels is not None:
        posOutputs = normalized_outputs[labels == 0]
        negOutputs = normalized_outputs[labels == 1]

        plt.hist(posOutputs, bins=100, label="Positive Examples", log=log)
        plt.hist(negOutputs, bins=100, label="Negative Examples", log=log)
    else:
        plt.hist(normalized_outputs, bins=100, label="Positive Examples", log=log)
    
    if lower_thresh is not None and upper_thresh is not None:
        plt.gca().axvline(lower_thresh, 0., 1., color='b', label="Lower Thresh")
        plt.gca().axvline(upper_thresh, 0., 1., color='r', label="Upper Thresh")

    plt.tight_layout(pad=2.)
    plt.xlim(0., 1.)
    plt.title(title)
    plt.legend()
    plt.xlabel("Normalized Score")
    plt.ylabel("Number of Examples")
    
    if save_path is not None:
        save_path = Path(save_path)
        dirs.create_folder(save_path.parent)

        # Save with desired format, and additional formats specified in save_formats
        plt.savefig(save_path)
        for ext in save_formats:
            if ext[0] == '.':
                plt.savefig(save_path.with_suffix(ext))
    if show and mpl.get_backend() != "agg":
        plt.show()
    return fig

## Dataset Image processing
def show_inputs(inputs, labels):
    '''
        Function to visualize dataset inputs
    '''
    for i in range(len(inputs)):
        print(np.shape(inputs.cpu().numpy()[i,:,:,:]))
        img = np.transpose(inputs.cpu().numpy()[i,:,:,:], (1, 2, 0))
        print(np.shape(img))
        print(labels.size())
        print("Label: ", labels[i])
        plt.imshow(img)
        plt.title("Label: {}".format(labels[i]))
        plt.show()


def show_image(image, title_string=None):
    '''
        Show Pillow or Pyplot input image.
    '''
    print("Title: ", title_string)
    plt.imshow(image)
    if title_string:
        plt.title(title_string)
    plt.show()
