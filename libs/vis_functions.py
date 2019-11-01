import matplotlib as mlp
import os
# Use non interactive backend if not running on Windows, what means it's on the remote server
if os.name != "nt":
    mlp.use("Agg")

from pathlib                    import Path
import matplotlib.pyplot        as plt

import libs.dirs                as dirs


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
    fig = plt.figure(figsize=(24, 18))
    
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
    if show and mlp.get_backend() != "agg":
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
    fig = plt.figure(figsize=(8, 4))
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
    if show and mlp.get_backend() != "agg":
        plt.show()
    return fig
