from pathlib                    import Path
import matplotlib.pyplot        as plt

import libs.dirs                as dirs

def plot_outputs_histogram(normalized_outputs,
                           labels=None,
                           lower_thresh=None,
                           upper_thresh=None,
                           title="Outputs Histogram",
                           show=True,
                           save_path=None,
                           save_formats=[".png", ".pdf"]):
    fig = plt.figure(figsize=(8, 4))
    # plt.subplots_adjust(left=0.09, bottom=0.09, right=0.95, top=0.80,
    #                     wspace=None, hspace=None)

    if labels is not None:
        posOutputs = normalized_outputs[labels == 0]
        negOutputs = normalized_outputs[labels == 1]

        plt.hist(posOutputs, bins=100, label="Positive Examples")
        plt.hist(negOutputs, bins=100, label="Negative Examples")
    else:
        plt.hist(normalized_outputs, bins=100, label="Positive Examples")
    
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
    if show:
        plt.show()
    return fig
