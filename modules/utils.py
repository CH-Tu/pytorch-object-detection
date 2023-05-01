import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import draw_bounding_boxes

def collate_fn(batch):
    """Customized collate_fn for dataloader.

    Args:
        batch (list): [dataset[0], dataset[1], ...].
    """
    return tuple(zip(*batch))

def print_loss(iteration, total_iterations, loss, loss_name='loss'):
    """Clean the line and print the loss.

    Args:
        iteration (int): This iteration.
        total_iterations (int): Total iterations.
        loss (number): Loss value.
        loss_name (str): Loss name.
    """
    string = f'{iteration}/{total_iterations} - {loss_name}: {loss:.3f}'
    space = '                              '
    print(f'\r{space}', end='\r')
    if iteration != total_iterations:
        print(string, end='', flush=True)
    else:
        print(string)

def output_plot(output_name, history, xlabel='iteration', ylabel='', ylim=None):
    """Output plot and values of the history.

    Args:
        output_name (str): Output name.
        history (numpy.ndarray): History.
        xlabel (str): X-label of the plot.
        ylabel (str): Y-label of the plot.
        ylim (list): Y-limit of the plot. [bottom, top].
    """
    np.savetxt(f'plots/{output_name}.txt', history)
    plt.plot(history)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(f'plots/{output_name}.png')
    plt.close()

def output_pred_image(image_path, image, output_dict, classes, colors, threshold=0.6):
    """Output an image of the detection result.

    Args:
        image_path (str): Path of output image.
        image (torch.Tensor): Input image in (C, H ,W) shape.
        output_dict (dict): Model output.
        classes (list): Class names.
        color (list): Bounding box colors.
        threshold (float): Score threshold.
    """
    for box, label, score in zip(*output_dict.values()):
        if score > threshold:
            image = draw_bounding_boxes(image, box.unsqueeze(0), labels=[classes[label-1]],
                                        colors=[colors[label-1]])
    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(image_path)
