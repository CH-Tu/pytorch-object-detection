import numpy as np
import matplotlib.pyplot as plt

def collate_fn(batch):
    """Customized collate_fn for dataloader.

    Args:
        batch (list): [dataset[0], dataset[1], ...].
    """
    return tuple(zip(*batch))

def print_loss_and_map(iteration, total_iterations, loss, map):
    """Clean the line and print the loss and mAP.

    Args:
        iteration (int): This iteration.
        total_iterations (int): Total iterations.
        loss (torch.Tensor): Loss value.
        map (torch.Tensor): mAP value.
    """
    string = f'{iteration}/{total_iterations} - loss: {loss:.3f} - mAP: {map:.3f}'
    space = '                                        '
    print(f'\r{space}', end='\r')
    if iteration != total_iterations:
        print(string, end='', flush=True)
    else:
        print(string)

def output_plot(output_name, history, xlabel='Iteration', ylabel='', ylim=None, linewidth=1.5):
    """Output the plot and values of the history.

    Args:
        output_name (str): Output name.
        history (numpy.ndarray): History.
        xlabel (str): X-label of the plot.
        ylabel (str): Y-label of the plot.
        ylim (list): Y-limit of the plot. [bottom, top]
    """
    np.savetxt(f'{output_name}.txt', history)
    plt.plot(history, linewidth=linewidth)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(f'{output_name}.png')
    plt.close()
