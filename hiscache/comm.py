import torch
import hiscache_backend


def all2allnaive(tensor_2d):
    """All-to-all communication.

    Args:
        tensor_2d: Tensors to be communicated, tensor_2d[i][j] is transfered from i to j

    Returns:
        output: Received tensors.
    """
    num_device = len(tensor_2d)
    output = [[None for i in range(num_device)] for j in range(num_device)]
    for i in range(num_device):
        for j in range(num_device):
            if i == j:
                output[j][i] = tensor_2d[i][j]
            else:
                output[j][i] = hiscache_backend.p2psend(tensor_2d[i][j], j)
    return output


# multi-rounded all2all
def all2all(tensor_2d):
    """All-to-all communication.

    Args:
        tensor_2d: Tensors to be communicated, tensor_2d[i][j] is transfered from i to j

    Returns:
        output: Received tensors.
    """
    num_device = len(tensor_2d)
    if num_device == 4:
        schedule_src = [0, 1, 2, 3, 0, 3, 2, 1, 0, 2, 3, 1]
        schedule_dst = [1, 0, 3, 2, 2, 1, 0, 3, 3, 1, 0, 2]
    elif num_device == 2:
        schedule_src = [0, 1]
        schedule_dst = [1, 0]
    elif num_device == 8:
        schedule_src = [
            0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 2, 1, 4, 7, 6, 5, 0, 2, 3, 1, 4, 6,
            7, 5
        ]
        schedule_dst = [
            1, 0, 3, 2, 5, 4, 7, 6, 6, 1, 0, 7, 2, 5, 4, 3, 4, 1, 0, 5, 6, 3,
            2, 7
        ]
        for i in range(num_device):
            assert schedule_src.count(i) == num_device - 1
            assert schedule_dst.count(i) == num_device - 1

    output = [[None for i in range(num_device)] for j in range(num_device)]
    for i in range(len(schedule_dst)):
        src = schedule_src[i]
        dst = schedule_dst[i]
        output[dst][src] = hiscache_backend.p2psend(tensor_2d[src][dst], dst)
    for i in range(num_device):
        output[i][i] = tensor_2d[i][i]
    return output