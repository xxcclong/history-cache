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
                output[j][i] = hiscache_backend.p2psend_gentensor(
                    tensor_2d[i][j], torch.numel(tensor_2d[i][j]), j)
    return output


def all2allnaive_prealloc(tensor_2d_src, tensor_2d_dst):
    """All-to-all communication.

    Args:
        tensor_2d: Tensors to be communicated, tensor_2d[i][j] is transfered from i to j

    Returns:
        output: Received tensors.
    """
    num_device = len(tensor_2d_src)
    for i in range(num_device):
        for j in range(num_device):
            if i == j:
                tensor_2d_dst[j][i] = tensor_2d_src[i][j]
            else:
                hiscache_backend.p2psend(tensor_2d_src[i][j],
                                         tensor_2d_dst[i][j],
                                         torch.numel(tensor_2d_src[i][j]), j)
    return tensor_2d_dst


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
        output[dst][src] = hiscache_backend.p2psend_gentensor(
            tensor_2d[src][dst], torch.numel(tensor_2d[src][dst]), dst)
    for i in range(num_device):
        output[i][i] = tensor_2d[i][i]
    return output


# multi-rounded all2all
def all2all_prealloc(tensor_2d_src, tensor_2d_dst):
    """All-to-all communication.

    Args:
        tensor_2d: Tensors to be communicated, tensor_2d[i][j] is transfered from i to j

    Returns:
        output: Received tensors.
    """
    num_device = len(tensor_2d_src)
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

    # schedule_src = [0, 1]
    # schedule_dst = [1, 0]

    for i in range(len(schedule_dst)):
        src = schedule_src[i]
        dst = schedule_dst[i]
        hiscache_backend.p2psend(tensor_2d_src[src][dst],
                                 tensor_2d_dst[src][dst],
                                 torch.numel(tensor_2d_src[src][dst]), dst)
    for i in range(num_device):
        tensor_2d_dst[i][i] = tensor_2d_src[i][i]
    return tensor_2d_dst


def prepare_transfer(batches, masks, feats):
    num_device = len(batches)
    masked_sub2full = [
        torch.masked_select(batches[i].sub_to_fulls[i], masks[i])
        for i in range(num_device)
    ]  # [num_device]
    tensor_list = [[] for i in range(num_device)]
    for src in range(num_device):
        for dst in range(num_device):
            tensor_list[src].append(
                torch.index_select(feats[src], 0,
                                   masked_sub2full[dst].to(src)))
    return tensor_list


def post_transfer(batches, masks, tensor_2d_dst):
    num_device = len(batches)
    for i in range(num_device):
        comm_feat = torch.cat(tensor_2d_dst[i], dim=1)
        batches[i].x[masks[i]] = comm_feat