import torch
from hiscache import all2all, all2allnaive, all2all_prealloc, all2allnaive_prealloc
import cxgnncomp as cxgc
import time
import hiscache_backend


def sync(num):
    for i in range(num):
        torch.cuda.synchronize(i)


def test_p2p():
    # tensor_size = [10000000, 64]
    tensor_size = [100000000]
    num_elem = 1
    for item in tensor_size:
        num_elem *= item
    byte_transfered = num_elem * 4
    tensor = torch.randn(tensor_size, device=0)
    tensor_dst = torch.randn(tensor_size, device=1)
    hiscache_backend.p2psend(tensor, tensor_dst, num_elem, 1)
    num_trial = 100
    sync(2)
    t0 = time.time()
    for i in range(num_trial):
        hiscache_backend.p2psend_gentensor(tensor, num_elem, 1)
    torch.cuda.synchronize(1)
    t1 = time.time()
    print(byte_transfered)
    print(f"p2p time: ", (t1 - t0) / num_trial)
    print(f"p2p bandwidth: ", byte_transfered / (t1 - t0) * num_trial / 1e9,
          "GB/s")

    sync(2)
    t0 = time.time()
    for i in range(num_trial):
        hiscache_backend.p2psend(tensor, tensor_dst, num_elem, 1)
    torch.cuda.synchronize(1)
    t1 = time.time()
    print(byte_transfered)
    print(f"p2p time: ", (t1 - t0) / num_trial)
    print(f"p2p bandwidth: ", byte_transfered / (t1 - t0) * num_trial / 1e9,
          "GB/s")


def test_all2all(func):
    num_device = 4
    tensor_size = [1000000, 64]
    num_elem = 1
    for item in tensor_size:
        num_elem *= item
    byte_transfered = num_elem * 4 * (num_device - 1) * num_device
    tensor_list = [[
        torch.randn(tensor_size, device=j) for i in range(num_device)
    ] for j in range(num_device)]
    tensor_list_dst = [[
        torch.empty(tensor_size, device=i) for i in range(num_device)
    ] for j in range(num_device)]
    # output = func(tensor_list)
    num_trial = 10

    sync(num_device)
    t0 = time.time()
    if "prealloc" in str(func):
        for i in range(num_trial):
            output = func(tensor_list, tensor_list_dst)
    else:
        for i in range(num_trial):
            output = func(tensor_list)

    sync(num_device)
    t1 = time.time()
    print(byte_transfered)
    print(f"all2all {str(func)} time: ", (t1 - t0) / num_trial)
    print(f"all2all {str(func)} bandwidth: ",
          byte_transfered / (t1 - t0) * num_trial / 1e9, "GB/s")


# test_p2p()
# test_all2all(all2all)
# test_all2all(all2allnaive)
test_all2all(all2all_prealloc)
# test_all2all(all2allnaive_prealloc)