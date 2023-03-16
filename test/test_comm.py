import torch
from hiscache import all2all, all2allnaive
import cxgnncomp as cxgc
import time


def sync(num):
    for i in range(num):
        torch.cuda.synchronize(i)


def test_all2all(func):
    num_device = 4
    tensor_size = [10000000, 64]
    num_elem = 1
    for item in tensor_size:
        num_elem *= item
    byte_transfered = num_elem * 4 * (num_device - 1) * num_device
    tensor_list = [[
        torch.randn(tensor_size, device=j) for i in range(num_device)
    ] for j in range(num_device)]
    output = func(tensor_list)
    num_trial = 10
    sync(num_device)
    t0 = time.time()
    for i in range(num_trial):
        output = func(tensor_list)
    sync(num_device)
    t1 = time.time()
    print(byte_transfered)
    print(f"all2all {str(func)} time: ", (t1 - t0) / num_trial)
    print(f"all2all {str(func)} bandwidth: ",
          byte_transfered / (t1 - t0) / num_trial / 1e9, "GB/s")


test_all2all(all2all)
test_all2all(all2allnaive)
test_all2all(all2all)
test_all2all(all2allnaive)