import cxgnncomp as cxgc
import cxgnncomp_backend
import time
import torch
import cxgnndl
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import logging

log = logging.getLogger(__name__)

base_dir = "/home/huangkz/data/dataset_diskgnn/{}/processed/"


def degree_metric(config, loader):
    ptr = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_ptr_undirected.dat",
                    dtype=np.int64)).cuda()
    deg = ptr[1:] - ptr[:-1]  # in degree
    return deg


def profile_metric(config, loader):
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    metric = torch.zeros(num_node, dtype=torch.int64).cuda()
    num_epochs = 1
    for epoch in range(num_epochs):
        for cur_loader in [loader.train_loader]:
            for batch in cur_loader:
                nodes = batch.sub_to_full
                metric[nodes] += 1
        log.info(f"epoch {epoch} metric {torch.sum(metric > 0)}")
    return metric


def pagerank(config, loader):
    ptr = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_ptr_undirected.dat",
                    dtype=np.int64)).cuda()
    idx = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_idx_undirected.dat",
                    dtype=np.int64)).cuda()
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    score = torch.ones([num_node], device=ptr.device) / num_node
    training_nodes = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "split/time/" +
                    "train_idx.dat",
                    dtype=np.int64)).cuda()
    num_train = training_nodes.shape[0]
    weight = num_node / num_train
    score[training_nodes] = weight / num_node
    deg = ptr[1:] - ptr[:-1]
    deg[deg == 0] = 1
    num_iter = 14 if config.dl.dataset.name == "papers100M" else 25
    # num_iter = 14
    d = 0.95
    for i in range(num_iter):
        score = score / deg
        score += cxgnncomp_backend.spmv(ptr, idx, score, ptr.shape[0] - 1)
        torch.cuda.synchronize()
        score = (1 - d) / num_node + d * score
    return score


def pagerank_metric2(config, loader):
    edge_index = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "edge_index.dat",
                    dtype=np.int64)).view(2, -1).cuda()
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    metric = torch.zeros(num_node, dtype=torch.int64, device=edge_index.device)
    training_nodes = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "split/time/" +
                    "train_idx.dat",
                    dtype=np.int64)).cuda()
    metric[training_nodes] = 1

    log.info("initial training nodes {}".format(torch.sum(metric > 0)))
    for l in range(3):
        metric[edge_index[1][metric[edge_index[0]] > 0]] += 1
        metric[edge_index[0][metric[edge_index[1]] > 0]] += 1
        log.info("layer {} training nodes {}".format(l, torch.sum(metric > 0)))


def analysis_training_nodes(config, loader):
    ptr = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_ptr_undirected.dat",
                    dtype=np.int64))
    idx = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) +
                    "csr_idx_undirected.dat",
                    dtype=np.int64))
    num_node = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    metric = torch.zeros(num_node, dtype=torch.int64, device=ptr.device)
    deg = ptr[1:] - ptr[:-1]  # in degree
    training_nodes = torch.from_numpy(
        np.fromfile(base_dir.format(config.dl.dataset.name) + "split/time/" +
                    "train_idx.dat",
                    dtype=np.int64))
    metric[training_nodes] = 1
    # Analysis how many nodes are needed in training for each layer, starting from the training nodes provided by dataset
    for l in range(3):
        for i in range(training_nodes.shape[0]):
            metric[idx[ptr[training_nodes[i]]:ptr[training_nodes[i] + 1]]] += 1
        training_nodes = (metric > 0).nonzero().squeeze()
        log.info("layer {} nodes needed in training {}".format(
            l, training_nodes.shape))


def get_cache_map(metric, cache_num):
    sorted, indices = torch.sort(metric, descending=True)
    import matplotlib.pyplot as plt
    plt.plot(sorted.cpu().numpy())
    plt.savefig("sorted.png")
    num_node = metric.shape[0]
    thres = metric[indices[cache_num]]
    print("thres", thres)
    cache_status = metric >= thres
    print("num cached", torch.sum(cache_status))
    log.info("rate of metric > 0 {}".format(
        torch.sum(metric > 0) / metric.shape[0]))
    log.info("percentage of cached metric {}".format(
        torch.sum(metric[cache_status]) / torch.sum(metric)))
    return cache_status


def test_cache(cache_status, loader):
    num_epochs = 3
    num_visit = 0
    num_hit = 0
    for epoch in range(num_epochs):
        for cur_loader in [
                # loader.train_loader, loader.val_loader, loader.test_loader
                loader.train_loader
        ]:
            for batch in cur_loader:
                nodes = batch.sub_to_full
                num_visit += nodes.shape[0]
                num_hit += int(torch.sum(cache_status[nodes]))
        log.info("raw feature cache hit rate {} {} {}".format(
            num_hit / num_visit, num_hit, num_visit))
        # print(nodes.shape)


    # print("hit rate", num_hit / num_visit, num_hit, num_visit)
def invalidate(cache, useful, sub2full, num1, num2, ptr, idx):
    status = cache[sub2full[:num1]]
    status = status.cpu()
    useful2 = torch.zeros(num2, dtype=torch.bool)
    for i in range(num1):
        if not status[i] and useful[i]:
            useful2[idx[ptr[i]:ptr[i + 1]]] = True
            # for j in range(ptr[i], ptr[i+1]):
            #     useful2[idx[j]] = True
    return useful2


# Can cache multiple layers of embeddings
def test_embedding_cache(cache_status, loader, config):
    num_epochs = 3
    num_visit = 0
    num_hit = 0
    num_layer = 3
    num_node_total = int(
        open(base_dir.format(config.dl.dataset.name) + "num_nodes.txt").read())
    embedding_cache_status = [
        torch.zeros([num_node_total], dtype=torch.int32, device="cuda")
        for _ in range(num_layer)
    ]
    # cached_embedding_ids = [torch.Tensor([], dtype=torch.int64, device="cuda") for _ in range(num_layer)]
    cached_embedding_ids = [[] for _ in range(num_layer)]
    glb_iter = 0
    staleness = 200
    for epoch in range(num_epochs):
        for cur_loader in [loader.train_loader]:
            for batch in cur_loader:
                glb_iter += 1
                num_layer = batch.num_node_in_layer.shape[0] - 1
                usefuls = [
                    torch.ones([batch.num_node_in_layer[0]], dtype=torch.bool)
                ]
                cpu_ptr = batch.ptr.cpu()
                cpu_idx = batch.idx.cpu()
                for k in range(num_layer):
                    usefuls.append(
                        invalidate(embedding_cache_status[k], usefuls[k],
                                   batch.sub_to_full,
                                   batch.num_node_in_layer[k],
                                   batch.num_node_in_layer[k + 1], cpu_ptr,
                                   cpu_idx))
                    # log.info("layer {} num_still_in_use {} num_node {}".format(
                    #     k, int(torch.sum(usefuls[-1])), usefuls[-1].shape[0]))
                useful = usefuls[-1]
                nodes = batch.sub_to_full
                num_visit = nodes.shape[0]
                num_hit = nodes.shape[0] - int(torch.sum(useful))
                # update cache
                for k in range(num_layer):
                    if k in [0, 2]:
                        continue  # only cache layer 1
                    num_node = batch.num_node_in_layer[k]
                    score = torch.randn([num_node])
                    rate = 0.9
                    thres = torch.quantile(score, rate)
                    to_cache = batch.sub_to_full[:num_node][torch.logical_and(
                        score < thres, usefuls[k])]
                    cached_embedding_ids[k].append(to_cache)
                    embedding_cache_status[k][to_cache] = glb_iter
                    if glb_iter >= staleness:
                        embedding_cache_status[k][cached_embedding_ids[k][0]][
                            glb_iter - embedding_cache_status[k][
                                cached_embedding_ids[k][0]] >= staleness] = 0
                        cached_embedding_ids[k] = cached_embedding_ids[k][1:]
                if glb_iter % 10 == 0:
                    log.info("glb_iter {}".format(glb_iter))
                    log.info("embedding hit rate {} {} {}".format(
                        num_hit / num_visit, num_hit, num_visit))
                    num_cached = torch.sum(
                        embedding_cache_status[1] != 0) + torch.sum(
                            embedding_cache_status[2] != 0)
                    log.info("num cached embeddings {} {}".format(
                        num_cached, num_cached / num_node_total))


@hydra.main(version_base=None,
            config_path="../../CxGNN-DL/configs",
            config_name="config")
def main(config: DictConfig):
    s = OmegaConf.to_yaml(config)
    log.info(s)
    new_file_name = "new_config.yaml"
    s_dl = OmegaConf.to_yaml(config.dl)
    with open(new_file_name, 'w') as f:
        s = s_dl.replace("-", "  -")  # fix cpp yaml interprete
        f.write(s)

    loader = cxgnndl.get_loader(config.dl)
    dataset = config.dl.dataset.name

    # pagerank_metric2(config, loader)
    if dataset == "papers100M":
        cache_size = 5e9
    elif dataset == "mag240M":
        cache_size = 2e10
    else:
        cache_size = 5e9
    feat_len = config.dl.dataset.feature_dim
    # feat_len = 128 if dataset == "papers100M" else 768
    feature_cache_num = int(cache_size // (feat_len * 4))
    embedding_cache_num = int(cache_size // (256 * 4))

    for func in [pagerank, degree_metric]:
        print(str(func))
        metric = func(config, loader)
        cache_status = get_cache_map(metric, cache_num=feature_cache_num)
        test_cache(cache_status, loader)

    # test_embedding_cache(cache_status, loader, config)


if __name__ == '__main__':
    main()
