# Historical Cache

What is this?
-------------
A high-performance cache system and algorithm for GNN training, with the use of historical embeddings and features.


## How to use

### 1. Install

```bash
python setup.py install
```

### 2. Dataset preparation
```
# download the dataset from ogb and run the training
# first time processing might take a long time
```

### 3. How to run

```bash
# run single GPU training with papers100M
python example/main.py dl/dataset=papers100M dl.type=cxg train.type=cxg dl.loading.feat_mode=history_uvm dl.sampler.train.batch_size=1000  train.train.num_epochs=30 dl.num_device=1
```

```bash
# run four GPU training with papers100M
python example/main.py dl/dataset=papers100M dl.type=cxg train.type=cxg dl.loading.feat_mode=history_uvm dl.sampler.train.batch_size=1000  train.train.num_epochs=30 dl.num_device=4
```

```bash
# training on heterogeneous graph mag240M 
python example/main.py train/model=rgcn dl/dataset=rmag240M dl.type=cxg train.type=cxg dl.loading.feat_mode=history_uvm dl.sampler.train.batch_size=1000  train.train.num_epochs=30 dl.num_device=1
```

### 4. Use hiscache in your own code

Before

```python

loader = init_loader() # e.g., DGL loader
for batch in loader:
    blocks = self.to_dgl_block(batch)
    if self.skip:
        continue
    batch_inputs = batch.x
    batch_labels = batch.y
    train(batch_inputs, batch_labels, blocks)
```

After 

```python
from hiscache import HistoryCache

loader = init_loader() # e.g., DGL loader
cache = HistoryCache(config=config)

for batch in loader:
    batch = cache.lookup_and_load(batch)
    train(batch)
```

## File structure

```
.
|-- example
|   `-- main.py
|-- hiscache
|   |-- comm.py
|   |-- history_cache.py
|   |-- history_model.py
|   |-- history_table.py
|   |-- history_trainer.py
|   |-- __init__.py
|   |-- multi_gpu_history_trainer.py
|   |-- README.md
|   `-- util.py
`-- src
    `-- cpp
        |-- include
        |   |-- comm.h
        |   |-- common.h
        |   |-- grad_check.h
        |   `-- history_aggr.h
        |-- python_bindings
        |   `-- binding.cpp
        `-- src
            |-- comm.cu
            |-- grad_check.cu
            `-- history_aggr.cu
```