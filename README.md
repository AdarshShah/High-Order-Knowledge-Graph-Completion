# Inductive Knowledge Hypergraph Completion

#### Datasets

The datasets are hypergraphs with categorical hyper edges. They are provided [here](https://www.cs.cornell.edu/~arb/data/) by Arthur R. Benson.

#### Training

The following shows the command to perform training on any of the datasets present in `./datasets`.

```
python train.py --dataset [Datasetname] --num_classes [Number of classes] --batch_size [ ] --iter [ number of samples ] --max_dim [ maximum dimension of simplex ] --dim [ particular simplex dimension to infere on] --gpu [GPU device number] --disable_cuda [Boolean]
```

For example :

```
python train.py --dataset cat_edge_cooking --num_classes 20 --iter 10000 --batch_size 32
```

The above will:

* generate 10000 positive and negative samples
* initialize Graph Attention based and Simplicial Message Passing based (our approach) models
* train them for 1 epoch i.e. 10000 iterations
* with batch size of 32
* save models in `./datasets/cat_edge_cooking/models/*`
* save log file as `./datasets/cat_edge_cooking/log_train.txt`
* generate tensorboad log files in `./datasets/cat_edge_cooking/logs/*` (Loss curves)

#### Testing

The following with test pretrained models generated from `train.py`.

```
python test.py --dataset [Datasetname] --num_classes [Number of classes] --batch_size [ ] --iter [ number of samples ] --max_dim [ maximum dimension of simplex ] --dim [ particular simplex dimension to infere on] --gpu [GPU device number] --disable_cuda [Boolean]
```

For example:

```
python test.py --dataset cat_edge_cooking --num_classes 20 --iter 10000 --batch_size 32
```

The above will:

* generate 10000 positive and negative samples
* load pretrained models from `./datasets/cat_edge_cooking/models/*`
* generate AUC and AUC_PR for 4 settings (Union, Intersection, GAT, Simplicial MPSN)
* save log file as `./datasets/cat_edge_cooking/log_test.txt` with the results.

#### Other Baselines

The baselines for GRAIL is present in the ./`grail/datasets/` folder within the log files.
