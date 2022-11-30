# Graph-learning-to-accelarate-MOIP



### To generate instances 0...49 in folder KP-6-100 by solving each with 1000s

```
for i in {0..49}; do python3 -m moiptimiser ../tests/examples/zgenerator/KP_6_100/6KP-100-$i.lp --runtime=1000; done
```

### Training on Knapsack with two-stage gnn

```
nohup python3 branch2learn_global_attention_gat_b/02_train.py -p KP -m "gin1" > XXX.log 2>&1 &
```

### Please refer to the ODA which our method is built on

https://github.com/bayan/python-moiptimiser

### Updating

The repository is being updated ...
