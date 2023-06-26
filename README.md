# Graph-learning-to-accelarate-MOIP

## Paper
```
Wu, Yaoxin, Wen Song, Zhiguang Cao, Jie Zhang, Abhishek Gupta, and Mingyan Lin. "Graph Learning Assisted Multi-Objective Integer Programming." Advances in Neural Information Processing Systems 35 (2022): 17774-17787.
```

### To generate instances
Run the following to generate KP instances in KP file. (cd python-moiptimiser/tests/examples/zgenerator)
```
for i in {0..49}; do python3 generate_KP.py KP/3KP-100-$i.lp 100 3; done
```

### To solve instances 0...49 in folder KP with 1000s for training data collection

```
for i in {0..49}; do python3 -m moiptimiser ../tests/examples/zgenerator/KP/3KP-100-$i.lp --runtime=1000; done
```

### Training on Knapsack with two-stage gnn

```
nohup python3 branch2learn_global_attention_gat_b/02_train.py -p KP -m "gin1" > XXX.log 2>&1 &
```

### Please refer to the ODA which our method is built on

https://github.com/bayan/python-moiptimiser

### Updating

The repository is being updated ...
