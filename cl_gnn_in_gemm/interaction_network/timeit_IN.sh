#!/bin/bash
BATCHSIZE=1
SETUP="import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import random
import yaml
import numpy as np
import torch
from torch.autograd import Variable
from models.interaction_network import InteractionNetwork
from models.graph import Graph, save_graphs, load_graph
def get_graphs(d):
    files = os.listdir(d)
    return [load_graph(d+f) for f in files]
def get_inputs(graphs):
    size = len(graphs)
    O = [Variable(torch.FloatTensor(graphs[i].X))
         for i in range(size)]
    Rs = [Variable(torch.FloatTensor(graphs[i].Ro))
          for i in range(size)]
    Rr = [Variable(torch.FloatTensor(graphs[i].Ri))
          for i in range(size)]
    Ra = [Variable(torch.FloatTensor(graphs[i].a)).unsqueeze(0)
          for i in range(size)]
    y  = [Variable(torch.FloatTensor(graphs[i].y)).unsqueeze(0).t()
          for i in range(size)]
    return O, Rs, Rr, Ra, y
config_name = 'configs/train_IN_LP_5.yaml'
with open(config_name) as f: config = yaml.load(f, yaml.FullLoader)
verbose = config['verbose']
graph_dir = config['graph_dir']
graphs = get_graphs(graph_dir)
object_dim, relation_dim, effect_dim = 3, 1, 1
interaction_network = InteractionNetwork(object_dim, relation_dim, effect_dim)
interaction_network.eval()
batch_size = $BATCHSIZE*10
test_O, test_Rs, test_Rr, test_Ra, test_y = get_inputs(graphs[:batch_size])
"
echo $SETUP
for i in {0..9}
do
    j=$((BATCHSIZE*i))
    k=$((BATCHSIZE*i+BATCHSIZE))
    echo $i
    echo $j
    echo $k
    python -m timeit -s "$SETUP" -n 100 -r 5 "with torch.no_grad(): interaction_network(test_O[$j:$k], test_Rs[$j:$k], test_Rr[$j:$k], test_Ra[$j:$k])"
done

