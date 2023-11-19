#include "gcn.h"

GCN::GCN(GCNParams params, GCNData *data)
  : params(params), 
    data(data), 
    input_feat({params.num_nodes, params.input_dim}, false)
{
  int num_layers = params.num_layers;
  layers.reserve(num_layers);

  // TODO: Graph partition
  PairSparseIndex pair_graph;

  // TODO: build gcn structure according to config file.
  // we pre-define gcn layers
  for (int i = 0; i < num_layers; i++) {
    layers.push_back(GCNLayer(pair_graph, &input_feat, params.hidden_dim, i, {AGG, MATMUL, RELU}));
  }

  // set input features
  set_input();
}

void GCN::set_input()
{

}

void GCN::train()
{

}

void GCN::inference()
{
  
}