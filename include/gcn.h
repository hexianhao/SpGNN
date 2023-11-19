#ifndef CPU_GCN_H
#define CPU_GCN_H

#include <vector>
#include "cpu/module.h"
#include "cpu/optim.h"
#include "cpu/gcn_layer.h"

struct GCNParams 
{
  int num_nodes, input_dim, hidden_dim, output_dim, num_layers;
  float dropout, learning_rate, weight_decay;
  int epochs, early_stopping;
  static GCNParams get_default();
};

struct GCNData
{
  SparseIndex feature_index, graph;
  std::vector<int> split;
  std::vector<int> label;
  std::vector<float> feature_value;
};

class GCN
{
public:
  GCN(GCNParams params, GCNData *data);

  GCNParams get_gcn_params() { return params; }

  void train();       // GCN training

  void inference();   // GCN inference

private:
  void set_input();

  std::vector<GCNLayer> layers;
  GCNParams params;         // gcn structure data
  GCNData *data;            // gcn input data
  Variable input_feat;     // input feature

  float loss;
  Adam optimizer;
};

#endif