#ifndef CPU_GCN_H
#define CPU_GCN_H

#include <vector>
#include "module.h"
#include "variable.h"
#include "optim.h"

typedef std::vector<Module *> ModuleList;
typedef std::vector<Variable *> VariableList;
typedef std::pair<SparseIndex*, SparseIndex*> PairSparseIndex;

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

class GCNLayer 
{
public:
  GCNLayer(PairSparseIndex pair_graph_index, 
           Variable *input_data, 
           int hidden_dim,
           int layer,
           std::vector<op_type_t> ops);

  void forward(bool);
  void backward();

  Variable *get_input() { return input; }
  Variable *get_output() { return output; }

private:
  void build_variables(VariableList *variables, std::vector<op_type_t> ops);
  void build_modules(SparseIndex *sp, ModuleList *mods, std::vector<op_type_t> ops);

  std::pair<ModuleList, ModuleList> pair_mods;
  VariableList variables;
  Variable *input;
  Variable *output;
  PairSparseIndex pair_graph;

  int layer;
  int input_dim;
  int hidden_dim;
  int num_nodes;
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
  GCNParams params;   // gcn structure data
  GCNData *input;     // gcn input data
  float loss;
  Adam optimizer;
};

#endif