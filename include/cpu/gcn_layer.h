#ifndef CPU_GCN_LAYER_H
#define CPU_GCN_LAYER_H

#include "module.h"

typedef std::vector<Module *> ModuleList;
typedef std::vector<Variable *> VariableList;
typedef std::pair<SparseIndex*, SparseIndex*> PairSparseIndex;

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

#endif