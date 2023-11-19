#include "cpu/gcn_layer.h"

GCNLayer::GCNLayer(PairSparseIndex pair_graph_index,
                   Variable *input_data, 
                   int hidden_dim,
                   int layer,
                   std::vector<op_type_t> ops)
  : pair_graph(pair_graph_index), 
    input(input_data), 
    hidden_dim(hidden_dim), 
    layer(layer)
{
  num_nodes = input->shape.front();
  input_dim = input->shape.back();
  
  variables.push_back(input);
  build_variables(&variables, ops);
  output = variables.back();

  // build inner_graph
  auto inner_graph = pair_graph.first;
  ModuleList* mods = &(pair_mods.first);
  if (inner_graph->indices.size() > 0) {
    build_modules(inner_graph, mods, ops);
  }

  // build outer_graph
  auto outer_graph = pair_graph.second;
  mods = &(pair_mods.second);
  build_modules(outer_graph, mods, ops);
}

void GCNLayer::build_variables(VariableList *variables, std::vector<op_type_t> ops)
{
  for (int i = 0; i < ops.size(); i++) {
    std::vector<int> shape;

    // initialize weights
    if (i > 0) {
      if (i == 1) {
        shape = {input_dim, hidden_dim};
      } else {
        shape = {hidden_dim, hidden_dim};
      }

      variables->emplace_back(shape, false);
    }

    // output of each op
    shape = {num_nodes, hidden_dim};
    variables->emplace_back(shape, false);
  }
}

void GCNLayer::build_modules(SparseIndex *sp, ModuleList *mods, std::vector<op_type_t> ops)
{
  int cur_var = 0;
  int n, m, p; // TODO: setting n, m, p
  for (int i = 0; i < ops.size(); i++) {
    switch (ops[i])
    {
    case MATMUL:
      mods->emplace_back(Matmul(variables[cur_var], 
                                variables[cur_var + 1], 
                                variables[cur_var + 2],
                                n,
                                m,
                                p));
      cur_var += 2;
      break;
    case SPMM:
      mods->emplace_back(SparseMatmul(variables[cur_var], 
                                      variables[cur_var + 1], 
                                      variables[cur_var + 2], 
                                      sp, 
                                      n,
                                      m,
                                      p));
      cur_var += 2;
      break;
    case AGG:
      mods->emplace_back(GraphSum(input, variables[cur_var], sp, hidden_dim));
      cur_var += 1;
      break;
    default:
      break;
    }
  }
}