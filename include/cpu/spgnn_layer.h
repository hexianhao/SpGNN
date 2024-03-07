#ifndef CPU_SpGNN_LAYER_H
#define CPU_SpGNN_LAYER_H

#include <functional>
#include "sparse.h"
#include "variable.h"

class SpGNNLayer 
{
public:
  SpGNNLayer(SparseIndex graph_index,
             Variable *input_data,
             Variable *weight_data,
             SparseIndex weight_index,
             int hidden_dim,
             int output_dim,
             int layer);

  void inference();
  void pruning();
  void auto_tuning();   // auto tuning
  // reorder row and column of sparse data
  void set_graph_index(SparseIndex &graph_index);
  void reorder_sparse_row(SparseIndex &sp_index, std::vector<int> &permutation);
  void reorder_sparse_col(SparseIndex &sp_index, std::vector<int> &permutation);

  Variable *get_input() { return input_; }
  Variable *get_output() { return output_; }
  SparseIndex &get_graph_index() { return graph_index_; }
  SparseIndex &get_weight_index() { return weight_index_; }

private:
  typedef std::tuple<int, int, int> TilePlan;
  typedef std::vector<TilePlan> TilePlans;
  typedef std::function<void(float*, float*, float*, int, int, int)> Kernel;
  typedef std::vector<Kernel> Kernels;

  void tile_plan_gen(TilePlans &tile_plan, int topk=5);  // generate tile plan
  void tiled_variable_gen(TilePlan &plan, 
                          TiledVariable &tiled_graph,
                          TiledVariable &tiled_weight);  // generate 
  void codegen(TiledVariable &tiled_graph, 
               TiledVariable &tiled_weight,
               Kernels &kerns);  // codegen by jit

  Variable *weight_;
  Variable *input_;
  Variable *tmp_;
  Variable *output_;
  SparseIndex graph_index_;
  SparseIndex weight_index_;

  // our tuning target
  TiledVariable tiled_graph;
  TiledVariable tiled_weight;
  Kernels kerns;

  int layer_;
  int input_dim_;
  int hidden_dim_;
  int output_dim_;
  int num_nodes_;
};

#endif