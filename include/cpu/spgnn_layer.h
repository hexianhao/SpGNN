#ifndef CPU_SpGNN_LAYER_H
#define CPU_SpGNN_LAYER_H

#include <functional>
#include "sparse.h"
#include "variable.h"

typedef std::function<void(int*, int*, float*, int, int,
                           float*, int, int, float*)> SxD_Kernel;
typedef std::vector<SxD_Kernel> SxD_Kernels;

typedef std::function<void(float*, int, int, int*, int*, float*,
                           int, int, float*, int, int, int)> DxS_Kernel;
typedef std::vector<DxS_Kernel> DxS_Kernels;

class SpGNNLayer 
{
public:
  SpGNNLayer(Variable *graph,
             Variable *input_data,
             Variable &weight_data,
             int hidden_dim,
             int output_dim,
             int layer);

  void inference();
  void pruning();
  void auto_tuning();   // auto tuning
  
  Variable *get_input() { return input_; }
  Variable *get_output() { return output_; }
  Variable *get_weight() { return &weight_; }
  SparseIndex *get_graph_index() { return graph_->index(); }
  SparseIndex *get_weight_index() { return weight_.index(); }

  static SxD_Kernels SxD_kerns;  // different macro kernels
  static DxS_Kernels DxS_kerns;

private:
  typedef std::vector<int> TileSeq;
  typedef std::tuple<TileSeq, TileSeq, TileSeq, TileSeq> TilePlan;
  typedef std::vector<TilePlan> TilePlans;

  void tile_plan_gen(TilePlans &tile_plan, int topk=5);  // generate tile plan
  void tiled_variable_gen(TilePlan &plan);      // generate tiled variables

  Variable weight_;
  Variable *input_;
  Variable *output_;
  Variable *graph_;

  // our tuning target
  TiledVariable tiled_graph_;
  TiledVariable tiled_weight_;
  TiledVariable tiled_tmp_;
  float *packed_input_;
  // tiled_input_info[i] has 4 fields [type(31), size(30), row(29-15), col(14-0)]. 
  // If type = 0, the tile is dense, otherwise is sparse. 
  // If size = 1, the tile size is over L1 cache, otherwise is below.
  int *tiled_input_info_;
  int tiled_input_rows_, tiled_input_cols_;

  int layer_;
  int input_dim_;
  int hidden_dim_;
  int output_dim_;
  int num_nodes_;
};

#endif