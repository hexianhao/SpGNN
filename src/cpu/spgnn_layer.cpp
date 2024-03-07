#include "cpu/spgnn_layer.h"

SpGNNLayer::SpGNNLayer(SparseIndex graph_index,
                       Variable *input_data,
                       Variable *weight_data,
                       SparseIndex weight_index, 
                       int input_dim,
                       int output_dim,
                       int layer)
  : graph_index_(graph_index), 
    input_(input_data),
    weight_(weight_data),
    weight_index_(weight_index),
    input_dim_(input_dim), 
    output_dim_(output_dim),
    layer_(layer)
{
  num_nodes_ = input_->shape.front();
  input_dim_ = input_->shape.back();
  
  // prepare tmp and output, dense data
  tmp_ = new Variable({num_nodes_, input_dim_});
  output_ = new Variable({num_nodes_, output_dim_});
}

SpGNNLayer::~SpGNNLayer()
{
  delete output_;
}

void SpGNNLayer::inference()
{

}

void SpGNNLayer::pruning()
{
  std::cout << "Pruning Not Implemented" << std::endl;
}

void SpGNNLayer::auto_tuning()
{
  TilePlans tile_plans;

  // generate tile_plan
  tile_plan_gen(tile_plans);

  // codegen by jit
  for (auto plan : tile_plans) {
    TiledVariable tiled_g, tiled_w;
    tiled_variable_gen(plan, tiled_g, tiled_w);
    // codegen under this tile var
    codegen(tiled_g, tiled_w, kerns);
    // measurement

  }

  // obtain optimized tile
  // tiled_graph = ...
  // tiled_weight = ...
  // codegen for optimized tile
  codegen(tiled_graph, tiled_weight, kerns);
}

void SpGNNLayer::reorder_sparse_row(SparseIndex &sp_index, std::vector<int> &permutation)
{

}

void SpGNNLayer::reorder_sparse_col(SparseIndex &sp_index, std::vector<int> &permutation)
{
  printf("Not Implemented Yet.\n");
}

void SpGNNLayer::tile_plan_gen(TilePlans &tile_plan, int topk)
{

}

void SpGNNLayer::tiled_variable_gen(TilePlan &plan, 
                                    TiledVariable &tiled_graph,
                                    TiledVariable &tiled_weight)
{

}

void SpGNNLayer::codegen(TiledVariable &tiled_graph, 
                         TiledVariable &tiled_weight,
                         Kernels &kerns)
{
  kerns.clear();
}
