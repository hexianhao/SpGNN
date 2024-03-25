#include "cpu/spgnn_layer.h"

SpGNNLayer::SpGNNLayer(Variable *graph,
                       Variable *input_data,
                       Variable &weight_data,
                       int input_dim,
                       int output_dim,
                       int layer)
  : graph_(graph), 
    input_(input_data),
    weight_(std::move(weight_data)),
    input_dim_(input_dim), 
    output_dim_(output_dim),
    layer_(layer)
{
  num_nodes_ = input_->row_num();
  input_dim_ = input_->column_num();
  
  // prepare tmp and output, dense data
  output_ = new Variable({num_nodes_, output_dim_}, "dense");
}

SpGNNLayer::~SpGNNLayer()
{
  delete packed_input_;
  delete tiled_input_info_;
  delete output_;
}

void SpGNNLayer::inference()
{
  auto tiled_graph_info = tiled_graph_.tile_info();
  auto tiled_graph_indptr = tiled_graph_.indptr();
  auto tiled_graph_indices = tiled_graph_.indices();
  auto tiled_graph_rows = tiled_graph_.row_tile_num();
  auto tiled_graph_cols = tiled_graph_.column_tile_num();
  auto graph_data = tiled_graph_.data();

  auto input_data = input_->data();
  auto tmp_data = tiled_tmp_.data();
  auto tiled_tmp_info = tiled_tmp_.tile_info();
  auto output_ld = output_->column_num();

  auto tiled_weight_info = tiled_weight_.tile_info();
  auto tiled_weight_indptr = tiled_weight_.indptr();
  auto tiled_weight_indices = tiled_weight_.indices();
  auto weight_data = tiled_weight_.data();
  auto tiled_weight_cols = tiled_weight_.column_tile_num();
  // we divide tiled_off into 4 parts
  // [input row off, input col off, output row off, output col off]
  // [63-44, 43-32, 31-12, 11-0]
  uint32_t tiled_off;
  uint64_t tile_nums;
  uint16_t idx;
  int graph_meta, graph_scope;
  int input_meta, tmp_meta, tmp_scope;
  int weight_meta, weight_scope;
  float *input_ptr;
  uint32_t packed_off;

  for (int i = 0; i < tiled_graph_rows; i++) {
    // reset tiled_off
    tiled_off = 0;
    for (int j = 0; j < tiled_input_cols_; j++) {
      for (int k = 0; k < tiled_graph_cols; k++) {
        graph_meta = tiled_graph_info[ROW_MAJOR_OFF(i, k, tiled_graph_cols) << 1];
        graph_scope = tiled_graph_info[ROW_MAJOR_OFF(i, k, tiled_graph_cols) << 1 | 1];
        input_meta = tiled_input_info_[COL_MAJOR_OFF(k, j, tiled_input_rows_)];
        tmp_scope = tiled_tmp_info[ROW_MAJOR_OFF(i, j, tiled_input_rows_) << 1 | 1];
        if (i == 0) {
          // pack input_data
          for (int m = 0; m < (input_meta >> 14 & 65535); m++) {
            for (int n = 0; n < (input_meta & 65535); n++) {
              input_ptr = packed_input_ + packed_off;
              input_ptr[COL_MAJOR_OFF(m, n, input_meta >> 14 & 65535)] = 
                input_data[COL_MAJOR_OFF((tiled_off >> 44) + m, ((tiled_off >> 32) & 4095) + n, num_nodes_)];
            }
          }
        }
        packed_off += (input_meta >> 14 & 65535) * (input_meta & 65535);
        
        idx = (graph_meta >> 30) << 2 | (input_meta >> 30);
        /*
          kernel(int *g_indptr, int *g_indices, float *g_data, int pos, int g_rows,
                 float *input, int i_rows, int i_cols, 
                 float *res);
        */
        SxD_kerns[idx](tiled_graph_indptr, tiled_graph_indices, graph_data, graph_scope, (graph_meta >> 14 & 65535),
                       packed_input_ + packed_off, (input_meta >> 14 & 65535), (input_meta & 65535),
                       tiled_tmp_.data() + tmp_scope);

        tiled_off += (input_meta >> 14 & 65535) << 44;
      }
      tiled_off &= (1 << 44) - 1;     // input_row_off = 0
      tiled_off += (input_meta & 65535) << 32;   // input_col_off += tiled_input_col

      for (int k = 0; k < tiled_weight_cols; k++) {
        weight_meta = tiled_weight_info[ROW_MAJOR_OFF(j, k, tiled_weight_cols) << 1];
        weight_scope = tiled_weight_info[ROW_MAJOR_OFF(j, k, tiled_weight_cols) << 1 | 1];
        tmp_meta = tiled_tmp_info[ROW_MAJOR_OFF(i, j, tiled_input_rows_) << 1];
        tmp_scope = tiled_tmp_info[ROW_MAJOR_OFF(i, j, tiled_input_rows_) << 1 | 1];
        idx = (tmp_meta >> 30) << 2 | (weight_meta >> 30);
        /*
          kernel(float *input, int i_rows, int i_cols,
                 int *w_indptr, int *w_indices, float *data, int pos,
                 int w_cols, float *res, int r_row, int r_col, int ld);
        */
        DxS_kerns[idx](tmp_data + tmp_scope, (tmp_meta >> 14 & 65535), (tmp_meta & 65535), tiled_weight_indptr, 
                       tiled_weight_indices, weight_data, weight_scope, (weight_meta & 65535), output_->data(), 
                       (tiled_off >> 12) & 65535, tiled_off & 4095, output_ld);
        tiled_off += weight_meta & 65535;       // output_col_of += tiled_output_col
      }
      tiled_off = (tiled_off >> 12) << 12;      // outut_col_off = 0
      tiled_off += ((graph_meta >> 14) & 65535) << 12;    // output_row_off += tiled_graph_row
    }
  }
}

void SpGNNLayer::pruning()
{
  std::cout << "Pruning Not Implemented" << std::endl;
}

void SpGNNLayer::auto_tuning()
{
  TilePlans tile_plans;
  // goal of our tuning
  TiledVariable optim_tiled_g;
  TiledVariable optim_tiled_w;
  TiledVariable optim_tiled_t;

  // generate tile_plan
  tile_plan_gen(tile_plans);

  // codegen by jit
  for (auto plan : tile_plans) {
    tiled_variable_gen(plan);
    inference();
  }

}

void SpGNNLayer::tile_plan_gen(TilePlans &tile_plan, int topk)
{

}

void SpGNNLayer::tiled_variable_gen(TilePlan &plan)
{
  TileSeq ts0 = std::get<0>(plan);
  TileSeq ts1 = std::get<1>(plan);
  TileSeq ts2 = std::get<2>(plan);
  TileSeq ts3 = std::get<3>(plan);

  // tiled_graph_ = TiledVariable(nullptr, graph_index_, ts0, ts1, true, "csr");
  // tiled_input_ = TiledVariable(nullptr, input_->index(), ts1, ts2, false, "csc");
  // tiled_weight_ = TiledVariable(weight_.data(), weight_.index(), ts2, ts3, true, "csr");
}