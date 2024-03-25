#include "spgnn.h"
#include <string.h>
#include <set>
#include <deque>
#include <math.h>

SpGNN::SpGNN(SpGNNParams params, SpGNNData *data)
  : params_(params), 
    data_(data), 
    input_feat_({params.num_nodes, params.input_dim}, false)
{
  int num_layers = params.num_layers;
  layers_.reserve(num_layers);

  // TODO: Graph partition
  SparseIndex graph;

  // TODO: build SpGNN structure according to config file.
  // we pre-define SpGNN layers
  for (int i = 0; i < num_layers; i++) {
    
  }

  // set input features
  set_input();
}

void SpGNN::set_input()
{

}

void SpGNN::pruning()
{
  printf("Not implemented yet\n");
}

void SpGNN::inference()
{

}

void SpGNN::auto_tuning()
{
  // data layout transform
  data_layout_transform();
  // kernel generation
  kernels_gen(SpGNNLayer::kerns);
  // each layer's auto tuning
  for (auto &layer : layers_) {
    layer.auto_tuning();
  }
}

void SpGNN::data_layout_transform()
{
  // reorder graph
  sparse_graph_reorder();
  // reorder weight
  sparse_weight_reorder(10);
}

void SpGNN::sparse_graph_reorder()
{
  int col = params_.num_nodes;
  int row = params_.num_nodes;
  AdjElem *adj_mat = (AdjElem *)malloc(sizeof(AdjElem) * row * col);
  memset(adj_mat, 0, sizeof(AdjElem) * row * col);
  // fill in adj_mat
  for (int i = 0; i < row; i++) {
    for (int j = graph_.indptr[i]; j < graph_.indptr[i + 1]; j++) {
      adj_mat[ROW_MAJOR_OFF(i, j, col)] = AdjValue::CONN;
    }
  }

  // reorder graph
  // first in row
  // choose the most nnz row
  std::vector<int> perm;
  for (int i = 0; i < row; i++) {
    perm.emplace_back(i);
  }
  typedef std::set<int> NNZ;
  NNZ *row_nnz;
  NNZ *row_nnz = (NNZ *)malloc(sizeof(NNZ) * row);
  int xchg_row = 0, max_nnz = 0;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (adj_mat[ROW_MAJOR_OFF(i, j, col)] == AdjValue::CONN) {
        row_nnz[i].insert(j);
      }
    }
    if (max_nnz < row_nnz[i].size()) {
      max_nnz = row_nnz[i].size();
      xchg_row = i;
    }
  }
  
  for (int i = 0; i < row; i++) {
    if (i > 0) {
      max_nnz = 0;
      for (int j = i + 1; j < row; j++) {
        NNZ intersection;
        std::set_intersection(row_nnz[perm[i]].begin(), row_nnz[perm[i]].end(),
                              row_nnz[perm[j]].begin(), row_nnz[perm[j]].end(),
                              std::inserter(intersection, intersection.begin()));
        if (max_nnz < intersection.size()) {
          max_nnz = intersection.size();
          xchg_row = j;
        }
      }
    }
    // swap
    for (int j = 0; j < col; j++) {
      std::swap(adj_mat[ROW_MAJOR_OFF(i, j, col)], adj_mat[ROW_MAJOR_OFF(xchg_row, j, col)]);
    }
    std::swap(perm[i], perm[xchg_row]);
  }

  // then in col
  perm.clear();
  for (int i = 0; i < col; i++) {
    perm.emplace_back(i);
  }
  NNZ *col_nnz;
  NNZ *col_nnz = (NNZ *)malloc(sizeof(NNZ) * col);
  int xchg_col = 0;
  max_nnz = 0;
  for (int j = 0; j < col; j++) {
    for (int i = 0; i < row; i++) {
      if (adj_mat[ROW_MAJOR_OFF(i, j, col)] == AdjValue::CONN) {
        col_nnz[j].insert(i);
      }
    }
    if (max_nnz < col_nnz[j].size()) {
      max_nnz = col_nnz[j].size();
      xchg_col = j;
    }
  }

  for (int j = 0; j < col; j++) {
    if (j > 0) {
      max_nnz = 0;
      for (int i = j + 1; i < col; i++) {
        NNZ intersection;
        std::set_intersection(col_nnz[perm[j]].begin(), col_nnz[perm[j]].end(),
                              col_nnz[perm[i]].begin(), col_nnz[perm[i]].end(),
                              std::inserter(intersection, intersection.begin()));
        if (max_nnz < intersection.size()) {
          max_nnz = intersection.size();
          xchg_col = j;
        }
      }
    }
    // swap
    for (int i = 0; i < row; i++) {
      std::swap(adj_mat[ROW_MAJOR_OFF(i, j, col)], adj_mat[ROW_MAJOR_OFF(i, xchg_col, col)]);
    }
    std::swap(perm[j], perm[xchg_col]);
  }

  // build new graph index
  graph_.indices.clear();
  graph_.indptr.clear();


  // SparseIndex new_graph;
  // new_graph.sparsity_rate = graph_->sparsity_rate;
  // for (int i = 0; i < row; i++) {
  //   new_graph.indptr.push_back(new_graph.indices.size());
  //   for (int j = 0; j < col; j++) {
  //     if (adj_mat[ROW_MAJOR_OFF(i, j, col)] == AdjValue::CONN) {
  //       new_graph.indices.push_back(j);
  //     }
  //   }
  // }
  // new_graph.indptr.push_back(new_graph.indices.size());

  

  free(adj_mat);
  free(row_nnz);
  free(col_nnz);
}

void SpGNN::sparse_weight_reorder(int wnd_size)
{
  int half_wnd_sz = wnd_size >> 1;
  for (int l = 0; l < layers_.size() - 1; l++) {
    auto cur_weight = layers_[l].get_weight();
    auto next_weight = layers_[l + 1].get_weight();
    // build graph to reorder col of the l-th weight 
    // and row of the (l+1)-th weight
    int n_vertex = cur_weight->column_num();
    float *scores = (float *)malloc(sizeof (float) * n_vertex * n_vertex);
    memset(scores, 0, sizeof(float) * n_vertex * n_vertex);
    cur_weight->build_relations(scores, half_wnd_sz, cur_weight->row_num(), false);
    next_weight->build_relations(scores, half_wnd_sz, next_weight->column_num(), true);
    // reorder the column of cur_weight and row of next_weight
    cur_weight->reorder(scores, false);
    next_weight->reorder(scores, true);
    // the row of the 1st weight can be reorder
    if (l == 0) {
      free(scores);
      n_vertex = cur_weight->row_num();
      scores = (float *)malloc(sizeof(float) * n_vertex * n_vertex);
      memset(scores, 0, sizeof(float) * n_vertex * n_vertex);
      cur_weight->build_relations(scores, half_wnd_sz, cur_weight->column_num(), true);
      cur_weight->reorder(scores, true);
    }
    free(scores);
  }
}
