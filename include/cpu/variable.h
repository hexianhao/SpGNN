#ifndef CPU_VARIABLE_H
#define CPU_VARIABLE_H

#include <vector>
#include <memory>
#include "cpu/sparse.h"

#define L1 (64 << 10)  // L1 cache size
#define ROW_MAJOR_OFF(i, j, ld) ((i) * (ld) + (j))
#define COL_MAJOR_OFF(i, j, ld) ((i) + (ld) * (j))

typedef enum {
  DENSE = -1,
  SPARSE = -2 
} TileType;

class Variable 
{
public:
  Variable() = default;
  Variable(std::vector<int> shape, std::string format="csr", bool init=true);

  void glorot(int in_size, int out_size);
  void zero();
  void zero_grad();
  void print(int col=0x7fffffff);
  float grad_norm();

  float *data() { return data_.data(); }
  int row_num() { return shape_[0]; }
  int column_num() { return shape_[1]; }
  SparseIndex *index() { return &index_; }
  void set_data(std::vector<float> &data) { data_.swap(data); }

  // transform
  void reorder(float *scores, bool xchg_row);
  void build_relations(float *scores, int half_wnd_sz,
                       int n_iter, bool col_dim);

private:
  void reorder_sparse(std::vector<int> &perm, bool xchg_row);
  void reorder_dense(std::vector<int> &perm, bool xchg_row);

  std::vector<float> data_, grad_;
  std::vector<int> shape_;
  SparseIndex index_;   // ignored, if format is "dense"
  std::string format_;
};

class TiledVariable
{
public:
  TiledVariable() = default;
  TiledVariable(float *data,
                SparseIndex *sp_index,
                std::vector<int> &row_tiles, 
                std::vector<int> &column_tiles,
                bool tile_row_major,
                std::string tile_format="csr");

  int row_tile_num() { return n_row_tiles_; }
  int column_tile_num() { return n_column_tiles_; }
  float *data() { return tiled_data_.data(); }
  int *indices() { return indices_.data(); }
  int *indptr() { return indptr_.data(); }
  int *tile_info() { return tile_info_.data(); }

private:
  int n_row_tiles_;
  int n_column_tiles_;
  std::vector<float> tiled_data_; // re-organize through tile plan
  // indices only for sparse matrix
  std::vector<int> indices_;
  std::vector<int> indptr_;
  // 2 x n_width_tiles x n_column_tiles, the tuple <meta, scope> of each
  // tile is stored in tile_info, therefore each tile occupies 2 elements in tile_info.
  // meta is 32-bit, which has 4 fields [type(31), size(30), row(29-15), col(14-0)]. 
  // If type = 0, the tile is dense, otherwise is sparse. If size = 1, the tile size is over 
  // L1 cache, otherwise is below. The dense tile will view 0 as non-zero element.
  // scope is the start position of indptr or tiled_data for the tile, 
  // If type = 0, the scope is start position of tiled_data for the tile, otherwise
  // is indptr (we ignore indptr and indices although we maintain these structures for dense tile).
  std::vector<int> tile_info_;
};


#endif