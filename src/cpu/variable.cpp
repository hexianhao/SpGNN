#include "cpu/variable.h"
#include <deque>
#include <cmath>

Variable::Variable(std::vector<int> shape, std::string format="csr", bool init=true)
  : shape_(std::move(shape)), format_(format)
{
  if (format.compare("dense")) {
    size_t data_size = 1;
    for (int i = 0; i < shape_.size(); i++) {
      data_size *= shape_[i];
    }
    data_.reserve(data_size);
    if (init) {
      for (int i = 0; i < data_size; i++) {
        data_[i] = 0;
      }
    }
  }
}

void Variable::glorot(int in_size, int out_size)
{
  printf("Not implemented yet\n");
}

void Variable::zero()
{
  for (int i = 0; i < data_.size(); i++) {
    data_[i] = 0;
  }
}

void Variable::zero_grad()
{
  for (int i = 0; i < grad_.size(); i++) {
    grad_[i] = 0;
  }
}

void Variable::print(int col=0x7fffffff)
{
  printf("Not implemented yet\n");
}

float Variable::grad_norm()
{
  printf("Not implemented yet\n");
}

void Variable::reorder(float *scores, bool xchg_row)
{
  int n_vertex = xchg_row ? row_num() : column_num();
  std::vector<int> perm;
  // we abstract the reorder as TSP problem.
  // TODO: TSP solver
  for (int i = 0; i < n_vertex; i++) {

  }

  if (format_.compare("dense")) {
    reorder_dense(perm, xchg_row);
  } else {
    reorder_sparse(perm, xchg_row);
  }
}

void Variable::build_relations(float *scores, int half_wnd_sz,
                               int n_iter, bool col_dim)
{
  int n_vertex = col_dim ? column_num() : row_num();
  for (int i = 0; i < n_vertex; i++) {
    for (int j = 0; j < n_vertex; j++) {
      if (i == j) continue;
      for (int k = 0; k < n_iter; k++) {
        if (col_dim) {
          if (!non_zero(i, k, index_)) continue;
          for (int t = std::max(0, k - half_wnd_sz); t < std::min(k + half_wnd_sz, n_iter); t++) {
            if (!non_zero(j, t, index_)) continue;
            scores[ROW_MAJOR_OFF(i, j, n_vertex)] += std::exp(-std::abs(k - t));
          }
        } else {
          if (!non_zero(k, i, index_)) continue;
          for (int t = std::max(0, k - half_wnd_sz); t < std::min(k + half_wnd_sz, n_iter); t++) {
            if (!non_zero(k, j, index_)) continue;
            scores[ROW_MAJOR_OFF(i, j, n_vertex)] += std::exp(-std::abs(k - t));
          }
        }
      }
    }
  }
}

void Variable::reorder_dense(std::vector<int> &perm, bool xchg_row)
{
  int row = row_num();
  int col = column_num();
  std::vector<float> data(row * col);

  if (xchg_row) {
    for (int i = 0; i < perm.size(); i++) {
      int r = perm[i];
      for (int j = 0; j < col; j++) {
        data[ROW_MAJOR_OFF(i, j, col)] = data_[ROW_MAJOR_OFF(r, j, col)];
      }
    }
  } else {
    for (int j = 0; j < perm.size(); j++) {
      int c = perm[j];
      for (int i = 0; i < row; i++) {
        data[ROW_MAJOR_OFF(i, j, col)] = data[ROW_MAJOR_OFF(i, c, col)];
      }
    }
  }
  data_.swap(data);
}

void Variable::reorder_sparse(std::vector<int> &perm, bool xchg_row)
{
  if (xchg_row) {
    std::vector<float> new_data;
    std::vector<int> new_indptr;
    std::vector<int> new_indices;
    for (int i = 0; i < perm.size(); i++) {
      int r = perm[i];
      new_indptr.push_back(new_indices.size());
      for (int j = index_.indptr[r]; j < index_.indptr[r + 1]; j++) {
        new_indices.push_back(index_.indices[j]);
        new_data.push_back(data_[j]);
      }
    }
    new_indptr.push_back(new_indices.size());
    // set new index
    index_.indices.swap(new_indices);
    index_.indptr.swap(new_indptr);
    data_.swap(new_data);
  } else {
    int row = row_num();
    for (int i = 0; i < row; i++) {
      std::deque<int> new_indces;
      std::deque<float> new_data;
      for (int j = 0; j < perm.size(); j++) {
        int c = perm[j];
        for (int k = index_.indptr[i]; k < index_.indptr[i + 1]; k++) {
          if (c == index_.indices[k]) {
            new_indces.push_back(j);
            new_data.push_back(data_[k]);
          }
        }
      }
      // update indices and data
      for (int k = index_.indptr[i]; k < index_.indptr[i + 1]; k++) {
        index_.indices[k] = new_indces.front();
        new_indces.pop_front();
        data_[k] = new_data.front();
        new_data.pop_front();
      }
    }
  }
}

TiledVariable::TiledVariable(float *data,
                             SparseIndex *sp_index,
                             std::vector<int> &row_tiles, 
                             std::vector<int> &column_tiles,
                             bool tile_row_major,
                             std::string tile_format)
{
  int row_accum = 0, col_accum = 0;

  auto tile_process = [this, &data, &sp_index, &row_accum, &col_accum, &tile_format]
                      (int row_size, int col_size) {
    int meta, s, e;
    if (tile_format.compare("csr") == 0) {
      // first make statistics
      int nnz = 0;
      for (int i = row_accum; i < row_accum + row_size; i++) {
        for (int j = sp_index->indptr[i]; j < sp_index->indptr[i + 1]; j++) {
          if (sp_index->indices[j] >= col_accum && 
              sp_index->indices[j] < col_accum + col_size) {
            nnz++;
          }
        }
      }
      // fill the meta in tile_info
      bool sparse = ((float)nnz / (row_size * col_size) < 0.6);
      if (sparse) {
        meta = (1 << 30) | (row_size << 14) | col_size;
        tile_info_.push_back(meta);
      } else {
        int over_l1 = 0;
        if (row_size * col_size > L1) { // whether over L1 cache
          over_l1 = 1;
        }
        meta = (over_l1 << 29) | (row_size << 14) | col_size;
        tile_info_.push_back(meta);
      }
      // fill tile_data and indices according to tile type
      int scope = (indptr_.size() << 16) | indices_.size(); // record the start position of the tile
      tile_info_.push_back(scope);
      for (int i = row_accum; i < row_accum + row_size; i++) {
        indptr_.push_back(indices_.size());
        for (int j = col_accum; j < col_accum + col_size; j++) {
          for (int k = sp_index->indptr[i]; k < sp_index->indptr[i + 1]; k++) {
            if (j == sp_index->indices[k]) {
              tiled_data_.emplace_back(data[k]);
              indices_.emplace_back(j - col_accum);
            } else if (!sparse) {
              // dense tile
              // fill in zero
              tiled_data_.emplace_back(0);
              indices_.emplace_back(j - col_accum);
            }
          }
        }
        indptr_.push_back(indices_.size());
      }
      indptr_.push_back(indices_.size());
    } else {
      printf("Not implemented yet\n");
    }
  };

  if (tile_row_major) {
    for (auto row_size : row_tiles) {
      col_accum = 0;
      for (auto col_size : column_tiles) {
        tile_process(row_size, col_size);
        col_accum += col_size;
      }
      row_accum += row_size;
    }
  } else {
    for (auto col_size : column_tiles) {
      row_accum = 0;
      for (auto row_size : row_tiles) {
        tile_process(row_size, col_size);
        row_accum += row_size;
      }
      col_accum += col_size;
    }
  }
}