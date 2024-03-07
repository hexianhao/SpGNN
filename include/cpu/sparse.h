#ifndef CPU_SPARSE_H
#define CPU_SPARSE_H

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

/* For both sparse matrix and graph. Sparse matrix = sparse index + variable
 * Compressed Sparse Row (CSR) format: len(indices) = nnz, len(indptr) = nrow+1
 * indices[j] (j in [indptr[i], indptr[i+1])) stores column indices of nonzero
 * elements in the i-th row.
 */
struct SparseIndex 
{
  double sparsity_rate;
  int row, column;
  std::vector<int> indices;
  std::vector<int> indptr;
};


#endif