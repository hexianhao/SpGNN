#include "cpu/sparse.h"

bool non_zero(int row, int col, SparseIndex &sp)
{
  for (int i = sp.indptr[row]; i < sp.indptr[row + 1]; i++) {
    if (sp.indices[i] == col) return true;
  }
  return false;
}