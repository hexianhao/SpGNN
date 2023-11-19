#include "cpu/module.h"
#include <math.h>

Matmul::Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p) 
  : a(a), b(b), c(c), m(m), n(n), p(p) {}

void Matmul::forward(bool training) 
{
  c->zero();

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
        c->data[i * p + k] += a->data[i * n + j] * b->data[j * p + k];
}

void Matmul::backward()
{
  return;
}

SparseMatmul::SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p) 
  : a(a), b(b), c(c), sp(sp), m(m), n(n), p(p) {}

void SparseMatmul::forward(bool training) 
{
  c->zero();

  for (int i = 0; i < sp->indptr.size() - 1; i++) {
    for (int jj = sp->indptr[i]; jj < sp->indptr[i + 1]; jj++) {
      int j = sp->indices[jj];
      for (int k = 0; k < p; k++)
        c->data[i * p + k] += a->data[jj] * b->data[j * p + k];
    }
  }
}

void SparseMatmul::backward() 
{
  return;
}

GraphSum::GraphSum(Variable *in, Variable *out, SparseIndex *graph, int dim) 
  : in(in), out(out), graph(graph), dim(dim) {}

void GraphSum::forward(bool training) 
{
  out->zero();

  for (int src = 0; src < graph->indptr.size() - 1; src++) {
    for (int i = graph->indptr[src]; i < graph->indptr[src + 1]; i++) {
      int dst = graph->indices[i];
      float coef = 1.0 / sqrtf(
              (graph->indptr[src + 1] - graph->indptr[src]) * (graph->indptr[dst + 1] - graph->indptr[dst])
      );
      for (int j = 0; j < dim; j++)
        // This only works for undirected graphs. Should be out[dst] += coef * in[src]
        out->data[src * dim + j] += coef * in->data[dst * dim + j];
    }
  }
}

void GraphSum::backward() 
{
  return;
}

ReLU::ReLU(Variable *in) 
{
  this->in = in;
  mask = new bool[in->data.size()];
}

ReLU::~ReLU() 
{
  delete[] mask;
}

void ReLU::forward(bool training) 
{
  for (int i = 0; i < in->data.size(); i++) {
    bool keep = in->data[i] > 0;
    if (training) mask[i] = keep;
    if (!keep) in->data[i] = 0;
  }
}

void ReLU::backward() 
{
  for (int i = 0; i < in->data.size(); i++)
    if (!mask[i]) in->grad[i] = 0;
}