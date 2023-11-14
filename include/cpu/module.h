#ifndef CPU_MODULE_H
#define CPU_MODULE_H

#include "variable.h"
#include "sparse.h"

typedef enum {
  MATMUL,
  SPMM,
  AGG,
  RELU
} op_type_t;

class Module 
{
public:
  virtual void forward(bool) = 0;
  virtual void backward() = 0;
  virtual ~Module() {};
};

class Matmul: public Module 
{
public:
  Matmul(Variable *a, Variable *b, Variable *c, int m, int n, int p);
  ~Matmul() {}
  void forward(bool);
  void backward();

private:
  Variable *a, *b, *c;
  int m, n, p;
};

class SparseMatmul: public Module {
public:
  SparseMatmul(Variable *a, Variable *b, Variable *c, SparseIndex *sp, int m, int n, int p);
  ~SparseMatmul() {}
  void forward(bool);
  void backward();

private:
  Variable *a, *b, *c;
  SparseIndex *sp;
  int m, n, p;
};

#endif