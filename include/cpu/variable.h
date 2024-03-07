#ifndef CPU_VARIABLE_H
#define CPU_VARIABLE_H

#include <vector>
#include <memory>

struct Variable 
{
  std::vector<float> data, grad;
  std::vector<int> shape;
  Variable(std::vector<int> shape, bool requires_grad=true);
  void glorot(int in_size, int out_size);
  void zero();
  void zero_grad();
  void print(int col=0x7fffffff);
  float grad_norm();
};

class TiledVariable
{
public:
  TiledVariable() = default;
  TiledVariable(std::vector<float> data, int width_tiles, int column_tiles);

private:
  std::vector<float> tiled_data_;           // re-organize through tile plan
  int width_tiles_;
  int column_tiles_;
  std::unique_ptr<SparseIndex> index_ptr_;  // size of width_tiles_ x column_tiles_
};

#endif