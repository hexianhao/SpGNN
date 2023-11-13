#ifndef CPU_OPTIM_H
#define CPU_OPTIM_H

#include <vector>
#include <utility>
#include "variable.h"

struct AdamParams 
{
  float lr, beta1, beta2, eps, weight_decay;
  static AdamParams get_default();
};

struct AdamVariable 
{
public:
  int size();
  AdamVariable(Variable*, bool);

private:
  std::vector<float> *data, *grad, m, v;
  bool decay;
};

class Adam 
{
public:
  Adam() {}
  Adam(std::vector<std::pair<Variable*, bool>> vars, AdamParams params);
  void step();
private:
  AdamParams params;
  int step_count;
  std::vector<AdamVariable> vars;
};

#endif