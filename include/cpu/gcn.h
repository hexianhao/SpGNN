#ifndef CPU_GCN_H
#define CPU_GCN_H

struct GCNParams {
  int num_nodes, input_dim, hidden_dim, output_dim;
  float dropout, learning_rate, weight_decay;
  int epochs, early_stopping;
  static GCNParams get_default();
};

class GCNData {
public:
  
private:
  GCNParams *params;
};

#endif