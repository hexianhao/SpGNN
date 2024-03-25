#ifndef CPU_SpGNN_H
#define CPU_SpGNN_H

#include <vector>
#include "cpu/module.h"
#include "cpu/optim.h"
#include "cpu/spgnn_layer.h"

struct SpGNNParams 
{
  int num_nodes, input_dim, hidden_dim, output_dim, num_layers;
  float dropout, learning_rate, weight_decay;
  int epochs, early_stopping;
  static SpGNNParams get_default();
};

struct SpGNNData
{
  Variable graph;
  Variable input_data;
};

class SpGNN
{
public:
  SpGNN(SpGNNParams params, SpGNNData *data);

  SpGNNParams get_spgnn_params() { return params_; }

  void set_input();
  void inference();         // SpGNN inference
  void pruning();
  void auto_tuning();       // auto tuning

private:
  typedef int8_t AdjElem;
  typedef enum {
    NONE,
    CONN
  } AdjValue;

  void data_layout_transform(); // change data layout
  void sparse_graph_reorder();
  void sparse_weight_reorder(int wnd_size);
  
  std::vector<SpGNNLayer> layers_;
  SpGNNParams params_;         // spgnn structure data
  SpGNNData *data_;            // spgnn input data
  Variable input_feat_;        // input feature
  Variable graph_;
};

#endif