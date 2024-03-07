#ifndef CPU_SpGNN_H
#define CPU_SpGNN_H

#include <vector>
#include "cpu/module.h"
#include "cpu/optim.h"
#include "cpu/spgnn_layer.h"

#define ROW_MAJOR_OFF(i, j, ld) ((i) * (ld) + (j))
#define COL_MAJOR_OFF(i, j, ld) ((i) + (ld) * (r))

struct SpGNNParams 
{
  int num_nodes, input_dim, hidden_dim, output_dim, num_layers;
  float dropout, learning_rate, weight_decay;
  int epochs, early_stopping;
  static SpGNNParams get_default();
};

struct SpGNNData
{
  SparseIndex feature_index, graph;
  std::vector<int> split;
  std::vector<int> label;
  std::vector<float> feature_value;
};

class SpGNN
{
public:
  SpGNN(SpGNNParams params, SpGNNData *data);

  SpGNNParams get_spgnn_params() { return params_; }

  void set_input();
  void inference();         // SpGNN inference
  void pruning();
  void compile();

private:
  typedef int8_t AdjElem;
  typedef enum {
    NONE,
    CONN
  } AdjValue;

  void auto_tuning();           // auto tuning
  void data_layout_transform(); // change data layout
  void sparse_graph_reorder();
  void sparse_weight_reorder();
  void reorder(AdjElem *adj, int row, int col, bool is_row);
  void reorder(AdjElem *adj1, AdjElem *adj2, int r1, int c1, int c2);
  
  std::vector<SpGNNLayer> layers_;
  SpGNNParams params_;         // spgnn structure data
  SpGNNData *data_;            // spgnn input data
  Variable input_feat_;        // input feature
};

#endif