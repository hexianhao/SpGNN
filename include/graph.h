#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <list>
#include <metis.h>

class Graph
{
public:
  Graph() = default;
  ~Graph() = default;

  void set_num(int n) { adj_list_.reserve(n); }
  void set_edge(int u, int v) { adj_list_[u].push_back(v); }

  void partition(int kway);   // 图划分算法

private:
  std::vector<std::list<int>> adj_list_;
  
};

#endif