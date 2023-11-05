#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <list>

class Graph
{
public:
  Graph() = default;
  ~Graph() = default;

private:
  std::vector<std::list<int>> adj_list_;
};

#endif