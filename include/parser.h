#ifndef PARSER_H
#define PARSER_H

#include "graph.h"

class Parser
{
public:
  Parser() = default;
  ~Parser() = default;

  void Parse(Graph &g);

};

#endif