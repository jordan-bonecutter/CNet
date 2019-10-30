/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* net.h * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* created by: jordan bonecutter * * * * * * * * * * * * * * * */
/* 25 october 2019 * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef NET_H
#define NET_H

#include <stdio.h>
#include "matmath.h"

typedef struct{
  unsigned *topo;
  unsigned tlen;
  Matrix** a;
  Matrix** W;
  Matrix** b;

  Matrix** W_;
  Matrix** b_;

  double cost;
  double acc;
} Net;

Net* net_new(unsigned* topo, unsigned tlen); /* creates new nn */
void net_del(Net* net); /* frees allocd space for nn */
void net_feed(Net* n, Matrix* input, Matrix* desired); /* does forward and backprop */
void net_learn(Net* n, unsigned N, double lrate); /* adds scaled gradient to W and b */
void net_eval(Matrix* input, Matrix* out); /* feeds fwd and saves output in out */
void net_dump(Net* n, FILE* fp); /* dumps to json file */
Net* net_res(FILE* fp); /* resurrect from json */

#endif

/* eof */
