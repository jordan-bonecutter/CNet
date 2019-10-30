/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* net.c * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* created by: jordan bonecutter * * * * * * * * * * * * * * * */
/* 26 october 2019 * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "net.h"
#include "matmath.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define sqr(x) ((x)*(x))

/* for now, we'll just have the standard sigmoid activation function */
double sig (double x){return (1./(1+pow(M_E, -x)));}
/* ds/dx(s) = x*(1-x) */
double sigd(double x){return (x*(1.-x));}

Net* net_new(unsigned* topo, unsigned tlen)
{
  // Valid args
  assert(topo);
  assert(tlen > 1);

  // Return DS
  unsigned l;
  Net* ret = malloc(sizeof(Net));
  assert(ret);

  // Save the topology
  ret->topo = malloc(sizeof(unsigned)*tlen);
  assert(ret->topo);
  memcpy(ret->topo, topo, tlen*sizeof(unsigned));
  ret->tlen = tlen;

  // Space for weights and biases
  ret->a = malloc(sizeof(Matrix*)*(tlen-1));
  ret->W = malloc(sizeof(Matrix*)*(tlen-1));
  ret->b = malloc(sizeof(Matrix*)*(tlen-1));

  ret->W_ = malloc(sizeof(Matrix*)*(tlen-1));
  ret->b_ = malloc(sizeof(Matrix*)*(tlen-1));

  for(l = 1; l < tlen; l++)
  {
    ret->a[l-1] = matrix_new(topo[l], 1);
    ret->W[l-1] = matrix_newr(topo[l], topo[l-1]);
    ret->b[l-1] = matrix_newr(topo[l], 1);

    ret->W_[l-1] = matrix_new(topo[l], topo[l-1]);
    ret->b_[l-1] = matrix_new(topo[l], 1);
  }
  ret->cost = 0.;
  ret->acc = 0.;

  return ret;
}

void net_del(Net* net)
{
  // Make sure we dont try to free NULL
  assert(net);
  unsigned i;

  for(i = 0; i < net->tlen - 1; i++)
  {
    matrix_del(net->a[i]);
    matrix_del(net->W[i]);
    matrix_del(net->b[i]);
    
    matrix_del(net->W_[i]);
    matrix_del(net->b_[i]);
  }

  free(net->a);
  free(net->W);
  free(net->b);

  free(net->W_);
  free(net->b_);

  assert(net->topo);
  free(net->topo);
  free(net);
}

void net_eval(Net* n, Matrix* input, Matrix* out)
{
  assert(n);
  assert(input);
  assert(out);

  long l, k;
  double s;
  
  // Feed fwd
  //
  // a_n = s([W_n][a_n-1] + [b_n])
  matrix_mul(n->W[0], input, n->a[0]);
  matrix_add(n->a[0], n->b[0], n->a[0]);
  matrix_fnc(n->a[0], n->a[0], sig);
  for(l = 1; l < n->tlen-2; l++)
  {
    matrix_mul(n->W[l], n->a[l-1], n->a[l]);
    matrix_add(n->a[l], n->b[l], n->a[l]);
    matrix_fnc(n->a[l], n->a[l], sig);
  }
  // Last layer is softmax
  matrix_mul(n->W[l], n->a[l-1], out);
  matrix_add(out, n->b[l], out);
  s = 0.;
  for(k = 0; k < n->topo[l+1]; k++)
  {
    s += pow(M_E, out->weights[k][0]);
  }
  for(k = 0; k < n->topo[l+1]; k++)
  {
    out->weights[k][0] = (pow(M_E, out->weights[k][0]))/s;
  }

  return;
}

void net_feed(Net* n, Matrix* input, Matrix* y)
{
  assert(n);
  assert(input);
  assert(y);

  long l, j, k, maxi;
  double maxo, s;
  
  // Feed fwd
  //
  // a_n = s([W_n][a_n-1] + [b_n])
  matrix_mul(n->W[0], input, n->a[0]);
  matrix_add(n->a[0], n->b[0], n->a[0]);
  matrix_fnc(n->a[0], n->a[0], sig);
  for(l = 1; l < n->tlen-2; l++)
  {
    matrix_mul(n->W[l], n->a[l-1], n->a[l]);
    matrix_add(n->a[l], n->b[l], n->a[l]);
    matrix_fnc(n->a[l], n->a[l], sig);
  }
  // Last layer is softmax
  matrix_mul(n->W[l], n->a[l-1], n->a[l]);
  matrix_add(n->a[l], n->b[l], n->a[l]);
  s = 0.;
  for(k = 0; k < n->topo[l+1]; k++)
  {
    s += pow(M_E, n->a[l]->weights[k][0]);
  }
  for(k = 0; k < n->topo[l+1]; k++)
  {
    n->a[l]->weights[k][0] = (pow(M_E, n->a[l]->weights[k][0]))/s;
  }

  // Backprop
  //
  // For my implementation, I will not find the gradient due
  // to the activation values. What I have realized is that the
  // gradient due to the bias vector elements is more useful as it 
  // is also used in calculation of the weight gradients. So in 
  // the first pass, I find the gradient due to all of the bias 
  // vector elements and then use those values to find the gradient 
  // due to each of the weights.
  //
  // I also denote the activation before being squished by the sigmoid
  // as z (the same as in 3Blue1Brown's video).
  //
  // Start with output layer as it is a special case
  //
  // dC/db_i = 2*(a_i - y_i)*s'(z_i)
  l = n->tlen-1;
  maxi = 0;
  maxo = 0;
  for(k = 0; k < n->topo[l]; k++)
  {
    n->b_[l-1]->weights[k][0] = 2.*(n->a[l-1]->weights[k][0] - y->weights[k][0])*
      sigd(n->a[l-1]->weights[k][0]);
    n->cost += sqr(n->a[l-1]->weights[k][0] - y->weights[k][0]);
    if(n->a[l-1]->weights[k][0] > maxo)
    {
      maxi = k;
      maxo = n->a[l-1]->weights[k][0];
    }
  }
  if(y->weights[maxi][0] == 1.0)
  {
    n->acc++;
  }

  // Now we will do the remaining bias vectors
  // 
  // We implement:
  //  dC/db_j = SUM_{k = 0}^{N_l+1 - 1}[]
  for(l = n->tlen-2; l > 0; l--)
  {
    for(j = 0; j < n->topo[l]; j++) 
    {
      n->b_[l-1]->weights[j][0] = 0.;
      for(k = 0; k < n->topo[l+1]; k++) 
      {
        n->b_[l-1]->weights[j][0] += n->b_[l]->weights[k][0]*
          sigd(n->a[l]->weights[k][0])*
          n->W[l]->weights[k][j]*
          sigd(n->a[l-1]->weights[j][0]);
      }
    }
  }

  // Now for the weights
  for(l = n->tlen-1; l > 1; l--)
  {
    for(j = 0; j < n->topo[l-1]; j++) 
    {
      for(k = 0; k < n->topo[l]; k++) 
      {
        n->W_[l-1]->weights[k][j] += n->b_[l-1]->weights[k][0]*
          n->a[l-2]->weights[j][0];
      }
    }
  }
  for(j = 0; j < n->topo[0]; j++)
  {
    for(k = 0; k < n->topo[1]; k++) 
    {
      n->W_[0]->weights[k][j] += n->b_[0]->weights[k][0]*
        input->weights[j][0];
    }
  }
}

void net_learn(Net* n, unsigned N, double lrate)
{
  // Essentially just subtract the gradient from the values
  long i, j, l;
  for(l = 0; l < n->tlen-2; l++) 
  {
    for(i = 0; i < n->topo[l]; i++) 
    {
      for(j = 0; j < n->topo[l+1]; j++) 
      {
        n->W[l]->weights[j][i] -= (n->W_[l]->weights[j][i]/N)*lrate;
        n->W_[l]->weights[j][i] = 0.;
      }
    }
    for(i = 0; i < n->topo[l+1]; i++)
    {
      n->b[l]->weights[i][0] -= (n->b_[l]->weights[i][0]/N)*lrate;
      n->b_[l]->weights[i][0] = 0.;
    }
  }

  // Nicety
  printf("%lf %lf\r", n->cost/N, n->acc/N);
  fflush(stdout);
  n->cost = 0.;
  n->acc = 0.;
}

void net_dump(Net* n, FILE* fp)
{
  // Dump to a json file
  fprintf(fp, "{\"tlen\": %u, \"topo\": [", n->tlen);
  for(int i = 0; i < n->tlen; i++)
  {
    if(i == 0) 
    {
      fprintf(fp, "%u", n->topo[i]);
    }
    else
    {
      fprintf(fp, ", %u", n->topo[i]);
    }
  }
  fprintf(fp, "]");

  fprintf(fp, ", \"a\": [");
  for(int i = 0; i < n->tlen-1; i++)
  {
    if(i == 0)
    {
      matrix_dump(n->a[i], fp);
    }
    else
    {
      fprintf(fp, ", ");
      matrix_dump(n->a[i], fp);
    }
  }
  fprintf(fp, "], \"W\": [");
  for(int i = 0; i < n->tlen-1; i++)
  {
    if(i == 0)
    {
      matrix_dump(n->W[i], fp);
    }
    else
    {
      fprintf(fp, ", ");
      matrix_dump(n->W[i], fp);
    }
  }
  fprintf(fp, "], \"b\": [");
  for(int i = 0; i < n->tlen-1; i++)
  {
    if(i == 0)
    {
      matrix_dump(n->b[i], fp);
    }
    else
    {
      fprintf(fp, ", ");
      matrix_dump(n->b[i], fp);
    }
  }
  
  fprintf(fp, "]}");
}

Net* net_res(FILE* fp)
{
  // Resurrect from a json file
  int i, j, _;
  char buff[100];

  for(_ = 0; _ < 8; _++, fgetc(fp));
  for(i = 0; i < 100; i++){buff[i] = fgetc(fp); if(buff[i] == ','){buff[i] = 0; break;}}
  Net* ret = malloc(sizeof(Net));
  ret->tlen = atoi(buff);
  ret->a    = malloc(sizeof(Matrix*)*(ret->tlen-1));
  ret->W    = malloc(sizeof(Matrix*)*(ret->tlen-1));
  ret->b    = malloc(sizeof(Matrix*)*(ret->tlen-1));
  ret->W_   = malloc(sizeof(Matrix*)*(ret->tlen-1));
  ret->b_   = malloc(sizeof(Matrix*)*(ret->tlen-1));
  ret->topo = malloc(sizeof(unsigned)*ret->tlen);

  for(_ = 0; _ < 10; _++, fgetc(fp));
  for(i = 0; i < ret->tlen; i++)
  {
    for(j = 0; j < 100; j++) 
    {
      buff[j] = fgetc(fp);
      if(buff[j] == ',' || buff[j] == ']')
      {
        buff[j] = 0;
        break;
      }
    }
    ret->topo[i] = atoi(buff);
  }

  for(_ = 0; _ < 8; _++, fgetc(fp));
  for(i = 0; i < ret->tlen-2; i++)
  {
    ret->a[i] = matrix_res(fp);
    fgetc(fp);
    fgetc(fp);
  }
  ret->a[i] = matrix_res(fp);

  for(_ = 0; _ < 9; _++, fgetc(fp));
  for(i = 0; i < ret->tlen-2; i++)
  {
    ret->W[i] = matrix_res(fp);
    fgetc(fp);
    fgetc(fp);
  }
  ret->W[i] = matrix_res(fp);

  for(_ = 0; _ < 9; _++, fgetc(fp));
  for(i = 0; i < ret->tlen-1; i++)
  {
    ret->b[i] = matrix_res(fp);
    ret->b_[i] = matrix_new(ret->b[i]->rows, ret->b[i]->cols);
    ret->W_[i] = matrix_new(ret->W[i]->rows, ret->W[i]->cols);
    fgetc(fp);
    fgetc(fp);
  }

  return ret;
}

#ifdef UNITTEST

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// Read in an integer from the file
int fgeti(FILE* fp)
{
  int ret = 0;
  ret = ((int)fgetc(fp));
  ret<<=8;
  ret |= ((int)fgetc(fp));
  ret<<=8;
  ret |= ((int)fgetc(fp));
  ret<<=8;
  ret |= ((int)fgetc(fp));

  return ret;
}

int main()
{
  // Imitate 3b1b's network
  unsigned topo[] = {28*28, 17, 16, 10}, i, N, R, C, x, done;
  Net* n = net_new(&topo[0], sizeof(topo)/sizeof(unsigned));
  FILE* images, * labels, *save;
 
  images = fopen("mnist/train-images.idx3-ubyte", "r");
  labels = fopen("mnist/train-labels.idx1-ubyte", "r");

  if(!images || !labels)
  {
    printf("Please goto http://yann.lecun.com/exdb/mnist/ & download the dataset\n");
    return 1;
  }

  // Read in the files
  fgeti(images);
  N = fgeti(images);
  R = fgeti(images);
  C = fgeti(images);
  fgeti(labels);
  i = fgeti(labels);
  assert(i == N);
  
  // Input & output vectors for the matrix
  Matrix* in = matrix_new(28*28, 1);
  Matrix* out = matrix_new(10, 1);
  done = 0;
  for(;!done;)
  {
    fseek(images, 16, SEEK_SET);
    fseek(labels, 8, SEEK_SET);
    for(i = 0; i < N; i++)
    {
      // Read in training data
      for(x = 0; x < R*C; x++) 
      {
        in->weights[x][0] = ((double)((unsigned char)fgetc(images)))/255.;
      }
      for(x = 0; x < 10; x++)
      {
        out->weights[0][x] = 0.;
      }
      x = fgetc(labels);
      out->weights[0][x] = 1.;

      // Feed to the net
      net_feed(n, in, out);
      if((i+1)%300 == 0)
      {
        // If ever the accuracy is greater than 98% for 
        // a mini-batch, then we done training
        if(n->acc/300 > 0.98)
        {
          done = 1; 
        }
        // Learn every 300 data points
        net_learn(n, 300, n->cost/270);
      }
    }
  }

  // Close fp's & save
  fclose(images);
  fclose(labels);
  save = fopen("save.net", "w");
  net_dump(n, save);
  fclose(save);

  // Free up space
  matrix_del(in);
  matrix_del(out);
  net_del(n);
}

#endif
