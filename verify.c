/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* verify.c  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* created by: jordan bonecutter * * * * * * * * * * * * * * * */
/* 29 october 2019 * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "net.h"
#include <stdio.h>
#include <assert.h>

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
  FILE* images, *labels, *net;

  net = fopen("trained.txt", "r");
  Net* trained = net_res(net);
  fclose(net);

  Matrix* in, *out;
  int maxi, i, x, j, N, R, C, correct = 0;
  double maxo;

  in = matrix_new(28*28, 1);
  out = matrix_new(10, 1);

  images = fopen("mnist/t10k-images.idx3-ubyte", "r");
  labels = fopen("mnist/t10k-labels.idx1-ubyte", "r");
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

  fseek(images, 16, SEEK_SET);
  fseek(labels, 8, SEEK_SET);
  for(i = 0; i < N; i++)
  {
    // Read in training data
    for(x = 0; x < R*C; x++) 
    {
      in->weights[x][0] = ((double)((unsigned char)fgetc(images)))/255.;
    }
    x = fgetc(labels);

    // Feed to the net
    net_eval(trained, in, out);

    maxo = 0.;
    for(j = 0; j < out->rows; j++)
    {
      if(out->weights[j][0] > maxo)
      {
        maxi = j;
        maxo = out->weights[j][0];
      }
    }

    if(maxi == x)
    {
      correct++; 
    }
  }

  net_del(trained);
  matrix_del(in);
  matrix_del(out);
  fclose(images);
  fclose(labels);

  printf("Accuracy was %lf\n", (double)correct/(double)N);

  return 0;
}
