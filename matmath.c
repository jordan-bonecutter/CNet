/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* matmul.h  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* created by: jordan bonecutter * * * * * * * * * * * * * * * */
/* 25 october 2019 * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "matmath.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#define RANGE 2.

/* init w random weights */
Matrix* matrix_newr(unsigned rows, unsigned cols)
{
  unsigned i, j;
  
  // Alloc matrix pointer space
  Matrix* ret = malloc(sizeof(Matrix));
  ret->rows = rows;
  ret->cols = cols;

  // alloc row pointers
  assert(ret);
  ret->weights = malloc(sizeof(double*)*rows);

  // setup matrix data structure
  assert(ret->weights);
  ret->weights[0] = malloc(sizeof(double)*rows*cols);
  for(i = 1; i < rows; i++)
  {
    ret->weights[i] = ret->weights[i-1] + cols;
  }

  // randomize entries
  for(i = 0; i < rows; i++)
  {
    for(j = 0; j < cols; j++)
    {
      ret->weights[i][j] = RANGE*((double)rand()/(double)RAND_MAX - 0.5);
    }
  }

  return ret;
}

Matrix* matrix_new(unsigned rows, unsigned cols)
{
  unsigned i, j;
  
  // Alloc matrix pointer space
  Matrix* ret = malloc(sizeof(Matrix));
  ret->rows = rows;
  ret->cols = cols;

  // alloc row pointers
  assert(ret);
  ret->weights = malloc(sizeof(double*)*rows);

  // setup matrix data structure
  assert(ret->weights);
  ret->weights[0] = calloc(rows*cols, sizeof(double));
  for(i = 1; i < rows; i++)
  {
    ret->weights[i] = ret->weights[i-1] + cols;
  }

  return ret;
}

void matrix_del(Matrix* A)
{
  // Ensure A is valid
  assert(A);
  assert(A->weights);
  assert(*A->weights);

  // Free mem
  free(*A->weights);
  free(A->weights);
  free(A);
}

void matrix_add(Matrix* A, Matrix* B, Matrix* C)
{
  // Ensure valid args
  assert(A);
  assert(B);
  assert(C);
  assert(A->rows == B->rows);
  assert(A->cols == B->cols);
  assert(A->rows == C->rows);
  assert(A->cols == C->cols);

  // Iterators
  unsigned i, j;

  // Loop and add
  for(i = 0; i < A->rows; i++)
  {
    for(j = 0; j < A->cols; j++) 
    {
      C->weights[i][j] = A->weights[i][j] + B->weights[i][j];
    }
  }
}

void matrix_mul(Matrix* A, Matrix* B, Matrix* C)
{
  // Ensure valid args
  assert(A);
  assert(B);
  assert(C);
  assert(A->cols == B->rows);
  assert(A->rows == C->rows);
  assert(B->cols == C->cols);
  
  // Iterators
  unsigned i, j, k;

  // Multiply
  for(i = 0; i < C->rows; i++)
  {
    for(j = 0; j < C->cols; j++)
    {
      C->weights[i][j] = 0;
      for(k = 0; k < A->cols; k++)
      {
        C->weights[i][j] += A->weights[i][k]*B->weights[k][j];
      }
    }
  }
}

void matrix_fnc(Matrix* A, Matrix* C, double (*f)(double))
{
  // Ensure valid args
  assert(A);
  assert(C);
  assert(A->rows == C->rows);
  assert(A->cols == C->cols);

  // Iterators
  unsigned i, j;

  // Do the function
  for(i = 0; i < A->rows; i++)
  {
    for(j = 0; j < A->cols; j++) 
    {
      C->weights[i][j] = f(A->weights[i][j]);
    }
  }
}

void matrix_dump(Matrix* A, FILE* fp)
{
  fprintf(fp, "{\"rows\":%d, \"cols\": %d, \"weights\": [", A->rows, A->cols);
  unsigned i, j;
  for(i = 0; i < A->rows; i++)
  {
    for(j = 0; j < A->cols; j++) 
    {
      if(i == 0 && j == 0)
      {
        fprintf(fp, "%lf", A->weights[i][j]);
      }
      else
      {
        fprintf(fp, ", %lf", A->weights[i][j]);
      }
    }
  }
  fprintf(fp, "]}");
}

Matrix* matrix_res(FILE* fp)
{
  char buff[100];
  int _, i, j, rows, cols;
  for(_ = 0; _ < 8; _++, fgetc(fp));
  for(i = 0; i < 100; i++){buff[i] = fgetc(fp); if(buff[i] == ','){buff[i] = 0; break;}}
  rows = atoi(buff);

  for(_ = 0; _ < 8; _++, fgetc(fp));
  for(i = 0; i < 100; i++){buff[i] = fgetc(fp); if(buff[i] == ','){buff[i] = 0; break;}}
  cols = atoi(buff);

  Matrix* ret = matrix_new(rows, cols);
  for(_ = 0; _ < 13; _++, fgetc(fp));

  for(i = 0; i < ret->rows; i++)
  {
    for(j = 0; j < ret->cols; j++) 
    {
      for(_ = 0; _ < 100; _++){
        buff[_] = fgetc(fp); 
        if(buff[_] == ']' || buff[_] == ','){
        buff[_] = 0; break;}
      }
      ret->weights[i][j] = atof(buff);
    }
  }
  fgetc(fp);

  return ret;
}

#ifdef UNITTEST

#include <stdio.h>

int main()
{
  // Create the 3 matrices
  Matrix* A = matrix_new(3, 5);
  Matrix* B = matrix_new(5, 7);
  Matrix* C = matrix_new(3, 7);
  unsigned i, j;

  // Fill A's values
  printf("A = \n");
  for(i = 0; i < 3; i++)
  {
    for(j = 0; j < 5; j++)   
    {
      A->weights[i][j] = ((double)i+1)/(j+1+i);
      printf("%.2lf ", ((double)i+1)/(j+1+i));
    }
    printf("\n");
  }

  // Fill B's values
  printf("B = \n");
  for(i = 0; i < 5; i++)
  {
    for(j = 0; j < 7; j++) 
    {
      B->weights[i][j] = ((double)j+1)/(i+j+2);
      printf("%.2lf ", ((double)j+1)/(j+2+i));
    }
    printf("\n");
  }

  // Run mutiplication subroutine
  matrix_mul(A, B, C);

  // Print the resultant matrix
  printf("AB = \n");
  for(i = 0; i < 3; i++)
  {
    for(j = 0; j < 7; j++) 
    {
      printf("%.2lf ", C->weights[i][j]);
    }
    printf("\n");
  }

  // Free up used space
  matrix_del(A);
  matrix_del(B);
  matrix_del(C);
}

#endif

/* eof */
