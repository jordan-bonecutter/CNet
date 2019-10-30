/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* matmul.h  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* created by: jordan bonecutter * * * * * * * * * * * * * * * */
/* 25 october 2019 * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef MATMATH_H
#define MATMATH_H

#include <stdio.h>

typedef struct{
  double** weights;
  unsigned rows, cols;
} Matrix;

Matrix* matrix_new(unsigned rows, unsigned cols); /* new matrix with weights init to 0*/
Matrix* matrix_newr(unsigned rows, unsigned cols); /* new matrix with random weights */
void    matrix_del(Matrix* A); /* free memory from allocd matrix */
void    matrix_add(Matrix* A, Matrix* B, Matrix* C);/* C = A + B */
void    matrix_mul(Matrix* A, Matrix* B, Matrix* C);/* C = AB    */
void    matrix_fnc(Matrix* A, Matrix* C, double (*f)(double)); /* C = f(A) */
void    matrix_dump(Matrix* A, FILE* fp); /* dump to json file */
Matrix* matrix_res(FILE* fp); /* resurrect from json file */

#endif

/* eof */
