#include "matrix.h"
#include <stdio.h>

void error(char* str) {
	fprintf(stderr, str);
	exit(0);
}

Matrix* mcreate(int m, int n) {
	Matrix* mp = (Matrix*)malloc(sizeof(Matrix));
	mp->nrows = m;
	mp->ncols = n;
	mp->val = NULL;
	mp->val = (double**)malloc(m * sizeof(double*));
	for (int i = 0; i < m; i++) {
		mp->val[i] = (double*)malloc(n * sizeof(double));
		for (int j = 0; j < n; j++) {
			mp->val[i][j] = 0;
		}
	}
	return mp;
}

Matrix* arr2matrix(int m, int n, double* arr) {
	Matrix* mp = (Matrix*)malloc(sizeof(Matrix));
	mp->nrows = m;
	mp->ncols = n;
	mp->val = NULL;
	mp->val = (double**)malloc(m * sizeof(double*));
	for (int i = 0; i < m; i++) {
		mp->val[i] = (double*)malloc(n * sizeof(double));
		for (int j = 0; j < n; j++) {
			mp->val[i][j] = *(arr + (m*j) + i);
		}
	}
	return mp;
}

Matrix* mdiag(int m, int n, double val) {
	Matrix* mp = mcreate(m, n);
	for (int i = 0; i < m; i++)
		mp->val[i][i] = val;
	return mp;
}

Matrix* ident(int n) {
	Matrix* mp = mdiag(n, n, 1);
	return mp;

}

Matrix* transpose(Matrix* mtrx) {
	int m = mtrx->nrows;
	int n = mtrx->ncols;
	Matrix* mp = mcreate(n, m);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mp->val[j][i] = mtrx->val[i][j];
	return mp;
}

Matrix* add(Matrix* m1, Matrix* m2) {
	if (m1->nrows != m2->nrows || m1->ncols != m2->ncols) {
		error("Size incompability!");
	}
	int m = m1->nrows;
	int n = m1->ncols;
	Matrix* mp = mcreate(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mp->val[i][j] = m1->val[i][j] + m2->val[i][j];
	return mp;
}

Matrix* dotp(Matrix* m1, Matrix* m2) {
	if (m1->ncols != m2->ncols) { //look later
		error("Size incompability!");
	}
	int m = m1->nrows;
	int n = m2->ncols;
	int k = m1->ncols;
	Matrix* mp = mcreate(m, n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int e = 0; e < k; e++) {
				sum += m1->val[i][e] * m2->val[e][j];
			}
			mp->val[i][j] = sum;
		}
	}
	return mp;
}

Matrix* scalarp(double scalar, Matrix* mtrx) {
	int m = mtrx->nrows;
	int n = mtrx->ncols;
	Matrix* mp = mcreate(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			mp->val[i][j] = scalar * mtrx->val[i][j];
	return mp;
}

void print_matrix(Matrix* mtrx) {
	for (int i = 0; i < mtrx->nrows; i++) {
		for (int j = 0; j < mtrx->ncols; j++) {
			printf("%lf\t ", mtrx->val[i][j]);
		}
		printf("\n");
	}
}