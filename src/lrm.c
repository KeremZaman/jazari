#define _USE_MATH_DEFINES
#include <math.h>
#include "lrm.h"

double sigmoid(double x) {
	return 1 / (1 + pow(M_E, (-1 *x)));
}

double lin_hyp(Matrix* params, Matrix* features) {
	return dotp(params, features)->val[0][0];
}

double log_hyp(Matrix* params, Matrix* features) {
	return sigmoid(dotp(params, features)->val[0][0]);
}

double linRM_cost(LRModel* lrm) {
	Matrix* x = transpose(lrm->tr_data);
	double sum = 0;
	for (int i = 0; i < x->nrows; i++) {
		double a = lrm->hyp(lrm->params, get_vec(x->ncols, x->val[i])) - lrm->targets->val[i][0];
		sum += (a*a);
	}
	return sum / (2*x->nrows);
}

double logRM_cost(LRModel* lrm) {
	Matrix* x = transpose(lrm->tr_data);
	double sum = 0;
	for (int i = 0; i < x->nrows; i++) {
		double a = 0.0;
		if (lrm->targets->val[i][0] == 1.0)
			a = -1 * log(lrm->hyp(lrm->params, get_vec(x->ncols, x->val[i])));
		else if (lrm->targets->val[i][0] == 0.0)
			a = -1 * (1 - log(lrm->hyp(lrm->params, get_vec(x->ncols, x->val[i]))));
		sum += (a*a);
	}
	return sum / (x->nrows);
}

void sgd(LRModel* lrm) {
	Matrix* x = transpose(lrm->tr_data); //feature matrix
	Matrix* updated_params;
	double lr = lrm->lr;
	for (int e = 0; e < lrm->epoch_num; e++) {
		for (int i = 0; i < lrm->tr_data->ncols; i++) {
			double diff = (lrm->targets->val[i][0]) - (lrm->hyp(lrm->params, get_vec(x->ncols,x->val[i])));
			updated_params = add(lrm->params, scalarp(lr * diff, get_vec(x->ncols, x->val[i])));
			lrm->params = updated_params;
			printf("LOSS: %lf\n", lrm->cost_f(lrm));
		}
	}
}

LRModel* init_linRM(Matrix* params, Matrix* tr_data, Matrix* targets, double lr, int epoch_num) {
	LRModel *lrm = (LRModel*)malloc(sizeof(LRModel));
	lrm->params = params;
	lrm->tr_data = tr_data;
	lrm->targets = targets;
	lrm->lr = lr;
	lrm->epoch_num = epoch_num;
	lrm->run = sgd;
	lrm->hyp = lin_hyp;
	lrm->cost_f = linRM_cost;
	return lrm;
}

LRModel* init_logRM(Matrix* params, Matrix* tr_data, Matrix* targets, double lr, int epoch_num) {
	LRModel *lrm = (LRModel*)malloc(sizeof(LRModel));
	lrm->params = params;
	lrm->tr_data = tr_data;
	lrm->targets = targets;
	lrm->lr = lr;
	lrm->epoch_num = epoch_num;
	lrm->run = sgd;
	lrm->hyp = log_hyp;
	lrm->cost_f = logRM_cost;
	return lrm;
}
