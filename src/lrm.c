#include "lrm.h"
double hyp(Matrix* params, Matrix* features) {
	return dotp(params, features)->val[0][0];
}

double cost_func(LinRegModel* lrm) {
	Matrix* x = transpose(lrm->tr_data);
	double sum = 0;
	for (int i = 0; i < x->nrows; i++) {
		double a = hyp(lrm->params, get_vec(x->ncols, x->val[i])) - lrm->targets->val[i][0];
		sum += (a*a);
	}
	return sum / 2;
}

void sgd(LinRegModel* lrm) {
	Matrix* x = transpose(lrm->tr_data); //feature matrix
	Matrix* updated_params;
	double lr = lrm->lr;
	for (int e = 0; e < lrm->epoch; e++) {
		for (int i = 0; i < lrm->tr_data->ncols; i++) {
			double diff = (lrm->targets->val[i][0]) - (hyp(lrm->params, get_vec(x->ncols,x->val[i])));
			updated_params = add(lrm->params, scalarp(lr * diff, get_vec(x->ncols, x->val[i])));
			lrm->params = updated_params;
			printf("LOSS: %lf\n", lrm->cost_f(lrm));
		}
	}
}

LinRegModel* init_lrm(Matrix* params, Matrix* tr_data, Matrix* targets, double lr, int epoch) {
	LinRegModel *lrm = (LinRegModel*)malloc(sizeof(LinRegModel));
	lrm->params = params;
	lrm->tr_data = tr_data;
	lrm->targets = targets;
	lrm->lr = lr;
	lrm->epoch = epoch;
	lrm->run = sgd;
	lrm->cost_f = cost_func;
	return lrm;
}