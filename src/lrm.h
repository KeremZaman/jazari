#include "matrix.h"

struct lrm {
	Matrix* params;
	Matrix* tr_data;
	Matrix* targets;
	int epoch_num;
	double lr; //learning rate
	void(*run) (struct lrm*); //function pointer to run
	double(*cost_f) (struct lrm*);
	double(*hyp) (Matrix* params, Matrix* features); //hyphothesis
};
typedef struct lrm LRModel;

double lin_hyp(Matrix* params, Matrix* features);
double log_hyp(Matrix* params, Matrix* features);
double linRM_cost(LRModel* lrm);
double logRM_cost(LRModel* lrm);
void sgd(LRModel* lrm);
LRModel* init_linRM(Matrix* params, Matrix* tr_data, Matrix* targets, double lr, int epoch_num);
LRModel* init_logRM(Matrix* params, Matrix* tr_data, Matrix* targets, double lr, int epoch_num);
