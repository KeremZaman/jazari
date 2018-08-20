#include "matrix.h"

struct lrm {
	Matrix* params;
	Matrix* tr_data;
	Matrix* targets;
	int epoch;
	double lr; //learning rate
	void(*run) (struct lrm*); //function pointer to run
	double(*cost_f) (struct lrm*);
	double(*hyp) (void); //hyphothesis
};
typedef struct lrm LinRegModel;

double hyp(Matrix* params, Matrix* features);
void sgd(LinRegModel* lrm);
LinRegModel* init_lrm(Matrix* params, Matrix* tr_data, Matrix* targets, double lr, int epoch);
