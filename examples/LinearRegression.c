#include <stdio.h>
#include "lrm.h"
int main()
{
	/* Consumption of petrol dataset from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x16.txt */
	double tr_features[][4] = {
		{ 9.00,3571,1976,0.5250 },
		{ 9.00,4092,1250,0.5720 },
		{ 9.00,3865,1586,0.5800 },
		{ 7.50,4870,2351,0.5290 },
		{ 8.00,4399,431,0.5440 },
		{ 10.00,5342,1333,0.5710},
		{ 8.00,5319,11868,0.4510 },
		{ 8.00,5126,2138,0.5530 },
		{ 8.00,4447,8577,0.5290 },
		{ 7.00,4512,8507,0.5520 },
		{ 8.00,4391,5939,0.5300 },
		{ 7.50,5126,14186,0.5250 },
		{ 7.00,4817,6930,0.5740 },
		{ 7.00,4207,6580,0.5450 },
		{ 7.00,4332,8159,0.6080 },
		{ 7.00,4318,10340,0.5860 },
		{ 7.00,4206,8508,0.5720 },
		{ 7.00,3718,4725,0.5400 },
		{ 7.00,4716,5915,0.7240 },
		{ 8.50,4341,6010,0.6770 },
		{ 7.00,4593,7834,0.6630 },
		{ 8.00,4983,602,0.6020  },
		{ 9.00,4897,2449,0.5110 },
		{ 9.00,4258,4686,0.5170 },
		{ 8.50,4574,2619,0.5510 },
		{ 9.00,3721,4746,0.5440 },
		{ 8.00,3448,5399,0.5480 },
		{ 7.50,3846,9061,0.5790 },
		{ 8.00,4188,5975,0.5630 },
		{ 9.00,3601,4650,0.4930 }
	};

	double test_features[][4] = {
		{ 7.00,3640,6905,0.5180 },
		{ 7.00,3333,6594,0.5130 },
		{ 8.00,3063,6524,0.5780 },
		{ 7.50,3357,4121,0.5470 },
		{ 8.00,3528,3495,0.4870 },
		{ 6.58,3802,7834,0.6290 },
		{ 5.00,4045,17782,0.5660 },
		{ 7.00,3897,6385,0.5860 },
		{ 8.50,3635,3274,0.6630 },
		{ 7.00,4345,3905,0.6720 },
		{ 7.00,4449,4639,0.6260 },
		{ 7.00,3656,3985,0.5630 },
		{ 7.00,4300,3635,0.6030 },
		{ 7.00,3745,2611,0.5080 },
		{ 6.00,5215,2302,0.6720 },
		{ 9.00,4476,3942,0.5710 },
		{ 7.00,4296,4083,0.6230 },
		{ 7.00,5002,9794,0.5930 }
	};

	double tr_outputs[][30] = { {541,524,561,414,410,457,344,467,464,498,580,471,525,508,566,635,603,714,865,640,649,540,464,547,460,566,577,631,574,534} };
	double test_outputs[][18] = { {571,554,577,628,487,644,640,704,648,968,587,699,632,591,782,510,610,524} };
	
	Matrix* training_data = arr2matrix(4,30, tr_features);
	Matrix* targets = arr2matrix(30, 1, tr_outputs);
	Matrix* test_data = arr2matrix(4, 18, test_features);
	Matrix* test_targets = arr2matrix(18, 1, test_outputs);
	
	Matrix* params = mcreate(4, 1);
	LRModel* lrm = init_linRM(params, training_data, targets, 0.003, 100);
	lrm->run(lrm);
	
	printf("\nDATASET OUTPUTS:\n");
	for (int i = 0; i < 18; i++) {
		printf("%lf\t", test_outputs[0][i]);
	}
	printf("\nPREDICTED OUTPUTS\n");
	for (int i = 0; i < 18; i++) {
		double y = lrm->hyp(lrm->params, arr2matrix(18, 1, test_features[i]));
		printf("%lf\t", y);
	}
	
	getch();
	return 0;
}
