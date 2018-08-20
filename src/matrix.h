struct matrix {
	int nrows;
	int ncols;
	double** val;
};
typedef struct matrix Matrix;

void error(char* str);
Matrix* mcreate(int m, int n);
Matrix* arr2matrix(int m, int n, double* arr);
Matrix* mdiag(int m, int n, double val);
Matrix* ident(int n);
Matrix* transpose(Matrix* mtrx);
Matrix* add(Matrix* m1, Matrix* m2);
Matrix* dotp(Matrix* m1, Matrix* m2);
Matrix* scalarp(double scalar, Matrix* mtrx);
void print_matrix(Matrix* mtrx);
