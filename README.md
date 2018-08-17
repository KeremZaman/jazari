# jazari
**jazari** is an educational machine learning library for C. Main purpose of the project is implementing some machine learning stuff from basic matrix operations to advanced models in a clean way.

>What I cannot create, I do not understand. - Richard Feynman

**p.s.** It will be avoided to use 3rd-party libraries unless implementation is not related to machine learning concepts.


# Matrix Operations

## Creation
Creating a zero matrix with `i x j` size
 `Matrix* m = mcreate(i,j);`

Converting 2d array to a matrix
```
double arr[2][3] = { {3,4,5},{8,1,5}};
Matrix* m = arr2matrix(3,2,arr);
```
Note that each sub-array corresponds to a vector in matrix. So you must reverse indices while creating matrix.

`A[0][1]` corresponds to A<sub>12</sub> in mathematical notation.

## Diagonal and Identity Matrices
Creating `m x n` diagonal matrix with number `e`

`Matrix* m = mdiag(m,n,e);`

Creating a `n x n` identity matrix

`Matrix* m = ident(n);`

## Transpose, Addition and Dot Product
`Matrix* m = tranpose(n);`

`Matrix* m = add(m1,m2);`

`Matrix* m = dotp(m1,m2);`

m1 and m2 are `Matrix` pointers in the examples above.

## Visualizing Matrix
For now you can print matrix to console by `print_matrix(m);`
