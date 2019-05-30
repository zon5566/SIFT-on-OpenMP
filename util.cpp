#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "util.hpp"

int inverse_3x3(double mat[][3], double inv_mat[][3]) {

	double determinant = 0.0;
	for (int d = 0; d < 3; d++)
		determinant += mat[0][d] * (mat[1][(d+1)%3] * mat[2][(d+2)%3] - mat[1][(d+2)%3] * mat[2][(d+1)%3]);
	
	if (determinant == 0.0) return -1;

	for (int p = 0; p < 3; p++)
		for (int q = 0; q < 3; q++){
			inv_mat[p][q] = (
					(mat[(q+1)%3][(p+1)%3] * mat[(q+2)%3][(p+2)%3]) - (mat[(q+1)%3][(p+2)%3] * mat[(q+2)%3][(p+1)%3])
				) / determinant;
		}

	return 0;

	/*
	// psuedo inverse
	
	double matT_mat[3][3];
	
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			matT_mat[i][j] = 0;
	

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				matT_mat[i][j] += mat[k][i] * mat[k][j];
			}
		}
	}

	double matT_mat_inv[3][3];
	inverse_3x3(matT_mat, matT_mat_inv);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				pinv_mat[i][j] += matT_mat_inv[i][k] * mat[j][k];
	*/
}

void MM_mul(double** m1, double** m2, double** res, int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			for (int k = 0; k < size; k++){
				res[i][j] += m1[i][k] * m2[k][j];
			}
}

void MM_mul_3x3(double m1[][3], double m2[][3], double res[][3]) {
	int size = 3;
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			for (int k = 0; k < size; k++){
				res[i][j] += m1[i][k] * m2[k][j];
			}
}

int sub2index(double subpoint, int octave) {
	return pow(2, octave) * (subpoint) + pow(2, octave) / 2;
}
