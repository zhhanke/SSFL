#pragma once
#ifndef PEFLMATH_H
#define PEFLMATH_H

#include "SsflCommon.h"

vector<Int> vector_add(vector<Int> X, vector<Int> garb);
vector<Int> vector_sub(vector<Int> X, vector<Int> garb);
vector<Int> get_garble(int lenth);

vector<Int> vecnum_sub(vector<Int> X, Int num);
vector<Int> vecflo_mul(vector<Int> &X, double Y);
vector<Int> vecmat_mul(vector<Int> &X, vector<Int> &Y);
Int vector_mul(vector<Int> &X, vector<Int> &Y);

#endif