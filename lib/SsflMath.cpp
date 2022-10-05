#include "SsflMath.h"

vector<Int> vector_add(vector<Int> X, vector<Int> garb)
{
    vector<Int> Y;
    for(int i = 0; i < X.size(); i++)
    {
        Y.push_back(X[i] + garb[i]);
    }
    return Y;
}

vector<Int> vector_sub(vector<Int> X, vector<Int> garb)
{
    vector<Int> Y;
    for(int i = 0; i < X.size(); i++)
    {
        Y.push_back(Int(X[i] - garb[i]));
    }
    return Y;
}

vector<Int> get_garble(int lenth)
{
    vector<Int> Res;
    default_random_engine e;
    uniform_int_distribution<Int> u(0, 184467440737);
    for(int i = 0; i < lenth; i++)
    {
        Res.push_back(Int(u(e)));
    }
    return Res;
}

Int vector_mul(vector<Int> &X, vector<Int> &Y)
{
    Int Res = 0;
    for(int i = 0; i < X.size(); i++)
    {
        Res += X[i] * Y[i];
    }
    return Res;
}

vector<Int> vecmat_mul(vector<Int> &X, vector<Int> &Y)
{
    vector<Int> Res;
    int xl = X.size();
    int yl = Y.size();
    int Rdim;
    Rdim = yl / xl;

    for(int i = 0; i < Rdim; i++)
    {
        vector<Int> temp(Y.begin() + (i * xl), Y.begin() + (i * xl) + xl);
        Res.push_back(vector_mul(X, temp));
    }

    return Res;
}

vector<Int> vecnum_sub(vector<Int> X, Int num)
{
    vector<Int> Res;
    for(uInt i = 0; i < X.size(); i++)
    {
        Res.push_back(Int(X[i]- num));
    }
    return Res;
}

vector<Int> vecflo_mul(vector<Int> &X, double Y)
{
    vector<Int> Res;
    for(uInt i = 0; i < X.size(); i++)
    {
        Res.push_back(X[i] * Y);
    }
    return Res;
}