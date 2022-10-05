#pragma once
#ifndef PEFLCOMMON_H
#define PEFLCOMMON_H

#include <vector>
#include <iostream>
#include <string>

#include <math.h>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

typedef int64_t Int;
typedef uint64_t uInt;

namespace py = pybind11;
using namespace std;
PYBIND11_MAKE_OPAQUE(std::vector<Int>);
PYBIND11_MAKE_OPAQUE(std::vector<uInt>);

#define CLIENT_NUM 20

#define ROLE_SP0 0
#define ROLE_SP1 1
#define ROLE_CP 2
#define ROLE_CLIENTS 3

#define CP_PORT 8000
#define SP0_PORT 8001
#define SP1_PORT 8002
#endif