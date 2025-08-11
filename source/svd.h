#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include "structs.h"
#include "log.h"
#include <chrono>
#include <omp.h>

namespace spm {
  SVD recursive_svd(const PMatrix & A, double tol = -1);
  SVD centrosymmetric_matrix_svd(const LMatrix & A, double tol = -1);
  PMatrix get_j_matrix(int n);
  PVector symmetric_linspace(int n, PScalar max, PScalar offset = 0.0);
  void test_centrosymmetric();
}
