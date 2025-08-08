#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include "structs.h"
#include "log.h"

namespace spm {
  SVD recursive_svd(const LMatrix & A, double tol = -1);
  SVD centrosymmetric_matrix_svd(const LMatrix & A, double tol = -1);
  LMatrix get_j_matrix(int n);
  Vector symmetric_linspace(int n, Scalar max, Scalar offset = 0.0);
  void test_centrosymmetric();
}
