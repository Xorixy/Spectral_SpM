#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include "structs.h"
#include "log.h"

namespace spm {
  SVD recursive_svd(const Eigen::MatrixXd& M, double tol = -1);
}
