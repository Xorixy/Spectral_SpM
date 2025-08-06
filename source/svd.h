#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include "structs.h"
#include "log.h"
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>

namespace spm {
  SVD recursive_svd(const Matrix & A, double tol = -1);
  MPSVD recursive_svd(const MPMatrix & A, double tol = -1);
}
