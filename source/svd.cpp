#include "svd.h"


void spm::test_svd() {
    Eigen::MatrixXd C {
        {5, 3},
            {-3.2, 2.1},
            {4.1, 1.0}
    };


    // Calculate SVD using the built in functionality
    Eigen::BDCSVD svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);

    svd.setThreshold(4);

    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto sigma = svd.singularValues().asDiagonal().toDenseMatrix();

    std::cout << "C = \n" << C << "\n\n";

    std::cout << "U = \n" << U << "\n\n";

    std::cout << "sigma = \n" << sigma << "\n\n";

    std::cout << "V = \n" << V << "\n\n";

    std::cout << "U * sigma * VT = \n" << U * sigma * V.transpose() << "\n\n";

    std::cout << "U * sigma = \n" << U * sigma << "\n\n";

    std::cout << "U * V = \n" << U * V << "\n\n";


}