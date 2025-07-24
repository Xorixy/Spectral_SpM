#include "admm.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

spm::Vector spm::positive_projection(Vector &input) {
    Vector output(input.size());
    for (int i = 0; i < input.size(); ++i) {
        output(i) = input(i) * (input(i) > 0);
    }
    return output;
}

spm::Vector spm::soft_threshold(Vector &input, double threshold) {
    Vector output(input.size());
    for (int i = 0; i < input.size(); ++i) {
        double positive = input(i) - threshold;
        double negative = -input(i) - threshold;
        output(i) = positive * (positive > 0) - negative * (negative > 0);
    }
    return output;
}
