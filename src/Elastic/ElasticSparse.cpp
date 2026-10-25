#include "ElasticRun.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdexcept>

namespace elastic {
void AppendBlock(Entries &entries, const Mat &block, int row, int col) {
    for (int r = 0; r < block.rows(); ++r)
        for (int c = 0; c < block.cols(); ++c)
            if (block(r, c) != 0) entries.emplace_back(row + r, col + c, block(r, c));
}
Sparse MakeSparse(int rows, int cols, const Entries &entries) {
    Sparse result(rows, cols);
    result.setFromTriplets(entries.begin(), entries.end());
    result.makeCompressed();
    return result;
}
double SpectralNorm(Mat matrix) {
    if (!matrix.allFinite()) throw std::runtime_error("Nonfinite CFL operator block");
    __LAPACK_int rows = matrix.rows(), cols = matrix.cols(), one = 1, info = 0, size = -1;
    Vec values(std::min(rows, cols));
    double query, unused;
    dgesvd_("N", "N", &rows, &cols, matrix.data(), &rows, values.data(), &unused, &one, &unused, &one, &query, &size, &info);
    if (info) throw std::runtime_error("CFL singular-value workspace query failed");
    size = __LAPACK_int(query);
    Vec work(size);
    dgesvd_("N", "N", &rows, &cols, matrix.data(), &rows, values.data(), &unused, &one, &unused, &one, work.data(), &size, &info);
    if (info || !values.allFinite()) throw std::runtime_error("CFL singular-value computation failed");
    return values[0];
}
Vec Model::Apply(const Vec &state) const {
    if (state.size() != Size) throw std::invalid_argument("Wrong state size");
    if (System.rows() != Size) throw std::logic_error("CPU reference operator was not assembled");
    Vec result(Size);
    SparseAttributes_t attributes{};
    attributes.transpose = true;
    const SparseMatrix_Double matrix{{Size, Size, const_cast<long *>(RowStarts.data()), const_cast<int *>(System.innerIndexPtr()), attributes, 1}, const_cast<double *>(System.valuePtr())};
    SparseMultiply(matrix, DenseVector_Double{Size, const_cast<double *>(state.data())}, DenseVector_Double{Size, result.data()});
    return result;
}
} // namespace elastic
