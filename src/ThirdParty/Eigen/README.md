# Eigen 3.4.0 (vendored)

Copied by `script/VendorEigen`, the only supported way to change this directory. Sources
are upstream's, unmodified.

- Release: <https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz>
- SHA256: `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`
- Modules kept: Dense Core LU Cholesky QR SVD Geometry Eigenvalues Householder Jacobi

Everything else is dropped: the sparse solvers, the external library backends, the Qt/STL
container headers, and upstream's `unsupported/`, `doc/`, and `test/` trees.

Eigen is MPL2 (`COPYING.MPL2`). The Intel-authored `*_BLAS.h`, `*_LAPACKE.h`, and
`MKL_support.h` shims are BSD (`COPYING.BSD`), and `src/Core/arch/Default/Half.h` and
`BFloat16.h` are Apache-2.0 (`COPYING.APACHE`).
