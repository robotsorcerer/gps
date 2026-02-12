/**
 * @file dynamics.cpp
 * @brief Implementation of Dynamics classes.
 */

#include "dynamics.h"

namespace gps {

// ============================================================================
// LinearDynamics Implementation
// ============================================================================

LinearDynamics LinearDynamics::zeros(int T, int dX, int dU) {
    LinearDynamics dyn;

    dyn.Fm.resize(T);
    dyn.dyn_covar.resize(T);

    for (int t = 0; t < T; ++t) {
        dyn.Fm[t] = Matrix::Zero(dX, dX + dU);
        dyn.dyn_covar[t] = Matrix::Zero(dX, dX);
    }

    dyn.fv = Matrix::Zero(T, dX);

    return dyn;
}

Vector LinearDynamics::predict(const Vector& x, const Vector& u, int t) const {
    // Concatenate [x; u]
    const int dX = static_cast<int>(x.size());
    const int dU = static_cast<int>(u.size());

    Vector xu(dX + dU);
    xu.head(dX) = x;
    xu.tail(dU) = u;

    // x_{t+1} = Fm * [x; u] + fv
    return Fm[t] * xu + fv.row(t).transpose();
}

// ============================================================================
// Dynamics Implementation
// ============================================================================

Dynamics::Dynamics(const Hyperparams& hyperparams)
    : hyperparams_(hyperparams) {
}

}  // namespace gps
