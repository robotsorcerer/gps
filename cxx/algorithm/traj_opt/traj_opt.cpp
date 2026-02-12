/**
 * @file traj_opt.cpp
 * @brief Implementation of TrajOpt base class.
 */

#include "traj_opt.h"

#include <stdexcept>

namespace gps {

TrajOpt::TrajOpt(const Hyperparams& hyperparams)
    : hyperparams_(hyperparams) {
}

TrajOptResultRobust TrajOpt::update_robust(
    int /*condition*/, const Algorithm& /*algorithm*/) {
    throw std::runtime_error(
        "update_robust not implemented for this TrajOpt type");
}

}  // namespace gps
