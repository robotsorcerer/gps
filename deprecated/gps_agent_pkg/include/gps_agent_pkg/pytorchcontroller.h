/*
Controller that executes a trial using a TorchScript neural-network policy.

The Python side exports the trained model with:
    torch.jit.script(net)  →  model_bytes (via BytesIO / save())

The bytes are shipped over ROS in TorchParams.model_bytes and loaded here
with torch::jit::load() from a std::istringstream, so no filesystem I/O is
required at inference time.

Input normalisation (scale / bias) is applied in C++ to match the Python
training normalisation:
    obs_scaled = obs @ scale + bias

Pre-sampled noise vectors for each timestep are transmitted in
TorchParams.noise (shape [T * dU], row-major) to keep the C++ controller
free of any random-number generation.
*/
#pragma once

// Standard library
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Eigen
#include <Eigen/Dense>

// LibTorch (must come before ROS / other headers to avoid symbol conflicts)
#include <torch/script.h>

// GPS superclass
#include "gps_agent_pkg/trialcontroller.h"

namespace gps_control
{

class PyTorchController : public TrialController
{
private:
    // Loaded TorchScript module.
    torch::jit::Module module_;
    // Whether module_ has been loaded successfully.
    bool module_loaded_ = false;

    // Observation normalisation: obs_scaled = obs * scale_diag_ + bias_
    // scale_diag_ is a diagonal matrix stored as a vector (length dO).
    Eigen::VectorXd scale_diag_;
    Eigen::VectorXd bias_;

    // Pre-sampled noise vectors for each timestep.  noise_[t] has length dU.
    std::vector<Eigen::VectorXd> noise_;

    // Action dimension (dU), inferred from the noise array.
    int dU_ = 0;

public:
    // Constructor / destructor.
    PyTorchController();
    virtual ~PyTorchController();

    // Compute the action at the current time step.
    //   t   — current timestep index
    //   X   — full state vector  (unused; obs contains the normalised subset)
    //   obs — observation vector
    //   U   — output action vector (written by this function)
    virtual void get_action(int t, const Eigen::VectorXd &X,
                            const Eigen::VectorXd &obs, Eigen::VectorXd &U);

    // Configure the controller from an OptionsMap populated by robotplugin.cpp.
    // Expected keys:
    //   "model_bytes" (std::string) — raw TorchScript bytes
    //   "scale"       (Eigen::VectorXd) — diagonal of the scale matrix (length dO)
    //   "bias"        (Eigen::VectorXd) — bias vector (length dO)
    //   "T"           (int)             — trial length
    //   "noise_t"     (Eigen::VectorXd) — noise for timestep t, t=0..T-1
    virtual void configure_controller(OptionsMap &options);
};

}  // namespace gps_control
