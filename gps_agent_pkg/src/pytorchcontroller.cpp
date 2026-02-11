#include "gps_agent_pkg/pytorchcontroller.h"
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/util.h"

#include <ros/ros.h>

using namespace gps_control;

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

PyTorchController::PyTorchController()
    : TrialController()
{
    is_configured_ = false;
}

PyTorchController::~PyTorchController() = default;

// ---------------------------------------------------------------------------
// configure_controller
// ---------------------------------------------------------------------------

void PyTorchController::configure_controller(OptionsMap &options)
{
    // Let the base class handle T, state/obs datatypes, ee_tgt, etc.
    TrialController::configure_controller(options);

    // ---- Load TorchScript model from raw bytes -------------------------
    const std::string model_bytes =
        std::get<std::string>(options.at("model_bytes"));

    try {
        std::istringstream stream(model_bytes);
        module_ = torch::jit::load(stream, torch::kCPU);
        module_.eval();
        module_loaded_ = true;
    } catch (const c10::Error &e) {
        ROS_ERROR("PyTorchController: failed to load TorchScript model: %s",
                  e.what());
        module_loaded_ = false;
        return;
    }

    // ---- Observation normalisation ------------------------------------
    // scale is transmitted as the diagonal of the scale matrix (length dO).
    scale_diag_ = std::get<Eigen::VectorXd>(options.at("scale"));
    bias_        = std::get<Eigen::VectorXd>(options.at("bias"));

    // ---- Per-timestep noise ------------------------------------------
    int T = std::get<int>(options.at("T"));
    noise_.resize(T);
    for (int t = 0; t < T; ++t) {
        noise_[t] = std::get<Eigen::VectorXd>(options.at("noise_" + std::to_string(t)));
    }

    // Infer dU from the first noise vector.
    dU_ = (T > 0) ? static_cast<int>(noise_[0].size()) : 0;

    ROS_INFO_STREAM("PyTorchController: loaded TorchScript model, dU=" << dU_
                    << ", T=" << T);
    is_configured_ = true;
}

// ---------------------------------------------------------------------------
// get_action
// ---------------------------------------------------------------------------

void PyTorchController::get_action(int t,
                                   const Eigen::VectorXd & /*X*/,
                                   const Eigen::VectorXd &obs,
                                   Eigen::VectorXd &U)
{
    if (!is_configured_ || !module_loaded_) {
        // Safety: output zero torques until we are properly configured.
        if (dU_ > 0) U.setZero(dU_);
        return;
    }

    const int dO = static_cast<int>(obs.size());

    // ---- Apply observation normalisation ----
    // obs_scaled = obs .* scale_diag + bias
    // scale_diag_ stores the diagonal of the scale matrix (pointwise multiply).
    Eigen::VectorXd obs_scaled(dO);
    for (int i = 0; i < dO; ++i) {
        obs_scaled(i) = obs(i) * scale_diag_(i) + bias_(i);
    }

    // ---- Build input tensor [1, dO] on CPU ----
    auto options_t = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::zeros({1, dO}, options_t);
    {
        auto acc = input.accessor<float, 2>();
        for (int i = 0; i < dO; ++i) {
            acc[0][i] = static_cast<float>(obs_scaled(i));
        }
    }

    // ---- Forward pass (no_grad, CPU, eval mode) ----
    torch::Tensor output;
    try {
        torch::NoGradGuard no_grad;
        output = module_.forward({input}).toTensor();
    } catch (const c10::Error &e) {
        ROS_ERROR_THROTTLE(1.0, "PyTorchController: forward pass failed: %s",
                           e.what());
        U.setZero(dU_);
        return;
    }

    // ---- Unpack output [1, dU] → U ----
    U.resize(dU_);
    {
        auto acc = output.accessor<float, 2>();
        for (int i = 0; i < dU_; ++i) {
            U(i) = static_cast<double>(acc[0][i]);
        }
    }

    // ---- Add pre-sampled noise ----
    if (t >= 0 && t < static_cast<int>(noise_.size())) {
        U += noise_[t];
    }
}
