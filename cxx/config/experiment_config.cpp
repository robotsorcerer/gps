/**
 * @file experiment_config.cpp
 * @brief Implementation of experiment configuration factory methods.
 */

#include "experiment_config.h"

#include <cmath>

namespace gps {
namespace config {

ExperimentConfig ExperimentConfig::create_box2d_mdgps(
    double gamma, const std::string& mode) {

    ExperimentConfig config;

    // Common settings
    config.common.experiment_name = "box2d_mdgps_gamma" + std::to_string(gamma);
    config.common.conditions = 4;
    config.common.mode = mode;
    config.common.gamma = gamma;

    // Sensor dimensions for Box2D arm
    config.agent.sensor_dims.joint_angles = 2;
    config.agent.sensor_dims.joint_velocities = 2;
    config.agent.sensor_dims.end_effector_points = 3;
    config.agent.sensor_dims.action = 2;

    // Agent settings
    config.agent.type = "AgentBox2D";
    config.agent.target_state = Vector::Zero(2);
    config.agent.world = "ArmWorld";
    config.agent.rk = 0;
    config.agent.dt = 0.05;
    config.agent.substeps = 1;
    config.agent.conditions = 4;
    config.agent.T = 100;

    // Initial states for 4 conditions
    config.agent.x0.resize(4);
    config.agent.x0[0] = Vector::Zero(7);
    config.agent.x0[0](0) = 0.5 * M_PI;

    config.agent.x0[1] = Vector::Zero(7);
    config.agent.x0[1](0) = 0.75 * M_PI;
    config.agent.x0[1](1) = 0.5 * M_PI;

    config.agent.x0[2] = Vector::Zero(7);
    config.agent.x0[2](0) = M_PI;
    config.agent.x0[2](1) = -0.5 * M_PI;

    config.agent.x0[3] = Vector::Zero(7);
    config.agent.x0[3](0) = 1.25 * M_PI;

    config.agent.state_include = {
        SampleType::JOINT_ANGLES,
        SampleType::JOINT_VELOCITIES,
        SampleType::END_EFFECTOR_POINTS
    };
    config.agent.obs_include = config.agent.state_include;

    // Algorithm settings
    config.algorithm.type = "AlgorithmMDGPS";
    config.algorithm.conditions = 4;
    config.algorithm.iterations = 10;

    config.algorithm.lg_step_schedule = Vector(4);
    config.algorithm.lg_step_schedule << 1e-4, 1e-3, 1e-2, 1e-2;

    config.algorithm.policy_dual_rate = 0.2;

    config.algorithm.ent_reg_schedule = Vector(4);
    config.algorithm.ent_reg_schedule << 1e-3, 1e-3, 1e-2, 1e-1;

    config.algorithm.fixed_lg_step = 3;
    config.algorithm.kl_step = 5.0;
    config.algorithm.min_step_mult = 0.01;
    config.algorithm.max_step_mult = 1.0;
    config.algorithm.sample_decrease_var = 0.05;
    config.algorithm.sample_increase_var = 0.1;

    // Initial trajectory distribution
    config.algorithm.init_traj_distr.type = "init_lqr";
    config.algorithm.init_traj_distr.init_gains = Vector::Zero(2);
    config.algorithm.init_traj_distr.init_acc = Vector::Zero(2);
    config.algorithm.init_traj_distr.init_var = 0.1;
    config.algorithm.init_traj_distr.stiffness = 0.01;
    config.algorithm.init_traj_distr.dt = config.agent.dt;
    config.algorithm.init_traj_distr.T = config.agent.T;

    // Cost configuration
    config.algorithm.cost.action_cost.wu = Vector::Ones(2);
    config.algorithm.cost.action_cost.gamma = gamma;
    config.algorithm.cost.action_cost.mode = mode;

    StateCostDataConfig joint_angles_cost;
    joint_angles_cost.wp = Vector::Ones(2);
    joint_angles_cost.target_state = config.agent.target_state;
    config.algorithm.cost.state_cost.data_types[SampleType::JOINT_ANGLES] = joint_angles_cost;
    config.algorithm.cost.state_cost.mode = mode;
    config.algorithm.cost.state_cost.gamma = gamma;

    config.algorithm.cost.action_weight = 1e-5;
    config.algorithm.cost.state_weight = 1.0;
    config.algorithm.cost.mode = mode;
    config.algorithm.cost.gamma = gamma;

    // Dynamics
    config.algorithm.dynamics.type = "DynamicsLRPrior";
    config.algorithm.dynamics.regularization = 1e-6;
    config.algorithm.dynamics.prior.type = "DynamicsPriorGMM";
    config.algorithm.dynamics.prior.max_clusters = 20;
    config.algorithm.dynamics.prior.min_samples_per_cluster = 40;
    config.algorithm.dynamics.prior.max_samples = 20;

    // Trajectory optimization
    config.algorithm.traj_opt.type = "TrajOptLQRPython";

    // Policy prior
    config.algorithm.policy_prior.type = "PolicyPriorGMM";
    config.algorithm.policy_prior.max_clusters = 20;
    config.algorithm.policy_prior.min_samples_per_cluster = 40;
    config.algorithm.policy_prior.max_samples = 20;

    // Top-level settings
    config.iterations = 10;
    config.num_samples = 5;
    config.verbose_trials = 5;
    config.verbose_policy_trials = 0;
    config.gui_on = true;

    return config;
}

ExperimentConfig ExperimentConfig::create_mjc_mdgps(
    double gamma, const std::string& mode) {

    // Start with Box2D config as base
    ExperimentConfig config = create_box2d_mdgps(gamma, mode);

    // Override for MuJoCo
    config.common.experiment_name = "mjc_mdgps_gamma" + std::to_string(gamma);
    config.agent.type = "AgentMuJoCo";
    config.agent.world = "MuJoCoWorld";

    // MuJoCo typically has higher-dimensional state
    config.agent.sensor_dims.joint_angles = 7;
    config.agent.sensor_dims.joint_velocities = 7;
    config.agent.sensor_dims.end_effector_points = 6;
    config.agent.sensor_dims.action = 7;

    // Update related parameters
    config.algorithm.init_traj_distr.init_gains = Vector::Zero(7);
    config.algorithm.init_traj_distr.init_acc = Vector::Zero(7);
    config.algorithm.cost.action_cost.wu = Vector::Ones(7);

    return config;
}

}  // namespace config
}  // namespace gps
