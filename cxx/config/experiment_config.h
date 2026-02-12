/**
 * @file experiment_config.h
 * @brief C++ experiment configuration structures.
 *
 * Replaces Python hyperparams.py files with type-safe C++ configuration.
 * Supports YAML/JSON loading for experiment parameters.
 */

#ifndef GPS_CXX_CONFIG_EXPERIMENT_CONFIG_H_
#define GPS_CXX_CONFIG_EXPERIMENT_CONFIG_H_

#include <string>
#include <vector>
#include <map>
#include <optional>

#include "../utility/types.h"

namespace gps {
namespace config {

/**
 * @brief Sensor dimension configuration.
 */
struct SensorDims {
    int joint_angles = 2;
    int joint_velocities = 2;
    int end_effector_points = 3;
    int action = 2;

    [[nodiscard]] int state_dim() const {
        return joint_angles + joint_velocities + end_effector_points;
    }
};

/**
 * @brief Common experiment parameters.
 */
struct CommonConfig {
    std::string experiment_name;
    std::string experiment_dir;
    std::string data_files_dir;
    std::string log_filename;
    std::string costs_filename;
    int conditions = 4;
    std::string mode = "protagonist";  // "protagonist", "antagonist", "robust"
    double gamma = 1.0;
};

/**
 * @brief Agent configuration.
 */
struct AgentConfig {
    std::string type = "AgentBox2D";
    Vector target_state;
    std::string world = "ArmWorld";
    std::vector<Vector> x0;  // Initial states per condition
    int rk = 0;
    double dt = 0.05;
    int substeps = 1;
    int conditions = 4;
    int T = 100;  // Time horizon
    SensorDims sensor_dims;
    std::vector<SampleType> state_include;
    std::vector<SampleType> obs_include;
};

/**
 * @brief Initial trajectory distribution parameters.
 */
struct InitTrajDistrConfig {
    std::string type = "init_lqr";
    Vector init_gains;
    Vector init_acc;
    double init_var = 0.1;
    double stiffness = 0.01;
    double dt = 0.05;
    int T = 100;
};

/**
 * @brief Action cost configuration.
 */
struct ActionCostConfig {
    Vector wu;  // Action weights
    double gamma = 1.0;
    std::string mode = "protagonist";
};

/**
 * @brief State cost configuration for a single data type.
 */
struct StateCostDataConfig {
    Vector wp;  // State weights
    Vector target_state;
};

/**
 * @brief State cost configuration.
 */
struct StateCostConfig {
    std::map<SampleType, StateCostDataConfig> data_types;
    std::string mode = "protagonist";
    double gamma = 1.0;
};

/**
 * @brief Combined cost configuration.
 */
struct CostSumConfig {
    ActionCostConfig action_cost;
    StateCostConfig state_cost;
    double action_weight = 1e-5;
    double state_weight = 1.0;
    std::string mode = "protagonist";
    double gamma = 1.0;
};

/**
 * @brief Dynamics prior configuration.
 */
struct DynamicsPriorConfig {
    std::string type = "DynamicsPriorGMM";
    int max_clusters = 20;
    int min_samples_per_cluster = 40;
    int max_samples = 20;
};

/**
 * @brief Dynamics configuration.
 */
struct DynamicsConfig {
    std::string type = "DynamicsLRPrior";
    double regularization = 1e-6;
    DynamicsPriorConfig prior;
};

/**
 * @brief Trajectory optimization configuration.
 */
struct TrajOptConfig {
    std::string type = "TrajOptLQRPython";
};

/**
 * @brief Policy optimization configuration.
 */
struct PolicyOptConfig {
    std::string type = "PolicyOptCaffe";
    std::string weights_file_prefix;
};

/**
 * @brief Policy prior configuration.
 */
struct PolicyPriorConfig {
    std::string type = "PolicyPriorGMM";
    int max_clusters = 20;
    int min_samples_per_cluster = 40;
    int max_samples = 20;
};

/**
 * @brief Algorithm configuration.
 */
struct AlgorithmConfig {
    std::string type = "AlgorithmMDGPS";
    int conditions = 4;
    int iterations = 10;
    Vector lg_step_schedule;
    double policy_dual_rate = 0.2;
    Vector ent_reg_schedule;
    int fixed_lg_step = 3;
    double kl_step = 5.0;
    double min_step_mult = 0.01;
    double max_step_mult = 1.0;
    double sample_decrease_var = 0.05;
    double sample_increase_var = 0.1;

    InitTrajDistrConfig init_traj_distr;
    CostSumConfig cost;
    DynamicsConfig dynamics;
    TrajOptConfig traj_opt;
    PolicyOptConfig policy_opt;
    PolicyPriorConfig policy_prior;
};

/**
 * @brief Complete experiment configuration.
 */
struct ExperimentConfig {
    int iterations = 10;
    int num_samples = 5;
    int verbose_trials = 5;
    int verbose_policy_trials = 0;
    bool gui_on = true;

    CommonConfig common;
    AgentConfig agent;
    AlgorithmConfig algorithm;

    /**
     * @brief Create default Box2D MDGPS configuration.
     *
     * @param gamma Adversarial weight parameter
     * @param mode Cost mode ("protagonist", "antagonist", "robust")
     * @return ExperimentConfig
     */
    static ExperimentConfig create_box2d_mdgps(double gamma, const std::string& mode);

    /**
     * @brief Create default MuJoCo MDGPS configuration.
     *
     * @param gamma Adversarial weight parameter
     * @param mode Cost mode
     * @return ExperimentConfig
     */
    static ExperimentConfig create_mjc_mdgps(double gamma, const std::string& mode);
};

}  // namespace config
}  // namespace gps

#endif  // GPS_CXX_CONFIG_EXPERIMENT_CONFIG_H_
