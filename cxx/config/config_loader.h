/**
 * @file config_loader.h
 * @brief YAML/JSON configuration file loader.
 *
 * Provides utilities to load experiment configurations from external files.
 */

#ifndef GPS_CXX_CONFIG_CONFIG_LOADER_H_
#define GPS_CXX_CONFIG_CONFIG_LOADER_H_

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>
#include <vector>

#include "experiment_config.h"

namespace gps {
namespace config {

/**
 * @brief Simple key-value configuration parser.
 *
 * Parses simple YAML-like configuration files with format:
 *   key: value
 *   nested:
 *     key: value
 */
class ConfigLoader {
public:
    /**
     * @brief Load configuration from file.
     *
     * @param filepath Path to configuration file
     * @return ExperimentConfig Parsed configuration
     * @throws std::runtime_error if file cannot be read or parsed
     */
    static ExperimentConfig load(const std::string& filepath);

    /**
     * @brief Load configuration from string.
     *
     * @param content Configuration string content
     * @return ExperimentConfig Parsed configuration
     */
    static ExperimentConfig parse(const std::string& content);

    /**
     * @brief Save configuration to file.
     *
     * @param config Configuration to save
     * @param filepath Output file path
     */
    static void save(const ExperimentConfig& config, const std::string& filepath);

    /**
     * @brief Generate YAML string from configuration.
     *
     * @param config Configuration to serialize
     * @return std::string YAML representation
     */
    static std::string to_yaml(const ExperimentConfig& config);

private:
    /**
     * @brief Parse a double value from string.
     */
    static double parse_double(const std::string& value);

    /**
     * @brief Parse an integer value from string.
     */
    static int parse_int(const std::string& value);

    /**
     * @brief Parse a vector from string format "[1.0, 2.0, 3.0]".
     */
    static Vector parse_vector(const std::string& value);

    /**
     * @brief Trim whitespace from string.
     */
    static std::string trim(const std::string& str);
};

/**
 * @brief Pre-defined experiment configurations.
 *
 * Provides factory methods for common experiment setups matching
 * the Python hyperparams.py files.
 */
class ExperimentPresets {
public:
    // Box2D experiments
    static ExperimentConfig box2d_arm_example();
    static ExperimentConfig box2d_pointmass_example();
    static ExperimentConfig box2d_mdgps_protagonist();

    // Box2D MDGPS with various gamma values
    static ExperimentConfig box2d_mdgps_y1e0();    // gamma = 1e0
    static ExperimentConfig box2d_mdgps_y1e2();    // gamma = 1e2
    static ExperimentConfig box2d_mdgps_y1e4();    // gamma = 1e4
    static ExperimentConfig box2d_mdgps_y1e6();    // gamma = 1e6
    static ExperimentConfig box2d_mdgps_y1e8();    // gamma = 1e8
    static ExperimentConfig box2d_mdgps_y1e_2();   // gamma = 1e-2
    static ExperimentConfig box2d_mdgps_y1e_4();   // gamma = 1e-4
    static ExperimentConfig box2d_mdgps_y1e_6();   // gamma = 1e-6
    static ExperimentConfig box2d_mdgps_y1e_8();   // gamma = 1e-8
    static ExperimentConfig box2d_mdgps_y0_5();    // gamma = 0.5
    static ExperimentConfig box2d_mdgps_y1_5();    // gamma = 1.5

    // MuJoCo experiments
    static ExperimentConfig mjc_example();
    static ExperimentConfig mjc_mdgps_example();
    static ExperimentConfig mjc_mdgps_idg();

    // MuJoCo MDGPS antagonist with various gamma values
    static ExperimentConfig mjc_mdgps_antagonist_y0_5();
    static ExperimentConfig mjc_mdgps_antagonist_y1();
    static ExperimentConfig mjc_mdgps_antagonist_y2();
    static ExperimentConfig mjc_mdgps_antagonist_y3();
    static ExperimentConfig mjc_mdgps_antagonist_y5();
    static ExperimentConfig mjc_mdgps_antagonist_y7();
    static ExperimentConfig mjc_mdgps_antagonist_y1e_1();
    static ExperimentConfig mjc_mdgps_antagonist_y1e_2();
    static ExperimentConfig mjc_mdgps_antagonist_y1e_4();
    static ExperimentConfig mjc_mdgps_antagonist_y1e_5();
    static ExperimentConfig mjc_mdgps_antagonist_y1e_6();
    static ExperimentConfig mjc_mdgps_antagonist_y1e8();
    static ExperimentConfig mjc_mdgps_antagonist_y1e10();

    /**
     * @brief Get preset by name.
     *
     * @param name Preset name (e.g., "box2d_mdgps_y1e0", "mjc_mdgps_antagonist_y5")
     * @return ExperimentConfig Preset configuration
     * @throws std::invalid_argument if preset not found
     */
    static ExperimentConfig get(const std::string& name);

    /**
     * @brief List all available preset names.
     */
    static std::vector<std::string> list();
};

}  // namespace config
}  // namespace gps

#endif  // GPS_CXX_CONFIG_CONFIG_LOADER_H_
