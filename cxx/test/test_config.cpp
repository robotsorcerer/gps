/**
 * @file test_config.cpp
 * @brief Unit tests for experiment configuration system.
 */

#include <gtest/gtest.h>

#include "../config/experiment_config.h"
#include "../config/config_loader.h"

namespace gps {
namespace config {
namespace test {

// =============================================================================
// ExperimentConfig Tests
// =============================================================================

TEST(ExperimentConfigTest, CreateBox2dMdgps) {
    auto config = ExperimentConfig::create_box2d_mdgps(1.0, "antagonist");

    EXPECT_EQ(config.common.mode, "antagonist");
    EXPECT_DOUBLE_EQ(config.common.gamma, 1.0);
    EXPECT_EQ(config.agent.T, 100);
    EXPECT_EQ(config.agent.conditions, 4);
    EXPECT_EQ(config.algorithm.type, "AlgorithmMDGPS");
}

TEST(ExperimentConfigTest, CreateBox2dMdgpsWithDifferentGamma) {
    auto config1 = ExperimentConfig::create_box2d_mdgps(1e-4, "antagonist");
    auto config2 = ExperimentConfig::create_box2d_mdgps(1e8, "antagonist");

    EXPECT_DOUBLE_EQ(config1.common.gamma, 1e-4);
    EXPECT_DOUBLE_EQ(config2.common.gamma, 1e8);
    EXPECT_DOUBLE_EQ(config1.algorithm.cost.gamma, 1e-4);
    EXPECT_DOUBLE_EQ(config2.algorithm.cost.gamma, 1e8);
}

TEST(ExperimentConfigTest, CreateMjcMdgps) {
    auto config = ExperimentConfig::create_mjc_mdgps(5.0, "antagonist");

    EXPECT_EQ(config.agent.type, "AgentMuJoCo");
    EXPECT_EQ(config.agent.sensor_dims.joint_angles, 7);
    EXPECT_EQ(config.agent.sensor_dims.action, 7);
    EXPECT_DOUBLE_EQ(config.common.gamma, 5.0);
}

TEST(ExperimentConfigTest, SensorDimsStateDim) {
    SensorDims dims;
    dims.joint_angles = 7;
    dims.joint_velocities = 7;
    dims.end_effector_points = 6;

    EXPECT_EQ(dims.state_dim(), 20);
}

// =============================================================================
// ConfigLoader Tests
// =============================================================================

TEST(ConfigLoaderTest, ToYaml) {
    auto config = ExperimentConfig::create_box2d_mdgps(1.0, "protagonist");
    std::string yaml = ConfigLoader::to_yaml(config);

    EXPECT_TRUE(yaml.find("experiment_name:") != std::string::npos);
    EXPECT_TRUE(yaml.find("gamma: 1") != std::string::npos);
    EXPECT_TRUE(yaml.find("mode: protagonist") != std::string::npos);
}

TEST(ConfigLoaderTest, ParseSimpleConfig) {
    std::string config_str = R"(
iterations: 20
num_samples: 10
gui_on: false

common:
  experiment_name: test_experiment
  conditions: 8
  mode: antagonist
  gamma: 2.5

agent:
  type: AgentBox2D
  T: 50
  dt: 0.1

algorithm:
  type: AlgorithmMDGPS
  iterations: 15
  kl_step: 10.0

cost:
  mode: antagonist
  gamma: 2.5
  action_weight: 0.001
  state_weight: 2.0
)";

    auto config = ConfigLoader::parse(config_str);

    EXPECT_EQ(config.iterations, 20);
    EXPECT_EQ(config.num_samples, 10);
    EXPECT_FALSE(config.gui_on);
    EXPECT_EQ(config.common.experiment_name, "test_experiment");
    EXPECT_EQ(config.common.conditions, 8);
    EXPECT_EQ(config.common.mode, "antagonist");
    EXPECT_DOUBLE_EQ(config.common.gamma, 2.5);
    EXPECT_EQ(config.agent.T, 50);
    EXPECT_DOUBLE_EQ(config.agent.dt, 0.1);
}

// =============================================================================
// ExperimentPresets Tests
// =============================================================================

TEST(ExperimentPresetsTest, ListPresets) {
    auto presets = ExperimentPresets::list();

    EXPECT_GT(presets.size(), 20u);
    EXPECT_TRUE(std::find(presets.begin(), presets.end(), "box2d_mdgps_y1e0") != presets.end());
    EXPECT_TRUE(std::find(presets.begin(), presets.end(), "mjc_mdgps_antagonist_y5") != presets.end());
}

TEST(ExperimentPresetsTest, GetPreset) {
    auto config = ExperimentPresets::get("box2d_mdgps_y1e0");

    EXPECT_DOUBLE_EQ(config.common.gamma, 1e0);
    EXPECT_EQ(config.common.mode, "antagonist");
}

TEST(ExperimentPresetsTest, GetPresetMjc) {
    auto config = ExperimentPresets::get("mjc_mdgps_antagonist_y5");

    EXPECT_DOUBLE_EQ(config.common.gamma, 5.0);
    EXPECT_EQ(config.agent.type, "AgentMuJoCo");
}

TEST(ExperimentPresetsTest, GetPresetThrowsOnUnknown) {
    EXPECT_THROW(ExperimentPresets::get("nonexistent_preset"), std::invalid_argument);
}

TEST(ExperimentPresetsTest, Box2dPresets) {
    // Test various Box2D presets
    auto y1e2 = ExperimentPresets::box2d_mdgps_y1e2();
    EXPECT_DOUBLE_EQ(y1e2.common.gamma, 1e2);

    auto y1e_4 = ExperimentPresets::box2d_mdgps_y1e_4();
    EXPECT_DOUBLE_EQ(y1e_4.common.gamma, 1e-4);

    auto y0_5 = ExperimentPresets::box2d_mdgps_y0_5();
    EXPECT_DOUBLE_EQ(y0_5.common.gamma, 0.5);
}

TEST(ExperimentPresetsTest, MjcPresets) {
    // Test various MuJoCo presets
    auto y2 = ExperimentPresets::mjc_mdgps_antagonist_y2();
    EXPECT_DOUBLE_EQ(y2.common.gamma, 2.0);
    EXPECT_EQ(y2.agent.type, "AgentMuJoCo");

    auto y1e_6 = ExperimentPresets::mjc_mdgps_antagonist_y1e_6();
    EXPECT_DOUBLE_EQ(y1e_6.common.gamma, 1e-6);

    auto y1e10 = ExperimentPresets::mjc_mdgps_antagonist_y1e10();
    EXPECT_DOUBLE_EQ(y1e10.common.gamma, 1e10);
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ConfigIntegrationTest, RoundTripYaml) {
    // Create config, serialize to YAML, parse back
    auto original = ExperimentConfig::create_box2d_mdgps(3.14, "robust");
    std::string yaml = ConfigLoader::to_yaml(original);
    auto parsed = ConfigLoader::parse(yaml);

    // Verify key fields survive round-trip
    EXPECT_EQ(parsed.common.mode, original.common.mode);
    EXPECT_NEAR(parsed.common.gamma, original.common.gamma, 0.01);
}

TEST(ConfigIntegrationTest, AllPresetsValid) {
    // Verify all presets can be loaded without error
    auto preset_names = ExperimentPresets::list();

    for (const auto& name : preset_names) {
        EXPECT_NO_THROW({
            auto config = ExperimentPresets::get(name);
            EXPECT_GT(config.agent.T, 0);
            EXPECT_GT(config.common.conditions, 0);
        }) << "Preset failed: " << name;
    }
}

}  // namespace test
}  // namespace config
}  // namespace gps
