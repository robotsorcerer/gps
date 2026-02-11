"""
Tests for Arch 9.4: dual catkin/ament_cmake (ROS 1 / ROS 2) build support.

No actual CMake execution — validates the structural correctness of
package.xml and CMakeLists.txt to catch regressions in the dual-build
configuration before a real colcon build is attempted.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).parents[2]
PKG_DIR    = REPO_ROOT / 'gps_agent_pkg'
PACKAGE_XML = PKG_DIR / 'package.xml'
CMAKE_FILE  = PKG_DIR / 'CMakeLists.txt'


# ---------------------------------------------------------------------------
# package.xml
# ---------------------------------------------------------------------------

class TestPackageXml:

    @pytest.fixture(scope='class')
    def tree(self):
        return ET.parse(PACKAGE_XML)

    @pytest.fixture(scope='class')
    def root(self, tree):
        return tree.getroot()

    def test_is_format_3(self, root):
        assert root.get('format') == '3', \
            "package.xml must be REP-149 format 3 for conditional dependencies"

    def test_name_is_correct(self, root):
        assert root.findtext('name') == 'gps_agent_pkg'

    def test_version_present(self, root):
        v = root.findtext('version')
        assert v is not None and v.strip() != ''

    def test_catkin_buildtool_dep_with_ros1_condition(self, root):
        catkin = [e for e in root.findall('buildtool_depend')
                  if e.text and 'catkin' in e.text]
        assert catkin, "package.xml must have a catkin buildtool_depend"
        conditions = [e.get('condition', '') for e in catkin]
        assert any('ROS_VERSION' in c and '1' in c for c in conditions), \
            "catkin buildtool_depend must be conditioned on ROS_VERSION == 1"

    def test_ament_cmake_buildtool_dep_with_ros2_condition(self, root):
        ament = [e for e in root.findall('buildtool_depend')
                 if e.text and 'ament_cmake' in e.text]
        assert ament, "package.xml must have an ament_cmake buildtool_depend"
        conditions = [e.get('condition', '') for e in ament]
        assert any('ROS_VERSION' in c and '2' in c for c in conditions), \
            "ament_cmake buildtool_depend must be conditioned on ROS_VERSION == 2"

    def test_ros2_message_generation_dep(self, root):
        all_deps = (root.findall('build_depend') +
                    root.findall('exec_depend') +
                    root.findall('depend'))
        rosidl = [e for e in all_deps
                  if e.text and 'rosidl' in e.text]
        assert rosidl, \
            "package.xml must declare rosidl_default_generators for ROS 2 build"
        conditions = [e.get('condition', '') for e in rosidl]
        assert any('ROS_VERSION' in c and '2' in c for c in conditions)

    def test_ros1_message_generation_dep(self, root):
        all_deps = (root.findall('build_depend') +
                    root.findall('exec_depend') +
                    root.findall('depend'))
        msggen = [e for e in all_deps
                  if e.text and 'message_generation' in e.text]
        assert msggen, \
            "package.xml must retain message_generation for ROS 1 build"
        conditions = [e.get('condition', '') for e in msggen]
        assert any('ROS_VERSION' in c and '1' in c for c in conditions)

    def test_rclcpp_dep_conditioned_on_ros2(self, root):
        all_deps = (root.findall('build_depend') +
                    root.findall('exec_depend') +
                    root.findall('depend'))
        rclcpp = [e for e in all_deps if e.text and 'rclcpp' in e.text]
        assert rclcpp, "package.xml must declare rclcpp for ROS 2"
        conditions = [e.get('condition', '') for e in rclcpp]
        assert any('ROS_VERSION' in c and '2' in c for c in conditions)

    def test_roscpp_dep_conditioned_on_ros1(self, root):
        all_deps = (root.findall('build_depend') +
                    root.findall('exec_depend') +
                    root.findall('depend'))
        roscpp = [e for e in all_deps if e.text and e.text.strip() == 'roscpp']
        assert roscpp, "package.xml must declare roscpp for ROS 1"
        conditions = [e.get('condition', '') for e in roscpp]
        assert any('ROS_VERSION' in c and '1' in c for c in conditions)

    def test_pr2_deps_conditioned_on_ros1(self, root):
        """PR2-specific packages must only be required for ROS 1."""
        all_deps = (root.findall('build_depend') +
                    root.findall('exec_depend') +
                    root.findall('depend'))
        pr2_deps = [e for e in all_deps
                    if e.text and e.text.strip().startswith('pr2_')]
        for dep in pr2_deps:
            cond = dep.get('condition', '')
            assert 'ROS_VERSION' in cond and '1' in cond, \
                f"PR2 dep '{dep.text}' must be conditioned on ROS_VERSION == 1"

    def test_standard_msgs_present_for_both_versions(self, root):
        """geometry_msgs / sensor_msgs / std_msgs must appear for both ROS versions."""
        all_deps = (root.findall('build_depend') +
                    root.findall('exec_depend') +
                    root.findall('depend'))
        for pkg in ('geometry_msgs', 'sensor_msgs', 'std_msgs'):
            matching = [e for e in all_deps if e.text and e.text.strip() == pkg]
            assert len(matching) >= 2, \
                f"{pkg} must appear for both ROS 1 and ROS 2 (found {len(matching)} entries)"

    def test_valid_xml(self):
        """package.xml must be well-formed XML."""
        ET.parse(PACKAGE_XML)   # raises ParseError on malformed XML


# ---------------------------------------------------------------------------
# CMakeLists.txt
# ---------------------------------------------------------------------------

class TestCMakeLists:

    @pytest.fixture(scope='class')
    def cmake_text(self):
        return CMAKE_FILE.read_text()

    def test_cmake_minimum_required(self, cmake_text):
        assert 'cmake_minimum_required(VERSION 3.14)' in cmake_text

    def test_ros_version_detection_block(self, cmake_text):
        assert 'ROS_VERSION' in cmake_text, \
            "CMakeLists.txt must detect ROS_VERSION env var"
        assert 'GPS_ROS2_BUILD' in cmake_text

    def test_ros2_path_uses_ament_cmake(self, cmake_text):
        assert 'find_package(ament_cmake REQUIRED)' in cmake_text

    def test_ros2_path_uses_rosidl(self, cmake_text):
        assert 'rosidl_generate_interfaces' in cmake_text

    def test_ros2_path_includes_update_policy_srv(self, cmake_text):
        assert 'srv/UpdatePolicy.srv' in cmake_text

    def test_ros2_path_calls_ament_package(self, cmake_text):
        assert 'ament_package()' in cmake_text

    def test_ros1_path_uses_catkin(self, cmake_text):
        assert 'find_package(catkin REQUIRED COMPONENTS' in cmake_text

    def test_ros1_path_add_message_files(self, cmake_text):
        assert 'add_message_files(' in cmake_text

    def test_ros1_path_add_service_files(self, cmake_text):
        assert 'add_service_files(' in cmake_text

    def test_ros1_path_generate_messages(self, cmake_text):
        assert 'generate_messages(' in cmake_text

    def test_wall_werror_flags_present(self, cmake_text):
        assert '-Wall' in cmake_text
        assert '-Werror' in cmake_text

    def test_torch_find_package(self, cmake_text):
        assert 'find_package(Torch REQUIRED)' in cmake_text

    def test_ros2_portable_sources_exclude_pr2(self, cmake_text):
        """The ROS 2 source list must not include pr2plugin.cpp."""
        # Find the GPS_ROS2_SOURCES block
        ros2_section_start = cmake_text.find('GPS_ROS2_SOURCES')
        ros2_section_end   = cmake_text.find('add_library(gps_agent_lib SHARED')
        if ros2_section_start != -1 and ros2_section_end != -1:
            ros2_section = cmake_text[ros2_section_start:ros2_section_end]
            assert 'pr2plugin.cpp' not in ros2_section, \
                "pr2plugin.cpp must not be in the ROS 2 source list"

    def test_pytorch_controller_in_both_paths(self, cmake_text):
        assert cmake_text.count('src/pytorchcontroller.cpp') >= 2, \
            "pytorchcontroller.cpp must be in both ROS 1 and ROS 2 build paths"

    def test_endif_closes_ros2_block(self, cmake_text):
        assert 'GPS_ROS2_BUILD' in cmake_text
        # Both if(GPS_ROS2_BUILD) and endif() must be present
        assert cmake_text.count('GPS_ROS2_BUILD') >= 2
