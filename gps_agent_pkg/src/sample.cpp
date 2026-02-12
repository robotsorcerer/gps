/**
 * @file sample.cpp
 * @brief Implementation of Sample class for trajectory data storage.
 *
 * C++20 modernization:
 * - nullptr instead of NULL
 * - Range-based for loops
 * - Modern casting
 * - std::span where appropriate
 */

#include "gps_agent_pkg/sample.h"
#include "gps/proto/gps.pb.h"
#include "ros/ros.h"

#include <algorithm>
#include <cstring>
#include <ranges>

using namespace gps_control;

Sample::Sample(int T)
    : T_(T)
{
    ROS_INFO("Initializing Sample with T=%d", T);

    const auto total_types = static_cast<std::size_t>(gps::TOTAL_DATA_TYPES);
    internal_data_size_.resize(total_types);
    internal_data_format_.resize(total_types);
    meta_data_.resize(total_types);

    // Fill in all possible sample types
    for (std::size_t i = 0; i < total_types; ++i) {
        internal_data_[static_cast<gps::SampleType>(i)].resize(static_cast<std::size_t>(T));
        internal_data_size_[i] = -1;  // initialize to -1 (unset)
    }
    ROS_INFO("done sample constructor");
}

Sample::~Sample() = default;

void* Sample::get_data_pointer(int t, gps::SampleType type)
{
    return nullptr;
}

void Sample::set_data_vector(int t, gps::SampleType type, double* data,
                             int data_size, SampleDataFormat data_format)
{
    set_data_vector(t, type, data, data_size, 1, data_format);
}

void Sample::set_data_vector(int t, gps::SampleType type, double* data,
                             int data_rows, int data_cols, SampleDataFormat data_format)
{
    if (t >= T_) {
        ROS_ERROR("Out of bounds t: %d/%d", t, T_);
        return;
    }

    if (data_format == SampleDataFormat::EigenVector) {
        auto& vector = std::get<Eigen::VectorXd>(internal_data_[type][static_cast<std::size_t>(t)]);
        if (vector.rows() != data_rows || data_cols != 1) {
            ROS_ERROR("Invalid size in set_data_vector! %li vs %i and cols %i for type %i",
                      vector.rows(), data_rows, data_cols, static_cast<int>(type));
        }
        std::memcpy(vector.data(), data, sizeof(double) * static_cast<std::size_t>(data_rows * data_cols));
    }
    else if (data_format == SampleDataFormat::EigenMatrix) {
        auto& matrix = std::get<Eigen::MatrixXd>(internal_data_[type][static_cast<std::size_t>(t)]);
        if (matrix.rows() != data_rows || matrix.cols() != data_cols) {
            ROS_ERROR("Invalid size in set_data_vector! %ld vs %d and %ld vs %d for type %d",
                      matrix.rows(), data_rows, matrix.cols(), data_cols, static_cast<int>(type));
        }
        std::memcpy(matrix.data(), data, sizeof(double) * static_cast<std::size_t>(data_rows * data_cols));
    }
    else {
        ROS_ERROR("Cannot use set_data_vector with non-Eigen types! Use set_data instead.");
    }
}

void Sample::set_data(int t, gps::SampleType type, SampleVariant data,
                      int data_size, SampleDataFormat data_format)
{
    if (t >= T_) {
        ROS_ERROR("Out of bounds t: %d/%d", t, T_);
        return;
    }
    internal_data_[type][static_cast<std::size_t>(t)] = std::move(data);
}

void Sample::get_data(int t, gps::SampleType type, void* data,
                      int data_size, SampleDataFormat data_format) const
{
    ROS_ERROR("Not supported!");
}

void Sample::set_meta_data(gps::SampleType type, int data_size,
                           SampleDataFormat data_format, OptionsMap meta_data)
{
    // A simplified version of set_meta_data for non-matrix types
    set_meta_data(type, data_size, 1, data_format, std::move(meta_data));
}

void Sample::set_meta_data(gps::SampleType type, int data_size_rows, int data_size_cols,
                           SampleDataFormat data_format, OptionsMap meta_data)
{
    const auto type_key = static_cast<std::size_t>(type);
    internal_data_size_[type_key] = data_size_rows * data_size_cols;
    internal_data_format_[type_key] = data_format;
    meta_data_[type_key] = std::move(meta_data);

    // If this is a matrix or vector type, preallocate for fast copy later
    if (data_format == SampleDataFormat::EigenVector) {
        for (int t = 0; t < T_; ++t) {
            internal_data_[type][static_cast<std::size_t>(t)] = Eigen::VectorXd(data_size_rows);
        }
    }
    else if (data_format == SampleDataFormat::EigenMatrix) {
        for (int t = 0; t < T_; ++t) {
            internal_data_[type][static_cast<std::size_t>(t)] = Eigen::MatrixXd(data_size_rows, data_size_cols);
        }
    }
}

void Sample::get_available_dtypes(std::vector<gps::SampleType>& types)
{
    const auto total_types = static_cast<int>(gps::TOTAL_DATA_TYPES);
    for (int i = 0; i < total_types; ++i) {
        if (internal_data_size_[static_cast<std::size_t>(i)] != -1) {
            types.push_back(static_cast<gps::SampleType>(i));
        }
    }
}

void Sample::get_meta_data(gps::SampleType type, int& data_size,
                           SampleDataFormat& data_format, OptionsMap& meta_data_) const
{
    ROS_ERROR("Not implemented!");
}

void Sample::get_state(int t, Eigen::VectorXd& x) const
{
    x.fill(0.0);
}

void Sample::get_obs(int t, Eigen::VectorXd& obs) const
{
    obs.fill(0.0);
}

void Sample::get_data_all_timesteps(Eigen::VectorXd& data, gps::SampleType datatype)
{
    const int size = internal_data_size_[static_cast<std::size_t>(datatype)];
    data.resize(size * T_);

    std::vector<gps::SampleType> dtype_vector{datatype};
    Eigen::VectorXd tmp_data;

    for (int t = 0; t < T_; ++t) {
        get_data(t, tmp_data, dtype_vector);
        // Fill in original data using Eigen segment assignment
        data.segment(t * size, size) = tmp_data;
    }
}

void Sample::get_data(int T, Eigen::VectorXd& data, gps::SampleType datatype)
{
    const int size = internal_data_size_[static_cast<std::size_t>(datatype)];
    data.resize(size * T);

    std::vector<gps::SampleType> dtype_vector{datatype};
    Eigen::VectorXd tmp_data;

    for (int t = 0; t < T; ++t) {
        get_data(t, tmp_data, dtype_vector);
        // Fill in original data using Eigen segment assignment
        data.segment(t * size, size) = tmp_data;
    }
}

void Sample::get_shape(gps::SampleType sample_type, std::vector<int>& shape)
{
    const auto dtype = static_cast<std::size_t>(sample_type);
    const int size = internal_data_size_[dtype];
    shape.clear();

    if (internal_data_format_[dtype] == SampleDataFormat::EigenVector) {
        shape.push_back(size);
    }
    else if (internal_data_format_[dtype] == SampleDataFormat::EigenMatrix) {
        // Grab shape from first entry at T=0
        const auto& sensor_data = std::get<Eigen::MatrixXd>(internal_data_.at(sample_type)[0]);
        shape.push_back(static_cast<int>(sensor_data.rows()));
        shape.push_back(static_cast<int>(sensor_data.cols()));
    }
}

void Sample::get_data(int t, Eigen::VectorXd& data, std::vector<gps::SampleType> datatypes)
{
    if (t >= T_) {
        ROS_ERROR("Out of bounds t: %d/%d", t, T_);
        return;
    }

    // Calculate total size
    int total_size = 0;
    for (const auto& dtype_enum : datatypes) {
        const auto dtype = static_cast<std::size_t>(dtype_enum);
        if (dtype >= internal_data_size_.size()) {
            ROS_ERROR("Requested size of dtype %zu, but internal_data_size_ only has %zu elements",
                      dtype, internal_data_size_.size());
            continue;
        }
        total_size += internal_data_size_[dtype];
    }

    data.resize(total_size);
    data.fill(0.0);

    // Fill in data
    int current_idx = 0;
    for (const auto& dtype_enum : datatypes) {
        const auto dtype = static_cast<std::size_t>(dtype_enum);
        if (dtype >= internal_data_.size()) {
            ROS_ERROR("Requested internal data of dtype %zu, but internal_data_ only has %zu elements",
                      dtype, internal_data_.size());
            continue;
        }

        const auto& sample_list = internal_data_.at(dtype_enum);
        const auto& sample_variant = sample_list[static_cast<std::size_t>(t)];
        const int size = internal_data_size_[dtype];

        // Handling for specific datatypes
        if (internal_data_format_[dtype] == SampleDataFormat::EigenVector) {
            const auto& sensor_data = std::get<Eigen::VectorXd>(sample_variant);
            data.segment(current_idx, size) = sensor_data;
            current_idx += size;
        }
        else if (internal_data_format_[dtype] == SampleDataFormat::EigenMatrix) {
            Eigen::MatrixXd sensor_data = std::get<Eigen::MatrixXd>(sample_variant).transpose();
            Eigen::VectorXd flattened_mat(Eigen::Map<Eigen::VectorXd>(sensor_data.data(), sensor_data.size()));
            flattened_mat.resize(sensor_data.cols() * sensor_data.rows(), 1);
            data.segment(current_idx, size) = flattened_mat;
            current_idx += size;
        }
        else {
            ROS_ERROR("Datatypes currently must be in Eigen::Vector/Eigen::Matrix format. Offender: dtype=%zu", dtype);
        }
    }
}

void Sample::get_action(int, Eigen::VectorXd& u) const
{
    // No-op: action retrieval not implemented
}
