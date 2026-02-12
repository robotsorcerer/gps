/**
 * @file sample.h
 * @brief Sample data structure for trajectory storage and state assembly.
 *
 * The Sample object maintains the state, assembles state and observation vectors,
 * and keeps track of what is and is not included in the state. This object is
 * used both by the controller, to incrementally assemble the state during a
 * trial, to keep track of sample data and get the state and observation vectors
 * from it.
 *
 * C++20 modernization includes:
 * - [[nodiscard]] attributes for getter methods
 * - std::span for array views
 * - Concepts for type safety
 * - noexcept specifications where appropriate
 */
#pragma once

#include <concepts>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

// Proto definitions for sample types
#define RUN_ON_ROBOT
#include "gps/proto/gps.pb.h"

// Options for metadata
#include "options.h"

namespace gps_control
{

// ---------------------------------------------------------------------------
// Data format enumeration
// ---------------------------------------------------------------------------

/**
 * Types of data supported for internal data storage.
 */
enum class SampleDataFormat : uint8_t
{
    Bool,
    UInt8,
    UInt16,
    Int,
    Double,
    EigenMatrix,
    EigenVector
};

// Legacy compatibility alias
using SampleDataFormatEnum = SampleDataFormat;
constexpr auto SampleDataFormatBool = SampleDataFormat::Bool;
constexpr auto SampleDataFormatUInt8 = SampleDataFormat::UInt8;
constexpr auto SampleDataFormatUInt16 = SampleDataFormat::UInt16;
constexpr auto SampleDataFormatInt = SampleDataFormat::Int;
constexpr auto SampleDataFormatDouble = SampleDataFormat::Double;
constexpr auto SampleDataFormatEigenMatrix = SampleDataFormat::EigenMatrix;
constexpr auto SampleDataFormatEigenVector = SampleDataFormat::EigenVector;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/**
 * Variant type for sample data storage.
 */
using SampleVariant = std::variant<
    bool,
    uint8_t,
    std::vector<int>,
    int,
    double,
    Eigen::MatrixXd,
    Eigen::VectorXd
>;

/**
 * List of sample variants for time-series storage.
 */
using SampleList = std::vector<SampleVariant>;

/**
 * Map from sample type to time-series data.
 */
using SampleMap = std::map<gps::SampleType, SampleList>;

// ---------------------------------------------------------------------------
// C++20 Concepts
// ---------------------------------------------------------------------------

/**
 * Concept for types that can be stored in SampleVariant.
 */
template <typename T>
concept SampleDataType = std::same_as<T, bool> ||
                         std::same_as<T, uint8_t> ||
                         std::same_as<T, std::vector<int>> ||
                         std::same_as<T, int> ||
                         std::same_as<T, double> ||
                         std::same_as<T, Eigen::MatrixXd> ||
                         std::same_as<T, Eigen::VectorXd>;

// ---------------------------------------------------------------------------
// Sample class
// ---------------------------------------------------------------------------

/**
 * Sample data structure for trajectory storage.
 *
 * Stores sensor readings, actions, and observations for each timestep
 * of a trial. Provides methods for assembly of state and observation
 * vectors from multiple sensor sources.
 */
class Sample
{
private:
    // Length of sample (number of timesteps)
    int T_;

    // Sensor data for all time steps
    // IMPORTANT: data management on the internal data is done manually
    SampleMap internal_data_;

    // Metadata: size of each field (in number of entries, not bytes)
    std::vector<int> internal_data_size_;

    // Metadata: format of each field
    std::vector<SampleDataFormat> internal_data_format_;

    // Metadata: additional information about each field
    std::vector<OptionsMap> meta_data_;

    // State definition (pairs of sample type and history depth)
    std::vector<std::pair<gps::SampleType, int>> state_definition_;

    // Observation definition (pairs of sample type and history depth)
    std::vector<std::pair<gps::SampleType, int>> obs_definition_;

public:
    /**
     * Constructor.
     *
     * @param T Number of timesteps to allocate
     */
    explicit Sample(int T);

    /**
     * Destructor.
     */
    virtual ~Sample();

    // Non-copyable, movable
    Sample(const Sample&) = delete;
    Sample& operator=(const Sample&) = delete;
    Sample(Sample&&) noexcept = default;
    Sample& operator=(Sample&&) noexcept = default;

    // -----------------------------------------------------------------------
    // Metadata accessors
    // -----------------------------------------------------------------------

    /**
     * Get sensor meta-data for a given type.
     *
     * @param type The sample type to query
     * @param data_size Output: size of the data field
     * @param data_format Output: format of the data
     * @param meta_data_ Output: additional metadata
     */
    virtual void get_meta_data(
        gps::SampleType type,
        int& data_size,
        SampleDataFormat& data_format,
        OptionsMap& meta_data_) const;

    /**
     * Set sensor meta-data for non-matrix types.
     *
     * @note This resizes any fields that don't match the current format
     *       and deletes their data!
     */
    virtual void set_meta_data(
        gps::SampleType type,
        int data_size,
        SampleDataFormat data_format,
        OptionsMap meta_data_);

    /**
     * Set sensor meta-data for matrix types.
     *
     * @note This resizes any fields that don't match the current format
     *       and deletes their data!
     */
    virtual void set_meta_data(
        gps::SampleType type,
        int data_size_rows,
        int data_size_cols,
        SampleDataFormat data_format,
        OptionsMap meta_data_);

    /**
     * Get datatypes which have metadata set.
     *
     * @param types Output vector to store available types
     */
    virtual void get_available_dtypes(std::vector<gps::SampleType>& types);

    // -----------------------------------------------------------------------
    // Data accessors
    // -----------------------------------------------------------------------

    /**
     * Get pointer to internal data for given time step.
     *
     * @param t Timestep index
     * @param type Sample type
     * @return Pointer to data, or nullptr if not found
     */
    [[nodiscard]] virtual void* get_data_pointer(int t, gps::SampleType type);

    /**
     * Fill data vector from a list of datatypes.
     *
     * @param t Timestep index
     * @param data Output vector to fill
     * @param datatypes List of sample types to include
     */
    virtual void get_data(
        int t,
        Eigen::VectorXd& data,
        std::vector<gps::SampleType> datatypes);

    /**
     * Get sensor data for given timestep.
     *
     * @param t Timestep index
     * @param type Sample type
     * @param data Output buffer
     * @param data_size Size of output buffer
     * @param data_format Expected format
     */
    virtual void get_data(
        int t,
        gps::SampleType type,
        void* data,
        int data_size,
        SampleDataFormat data_format) const;

    /**
     * Get sensor data up to a given timestep for a particular datatype.
     *
     * @param T Number of timesteps
     * @param data Output vector (flattened)
     * @param datatype Sample type
     */
    virtual void get_data(
        int T,
        Eigen::VectorXd& data,
        gps::SampleType datatype);

    /**
     * Get data for all timesteps from a single datatype.
     *
     * @param data Output vector (flattened)
     * @param datatype Sample type
     */
    virtual void get_data_all_timesteps(
        Eigen::VectorXd& data,
        gps::SampleType datatype);

    // -----------------------------------------------------------------------
    // Data setters
    // -----------------------------------------------------------------------

    /**
     * Set sensor data for given timestep.
     *
     * @param t Timestep index
     * @param type Sample type
     * @param data Data to store (variant)
     * @param data_size Size of data
     * @param data_format Format of data
     */
    virtual void set_data(
        int t,
        gps::SampleType type,
        SampleVariant data,
        int data_size,
        SampleDataFormat data_format);

    /**
     * Set vector data for given timestep.
     *
     * @param t Timestep index
     * @param type Sample type
     * @param data Pointer to data array
     * @param data_size Size of data
     * @param data_format Format of data
     */
    virtual void set_data_vector(
        int t,
        gps::SampleType type,
        double* data,
        int data_size,
        SampleDataFormat data_format);

    /**
     * Set matrix data for given timestep.
     *
     * @param t Timestep index
     * @param type Sample type
     * @param data Pointer to data array
     * @param data_rows Number of rows
     * @param data_cols Number of columns
     * @param data_format Format of data
     */
    virtual void set_data_vector(
        int t,
        gps::SampleType type,
        double* data,
        int data_rows,
        int data_cols,
        SampleDataFormat data_format);

    // -----------------------------------------------------------------------
    // Shape and state queries
    // -----------------------------------------------------------------------

    /**
     * Get dimensions of data for a sample type.
     *
     * @param sample_type The sample type
     * @param shape Output vector of dimensions
     */
    virtual void get_shape(
        gps::SampleType sample_type,
        std::vector<int>& shape);

    /**
     * Get the state representation at timestep t.
     *
     * @param t Timestep index
     * @param x Output state vector
     */
    virtual void get_state(int t, Eigen::VectorXd& x) const;

    /**
     * Get the observation at timestep t.
     *
     * @param t Timestep index
     * @param obs Output observation vector
     */
    virtual void get_obs(int t, Eigen::VectorXd& obs) const;

    /**
     * Get the action at timestep t.
     *
     * @param t Timestep index
     * @param u Output action vector
     */
    virtual void get_action(int t, Eigen::VectorXd& u) const;

    /**
     * Get the number of timesteps.
     *
     * @return Number of timesteps T
     */
    [[nodiscard]] virtual int get_T() const noexcept { return T_; }
};

}  // namespace gps_control
