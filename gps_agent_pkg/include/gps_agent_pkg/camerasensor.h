/*
Camera sensor: records latest images from camera.
*/
#pragma once

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <vicon_bridge/Markers.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Point.h>


// Superclass.
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"
#include "gps_agent_pkg/positioncontroller.h"

// This sensor writes to the following data types:
// RGBImage
// DepthImage

// Default values for image dimensions
#define IMAGE_WIDTH_INIT 320
#define IMAGE_HEIGHT_INIT 240
#define IMAGE_WIDTH 240
#define IMAGE_HEIGHT 240
// Default vicon cloud dims
#define VICON_CLOUD_HEIGHT 1
#define VICON_CLOUD_WIDTH 4

namespace gps_control
{

class CameraSensor: public Sensor
{
    // friend class PositionController;
private:    
    geometry_msgs::Point forehead, leftcheek, rightcheek, chin;    
    // Latest image.
    std::vector<uint8_t> latest_rgb_image_;
    std::vector<uint16_t> latest_depth_image_;
    // //Latest vicons
    // std::vector<geometry_msgs::Point> latest_vicon_markers_;
    // std::vector<geometry_msgs::Twist> latest_vicon_pose_;
    std::vector<sensor_msgs::PointCloud2> latest_vicon_clouds_;

    // Time at which the image was first published.
    ros::Time latest_rgb_time_, latest_depth_time_;
    ros::Time latest_vicon_time_, latest_vicon_twist_time_;

    // Image subscribers
    ros::Subscriber depth_subscriber_, rgb_subscriber_;
    ros::Subscriber vicon_subscriber_, vicon_twist_subscriber_;

    // Image dimensions, before and after cropping, for both rgb and depth images
    int image_width_init_, image_height_init_, image_width_, image_height_, image_size_;
    // clouds dims
    int vicon_height_, vicon_width_, vicon_size_;

    std::string rgb_topic_name_, depth_topic_name_, \
                    vicon_topic_name, vicon_twist_topic_name_;          ;

public:
    // Constructor.
    CameraSensor(ros::NodeHandle& n, RobotPlugin *plugin);
    //copy constructor
    CameraSensor();
    // Destructor.
    virtual ~CameraSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
    void update_rgb_image(const sensor_msgs::Image::ConstPtr& msg);
    void update_depth_image(const sensor_msgs::Image::ConstPtr& msg);
    void update_vicon_markers(const vicon_bridge::Markers::ConstPtr& markers_msg);
    void update_vicon_pose(const geometry_msgs::Twist::ConstPtr& pose_msg);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to set resolution, cropping, topic to listen to...
    virtual void configure_sensor(const OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual void set_sample_data_format(boost::scoped_ptr<Sample> sample) const;
    // Set data on the provided sample.
    virtual void set_sample_data(boost::scoped_ptr<Sample> sample) const;
    
    std::vector<geometry_msgs::Point> latest_vicon_markers_;
    std::vector<geometry_msgs::Twist> latest_vicon_pose_;
};

}
