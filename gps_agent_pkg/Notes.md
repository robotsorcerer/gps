## Must-Dos
source catkin_ws/devel/setup.bash
Then do python python/gps/gps_main.py pr2_example


Check data_type_ in rostopic sensor. They might just be initializing the size of the data from cnn downsampling

Also check gps::IMAGE_FEAT in ros_topic_sensor->set_sample_data_format

Check PositionController::configure_controller in PositionController.cpp