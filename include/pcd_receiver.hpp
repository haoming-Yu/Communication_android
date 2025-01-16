#include <ros/ros.h>
#include <Eigen/Core>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZRGB;
// typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector; // an Eigen aligned vector of PointTypes

class PointCloudSubscriber
{
public:
    PointCloudSubscriber()
    {
        sub_ = nh_.subscribe("/cloud_registered", 100000, &PointCloudSubscriber::pointCloudCallback, this);
    }
    PointCloudXYZRGB::Ptr getCloud() {
        return received_cloud_;
    }
private:
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    PointCloudXYZRGB::Ptr received_cloud_;
};

class ImageSubscriber
{
public:
    ImageSubscriber()
    {
        sub_ = nh_.subscribe("/rgb_img", 100000, &ImageSubscriber::imageCallback, this);
    }
    cv::Mat getImage() {
        return received_image_;
    }
private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    cv::Mat received_image_;
};

class OdometrySubscriber
{
public:
    OdometrySubscriber()
    {
        sub_ = nh_.subscribe("/aft_mapped_to_init", 1000, &OdometrySubscriber::odometryCallback, this);
    }
    nav_msgs::Odometry getOdometry() {
        return received_odometry_;
    }
private:
    void odometryCallback(const nav_msgs::OdometryConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    nav_msgs::Odometry received_odometry_;
};

class PathSubscriber
{
public:
    PathSubscriber()
    {
        sub_ = nh_.subscribe("/path", 1000, &PathSubscriber::pathCallback, this);
    }
    nav_msgs::Path getPath() {
        return received_path_;
    }
private:
    void pathCallback(const nav_msgs::PathConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    nav_msgs::Path received_path_;
};