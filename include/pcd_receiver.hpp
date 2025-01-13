#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <cuda_runtime.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector; // an Eigen aligned vector of PointTypes

class PointCloudSubscriber
{
public:
    PointCloudSubscriber()
    {
        sub_ = nh_.subscribe("/cloud_registered", 100000, &PointCloudSubscriber::pointCloudCallback, this);
    }
private:
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    PointCloudXYZI::Ptr received_cloud_;
};
