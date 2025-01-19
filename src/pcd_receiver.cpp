#include "pcd_receiver.hpp"
#include "cuda_test.hpp"
#include <spdlog/spdlog.h>
#include "scanner_server.h"

// lidar to imu
std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);

// camera to lidar
std::vector<double> cameraextrinT(3, 0.0); 
std::vector<double> cameraextrinR(9, 0.0);

typedef Eigen::Vector3d V3D;
typedef Eigen::Vector2d V2D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

#define MD(a,b)  Eigen::Matrix<double, (a), (b)>
#define VD(a)    Eigen::Matrix<double, (a), 1>
#define MF(a,b)  Eigen::Matrix<float, (a), (b)>
#define VF(a)    Eigen::Matrix<float, (a), 1>

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

V3D Lidar_offset_to_IMU(Zero3d);
M3D Lidar_rot_to_IMU(Eye3d);

V3D Camera_offset_to_Lidar(Zero3d);
M3D Camera_rot_to_Lidar(Eye3d);

M3D state_rot(Eye3d);
V3D state_pos(Zero3d);

M3D state_cam_rot(Eye3d);
V3D state_cam_pos(Zero3d);
M3F state_cam_rot_f(Eye3f);

M3D state_lidar_rot(Eye3d);
V3D state_lidar_pos(Zero3d);
M3F state_lidar_rot_f(Eye3f);

int img_cnt = 0;
int lidar_cnt = 0;
int odom_cnt = 0;
int path_cnt = 0;


void readParameters(ros::NodeHandle &nh)
{
    nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
    nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());
    nh.param<std::vector<double>>("camera/Pcl", cameraextrinT, std::vector<double>());
    nh.param<std::vector<double>>("camera/Rcl", cameraextrinR, std::vector<double>());
}

void pointWorldToLidar(PointType const * const pi, PointType * const po)
{
    V3D p_world(pi->x, pi->y, pi->z);
    
    V3D p_lidar = Lidar_rot_to_IMU.inverse() * (state_rot.inverse() * (p_world - state_pos) - Lidar_offset_to_IMU);

    po->x = p_lidar(0);
    po->y = p_lidar(1);
    po->z = p_lidar(2);
    po->r = pi->r;
    po->g = pi->g;
    po->b = pi->b; 
    po->a = pi->a;
}

void pointWorldToCamera(PointType const * const pi, PointType * const po)
{
    V3D p_world(pi->x, pi->y, pi->z);
    
    V3D p_cam = Camera_rot_to_Lidar.inverse() * (Lidar_rot_to_IMU.inverse() * (state_rot.inverse() * (p_world - state_pos) - Lidar_offset_to_IMU) - Camera_offset_to_Lidar);

    po->x = p_cam(0);
    po->y = p_cam(1);
    po->z = p_cam(2);
    po->r = pi->r;
    po->g = pi->g;
    po->b = pi->b; 
    po->a = pi->a;
}

void BodyToCamera(V3D& imu_trans, M3D& imu_rot)
{
    state_cam_pos = Camera_rot_to_Lidar * (Lidar_rot_to_IMU * imu_trans + Lidar_offset_to_IMU) + Camera_offset_to_Lidar;
    state_cam_rot = Camera_rot_to_Lidar * Lidar_rot_to_IMU * imu_rot;
    state_cam_rot_f = state_cam_rot.cast<float>();
}

void BodyToLidar(V3D& imu_trans, M3D& imu_rot)
{
    state_lidar_pos = Lidar_rot_to_IMU * imu_trans + Lidar_offset_to_IMU;
    state_lidar_rot = Lidar_rot_to_IMU * imu_rot;
    state_lidar_rot_f = state_lidar_rot.cast<float>();
}

cv::Mat processDepthImage(cv::Mat depth_image)
{
    // Normalize and apply color map for visualization
    double min, max;
    cv::minMaxIdx(depth_image, &min, &max);
    cv::Mat adjMap;
    cv::convertScaleAbs(depth_image, adjMap, 255 / max);
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);

    // Apply bilateral filter
    cv::Mat filtered_image;
    cv::bilateralFilter(falseColorsMap, filtered_image, 9, 75, 75);

    return filtered_image;
}

int name_cnt = 0;
cv::Mat transformPointCloudToDepthImage(PointCloudXYZRGB::Ptr cloud)
{
    // transform point cloud to depth image
    Eigen::Affine3f sensorPose;
    int width = 640, height = 480;
    float fx = 525, fy = 525, cx = 320, cy = 240;  

    // set sensorPose to odometry
    BodyToCamera(state_pos, state_rot);
    sensorPose.translation() << state_cam_pos(0), state_cam_pos(1), state_cam_pos(2);
    sensorPose.rotate(state_cam_rot_f);
    pcl::RangeImagePlanar::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    pcl::RangeImagePlanar::Ptr rangeImage(new pcl::RangeImagePlanar);
    rangeImage->createFromPointCloudWithFixedSize(*cloud, width, height, cx, cy, fx, fy, sensorPose, coordinate_frame);

    cv::Mat depth_image(height, width, CV_32FC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float range = rangeImage->getPoint(x, y).range;
            depth_image.at<float>(y, x) = range;
        }
    }

    // Save depth image to file
    cv::imwrite("/home/hmy/catkin_ws/src/pcd_receiver/PCD/depth_img/depth_img_" + std::to_string(name_cnt) + ".png", processDepthImage(depth_image));
    name_cnt++;

    return depth_image;
}

void PointCloudSubscriber::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    PointCloudXYZRGB::Ptr tmp(new PointCloudXYZRGB(msg->width * msg->height, 1));
    this->received_cloud_ = tmp;
    pcl::fromROSMsg(*msg, *(this->received_cloud_));
    lidar_cnt++;
    ROS_INFO("Received PointCloud message with %ld points", this->received_cloud_->size());
}

void ImageSubscriber::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    this->received_image_ = cv::Mat(msg->height, msg->width, CV_8UC3, const_cast<uchar*>(msg->data.data()));
    img_cnt++;

    ROS_INFO("Received Image message with size %d x %d", this->received_image_.cols, this->received_image_.rows);
}

void OdometrySubscriber::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
    this->received_odometry_ = *msg;
    odom_cnt++;
    
    ROS_INFO("Received Odometry message with position %f, %f, %f, %f, %f, %f, %f", 
    this->received_odometry_.pose.pose.position.x, 
    this->received_odometry_.pose.pose.position.y, 
    this->received_odometry_.pose.pose.position.z, 
    this->received_odometry_.pose.pose.orientation.x, 
    this->received_odometry_.pose.pose.orientation.y, 
    this->received_odometry_.pose.pose.orientation.z, 
    this->received_odometry_.pose.pose.orientation.w);

    state_pos << this->received_odometry_.pose.pose.position.x, this->received_odometry_.pose.pose.position.y, this->received_odometry_.pose.pose.position.z;
    Eigen::Quaterniond q(this->received_odometry_.pose.pose.orientation.w, this->received_odometry_.pose.pose.orientation.x, this->received_odometry_.pose.pose.orientation.y, this->received_odometry_.pose.pose.orientation.z);
    state_rot = q.toRotationMatrix();
}

void PathSubscriber::pathCallback(const nav_msgs::PathConstPtr& msg)
{
    this->received_path_ = *msg;
    path_cnt++;
    
    ROS_INFO("Received Path message with %ld poses", this->received_path_.poses.size());
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcd_receiver");
    ros::NodeHandle nh;
    readParameters(nh);

    spdlog::set_level(spdlog::level::debug);

    ScannerServer app;
    app.setupServer();
    app.serve("tcp://0.0.0.0:8833");
    app.startCamera.connect([&] {
        // 雷达无所谓相机内参，随便发一个就行
        app.sendModel(525, 525, 319.5, 239);

        // TODO: 启动点云服务
    });
    app.stopCamera.connect([&] { 
        // TODO: 停止点云服务
    });
    // 关机
    app.shutdown.connect([&] { system("shutdown -P now"); });

    Lidar_offset_to_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_rot_to_IMU<<MAT_FROM_ARRAY(extrinR);
    Camera_offset_to_Lidar<<VEC_FROM_ARRAY(cameraextrinT);
    Camera_rot_to_Lidar<<MAT_FROM_ARRAY(cameraextrinR);

    PointCloudSubscriber pcd_receiver;
    ImageSubscriber image_receiver;
    OdometrySubscriber odometry_receiver;
    PathSubscriber path_receiver;

    ROS_INFO("Running CUDA test...");
    cuda_test::run_cuda_test();

    pcl::PointCloud<pcl::PointXYZRGB> whole_cloud;

    int store_cnt = 0;
    while (ros::ok())
    {
        ros::spinOnce();
        if (lidar_cnt == 1 && img_cnt == 1 && odom_cnt == 2 && path_cnt == 1)
        {
            ROS_INFO("All messages received\n");
            // transform point cloud to depth image
            PointCloudXYZRGB::Ptr cloud = pcd_receiver.getCloud();
            if (cloud->size() > 0)
            {
                // TODO: 调用 app.sendPointCloud 发送 pointcloud 和 pose
                

                cv::Mat image = image_receiver.getImage();
                nav_msgs::Odometry odometry = odometry_receiver.getOdometry();
                BodyToLidar(state_pos, state_rot);
                nav_msgs::Path path = path_receiver.getPath();

                // for (size_t i = 0; i < cloud->size(); i++)
                // {
                //     pointWorldToLidar(&cloud->points[i], &cloud->points[i]);
                // }
                
                // send world frame point cloud to visualizer
                pcl::io::savePCDFileASCII("/home/hmy/catkin_ws/src/pcd_receiver/PCD/pcd_files/frame_" + std::to_string(store_cnt) + ".pcd", *cloud);
                whole_cloud += *cloud;
                ROS_INFO("Saved frame %d to frame_%d.pcd", store_cnt, store_cnt);

                // send imu-based odometry to visualizer
                std::ofstream odom_file("/home/hmy/catkin_ws/src/pcd_receiver/PCD/odometry_files/frame_" + std::to_string(store_cnt) + ".txt");
                if (odom_file.is_open())
                {
                    odom_file << "Position:\n";
                    odom_file << "x: " << odometry.pose.pose.position.x << "\n";
                    odom_file << "y: " << odometry.pose.pose.position.y << "\n";
                    odom_file << "z: " << odometry.pose.pose.position.z << "\n";
                    odom_file << "Orientation:\n";
                    odom_file << "w: " << odometry.pose.pose.orientation.w << "\n";
                    odom_file << "x: " << odometry.pose.pose.orientation.x << "\n";
                    odom_file << "y: " << odometry.pose.pose.orientation.y << "\n";
                    odom_file << "z: " << odometry.pose.pose.orientation.z << "\n";
                    odom_file.close();
                    ROS_INFO("Saved odometry for frame %d to frame_%d.txt", store_cnt, store_cnt);
                }
                else
                {
                    ROS_ERROR("Unable to open file to save odometry for frame %d", store_cnt);
                }
                // transformPointCloudToDepthImage(cloud);
                store_cnt++;
            }
            lidar_cnt = 0; img_cnt = 0; odom_cnt = 0; path_cnt = 0;
        }
    }

    pcl::io::savePCDFileASCII("/home/hmy/catkin_ws/src/pcd_receiver/PCD/whole_cloud.pcd", whole_cloud);

    return 0;
}