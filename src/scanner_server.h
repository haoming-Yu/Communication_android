#ifndef __SCANNERSERVER_H__
#define __SCANNERSERVER_H__

#include <string>
#include <nng/nng.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/signals2.hpp>

enum class Command : int
{
    StartCamera,
    StopCamera,
    RGBD,
    PointCloud,
    Pose,
    GenerateMesh,
    Shutdown,
};

class ScannerServer
{
public:
    ScannerServer();
    ~ScannerServer();

    bool setupServer();
    void serve(const std::string &url);

    // 这几个 send 方法都是线程安全的
    void sendModel(double fx, double fy, double cx, double cy);
    void sendImage(const cv::Mat &rgb, const cv::Mat &depth, const Eigen::Matrix4f &pose);
    void sendPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const Eigen::Matrix4f &pose);
    void sendPose(const Eigen::Matrix4f &pose);

    // 回调在其他线程中执行
    boost::signals2::signal<void()> startCamera;
    boost::signals2::signal<void()> stopCamera;
    boost::signals2::signal<void()> generateMesh;
    boost::signals2::signal<void()> shutdown;

private:
    void sendString(const std::string &v);

    static void handle_connected(nng_pipe_s, nng_pipe_ev ev, void *user_data);
    static void handle_disconnected(nng_pipe_s, nng_pipe_ev ev, void *user_data);
    static void aio_recv(void *);

private:
    nng_socket sock_;
    nng_aio *recv_;
    bool connected_;
};

#endif // __ScannerServer_H__