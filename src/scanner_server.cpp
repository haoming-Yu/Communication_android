#include "scanner_server.h"
#include <nng/protocol/pair0/pair.h>
#include <spdlog/spdlog.h>
#include <cereal/archives/binary.hpp>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <cereal/types/string.hpp>
#include "serialize.h" // IWYU pragma: keep

ScannerServer::ScannerServer() : sock_(NNG_SOCKET_INITIALIZER), recv_(nullptr) {}

ScannerServer::~ScannerServer()
{
    if(recv_ && nng_aio_busy(recv_)) {
        nng_aio_cancel(recv_);
    }
    if(nng_socket_id(sock_) != -1) {
        nng_aio_free(recv_);
        nng_close(sock_);
    }
}

bool ScannerServer::setupServer()
{
    int code = 0;
    if((code = nng_pair0_open(&sock_)) != 0) {
        spdlog::error("create socket failed {}", nng_strerror(code));
        return false;
    }
    nng_pipe_notify(sock_, NNG_PIPE_EV_ADD_POST, &handle_connected, this);
    nng_pipe_notify(sock_, NNG_PIPE_EV_REM_POST, &handle_disconnected, this);
    if((code = nng_aio_alloc(&recv_, &aio_recv, this)) != 0) {
        nng_close(sock_);
        spdlog::error("alloc aio failed {}", nng_strerror(code));
        return false;
    }

    return true;
}

void ScannerServer::serve(const std::string &url)
{
    int code = 0;
    if((code = nng_listen(sock_, url.c_str(), nullptr, 0)) != 0) {
        spdlog::error("nng_listen failed {}", nng_strerror(code));
        return;
    }
}

void ScannerServer::sendModel(double fx, double fy, double cx, double cy)
{
    std::stringstream ss;
    cereal::BinaryOutputArchive output(ss);
    output(Command::StartCamera, fx, fy, cx, cy);
    sendString(ss.str());
}

void ScannerServer::sendImage(const cv::Mat &rgb, const cv::Mat &depth, const Eigen::Matrix4f &pose)
{
    std::stringstream ss;
    cereal::BinaryOutputArchive output(ss);
    output(Command::RGBD, rgb, depth, pose);
    sendString(ss.str());
}

void ScannerServer::sendPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const Eigen::Matrix4f &pose)
{
    pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB> encoder(
        pcl::io::MED_RES_ONLINE_COMPRESSION_WITH_COLOR, false);

    std::stringstream cloudOut;
    encoder.encodePointCloud(cloud, cloudOut);
    std::string cloudBytes = cloudOut.str();
    spdlog::debug("send cloudata size: {}", cloudBytes.size());

    std::stringstream ss;
    cereal::BinaryOutputArchive output(ss);
    output(Command::PointCloud, cloudBytes, pose);
    sendString(ss.str());
}

void ScannerServer::sendPose(const Eigen::Matrix4f &pose)
{
    std::stringstream ss;
    cereal::BinaryOutputArchive output(ss);
    output(Command::Pose, pose);
    sendString(ss.str());
}

void ScannerServer::sendString(const std::string &v)
{
    int code;
    if((code = nng_send(sock_, (void *)v.c_str(), v.size(), NNG_FLAG_NONBLOCK)) != 0) {
        spdlog::error("nng alloc msg failed {}", nng_strerror(code));
    }
}

void ScannerServer::handle_connected(nng_pipe_s, nng_pipe_ev ev, void *user_data)
{
    auto app = (ScannerServer *)user_data;
    spdlog::debug("client connected");
    nng_recv_aio(app->sock_, app->recv_);
}

void ScannerServer::handle_disconnected(nng_pipe_s, nng_pipe_ev ev, void *user_data)
{
    auto app = (ScannerServer *)user_data;
    spdlog::debug("client disconnected");
    nng_aio_cancel(app->recv_);
    app->stopCamera();
}

void ScannerServer::aio_recv(void *v)
{
    auto app = (ScannerServer *)v;
    int code;
    code = nng_aio_result(app->recv_);
    if(code != 0) {
        spdlog::error("nng_aio_result error: {}", nng_strerror(code));
        return;
    }

    nng_msg *msg = nng_aio_get_msg(app->recv_);
    size_t len = nng_msg_len(msg);
    void *body = nng_msg_body(msg);

    std::stringstream ss;
    ss.write((char *)body, len);

    Command cmd;
    {
        cereal::BinaryInputArchive input(ss);
        input(cmd);
    }

    switch(cmd) {
        case Command::StartCamera:
            app->startCamera();
            break;
        case Command::StopCamera:
            app->stopCamera();
            break;
        case Command::GenerateMesh:
            app->generateMesh();
            break;
        case Command::Shutdown:
            app->shutdown();
            break;
        default:
            spdlog::error("unknown command type: {}", int(cmd));
            break;
    }

    nng_recv_aio(app->sock_, app->recv_);
}
