#ifndef __SCAN_SERIALIZE_H__
#define __SCAN_SERIALIZE_H__

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cereal/cereal.hpp>

namespace cv {

template <class Archive>
void save(Archive &ar, const cv::Mat &mat)
{
    int rows, cols, type;
    bool continuous;

    rows = mat.rows;
    cols = mat.cols;
    type = mat.type();
    continuous = mat.isContinuous();

    ar & rows & cols & type & continuous;

    if(continuous) {
        const int data_size = rows * cols * static_cast<int>(mat.elemSize());
        auto mat_data = cereal::binary_data(mat.ptr(), data_size);
        ar & mat_data;
    } else {
        const int row_size = cols * static_cast<int>(mat.elemSize());
        for(int i = 0; i < rows; i++) {
            auto row_data = cereal::binary_data(mat.ptr(i), row_size);
            ar & row_data;
        }
    }
}

template <class Archive>
void load(Archive &ar, cv::Mat &mat)
{
    int rows, cols, type;
    bool continuous;

    ar & rows & cols & type & continuous;

    if(continuous) {
        mat.create(rows, cols, type);
        const int data_size = rows * cols * static_cast<int>(mat.elemSize());
        auto mat_data = cereal::binary_data(mat.ptr(), data_size);
        ar & mat_data;
    } else {
        mat.create(rows, cols, type);
        const int row_size = cols * static_cast<int>(mat.elemSize());
        for(int i = 0; i < rows; i++) {
            auto row_data = cereal::binary_data(mat.ptr(i), row_size);
            ar & row_data;
        }
    }
}

}

namespace Eigen {

template <class Archive>
void save(Archive &ar, const Matrix4f &matrix)
{
    int rows = matrix.rows();
    int cols = matrix.cols();
    ar(rows);
    ar(cols);
    ar(cereal::binary_data(matrix.data(), rows * cols * sizeof(float)));
}

template <class Archive>
void load(Archive &ar, Matrix4f &matrix)
{
    int rows, cols;
    ar(rows, cols);
    matrix.resize(rows, cols);
    ar(cereal::binary_data(matrix.data(), rows * cols * sizeof(float)));
}

}

#endif // __SCAN_SERIALIZE_H__