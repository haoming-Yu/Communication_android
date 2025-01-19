cmake_minimum_required(VERSION 2.8.3)
project(pcd_receiver)

SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3 -Wall" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)

find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  eigen_conversions
  geometry_msgs
  # livox_ros_driver
  message_generation
  nav_msgs
  pcl_ros
  roscpp
  rospy
  std_msgs
  tf
)

find_package(Eigen3 REQUIRED)

# 明确指定PCL的搜索路径
set(PCL_DIR /usr/lib/x86_64-linux-gnu/cmake/pcl)  # 使用系统PCL库
set(PCL_INCLUDE_DIRS /usr/include/pcl-1.10)  # 根据你的PCL版本调整

# 查找PCL包
find_package(PCL 1.10 REQUIRED PATHS ${PCL_ROOT} NO_DEFAULT_PATH)

find_package(OpenCV REQUIRED)

# 第三方依赖
set(NNG_TESTS OFF)
add_subdirectory(third_party/nng-1.9.0)
add_subdirectory(third_party/cereal-1.3.2)
find_package(spdlog REQUIRED)

# 添加PCL库的具体路径
link_directories(${PCL_ROOT})

# 设置运行时路径
set(CMAKE_INSTALL_RPATH ${PCL_ROOT})
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O3 
    --expt-relaxed-constexpr    # 解决Eigen constexpr警告
    -gencode arch=compute_89,code=sm_89
    -G                          # 只保留-G，移除-lineinfo
    --disable-warnings
    -Wno-deprecated-declarations
    
)

# 添加编译选项来抑制特定警告
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wno-unused-result -Wno-deprecated-declarations)
endif()

# 设置CUDA代码的编译选项
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
  include/voxel_hashing)

cuda_add_library(
  cuda_example 
  src/cuda_test.cu 
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

add_executable(pcd_receiver 
    src/pcd_receiver.cpp
    src/scanner_server.cpp
)
target_link_libraries(pcd_receiver 
    cuda_example
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${PYTHON_LIBRARIES} 
    ${OpenCV_LIBS}
    yaml-cpp
    nng
    cereal
    spdlog::spdlog
)
target_include_directories(pcd_receiver PRIVATE ${PYTHON_INCLUDE_DIRS})

# 明确指定目标的运行时搜索路径
set_target_properties(pcd_receiver PROPERTIES
    INSTALL_RPATH "${PCL_ROOT}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 添加编译选项
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(
        -Wno-unused-result 
        -Wno-deprecated-declarations
        -Wno-stringop-truncation
        -Wno-cpp
    )
endif()

