# Introduction
This node is used to do the node subscribe, online tsdf refreshment(voxel hashing) and offline mesh generation.

## Basic organization
The core subscribtion is done inside file `pcd_receiver.cpp`, the helper and voxel hashing data structure is put in other files.

## Node topic
1. The rgb image will be published in topic `/left_camera/image`, and the frequency will be 10Hz.
2. The track path will be published in topic `/path`, and the information type will be `nav_msgs::Path`
3. The depth image is still hard to generate, however, we can now get the per-frame point cloud in topic `/cloud_registered`, the frequency will also be 10Hz
4. The mesh will directly be generated by voxel hashing for now. And the generated mesh will be saved inside PCD file folder

## TODO
based on current subscribtion node, write a communication method to transmit the information into android system. 