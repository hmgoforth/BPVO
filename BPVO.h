#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/affine.h>
#include <bitplanes/core/viz.h>
#include <bitplanes/core/debug.h>
#include <bitplanes/utils/str2num.h>
#include <bitplanes/utils/config_file.h>
#include <bitplanes/utils/error.h>
#include <bitplanes/utils/timer.h>

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <string>
#include <array>
#include <memory>
#include <math.h>

#ifndef BPVO_H
#define BPVO_H
#include <BPVO.h>

class BPVO
{
  public:
    bp::BitPlanesTrackerPyramid<bp::Homography> tracker;
    cv::Mat K;
    bp::Matrix33f H_curr;
    double last_pose[6];
    cv::Mat last_img;
    double class_rp;
    bool last_img_set_fl;
    double max_from_down;
    int max_before_tracker_refresh;
    int skipped_in_a_row;

    BPVO(std::string, cv::Mat, double, int);
    std::tuple<double*,int> solver(double*, cv::Mat);
    cv::Mat pose_mat_from_telem(double* pose);
    cv::Mat rot_mat_from_telem(double roll, double pitch, double yaw);
    cv::Mat rot_mat_deg(double x, double y, double z);
    std::tuple<cv::Mat,int> find_H_sift(cv::Mat curr_img,bool visualize);
    cv::Mat combine_H_and_last_pose(cv::Mat H_last_to_curr, cv::Mat last_pose_mat);
    double* rpy_enu_from_pose_mat(cv::Mat pose_mat);
    cv::Vec3f rotation_matrix_to_euler_angles(cv::Mat R);

};

#endif