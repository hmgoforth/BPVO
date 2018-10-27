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
    double curr_pose[4];
    double curr_pose_stitch[7];
    double last_pose_stitch[6];
    double last_dpose[3];
    double class_rp;
    double max_t;
    bool template_set_fl;
    cv::Mat template_img;
    int max_iter;

    BPVO(std::string, cv::Mat);

    double* solver(double global_x, double global_y, double alt, double comp_heading, cv::Mat I_curr);
    double* solver_stitch(double* telem_pose, cv::Mat I_curr);
    std::tuple <cv::Mat,int,bool> batch_stitch(std::vector<cv::Mat> image_batch);

    void set_template(cv::Mat I_tmp, double rp);

    cv::Vec3f rotationMatrixToEulerAngles(cv::Mat R);
    cv::Vec3f rotationMatrixToEulerAnglesZXY(cv::Mat R);
};

#endif