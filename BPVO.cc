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

#include <BPVO.h>

BPVO::BPVO(std::string config_file, cv::Mat intrinsics)
  : tracker(bp::AlgorithmParameters::FromConfigFile(config_file)),
    K(intrinsics),
    H_curr(bp::Matrix33f::Identity()),
    curr_pose{0, 0, 0, 0},
    last_dpose{0, 0, 0},
    class_rp(0.2),
    max_t(1500),
    template_set_fl(false)
{
  bp::AlgorithmParameters params = bp::AlgorithmParameters::FromConfigFile(config_file);
  std::cout << "BitPlanes Parameters:" << std::endl;
  std::cout << params << std::endl << std::endl;
  // std::cout << "K = " << std::endl << intrinsics << std::endl << std::endl;
}

    
double* BPVO::solver(double global_x, double global_y, double alt, double comp_heading, cv::Mat I_curr)
{   
  /*
  Args:
    double global_x: Global x position (meters, if < 0 then ignored)
    double global_y: Global y position (meters, if < 0 then ignored)
    double alt: Altitude (meters, if < 0 then ignored)
    double comp_heading: Compass heading (degrees, N: 0, W: 90, S: 180, E: 270, if < 0 then ignored)
    cv::Mat I_curr: current UAV image (must be initialized Mat array, non-null)

  Returns:
    double*: pointer to array of refined telemetry (0:global_x, 1:global_y, 2:alt, 3:comp_heading)
  */

  // std::cout << "last_dpose[0] = " << last_dpose[0] << std::endl << std::endl;
  // std::cout << "last_dpose[1] = " << last_dpose[1] << std::endl << std::endl;
  // std::cout << "last_dpose[2] = " << last_dpose[2] << std::endl << std::endl;

  // std::cout << "curr_pose[0] = " << curr_pose[0] << std::endl << std::endl;
  // std::cout << "curr_pose[1] = " << curr_pose[1] << std::endl << std::endl;
  // std::cout << "curr_pose[2] = " << curr_pose[2] << std::endl << std::endl;

  // std::cout << "global_x = " << global_x << std::endl << std::endl;
  // std::cout << "global_y = " << global_y << std::endl << std::endl;
  // std::cout << "alt = " << alt << std::endl << std::endl;
  // std::cout << "comp_heading = " << comp_heading << std::endl << std::endl;

  if (!template_set_fl) {
    // std::cout << "No template yet, setting now..." << std::endl << std::endl;
    set_template(I_curr, class_rp);
  }

  if (cv::countNonZero(I_curr) < 1) { // empty image
    throw std::invalid_argument("Input image to solver is empty or uninitialized");
  }

  auto result = tracker.track(I_curr, H_curr);

  H_curr = result.T;
  cv::Mat H_curr_cv = cv::Mat(3, 3, CV_32F);

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      H_curr_cv.at<float>(i, j) = H_curr(i, j);
    }
  }

  // std::cout << "K = " << std::endl << K << std::endl << std::endl;
  // std::cout << "H_curr_cv: " << std::endl << H_curr_cv << std::endl << std::endl;

  cv::Mat H_noneuclid = K * H_curr_cv * K.inv(); 

  // std::cout << "H_noneuclid = " << std::endl << H_noneuclid << std::endl << std::endl;

  H_noneuclid /= H_noneuclid.at<float>(2,2);

  // std::cout << "H_noneuclid nmlz = " << std::endl << H_noneuclid << std::endl << std::endl;

  std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

  cv::decomposeHomographyMat(H_noneuclid, K, Rs_decomp, ts_decomp, normals_decomp);

  cv::Mat tvec;

  if (alt > 0) {
    tvec = ts_decomp[0].t() * alt;
  } else {
    tvec = ts_decomp[0].t();
  }

  // must negate tvec to get translation from template to frame
  // then must convert camera coord to world coord
  // however, this equates to identity operation for y and z coords
  // and negating x coord
  tvec.at<double>(0,0) = -tvec.at<double>(0,0);

  // std::cout << "ts_decomp = " << std::endl << ts_decomp[0].t() << std::endl << std::endl;
  // std::cout << "ts_decomp * alt = " << std::endl << tvec << std::endl << std::endl;

  double rad_heading;

  if (comp_heading < 0) {

    rad_heading = curr_pose[3] / 180 * M_PI;

  } else {

    rad_heading = comp_heading / 180 * M_PI;

  }

  // std::cout << "rad_heading = " << rad_heading << std::endl << std::endl;

  cv::Mat rot_mat = (cv::Mat_<double>(2,2) << cos(rad_heading), -sin(rad_heading), sin(rad_heading), cos(rad_heading));

  cv::Mat dx_dy = (cv::Mat_<double>(2,1) << tvec.at<double>(0,0), tvec.at<double>(0,1));

  // std::cout << "dx_dy = " << dx_dy << std::endl << std::endl;
  
  cv::Mat rot_dx_dy = rot_mat * dx_dy;
  
  // std::cout << "rot_dx_dy = " << rot_dx_dy << std::endl << std::endl;

  // delta translation from current template, will replace last_dpose at end of function
  float curr_dz_tmpl = tvec.at<double>(0,2);
  float curr_dx_tmpl = rot_dx_dy.at<double>(0,0);
  float curr_dy_tmpl = rot_dx_dy.at<double>(1,0);

  // std::cout << "curr_dx_tmpl = " << curr_dx_tmpl << std::endl << std::endl;
  // std::cout << "curr_dy_tmpl = " << curr_dy_tmpl << std::endl << std::endl;
  // std::cout << "curr_dz_tmpl = " << curr_dz_tmpl << std::endl << std::endl;

  double in_telem_wght = 0.5;
  double bp_telem_wght = 0.5;

  double refined_x;
  double refined_y;
  double refined_z;

  // check valid telemetry inputs
  if (global_x < 0 || global_y < 0) {

    refined_x = curr_dx_tmpl - last_dpose[0] + curr_pose[0];
    refined_y = curr_dy_tmpl - last_dpose[1] + curr_pose[1];

  } else {

    refined_x = in_telem_wght * global_x
      + bp_telem_wght * (curr_dx_tmpl - last_dpose[0] + curr_pose[0]);

    refined_y = in_telem_wght * global_y
      + bp_telem_wght * (curr_dy_tmpl - last_dpose[1] + curr_pose[1]);
  
  }

  if (alt < 0) {

    refined_z = curr_dz_tmpl - last_dpose[2] + curr_pose[2];

  } else {

    refined_z = in_telem_wght * alt
      + bp_telem_wght * (curr_dz_tmpl - last_dpose[2] + curr_pose[2]);

  }

  // std::cout << "refined_x = " << refined_x << std::endl << std::endl;
  // std::cout << "refined_y = " << refined_y << std::endl << std::endl;
  // std::cout << "refined_z = " << refined_z << std::endl << std::endl;

  // create new template if translation too large
  double t_mag = curr_dx_tmpl * curr_dx_tmpl + curr_dy_tmpl * curr_dy_tmpl;

  // std::cout << "t_mag = " << t_mag << std::endl << std::endl;

  if (t_mag > max_t) {

    set_template(I_curr, class_rp);
    H_curr = bp::Matrix33f::Identity();

    last_dpose[0] = 0;
    last_dpose[1] = 0;
    last_dpose[2] = 0;

  } else {

    last_dpose[0] = curr_dx_tmpl;
    last_dpose[1] = curr_dy_tmpl;
    last_dpose[2] = curr_dz_tmpl;

  }

  curr_pose[0] = refined_x;
  curr_pose[1] = refined_y;
  curr_pose[2] = refined_z;
  curr_pose[3] = rad_heading / M_PI * 180;

  // std::cout << "curr_pose[0] = " << curr_pose[0] << std::endl << std::endl;
  // std::cout << "curr_pose[1] = " << curr_pose[1] << std::endl << std::endl;
  // std::cout << "curr_pose[2] = " << curr_pose[2] << std::endl << std::endl;

  return curr_pose;
}

void BPVO::set_template(cv::Mat I_tmp, double rp)
{
  std::cout << "[Template Update]" << std::endl << std::endl;
  template_set_fl = true;
  cv::Rect bbox(rp*I_tmp.cols, rp*I_tmp.rows, I_tmp.cols-2*rp*I_tmp.cols, I_tmp.rows-2*rp*I_tmp.rows);
  tracker.setTemplate(I_tmp, bbox);
}