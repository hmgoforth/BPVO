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

#define PI 3.14159265

BPVO::BPVO(std::string config_file, cv::Mat intrinsics)
  : tracker(bp::AlgorithmParameters::FromConfigFile(config_file)),
    K(intrinsics),
    H_curr(bp::Matrix33f::Identity()),
    curr_pose{0, 0, 0, 0},
    curr_pose_stitch{0, 0, 0, 0, 0, 0, 1},
    last_pose_stitch{0, 0, 0, 0, 0, 0},
    last_dpose{0, 0, 0},
    class_rp(0.0),
    max_t(1500),
    template_set_fl(false),
    max_iter(10)
{
  bp::AlgorithmParameters params = bp::AlgorithmParameters::FromConfigFile(config_file);
  std::cout << "BitPlanes Parameters:" << std::endl;
  std::cout << params << std::endl << std::endl;
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
  bool pr_dbg = false;

  if (!template_set_fl) {
   
    if (pr_dbg) {
      std::cout << "\tNo template yet, setting now..." << std::endl;
      std::cout << "\tAlso setting initial pose using current inputs ..." << std::endl << std::endl;
    }

    set_template(I_curr, class_rp);
    
    curr_pose[0] = global_x;
    curr_pose[1] = global_y;
    curr_pose[2] = alt;
    curr_pose[3] = comp_heading;
  }

  if (pr_dbg) {
    std::cout << "\tlast_dpose[0] = " << last_dpose[0] << std::endl;
    std::cout << "\tlast_dpose[1] = " << last_dpose[1] << std::endl;
    std::cout << "\tlast_dpose[2] = " << last_dpose[2] << std::endl << std::endl;

    std::cout << "\tcurr_pose[0] = " << curr_pose[0] << std::endl;
    std::cout << "\tcurr_pose[1] = " << curr_pose[1] << std::endl;
    std::cout << "\tcurr_pose[2] = " << curr_pose[2] << std::endl << std::endl;

    std::cout << "\tglobal_x = " << global_x << std::endl;
    std::cout << "\tglobal_y = " << global_y << std::endl;
    std::cout << "\talt = " << alt << std::endl;
    std::cout << "\tcomp_heading = " << comp_heading << std::endl << std::endl;
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

  if (pr_dbg) {
    std::cout << "\tK = " << std::endl << K << std::endl << std::endl;
    std::cout << "\tH_curr_cv: " << std::endl << H_curr_cv << std::endl << std::endl;
  }

  cv::Mat H_noneuclid = K * H_curr_cv * K.inv();


  if (pr_dbg) {
    std::cout << "\tH_noneuclid = " << std::endl << H_noneuclid << std::endl << std::endl;
  }

  H_noneuclid /= H_noneuclid.at<float>(2,2);

  if (pr_dbg) {
    std::cout << "\tH_noneuclid nmlz = " << std::endl << H_noneuclid << std::endl << std::endl;
  }

  std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

  cv::decomposeHomographyMat(H_noneuclid, K, Rs_decomp, ts_decomp, normals_decomp);

  cv::Mat tvec;

  int decomp_ind = 0;
  int min_euler_z = 10;

  for (std::size_t i = 0, max = Rs_decomp.size(); i != max; ++i) {
    cv::Vec3f euler_ang = rotationMatrixToEulerAngles(Rs_decomp[i]);
    // std::cout << "euler_ang[" << i << "] = " << std::endl << euler_ang << std::endl << std::endl;
    // std::cout << "ts_decomp[" << i << "] = " << std::endl << ts_decomp[i] << std::endl << std::endl;
    // std::cout << "normals_decomp[" << i << "] = " << std::endl << normals_decomp[i] << std::endl << std::endl;
    // std::cout << "normals_decomp[" << 2 << "] = " << std::endl << normals_decomp[i].at<double>(2) << std::endl << std::endl;
    if (normals_decomp[i].at<double>(2) > 0) {
      if (euler_ang[2] < min_euler_z) {
        decomp_ind = i;
      }
    }

  }

  if ((K.at<float>(0,0) > 1) && (alt > 0)) {
    tvec = ts_decomp[decomp_ind].t() * alt;
  } else {
    tvec = ts_decomp[decomp_ind].t();
  }

  // must negate tvec to get translation from template to frame
  // then must convert camera coord to world coord
  // however, this equates to identity operation for y and z coords
  // and negating x coord
  tvec.at<double>(0,0) = -tvec.at<double>(0,0);

  cv::Vec3f euler_ang = rotationMatrixToEulerAngles(Rs_decomp[decomp_ind]);

  if (pr_dbg) {
    std::cout << "\tts_decomp = " << std::endl << ts_decomp[0].t() << std::endl << std::endl;
    std::cout << "\tts_decomp * alt = " << std::endl << tvec << std::endl << std::endl;
    std::cout << "\teuler_ang = " << std::endl << euler_ang << std::endl << std::endl;
    // std::cout << "euler_ang[2] = " << euler_ang[2] << std::endl << std::endl;
  }

  double rad_heading;

  if (std::isinf(comp_heading)) {

    rad_heading = curr_pose[3] / 180 * M_PI + euler_ang[2];

  } else {

    rad_heading = comp_heading / 180 * M_PI;

  }

  // wrap to 2pi
  if (rad_heading < 0) {
    rad_heading = 2 * M_PI + rad_heading;
  } else if (rad_heading > (2 * M_PI)) {
    rad_heading = rad_heading - 2 * M_PI;
  }

  cv::Mat rot_mat = (cv::Mat_<double>(2,2) << cos(rad_heading), -sin(rad_heading), sin(rad_heading), cos(rad_heading));

  cv::Mat dx_dy = (cv::Mat_<double>(2,1) << tvec.at<double>(0,0), tvec.at<double>(0,1));
  
  cv::Mat rot_dx_dy = rot_mat * dx_dy;
  
  if (pr_dbg) {
    std::cout << "\trad_heading = " << rad_heading << std::endl << std::endl;
    std::cout << "\tdx_dy = " << dx_dy << std::endl << std::endl;
    std::cout << "\trot_dx_dy = " << rot_dx_dy << std::endl << std::endl;
  }

  // delta translation from current template, will replace last_dpose at end of function
  float curr_dz_tmpl = tvec.at<double>(0,2);
  float curr_dx_tmpl = rot_dx_dy.at<double>(0,0);
  float curr_dy_tmpl = rot_dx_dy.at<double>(1,0);

  if (pr_dbg) {
    std::cout << "\tcurr_dx_tmpl = " << curr_dx_tmpl << std::endl << std::endl;
    std::cout << "\tcurr_dy_tmpl = " << curr_dy_tmpl << std::endl << std::endl;
    std::cout << "\tcurr_dz_tmpl = " << curr_dz_tmpl << std::endl << std::endl;
  }

  double in_telem_wght = 0.5;
  double bp_telem_wght = 0.5;

  double refined_x;
  double refined_y;
  double refined_z;

  // check valid telemetry inputs
  if (std::isinf(global_x) || std::isinf(global_y)) {

    refined_x = curr_dx_tmpl - last_dpose[0] + curr_pose[0];
    refined_y = curr_dy_tmpl - last_dpose[1] + curr_pose[1];

  } else {

    refined_x = in_telem_wght * global_x
      + bp_telem_wght * (curr_dx_tmpl - last_dpose[0] + curr_pose[0]);

    refined_y = in_telem_wght * global_y
      + bp_telem_wght * (curr_dy_tmpl - last_dpose[1] + curr_pose[1]);
  
  }

  if (std::isinf(alt)) {

    refined_z = curr_dz_tmpl - last_dpose[2] + curr_pose[2];

  } else {

    refined_z = in_telem_wght * alt
      + bp_telem_wght * (curr_dz_tmpl - last_dpose[2] + curr_pose[2]);

  }

  // create new template if translation too large
  double t_mag = curr_dx_tmpl * curr_dx_tmpl + curr_dy_tmpl * curr_dy_tmpl;

  if (pr_dbg) {
    std::cout << "\trefined_x = " << refined_x << std::endl << std::endl;
    std::cout << "\trefined_y = " << refined_y << std::endl << std::endl;
    std::cout << "\trefined_z = " << refined_z << std::endl << std::endl;
    std::cout << "\tt_mag = " << t_mag << std::endl << std::endl;
  }

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

double* BPVO::solver_stitch(double* telem_pose, cv::Mat I_curr) {
  /*
  Args:
    telem_pose: pointer to array containing (if any are inf they are ignored)
      [0]: global x position
      [1]: global y position
      [2]: global z position
      [3]: global yaw
      [4]: global pitch
      [5]: global roll
    I_curr: current UAV image (must be initialized Mat array, non-null)

  Out:
    pointer to array of refined telemetry in same format as telem_pose
      [0]: global x position
      [1]: global y position
      [2]: global z position
      [3]: global yaw
      [4]: global pitch
      [5]: global roll
    with flag at last index indicating good frame (1) or bad frame (0)
      [6]: good frame
  */

  bool pr_dbg = false;

  int Iw = I_curr.cols;
  int Ih = I_curr.rows;

  cv::Mat I_curr_resize, K_resize;

  if (pr_dbg) {
    std::cout << "Image size: " << Iw << "x" << Ih << std::endl;
  }

  if (Ih == 1080) { // resize 1080p images to 0.25 scale, rescale K as well

    if (pr_dbg) {
      std::cout << "Resizing..." << std::endl;
    }

    cv::resize(I_curr, I_curr_resize, cv::Size(), 0.25, 0.25);

    double K_resize_arr[3][3] = {
      {K.at<double>(0,0) * 0.25, 0, K.at<double>(0,2) * 0.25},
      {0, K.at<double>(1,1) * 0.25, K.at<double>(1,2) * 0.25},
      {0, 0, 1}};

    K_resize = cv::Mat(3, 3, CV_64F, &K_resize_arr);
  } else {

    if (pr_dbg) {
      std::cout << "Not resizing..." << std::endl;
    }

    I_curr_resize = I_curr.clone();
    K_resize = K.clone();
  }

  if (pr_dbg) {
    std::cout << "K_resize = " << std::endl << K_resize << std::endl;
  }

  double telem_x = telem_pose[0];
  double telem_y = telem_pose[1];
  double telem_z = telem_pose[2];
  double telem_yaw = telem_pose[3];
  double telem_pitch = telem_pose[4];
  double telem_roll = telem_pose[5];

  if (pr_dbg) {
    std::cout << 
    "telem_x = " << telem_x << ", diff: " << telem_x - curr_pose_stitch[0] << "\n" <<
    "telem_y = " << telem_y << ", diff: " << telem_y - curr_pose_stitch[1] <<"\n" <<
    "telem_z = " << telem_z << ", diff: " << telem_z - curr_pose_stitch[2] <<"\n" <<
    "telem_yaw = " << telem_yaw << ", diff: " << telem_yaw - curr_pose_stitch[3] << "\n" <<
    "telem_pitch = " << telem_pitch << ", diff: " << telem_pitch - curr_pose_stitch[4] << "\n" <<
    "telem_roll = " << telem_roll << ", diff: " << telem_roll - curr_pose_stitch[5] << std::endl << std::endl;
  }

  if (!template_set_fl) {
    if (pr_dbg) {
      std::cout << "No template yet, setting now..." << std::endl;
      std::cout << "Also setting initial pose using current inputs ..." << std::endl << std::endl;
    }

    set_template(I_curr_resize, class_rp);
    template_img = I_curr_resize.clone();
    
    curr_pose_stitch[0] = telem_x;
    curr_pose_stitch[1] = telem_y;
    curr_pose_stitch[2] = telem_z;
    curr_pose_stitch[3] = telem_yaw;
    curr_pose_stitch[4] = telem_pitch;
    curr_pose_stitch[5] = telem_roll;
    curr_pose_stitch[6] = 1;

    last_pose_stitch[0] = telem_x;
    last_pose_stitch[1] = telem_y;
    last_pose_stitch[2] = telem_z;
    last_pose_stitch[3] = telem_yaw;
    last_pose_stitch[4] = telem_pitch;
    last_pose_stitch[5] = telem_roll;

    return curr_pose_stitch;

  } else {

    set_template(template_img, class_rp);

  }

  // returns pixel H from template to curr
  auto result = tracker.track(I_curr_resize, H_curr);
  H_curr = result.T;
  int num_iter = result.num_iterations;
  float final_error = result.final_ssd_error;
  cv::Mat H_curr_cv = cv::Mat(3, 3, CV_32F);

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      H_curr_cv.at<float>(i, j) = H_curr(i, j);
    }
  }

  if (pr_dbg) {
    std::cout << "H_curr_cv: " << std::endl << H_curr_cv << std::endl;
    std::cout << "num iterations: " << num_iter << std::endl << std::endl;
    std::cout << "final error: " << final_error << std::endl;
  }

  if (num_iter > max_iter) {
    if (pr_dbg) {
      std::cout << "Iter was " << num_iter << ", greater than max iter " << max_iter << ", waiting for new template" << std::endl;
    }

    curr_pose_stitch[0] = telem_x;
    curr_pose_stitch[1] = telem_y;
    curr_pose_stitch[2] = telem_z;
    curr_pose_stitch[3] = telem_yaw;
    curr_pose_stitch[4] = telem_pitch;
    curr_pose_stitch[5] = telem_roll;
    curr_pose_stitch[6] = 0;

    last_pose_stitch[0] = telem_x;
    last_pose_stitch[1] = telem_y;
    last_pose_stitch[2] = telem_z;
    last_pose_stitch[3] = telem_yaw;
    last_pose_stitch[4] = telem_pitch;
    last_pose_stitch[5] = telem_roll;

    template_set_fl = false;

    return curr_pose_stitch;
  } else {

    if (pr_dbg) {
      std::cout << "last_pose_x: " << last_pose_stitch[0] << std::endl;
      std::cout << "last_pose_y: " << last_pose_stitch[1] << std::endl;
      std::cout << "last_pose_z: " << last_pose_stitch[2] << std::endl;
      std::cout << "last_pose_yaw: " << last_pose_stitch[3] << std::endl;
      std::cout << "last_pose_pitch: " << last_pose_stitch[4] << std::endl;
      std::cout << "last_pose_roll: " << last_pose_stitch[5] << std::endl << std::endl;
    }

    // compute refined pose for input camera based on H_curr_cv
    std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;
    cv::decomposeHomographyMat(H_curr_cv, K_resize, Rs_decomp, ts_decomp, normals_decomp);

    if (pr_dbg) {
      for (std::size_t i = 0; i < Rs_decomp.size(); i++) {
        cv::Vec3f euler_ang = rotationMatrixToEulerAngles(Rs_decomp[i]);
        std::cout << "euler_ang[" << i << "] = " << std::endl << "\t" << euler_ang.t() << std::endl << std::endl;
        std::cout << "ts_decomp[" << i << "] = " << std::endl << "\t" << ts_decomp[i].t() << std::endl << std::endl;
        std::cout << "normals_decomp[" << i << "] = " << std::endl << "\t" << normals_decomp[i].t() << std::endl << std::endl;
      }
    }

    // convert last input (corresponding to template) into euler angles, and reverse order to get extrinsic form, create R_pose,t_pose for vehicle pose
    // see 'General Rotations' section here: https://en.wikipedia.org/wiki/Rotation_matrix
    // and 'Converting Between Intrinsic and Extrinsic': https://www.cs.utexas.edu/~theshark/courses/cs354/lectures/cs354-14.pdf
    double last_yaw_rad = last_pose_stitch[3] / 180 * PI;
    double last_pitch_rad = last_pose_stitch[4] / 180 * PI;
    double last_roll_rad = last_pose_stitch[5] / 180 * PI; 

    double last_yaw_arr[3][3] = 
      {{cos(last_yaw_rad), -sin(last_yaw_rad), 0},
       {sin(last_yaw_rad), cos(last_yaw_rad), 0},
       {0, 0, 1}};

    double last_pitch_arr[3][3] = 
      {{1, 0, 0},
       {0, cos(last_pitch_rad), -sin(last_pitch_rad)},
       {0, sin(last_pitch_rad), cos(last_pitch_rad)}};

    double last_roll_arr[3][3] =
      {{cos(last_roll_rad), 0, sin(last_roll_rad)},
       {0, 1, 0},
       {-sin(last_roll_rad), 0, cos(last_roll_rad)}};

    double last_T_arr[3][1] =
      {{last_pose_stitch[0]},
       {last_pose_stitch[1]},
       {last_pose_stitch[2]}};

    cv::Mat last_yaw_mat = cv::Mat(3, 3, CV_64F, last_yaw_arr);
    cv::Mat last_pitch_mat = cv::Mat(3, 3, CV_64F, last_pitch_arr);
    cv::Mat last_roll_mat = cv::Mat(3, 3, CV_64F, last_roll_arr);

    cv::Mat last_R_extrinsic = last_yaw_mat * last_pitch_mat * last_roll_mat;
    cv::Mat last_T_extrinsic = cv::Mat(3, 1, CV_64F, last_T_arr);

    double pose_bottom_row_arr[1][4] = 
      {{0, 0, 0, 1}};
    cv::Mat pose_bottom_row_mat = cv::Mat(1, 4, CV_64F, pose_bottom_row_arr);
    
    cv::Mat last_pose, last_pose_4;
    cv::hconcat(last_R_extrinsic, last_T_extrinsic, last_pose);
    cv::vconcat(last_pose, pose_bottom_row_mat, last_pose_4);

    cv::Vec3f last_R_test_euler = rotationMatrixToEulerAnglesZXY(last_R_extrinsic);

    if (pr_dbg) {
      std::cout << "last_yaw_mat = " << std::endl << "\t" << last_yaw_mat << std::endl << std::endl;
      std::cout << "last_pitch_mat = " << std::endl << "\t" << last_pitch_mat << std::endl << std::endl;
      std::cout << "last_roll_mat = " << std::endl << "\t" << last_roll_mat << std::endl << std::endl;
      std::cout << "last_R_extrinsic = " << std::endl << "\t" << last_R_extrinsic << std::endl << std::endl;
      std::cout << "last_T_extrinsic = " << std::endl << "\t" << last_T_extrinsic << std::endl << std::endl;
      std::cout << "last_pose_4 = " << std::endl << "\t" << last_pose_4 << std::endl << std::endl;
      std::cout << "last_R_test_euler = " << last_R_test_euler.t() << std::endl;
      std::cout << "last_yaw_rad: " << last_yaw_rad << ", last_pitch_rad: " << last_pitch_rad << ", last_roll_rad: " << last_roll_rad << std::endl;
    }

    // convert euler ang and translation from H into changes in x,y,z,roll,pitch,yaw in vehicle frame, R_diff, t_diff
    // use solution which has largest value for 3rd coordinate of normal

    int decomp_ind = 0;
    double min_normal_z = 0;

    // find solution with smallest 3rd coordinate of the normal
    for (std::size_t i = 0; i < Rs_decomp.size(); i++) {
      if (normals_decomp[i].at<double>(2) < min_normal_z) {
        min_normal_z = normals_decomp[i].at<double>(2);
        decomp_ind = i;
      }
    }

    // to change euler angles of camera to euler angles of vehicle pose requires negating all angles
    cv::Vec3f euler_ang_diff = rotationMatrixToEulerAngles(Rs_decomp[decomp_ind]);
    double x_axis_rot_camera = euler_ang_diff[0];
    double y_axis_rot_camera = euler_ang_diff[1];
    double z_axis_rot_camera = euler_ang_diff[2];

    if (pr_dbg) {
      std::cout << "decomp_ind = " << decomp_ind << std::endl << std::endl;
      std::cout << "euler_ang_diff = " << euler_ang_diff << std::endl;
      std::cout << "x_axis_rot_camera = " << x_axis_rot_camera << std::endl;
      std::cout << "y_axis_rot_camera = " << y_axis_rot_camera << std::endl;
      std::cout << "z_axis_rot_camera = " << z_axis_rot_camera << std::endl << std::endl;
    }

    // change camera angle differences in the camera coordinate frame, into angles changes in the vehicle frame
    // requires negation of all rotations?
    double diff_yaw_pose = -z_axis_rot_camera;
    double diff_pitch_pose = -x_axis_rot_camera;
    double diff_roll_pose = -y_axis_rot_camera;

    // compose the 3 diff angles in the vehicle frame into second rotation matrix
    double diff_yaw_arr[3][3] = 
      {{cos(diff_yaw_pose), -sin(diff_yaw_pose), 0},
       {sin(diff_yaw_pose), cos(diff_yaw_pose), 0},
       {0, 0, 1}};

    double diff_pitch_arr[3][3] = 
      {{1, 0, 0},
       {0, cos(diff_pitch_pose), -sin(diff_pitch_pose)},
       {0, sin(diff_pitch_pose), cos(diff_pitch_pose)}};

    double diff_roll_arr[3][3] =
      {{cos(diff_roll_pose), 0, sin(diff_roll_pose)},
       {0, 1, 0},
       {-sin(diff_roll_pose), 0, cos(diff_roll_pose)}};

    cv::Mat diff_yaw_mat = cv::Mat(3, 3, CV_64F, diff_yaw_arr);
    cv::Mat diff_pitch_mat = cv::Mat(3, 3, CV_64F, diff_pitch_arr);
    cv::Mat diff_roll_mat = cv::Mat(3, 3, CV_64F, diff_roll_arr);

    cv::Mat diff_R_extrinsic = diff_yaw_mat * diff_pitch_mat * diff_roll_mat;

    // use altitude of template frame as distance from plane for scaling translation
    double scaled_d = last_pose_stitch[2]; // scale by altitude
    // changing translation from camera to vehicle, means keep X, negate Y, negate Z
    double diff_T_extrinsic_arr[3][1] = 
      {{ts_decomp[decomp_ind].at<double>(0) * scaled_d},
       {-ts_decomp[decomp_ind].at<double>(1) * scaled_d},
       {-ts_decomp[decomp_ind].at<double>(2) * scaled_d}};

    cv::Mat diff_T_extrinsic = cv::Mat(3, 1, CV_64F, diff_T_extrinsic_arr);

    // create 4x4 transformation matrix using diff R and diff T
    cv::Mat diff_pose, diff_pose_4;
    cv::hconcat(diff_R_extrinsic, diff_T_extrinsic, diff_pose);
    cv::vconcat(diff_pose, pose_bottom_row_mat, diff_pose_4);

    // update pose by multiplying previous pose by new update
    // [R_new_pose, t_new_pose] = [R_pose, t_pose] * [R_diff, t_diff]
    cv::Mat updated_pose = last_pose_4 * diff_pose_4;

    // decompose new R,t into new roll,pitch,yaw,x,y,z in Earth frame
    cv::Mat updated_R, updated_T;
    updated_pose(cv::Rect(0, 0, 3, 3)).copyTo(updated_R);
    updated_pose(cv::Rect(3, 0, 1, 3)).copyTo(updated_T);

    if (pr_dbg) {
      std::cout << "scaled_d = " << std::endl << scaled_d << std::endl << std::endl;
      std::cout << "diff_T_extrinsic = " << std::endl << diff_T_extrinsic << std::endl << std::endl;
      std::cout << "updated_pose = " << std::endl << "\t" << updated_pose << std::endl << std::endl;
      std::cout << "updated_R = " << std::endl << "\t" << updated_R << std::endl << std::endl;
      std::cout << "updated_T = " << std::endl << "\t" << updated_T << std::endl << std::endl;
    }

    cv::Vec3f updated_R_euler = rotationMatrixToEulerAnglesZXY(updated_R);

    // convert new earth frame roll, pitch, yaw back to intrinsic form, return
    double curr_pose_x = updated_T.at<double>(0,0);
    double curr_pose_y = updated_T.at<double>(0,1);
    double curr_pose_z = updated_T.at<double>(0,2);
    double curr_pose_yaw = updated_R_euler[0] / PI * 180;
    double curr_pose_pitch = updated_R_euler[1] / PI * 180;
    double curr_pose_roll = updated_R_euler[2] / PI * 180;

    if (curr_pose_yaw < 0) {
      curr_pose_yaw = 360 + curr_pose_yaw;
    }

    if (pr_dbg) {
      std::cout << "updated_T[0] (x) = " << curr_pose_x << std::endl;
      std::cout << "updated_T[1] (y) = " << curr_pose_y << std::endl;
      std::cout << "updated_T[2] (z) = " << curr_pose_z << std::endl;
      std::cout << "updated_R_euler[3] (yaw) = " << updated_R_euler[2] << std::endl;
      std::cout << "updated_R_euler[1] (pitch) = " << updated_R_euler[0] << std::endl;
      std::cout << "updated_R_euler[2] (roll) = " << updated_R_euler[1] << std::endl;
      std::cout << "curr_pose_yaw = " << curr_pose_yaw << std::endl;
      std::cout << "curr_pose_pitch = " << curr_pose_pitch << std::endl;
      std::cout << "curr_pose_roll = " << curr_pose_roll << std::endl;
    }

    // cv::Mat template_warped;
    // cv::warpPerspective(template_img, template_warped, H_curr_cv, template_img.size());
    // cv::imshow("Template", template_img);
    // cv::imshow("Image", I_curr_resize);
    // cv::imshow("Warped Template", template_warped);
    // cv::waitKey(0);

    curr_pose_stitch[0] = curr_pose_x;
    curr_pose_stitch[1] = curr_pose_y;
    curr_pose_stitch[2] = curr_pose_z;
    curr_pose_stitch[3] = curr_pose_yaw;
    curr_pose_stitch[4] = curr_pose_pitch;
    curr_pose_stitch[5] = curr_pose_roll;
    curr_pose_stitch[6] = 1;

    // save the current pose into last, for next iteration
    last_pose_stitch[0] = curr_pose_x;
    last_pose_stitch[1] = curr_pose_y;
    last_pose_stitch[2] = curr_pose_z;
    last_pose_stitch[3] = curr_pose_yaw;
    last_pose_stitch[4] = curr_pose_pitch;
    last_pose_stitch[5] = curr_pose_roll;

    template_img = I_curr_resize.clone();

    return curr_pose_stitch;
  }
}

void BPVO::set_template(cv::Mat I_tmp, double rp)
{
  // std::cout << "[Template Update]" << std::endl << std::endl;
  template_set_fl = true;

  double left_col, right_col, top_row, bottom_row;

  if (rp > 0.0) {
    left_col = rp*I_tmp.cols;
    top_row = rp*I_tmp.rows;
    right_col = I_tmp.cols-2*rp*I_tmp.cols;
    bottom_row = I_tmp.rows-2*rp*I_tmp.rows;
  } else {
    left_col = 10;
    top_row = 10;
    right_col = I_tmp.cols - 20;
    bottom_row = I_tmp.rows - 20;
  }

  // cv::Rect bbox(rp*I_tmp.cols, rp*I_tmp.rows, I_tmp.cols-2*rp*I_tmp.cols, I_tmp.rows-2*rp*I_tmp.rows);
  cv::Rect bbox(left_col, top_row, right_col, bottom_row);
  tracker.setTemplate(I_tmp, bbox);
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f BPVO::rotationMatrixToEulerAngles(cv::Mat R)
{
      
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);

}

// see table of Tait-Bryan angles here: https://en.wikipedia.org/wiki/Euler_angles
// decompose R in Z_1 * X_2 * Y_3
cv::Vec3f BPVO::rotationMatrixToEulerAnglesZXY(cv::Mat R)
{
      
    double sy = sqrt(R.at<double>(2,0) * R.at<double>(2,0) +  R.at<double>(2,2) * R.at<double>(2,2) );
  
    double x, y, z;

    x = atan2(R.at<double>(2,1) , sy);
    y = atan2(-R.at<double>(2,0), R.at<double>(2,2));
    z = atan2(-R.at<double>(0,1), R.at<double>(1,1));

    return cv::Vec3f(z, x, y);

}