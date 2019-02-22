#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

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

BPVO::BPVO(std::string config_file, cv::Mat intrinsics, double max_from_down, int max_before_tracker_refresh)
  : tracker(bp::AlgorithmParameters::FromConfigFile(config_file)),
    K(intrinsics),
    H_curr(bp::Matrix33f::Identity()),
    class_rp(0.2),
    last_img_set_fl(false),
    max_from_down(max_from_down),
    max_before_tracker_refresh(max_before_tracker_refresh),
    skipped_in_a_row(0)
{
	bp::AlgorithmParameters params = bp::AlgorithmParameters::FromConfigFile(config_file);
	std::cout << "BitPlanes Parameters:" << std::endl;
	std::cout << params << std::endl << std::endl;
}

std::tuple<double*,int> BPVO::solver(double* telem_curr_pose, cv::Mat curr_img) {
	// input:
	// telem_curr_pose: current telemetry [roll, pitch, yaw, east, north, up]
	// curr_img: current frame

	// output:
	// refined pose [roll, pitch, yaw, east, north, up]
	// error flag: no error (0), max from down exceeded (1), bad frame (2)

	bool visualize = false;

	if (skipped_in_a_row >= max_before_tracker_refresh) {
		std::cout << "refreshing tracker" << std::endl;
		skipped_in_a_row = 0;
		std::copy(telem_curr_pose, telem_curr_pose + 6, last_pose);
		curr_img.copyTo(last_img);
	}

	if (!last_img_set_fl) {
		curr_img.copyTo(last_img);
		std::copy(telem_curr_pose, telem_curr_pose + 6, last_pose);
		last_img_set_fl = true;
	} else {
		double abs_roll = std::abs(telem_curr_pose[0]);
		double abs_pitch = std::abs(telem_curr_pose[1]);

		if (abs_roll > max_from_down || abs_pitch > max_from_down) {
			std::cout << "exceeded max down angle" << std::endl;
			skipped_in_a_row += 1;
			return std::make_tuple(telem_curr_pose, 1);
		}

		std::tuple<cv::Mat, int> H_ret;
		H_ret = BPVO::find_H_sift(curr_img,visualize);

		cv::Mat H_last_to_curr = std::get<0>(H_ret);
		int H_bad = std::get<1>(H_ret);

		// double temparr[3][3] = {{1.01117713e+00,1.71099460e-02,5.93614009e+00},
		// 			      {2.14264573e-02,9.75923779e-01,1.24424284e+02},
		// 			 	  {9.56987712e-06,2.04678403e-05,1.00000000e+00}};
		// H_last_to_curr = cv::Mat(3,3,CV_64F);
		// std::memcpy(H_last_to_curr.data, temparr, 3*3*sizeof(double));

		if (H_bad == 0) {
			skipped_in_a_row = 0;

			cv::Mat curr_pose_mat; // [R | t] format
			cv::Mat last_pose_mat; // [R | t] format
			double* curr_pose; // rpy enu format

			last_pose_mat = pose_mat_from_telem(last_pose);
			curr_pose_mat = combine_H_and_last_pose(H_last_to_curr, last_pose_mat);
			curr_pose = rpy_enu_from_pose_mat(curr_pose_mat);

			// std::cout << "H" << std::endl;
			// std::cout << H_last_to_curr << std::endl;
			// std::cout << "last_pose" << std::endl;
			// std::cout <<last_pose[0]<<","<<last_pose[1]<<","<<last_pose[2] << ","<<last_pose[3]<<","<<last_pose[4]<<","<<last_pose[5]<< std::endl;
			// std::cout << "last_pose_mat" << std::endl;
			// std::cout << last_pose_mat << std::endl;
			// std::cout << "curr_pose_mat" << std::endl;
			// std::cout << curr_pose_mat << std::endl;
			// std::cout << "curr_pose" << std::endl;
			// std::cout << curr_pose << std::endl;

			std::copy(curr_pose, curr_pose + 6, last_pose);
			curr_img.copyTo(last_img);
		} else {
			// bad frame (corruption, too low texture)
			std::cout << "skipping bad frame" << std::endl;
			skipped_in_a_row += 1;

			return std::make_tuple(telem_curr_pose, 2);
		}
	}

	return std::make_tuple(last_pose, 0);
}

double* BPVO::rpy_enu_from_pose_mat(cv::Mat pose_mat) {
	// convert pose_mat [R | t] into RPY ENU [6] array

	static double pose[6];

	pose[3] = pose_mat.at<double>(0,3);
	pose[4] = pose_mat.at<double>(1,3);
	pose[5] = pose_mat.at<double>(2,3);

	// undo steps 1, 2, and 6 from rot_mat_from_telem
	// R_cam = [180deg_y][90deg_z][yaw][pitch][roll][90deg_z]
	// R_ypr = [90deg_z]^-1 * [180deg_y]^-1 * R_cam * [90deg_z]^-1

	cv::Mat R_cam;
	pose_mat.colRange(0,3).rowRange(0,3).copyTo(R_cam);

	cv::Mat R_uav_to_cam = rot_mat_deg(0,0,90);
	cv::Mat R_uav_yaw_correction = rot_mat_deg(0,0,90);
	cv::Mat R_world_to_uav_flip = rot_mat_deg(0,180,0);

	cv::Mat R_pose = R_uav_yaw_correction.inv() * R_world_to_uav_flip.inv() * R_cam * R_uav_to_cam.inv();

	cv::Vec3f rpy = rotation_matrix_to_euler_angles(R_pose);

	// convert to degrees
	pose[0] = rpy[0] / M_PI * 180;
	pose[1] = rpy[1] / M_PI * 180;
	pose[2] = rpy[2] / M_PI * 180;

	return pose;
}

cv::Mat BPVO::combine_H_and_last_pose(cv::Mat H_last_to_curr, cv::Mat last_pose_mat) {
	// determine curr_pose (R, t) based on last_pose (R,t) and homography (pixel-based)

	// invert pose to get extrinsics
	cv::Mat last_pose_cam_mat = last_pose_mat.inv();


	// decompose homography into R,t
	std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;
	cv::decomposeHomographyMat(H_last_to_curr, K, Rs_decomp, ts_decomp, normals_decomp);

	// choose correct solution based on normal. For nadir UAV, ground plane has normal (0,0,1)
	// so choose normal which has largest Z-component
	double largest_Z = 0;
	double largest_Z_ind = 0;
	for (size_t i=0; i < Rs_decomp.size(); i++) {
		if (normals_decomp[i].at<double>(2) > largest_Z) {
			largest_Z_ind = i;
			largest_Z = normals_decomp[i].at<double>(2);
		}
	}

	// For R,t between last frame and curr frame

	cv::Mat Rt_last_to_curr = cv::Mat::eye(4,4,CV_64F);

	double last_altitude = last_pose_mat.at<double>(2,3);
	cv::Mat rot = Rs_decomp[largest_Z_ind];

	// scale translation by distance from plane in last image i.e. altitude
	cv::Mat trans = ts_decomp[largest_Z_ind] * last_altitude;

	rot.copyTo(Rt_last_to_curr.colRange(0,3).rowRange(0,3));
	trans.copyTo(Rt_last_to_curr.colRange(3,4).rowRange(0,3));

	// odometry (inter-frame motion with current pose)
	cv::Mat curr_pose_cam_mat = Rt_last_to_curr * last_pose_cam_mat;

	// invert to go from extrinsics to pose
	cv::Mat curr_pose_mat = curr_pose_cam_mat.inv();

	return curr_pose_mat;
}

std::tuple<cv::Mat,int> BPVO::find_H_sift(cv::Mat curr_img,bool visualize) {
	// find H from last_img to curr_img using SIFT

	std::tuple<cv::Mat, int> ret;

	size_t MIN_MATCH_COUNT = 20;

	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
	cv::Mat curr_img_g, last_img_g, curr_img_g_rz, last_img_g_rz;

	cv::cvtColor(curr_img, curr_img_g, CV_BGR2GRAY);
	cv::cvtColor(last_img, last_img_g, CV_BGR2GRAY);

	double scaling = 0.6;

	cv::resize(curr_img_g, curr_img_g_rz, cv::Size(0,0), scaling, scaling);
	cv::resize(last_img_g, last_img_g_rz, cv::Size(0,0), scaling, scaling);

	std::vector<cv::KeyPoint> kp_1, kp_2;
	cv::Mat des_1, des_2;

	f2d->detectAndCompute(last_img_g_rz, cv::Mat(), kp_1, des_1);
	f2d->detectAndCompute(curr_img_g_rz, cv::Mat(), kp_2, des_2);

	const cv::Ptr<cv::flann::IndexParams>& index_params=new cv::flann::KDTreeIndexParams(5);
	const cv::Ptr<cv::flann::SearchParams>& search_params=new cv::flann::SearchParams(50,0,true);
	cv::FlannBasedMatcher matcher(index_params, search_params);
	std::vector<std::vector<cv::DMatch>> matches;
	matcher.knnMatch(des_1, des_2, matches, 2); // find 2 closest matches

	// std::cout << matches.size() << " total matches" << std::endl;

	// Lowe test
	std::vector<cv::DMatch> good_matches;

	const float ratio_thresh = 0.7;
	for (size_t i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < (ratio_thresh * matches[i][1].distance)) {
			good_matches.push_back(matches[i][0]);
		}
	}

	cv::Mat H;

	if (good_matches.size() > MIN_MATCH_COUNT) {

		std::cout << good_matches.size() << " good matches" << std::endl;

		std::vector<cv::Point2f> pts_1;
		std::vector<cv::Point2f> pts_2;

		for(size_t i = 0; i < good_matches.size(); i++) {
			//-- Get the keypoints from the good matches
			pts_1.push_back(kp_1[good_matches[i].queryIdx].pt);
			pts_2.push_back(kp_2[good_matches[i].trainIdx].pt);
		}

		H = findHomography(pts_1, pts_2, cv::RANSAC, 2.0);

		if (H.empty()) {
			std::cout << "Could not RANSAC H from good matches" << std::endl;
			H = cv::Mat::eye(3,3,CV_64F);
			return std::make_tuple(H, 1);
		}

		if (visualize) {
			cv::Mat img_matches;

			cv::drawMatches(last_img_g_rz, kp_1, curr_img_g_rz, kp_2,
			           good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
			           std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<cv::Point2f> last_corners(4);

			last_corners[0] = cvPoint(0,0); 
			last_corners[1] = cvPoint(last_img_g_rz.cols, 0);
			last_corners[2] = cvPoint(last_img_g_rz.cols, last_img_g_rz.rows);
			last_corners[3] = cvPoint(0, last_img_g_rz.rows);

			std::vector<cv::Point2f> curr_corners(4);
			perspectiveTransform(last_corners, curr_corners, H);

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			cv::line(img_matches, curr_corners[0] + cv::Point2f(last_img_g_rz.cols, 0), curr_corners[1] + cv::Point2f(last_img_g_rz.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			cv::line(img_matches, curr_corners[1] + cv::Point2f(last_img_g_rz.cols, 0), curr_corners[2] + cv::Point2f(last_img_g_rz.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			cv::line(img_matches, curr_corners[2] + cv::Point2f(last_img_g_rz.cols, 0), curr_corners[3] + cv::Point2f(last_img_g_rz.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			cv::line(img_matches, curr_corners[3] + cv::Point2f(last_img_g_rz.cols, 0), curr_corners[0] + cv::Point2f(last_img_g_rz.cols, 0), cv::Scalar( 0, 255, 0), 4 );
			//-- Show detected matches
			imshow( "matches", img_matches );
			cv::waitKey(0);
		}

	} else {
		std::cout << "Not enough matches" << std::endl;
		H = cv::Mat::eye(3,3,CV_64F);
		return std::make_tuple(H, 1);
	}

	// rescaling H
	cv::Mat S = cv::Mat::eye(3,3,CV_64F);
	S.at<double>(0,0) = scaling;
	S.at<double>(1,1) = scaling;

	cv::Mat H_rescaled = S.inv() * H * S;

	return std::make_tuple(H_rescaled, 0);
}

cv::Mat BPVO::pose_mat_from_telem(double* pose) {
	// create (R,t) from RPY ENY array

	cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);

	pose_mat.at<double>(0,3) = pose[3];
	pose_mat.at<double>(1,3) = pose[4];
	pose_mat.at<double>(2,3) = pose[5];

	cv::Mat R = rot_mat_from_telem(pose[0], pose[1], pose[2]);

	cv::Mat temp = pose_mat.colRange(0,3).rowRange(0,3);
	R.copyTo(temp);

	return pose_mat;
}

cv::Mat BPVO::rot_mat_from_telem(double roll, double pitch, double yaw) {
	// convert yaw, pitch, roll of UAV to a camera orientation R
	// return: R
	// 
	// operations needed to transform from ENU world coordinate system to camera coordinate system:
	// 	1. Rotate 180deg about y-axis
	// 	2. Rotate 90deg about z-axis, putting the x-axis in same direction as y-axis initially was
	// 	3. Rotate yaw degrees about z-axis
	// 	4. Rotate pitch degrees about y-axis
	// 	5. Rotate roll degrees about x-axis
	// 	6. Rotate 90 degrees about z-axis one final time to get into camera coordinate system (camera coordinate system is 90 deg from UAV coordinate)

	// coordinate transform from world to camera
	cv::Mat R_uav_to_cam = rot_mat_deg(0,0,90); // step 6
	cv::Mat R_uav_roll = rot_mat_deg(roll, 0, 0); // step 5
	cv::Mat R_uav_pitch = rot_mat_deg(0, pitch, 0); // step 4
	cv::Mat R_uav_yaw = rot_mat_deg(0, 0, yaw + 90); // step 2 and 3 together
	// add 90 above, since x-axis of UAV should face north for yawDegrees = 0, but after
	// world_to_uav_flip, the uav x-axis faces west

	cv::Mat R_world_to_uav_flip = rot_mat_deg(0, 180, 0); // step 1

	// pre-multiply form for rotation
	cv::Mat R_all = R_world_to_uav_flip * R_uav_yaw * R_uav_pitch * R_uav_roll * R_uav_to_cam;

	return R_all;
}

cv::Mat BPVO::rot_mat_deg(double x, double y, double z) {
	// create rotation matrix from x, y, z angles in degrees

	double x_rad = x / 180 * M_PI;
	double y_rad = y / 180 * M_PI;
	double z_rad = z / 180 * M_PI;

	cv::Mat Rx = cv::Mat::eye(3, 3, CV_64F);
	Rx.at<double>(1,1) = cos(x_rad);
	Rx.at<double>(2,2) = cos(x_rad);
	Rx.at<double>(1,2) = -sin(x_rad);
	Rx.at<double>(2,1) = sin(x_rad);

	cv::Mat Ry = cv::Mat::eye(3, 3, CV_64F);
	Ry.at<double>(0,0) = cos(y_rad);
	Ry.at<double>(2,2) = cos(y_rad);
	Ry.at<double>(0,2) = sin(y_rad);
	Ry.at<double>(2,0) = -sin(y_rad);

	cv::Mat Rz = cv::Mat::eye(3, 3, CV_64F);
	Rz.at<double>(0,0) = cos(z_rad);
	Rz.at<double>(1,1) = cos(z_rad);
	Rz.at<double>(0,1) = -sin(z_rad);
	Rz.at<double>(1,0) = sin(z_rad);

	return Rz * Ry * Rx;
}

cv::Vec3f BPVO::rotation_matrix_to_euler_angles(cv::Mat R) {
      
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
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