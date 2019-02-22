#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>

#include <BPVO.h>

static const char* USAGE = "%s <bp_config_file> <telemetry_file> <frames_dir> <intrinsics_file> <start_frame> <end_frame> \n";

/*
input:
  folder with .csv telemetry and frames
  path to intrinsics file
  start frame and end frame
output:
  print old and new rpy enu, mosaic figure with images pasted in
*/

cv::Mat add_frame_to_mosaic(cv::Mat mosaic, cv::Mat pose_cam, cv::Mat im_rgb, cv::Mat K, int res, int buffer_x, int buffer_y, double min_x, double min_y) {
	// "bake" a frame into the current mosaic based on the camera pose for the frame

	// '''
	// Steps for adding frame to mosaic:
	// 	1. Invert pose to get camera matrix version of R,t
	// 	2. M = K * [R | t]
	// 	3. Since all 3D points are assumed to be on world plane Z=0,
	// 	   then we can remove 3rd column of M, creating a homography
	// 	   between 3D points on world ground plane to the image plane of current frame
	// 	4. Invert this homography to get mapping from image pixels to world coordinates
	// 	5. Scale by the mosaic resolution
	// 	6. Translate before adding to mosaic, to make sure the image will be added within the mosaic boundaries
	// 	7. Warp image and add to mosaic
	// '''

	cv::Mat pose_cam_inv = pose_cam.inv(); // step 1
	cv::Mat camera_mat = K * pose_cam_inv.colRange(0,4).rowRange(0,3); // step 2

	// step 3
	cv::Mat H_world_to_pix = cv::Mat::eye(3,3,CV_64F);
	camera_mat.colRange(0,2).rowRange(0,3).copyTo(H_world_to_pix.colRange(0,2).rowRange(0,3));
	camera_mat.colRange(3,4).rowRange(0,3).copyTo(H_world_to_pix.colRange(2,3).rowRange(0,3));

	// std::cout << "pose_cam" << std::endl;
	// std::cout << pose_cam << std::endl;
	// std::cout << "pose_cam_inv" << std::endl;
	// std::cout << pose_cam_inv << std::endl;
	// std::cout << "camera_mat" << std::endl;
	// std::cout << camera_mat << std::endl;
	// std::cout << "H_world_to_pix" << std::endl;
	// std::cout << H_world_to_pix << std::endl;

	// step 4
	cv::Mat H_pix_to_world = H_world_to_pix.inv();

	// step 5
	cv::Mat res_mat = cv::Mat::eye(3,3,CV_64F);
	res_mat.at<double>(0,0) = (double)res;
	res_mat.at<double>(1,1) = (double)res;

	cv::Mat H_pix_to_world_scale = res_mat * H_pix_to_world;

	// step 6
	cv::Mat trans = cv::Mat::eye(3,3,CV_64F);
	trans.at<double>(0,2) = (double)(res * (buffer_x - min_x));
	trans.at<double>(1,2) = (double)(res * (buffer_y - min_y));

	cv::Mat H_pix_to_world_scale_trans = trans * H_pix_to_world_scale;

	int mosaic_y_size = mosaic.rows;
	int mosaic_x_size = mosaic.cols;

	cv::Mat mask = cv::Mat(im_rgb.rows,im_rgb.cols,CV_8U, cv::Scalar(1));

	cv::Mat warped_im, warped_mask;

	// std::cout << "H_pix_to_world_scale_trans" << std::endl;
	// std::cout << H_pix_to_world_scale_trans << std::endl;

	// step 7
	cv::warpPerspective(im_rgb, warped_im, H_pix_to_world_scale_trans, cv::Size(mosaic_x_size, mosaic_y_size));
	cv::warpPerspective(mask, warped_mask, H_pix_to_world_scale_trans, cv::Size(mosaic_x_size, mosaic_y_size));

	warped_im.copyTo(mosaic, warped_mask);

	// cv::Mat warped_im_sm, warped_im_sm_flip, warped_mask_sm, warped_mask_sm_flip, mosaic_sm, mosaic_sm_flip;
	// cv::resize(warped_im, warped_im_sm, cv::Size(mosaic.cols/4, mosaic.rows/4));
	// cv::resize(warped_mask, warped_mask_sm, cv::Size(mosaic.cols/4, mosaic.rows/4));
	// cv::resize(mosaic, mosaic_sm, cv::Size(mosaic.cols/4, mosaic.rows/4));
	// cv::flip(warped_im_sm, warped_im_sm_flip, 0);
	// cv::flip(warped_mask_sm, warped_mask_sm_flip, 0);
	// cv::flip(mosaic_sm, mosaic_sm_flip, 0);
	// cv::imshow("warped_im", warped_im_sm_flip);
	// cv::imshow("warped_mask", warped_mask_sm_flip);
	// cv::imshow("mosaic", mosaic_sm_flip);
	// cv::waitKey(0);

	return mosaic;
}

int main(int argc, char** argv) {
	if (argc < 7) {
		printf(USAGE, argv[0]);
		return 1;
	}

	std::cout << "bp_config_file = " << argv[1] << std::endl;
	std::cout << "telemetry_file = " << argv[2] << std::endl;
	std::cout << "frames_dir = " << argv[3] << std::endl;
	std::cout << "intrinsics_file = " << argv[4] << std::endl;
	std::cout << "start_frame = " << argv[5] << std::endl;
	std::cout << "end_frame = " << argv[6] << std::endl;
	std::cout << std::endl;

	std::string config_file(argv[1]);
	std::string telemetry_file(argv[2]);
	cv::String frames_dir = argv[3];
	std::string intrinsics(argv[4]);
	int start_frame = bp::str2num<int>(argv[5]);
	int end_frame = bp::str2num<int>(argv[6]);
	std::string img_fname;

	std::vector<cv::String> frames;
	cv::glob(frames_dir, frames);
	std::ifstream telem(telemetry_file);
	std::ifstream intrins(intrinsics);
	std::stringstream line_stream;
	std::string line, line_item;

	int num_data = end_frame - start_frame + 1;
	int frameCounter[num_data];
	long timestampMS[num_data];
	double latitudeDeg[num_data], longitudeDeg[num_data], altitudeAMSL[num_data],
	    relativeAltMeters[num_data], rollDeg[num_data], pitchDeg[num_data], yawDeg[num_data],
	    angleFromDownDeg[num_data], ENUEastMeters[num_data], ENUNorthMeters[num_data],
	    ENUAltMeters[num_data], ENUCompHead[num_data];
	std::vector<std::string> frameFilename;
	double curr_pose[6];
	double* out_pose;

	// read instrinsics
	double fc[2], cc[2];

	std::getline(intrins, line, '\n');
	line_stream.str(line);
	std::getline(line_stream, line_item, ',');
	fc[0] = std::stof(line_item);
	std::getline(line_stream, line_item, ',');
	fc[1] = std::stof(line_item);
	std::getline(line_stream, line_item, ',');
	cc[0] = std::stof(line_item);
	std::getline(line_stream, line_item, '\n');
	cc[1] = std::stof(line_item);

	line_stream.clear();

	double kkparams[3][3] = {{fc[0], 0, cc[0]}, {0, fc[1], cc[1]}, {0, 0, 1.0000}};
	cv::Mat K = cv::Mat(3, 3, CV_64F, &kkparams);
	std::cout << "K = " << std::endl << K << std::endl;

	double max_ang_from_down = 25.0;
	int max_frame_before_tracker_refresh = 6;

	BPVO bpvo(config_file, K, max_ang_from_down, max_frame_before_tracker_refresh);

	// read header
	std::getline(telem, line, '\n');

	// skip to starting frame
	std::getline(telem, line, '\n');
	line_stream.str(line);
	std::getline(line_stream, line_item, ',');
	int first_frame = std::stoi(line_item);

	telem.seekg(0, std::ios::beg); // undo above getline before starting to skip lines
	// read header
	std::getline(telem, line, '\n');

	for (int i=0; i<(start_frame - first_frame); i++) {
	std::getline(telem, line, '\n');
	}

	float min_x, min_y, max_x, max_y;

	// reading telem file
	for (int i=0; i<(end_frame - start_frame + 1); i++) {
		std::getline(telem, line, '\n');
	line_stream.str(line);

	std::getline(line_stream, line_item, ',');
	frameCounter[i] = std::stoi(line_item);

	std::getline(line_stream, line_item, ',');
	frameFilename.push_back(line_item);

	std::getline(line_stream, line_item, ',');
	timestampMS[i] = std::stol(line_item);

	std::getline(line_stream, line_item, ',');
	latitudeDeg[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	longitudeDeg[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	altitudeAMSL[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	relativeAltMeters[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	rollDeg[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	pitchDeg[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	yawDeg[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	angleFromDownDeg[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	ENUEastMeters[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	ENUNorthMeters[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	ENUAltMeters[i] = std::stof(line_item);

	std::getline(line_stream, line_item, ',');
	ENUCompHead[i] = std::stof(line_item);

	if (i == 0) {
		min_x = ENUEastMeters[i];
		max_x = ENUEastMeters[i];
		min_y = ENUNorthMeters[i];
		max_y = ENUNorthMeters[i];
	} else {
		if (ENUEastMeters[i] < min_x) {
			min_x = ENUEastMeters[i];
		}

		if (ENUEastMeters[i] > max_x) {
			max_x = ENUEastMeters[i];
		}

		if (ENUNorthMeters[i] < min_y) {
			min_y = ENUNorthMeters[i];
		}

		if (ENUNorthMeters[i] > max_y) {
			max_y = ENUNorthMeters[i];
		}
	}

	line_stream.clear();
	}

	int buffer_x = 100.0;
	int buffer_y = 100.0;
	int res = 10; // resolution pixels per meter

	int mosaic_x_size = res * (buffer_x + (int)std::round(max_x - min_x) + buffer_x);
	int mosaic_y_size = res * (buffer_y + (int)std::round(max_y - min_y) + buffer_y);

	std::cout<<"mosaic_x_size: "<<mosaic_x_size<<std::endl;
	std::cout<<"mosaic_y_size: "<<mosaic_y_size<<std::endl;

	cv::Mat mosaic = cv::Mat::zeros(mosaic_y_size, mosaic_x_size, CV_8UC3);
	cv::Mat curr_pose_mat;

	for (int i=0; i<(end_frame - start_frame + 1); i++) {
		std::cout << "- - - - - - - - - -" << std::endl << std::endl;

		std::cout <<
		"FrameCounter: " << frameCounter[i] << std::endl << std::endl;

		std::string frames_dir_std = frames_dir.operator std::string();
		img_fname = frames_dir_std + frameFilename[i];
		cv::Mat im = cv::imread(img_fname);

		curr_pose[0] = rollDeg[i];
		curr_pose[1] = pitchDeg[i];
		curr_pose[2] = yawDeg[i];
		curr_pose[3] = ENUEastMeters[i];
		curr_pose[4] = ENUNorthMeters[i];
		curr_pose[5] = ENUAltMeters[i];

		std::tuple<double*,int> ret;
		ret = bpvo.solver(curr_pose, im);
		out_pose = std::get<0>(ret);
		int retval = std::get<1>(ret);

		curr_pose_mat = bpvo.pose_mat_from_telem(out_pose);

		// std::cout << "rollDeg[i]: " << curr_pose[0] << std::endl;
		// std::cout << "pitchDeg[i]: " << curr_pose[1] << std::endl;
		// std::cout << "yawDeg[i]: " << curr_pose[2] << std::endl;
		// std::cout << "ENUEastMeters[i]: " << curr_pose[3] << std::endl;
		// std::cout << "ENUNorthMeters[i]: " << curr_pose[4] << std::endl;
		// std::cout << "ENUAltMeters[i]: " << curr_pose[5] << std::endl;

		if (retval == 0) {
			mosaic = add_frame_to_mosaic(mosaic, curr_pose_mat, im, K, res, buffer_x, buffer_y, min_x, min_y);
		}

		cv::Mat mosaic_sm, mosaic_sm_flip;
		cv::resize(mosaic, mosaic_sm, cv::Size(mosaic.cols/3, mosaic.rows/3));
		cv::flip(mosaic_sm, mosaic_sm_flip, 0);
		cv::imshow("mosaic", mosaic_sm_flip);
		cv::waitKey(0);
	}

	return 0;
}