#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <memory>
#include <dirent.h>
#include <fstream>

#include <BPVO.h>

static const char* USAGE = "%s <config_file> <img_dir> <csv_file>\n";

int main(int argc, char** argv)
{
	if(argc < 4) {
	    printf(USAGE, argv[0]);
	    return 1;
	}

  	std::string dir = argv[2];
  	
  	cv::String path(dir + "/*.jpg"); //select only jpg
	std::vector<cv::String> fn;
	std::vector<cv::Mat> I_orig;
	cv::glob(path,fn,true); // recurse

	std::cout << "reading images ... " << std::endl;

	for (size_t k=0; k<fn.size(); ++k)
	{
	     cv::Mat im = cv::imread(fn[k]);
	     if (im.empty()) continue;
	     I_orig.push_back(im);
	     // std::cout << "Image: " << fn[k] << ", Num: " << k << ", Size: " << im.rows << "x" << im.cols << std::endl;
	}

	std::cout << "reading csv ... " << std::endl;

	std::ifstream csv_file(argv[3]);

	int num_frames = 986;

	cv::Mat in_telem = cv::Mat(num_frames, 4, CV_32F);

	double global_x, global_y, alt, comp_heading;

	int i = 0;
	
	while (csv_file >> global_x >> global_y >> alt >> comp_heading) {
		in_telem.at<float>(i, 0) = global_x;
		in_telem.at<float>(i, 1) = global_y;
		in_telem.at<float>(i, 2) = alt;

		if (comp_heading < 0) {
			in_telem.at<float>(i, 3) = INFINITY;
		} else {
			in_telem.at<float>(i, 3) = comp_heading;
		}

		// std::cout << "i = " << i << std::endl;
		// std::cout << "x: " << global_x << ", y: " << global_y << ", alt: " << alt << ", ch: " << comp_heading << std::endl;
		// std::cout << "x: " << in_telem.at<float>(i, 0) << ", y: " << in_telem.at<float>(i, 1) << ", alt: " << in_telem.at<float>(i, 2) << ", ch: " << in_telem.at<float>(i, 3) << std::endl;
		i++;
	}

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	std::string config_file(argv[1]);
  	BPVO bpvo_module(config_file, K);

  	std::cout << "Starting loop" << std::endl;

  	cv::Mat I_gs;

  	for (int i = 0; i < num_frames; i++) {
  		cv::cvtColor(I_orig[i], I_gs, cv::COLOR_BGR2GRAY);

  		double global_x = in_telem.at<float>(i, 0);
	    double global_y = in_telem.at<float>(i, 1);
	    double alt = in_telem.at<float>(i, 2);
	    double comp_heading = in_telem.at<float>(i, 3);

	    std::cout << "frame " << i << " input telem = " << std::endl
	    	<< "\tx: " << global_x << std::endl
	    	<< "\ty: " << global_y << std::endl
	    	<< "\talt: " << alt << std::endl
	    	<< "\tch: " << comp_heading << std::endl;

	    double * pose;

	    pose = bpvo_module.solver(global_x, global_y, alt, comp_heading, I_gs);

	    std::cout << "refined pose = " << std::endl
			<< "\t x: " << pose[0] << std::endl
			<< "\t y: " << pose[1] << std::endl
			<< "\t z: " << pose[2] << std::endl
			<< "\t h: " << pose[3] << std::endl << std::endl;
  	}
}




