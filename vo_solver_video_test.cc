#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <memory>

#include <BPVO.h>

static const char* USAGE = "%s <config_file> <video_name> <frame_start> [output file]\n";

int main(int argc, char** argv)
{
  if(argc < 4) {
    printf(USAGE, argv[0]);
    return 1;
  }

  // std::cout << "EIGEN_WORLD_VERSION: " << EIGEN_WORLD_VERSION << std::endl;
  // std::cout << "EIGEN_MAJOR_VERSION: " << EIGEN_MAJOR_VERSION << std::endl;
  // std::cout << "EIGEN_MINOR_VERSION: " << EIGEN_MINOR_VERSION << std::endl;

  std::string config_file(argv[1]);
  std::string vname(argv[2]);
  int frame_start = bp::str2num<int>(argv[3]);

  std::string output_video;
  if(argc > 4)
    output_video = std::string(argv[4]);

  cv::VideoCapture vcap(vname);
  THROW_ERROR_IF(!vcap.isOpened(), ("failed to open " + vname).c_str());

  // skip the few first messed up frames from the video compression
  cv::Mat I_orig, I;
  for(int i = 0; i < frame_start; ++i) {
    vcap >> I_orig;
    THROW_ERROR_IF(I_orig.empty(), "video was cut too short\n");
  }

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);

  BPVO bpvo_module(config_file, K);

  double total_time = 0.0;
  int f_i = 0;

  vcap >> I_orig;

  std::cout << "Starting loop" << std::endl;

  char text_buf[128];

  while(!I_orig.empty()) {

    cv::cvtColor(I_orig, I, cv::COLOR_BGR2GRAY);

    bp::Timer timer;

    double global_x = -1;
    double global_y = -1;
    double alt = 1;
    double comp_heading = 0;

    double * pose;

    pose = bpvo_module.solver(global_x, global_y, alt, comp_heading, I);

    std::cout << "pose = " << std::endl
      << "\t x: " << pose[0] << std::endl
      << "\t y: " << pose[1] << std::endl
      << "\t z: " << pose[2] << std::endl
      << "\t h: " << pose[3] << std::endl;

    total_time += timer.stop().count() / 1000.0;

    snprintf(text_buf, sizeof(text_buf), "Frame %05d @ %0.2f Hz", f_i, f_i / total_time);

    Info("%s\n", text_buf);

    vcap >> I_orig;
    ++f_i;

  }

  return 0;
}
