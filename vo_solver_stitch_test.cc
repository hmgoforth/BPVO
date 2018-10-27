#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>

#include <BPVO.h>

static const char* USAGE = "%s <config_file> <frames_dir> <frame_start> <telemetry_csv>\n";

int main(int argc, char** argv)
{
  if(argc < 4) {
    printf(USAGE, argv[0]);
    return 1;
  }

  // test input
  std::cout << "\tconfig_file = " << argv[1] << std::endl;
  std::cout << "\tframes_dir = " << argv[2] << std::endl;
  std::cout << "\tframe_start = " << argv[3] << std::endl;
  std::cout << "\ttelemetry_csv = " << argv[4] << std::endl;
  std::cout << std::endl;

  std::string config_file(argv[1]);
  cv::String frames_dir = argv[2];
  int frame_start = bp::str2num<int>(argv[3]);
  std::string telemetry_csv = argv[4];

  std::vector<cv::String> frames;
  cv::glob(frames_dir, frames);
  std::ifstream infile(telemetry_csv);
  std::stringstream line_stream;
  std::string line, line_item;

  // csv variables
  int frameCounter, timestampMS;
  double latitudeDeg, longitudeDeg, altitudeAMSL,
        relativeAltMeters, rollDeg, pitchDeg, yawDeg,
        angleFromDownDeg, ENUEastMeters, ENUNorthMeters,
        ENUAltMeters, ENUCompHead;
  std::string frameFilename;

  cv::Mat imrz, imbw;

  double rescale_factor = 0.25;
  double fc[2] = {1371.02208647 * rescale_factor, 1370.86986366  * rescale_factor};
  double cc[2] = {960 * rescale_factor, 540 * rescale_factor};
  double kkparams[3][3] = {{fc[0], 0, cc[0]}, {0, fc[1], cc[1]}, {0, 0, 1.0000}};
  cv::Mat K = cv::Mat(3, 3, CV_64F, &kkparams);
  std::cout << "K = " << std::endl << K << std::endl;

  BPVO bpvo_module(config_file, K);

  double telem_pose[6];

  // read header
  std::getline(infile, line, '\n');

  // skip starting lines
  for (int i=0; i<frame_start - 1; i++) {
    std::getline(infile, line, '\n');
  }

  std::vector<cv::Mat> image_batch;
  std::vector<std::string> image_batch_fnames;
  cv::Mat stitch_output;

  int frames_read = 0;
  int max_frames_per_batch = 10;
  std::tuple <cv::Mat, int, bool> retval;

  for (size_t i=(frame_start - 1); i<frames.size(); i++) {
    std::getline(infile, line, '\n');
    line_stream.str(line);
    
    std::getline(line_stream, line_item, ',');
    frameCounter = std::stoi(line_item);

    std::getline(line_stream, line_item, ',');
    frameFilename = line_item;

    std::getline(line_stream, line_item, ',');
    timestampMS = std::stoi(line_item);

    std::getline(line_stream, line_item, ',');
    latitudeDeg = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    longitudeDeg = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    altitudeAMSL = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    relativeAltMeters = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    rollDeg = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    pitchDeg = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    yawDeg = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    angleFromDownDeg = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    ENUEastMeters = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    ENUNorthMeters = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    ENUAltMeters = std::stof(line_item);

    std::getline(line_stream, line_item, ',');
    ENUCompHead = std::stof(line_item);

    line_stream.clear();

    std::cout << "- - - - - - - - - -" << std::endl << std::endl;

    std::cout <<
    "FrameCounter: " << frameCounter << std::endl << std::endl;

    cv::Mat im = cv::imread(frames[i]);
    cv::resize(im, imrz, cv::Size(), rescale_factor, rescale_factor);
    cv::imshow("Current Image", imrz);
    cv::waitKey(0);

    std::cout << "Frames Read: " << frames_read << std::endl;

    if (frames_read == max_frames_per_batch) {
      std::cout << "batch full, calling batch stitch_output" << std::endl;
      retval = bpvo_module.batch_stitch(image_batch);
      cv::Mat stitch_output = std::get<0>(retval);
      int iden_ind = std::get<1>(retval);
      bool bad_out = std::get<2>(retval);

      if (!bad_out) {
        int half_height_img = image_batch[0].rows / 2;
        int half_width_img = image_batch[0].cols / 2;
        int half_width_stitch = stitch_output.cols / 2;
        int half_height_stitch = stitch_output.rows / 2;

        cv::Mat iden_img = image_batch[iden_ind].clone();
        cv::circle(iden_img, 
                  cv::Point(half_width_img,half_height_img),
                  6, cv::Scalar(0,0,255),CV_FILLED, 8,0);

        cv::imshow("Iden image", iden_img);

        cv::circle(stitch_output, 
              cv::Point(half_width_stitch,half_height_stitch),
              6, cv::Scalar(0,0,255),CV_FILLED, 8,0);

        cv::imshow("stitch output", stitch_output);
      }

      frames_read = 0;
      image_batch.clear();
      image_batch_fnames.clear();
    }

    image_batch.push_back(imrz.clone());
    image_batch_fnames.push_back(frames[i]);

    frames_read = frames_read + 1;

    std::cout << "- - - - - - - - - -" << std::endl << std::endl;
  }

  return 0;
}
