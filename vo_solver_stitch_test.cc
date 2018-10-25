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

  double rescale_factor = 1.00;
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

    // std::cout << frameCounter << " "
    // << frameFilename<< " "
    // << timestampMS << " "
    // << latitudeDeg << " "
    // << longitudeDeg << " "
    // << altitudeAMSL << " "
    // << relativeAltMeters << " "
    // << rollDeg << " "
    // << pitchDeg << " "
    // << yawDeg << " "
    // << angleFromDownDeg << " "
    // << ENUEastMeters << " "
    // << ENUNorthMeters << " "
    // << ENUAltMeters << " "
    // << ENUCompHead << std::endl;

    std::cout << "- - - - - - - - - -" << std::endl << std::endl;

    std::cout <<
    "FrameCounter: " << frameCounter << std::endl << std::endl;
    // "ENUEastMeters: " << ENUEastMeters << ", " <<
    // "ENUNorthMeters: " << ENUNorthMeters << ", " <<
    // "ENUAltMeters: " << ENUAltMeters << ", " <<
    // "ENUCompassHeading: " << ENUCompHead << std::endl;

    telem_pose[0] = ENUEastMeters;
    telem_pose[1] = ENUNorthMeters;
    telem_pose[2] = ENUAltMeters;
    telem_pose[3] = ENUCompHead;
    telem_pose[4] = pitchDeg;
    telem_pose[5] = rollDeg;

    cv::Mat im = cv::imread(frames[i]);
    cv::cvtColor(im, imbw, cv::COLOR_BGR2GRAY);

    double* bpvo_pose;
    bpvo_pose = bpvo_module.solver_stitch(telem_pose, imbw);

    std::cout << 
      "telem_pose: [" 
      << telem_pose[0] << ","
      << telem_pose[1] << ","
      << telem_pose[2] << ","
      << telem_pose[3] << ","
      << telem_pose[4] << ","
      << telem_pose[5] << "]" << std::endl;


    std::cout << 
      "bpvo_pose: [" 
      << bpvo_pose[0] << ","
      << bpvo_pose[1] << ","
      << bpvo_pose[2] << ","
      << bpvo_pose[3] << ","
      << bpvo_pose[4] << ","
      << bpvo_pose[5] << ","
      << bpvo_pose[6] << "]" << std::endl;

    std::cout << "- - - - - - - - - -" << std::endl << std::endl;

    // cv::imshow("Display window", imrz); 
    // cv::waitKey(0);
  }

  return 0;
}
