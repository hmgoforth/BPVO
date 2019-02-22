# BPVO

### Dependencies

- [bitplanes](https://github.com/halismai/bitplanes)

### Compiling

	./build_vo_solver_video_test.sh

### Running

	./vo_solver_video_test config/config_cust.cfg path/to/telemetry.csv path/to/imagery/directory/ intrinsics.csv start_frame end_frame

Examples:

	./vo_solver_video_test config/config_cust.cfg ~/Documents/2018_11_13_13_38_02_AERO5_imagery/2018_11_13_13_38_02_AERO5_imagery_telem.csv  ~/Documents/2018_11_13_13_38_02_AERO5_imagery/ intrinsics.csv 1964 1980

	./vo_solver_video_test config/config_cust.cfg ~/Documents/2018_11_13_13_38_02_AERO5_imagery/2018_11_13_13_38_02_AERO5_imagery_telem.csv  ~/Documents/2018_11_13_13_38_02_AERO5_imagery/ intrinsics.csv 1922 1942

	./vo_solver_video_test config/config_cust.cfg ~/Documents/2018_11_13_13_38_02_AERO5_imagery/2018_11_13_13_38_02_AERO5_imagery_telem.csv  ~/Documents/2018_11_13_13_38_02_AERO5_imagery/ intrinsics.csv 2032 2045

	./vo_solver_video_test config/config_cust.cfg ~/Documents/DroneAgentImageryLog_AERO3_2018_10_21_14_37_12/DroneAgentImageryLog_AERO3_2018_10_21_14_37_12_telem.csv  ~/Documents/DroneAgentImageryLog_AERO3_2018_10_21_14_37_12/ intrinsics.csv 236 255

### Output

Should bring up window with frames being pasted in (like videos I sent earlier). Press any key to keep adding frames.

### BPVO Usage

Instantiating:

	BPVO bpvo(config_file, K, max_ang_from_down, max_frame_before_tracker_refresh);

max_ang_from_down: as we've previously defined (I use 25 degrees in vo_solver_video_test)

max_frame_before_tracker_refresh: if there is this many corrupted/bad frames in a row, then BPVO will have to 'reset' tracking because there is now no overlap between the next frame and the last good frame. BPVO.solver will return the telemetry input as refined pose for the next frame, and then continue tracking. (I found that 6 works okay for this number.)

Solver:

	BPVO.solver(curr_pose, curr_img)

curr_pose: length 6 array of [roll deg, pitch deg, yaw deg, east meters, north meters, up meters] as defined in all telemetry csv files.

curr_img: current 1080p RGB frame

returns: tuple of refined pose (length 6 array of same format as curr_pose) and int flag. Flag values: refined pose is good (0), max angle from down exceeded, don't use refined pose (1), corrupted/bad frame, don't use refined pose (2).

<!-- ### More Details

Instantiating a BPVO module:

BPVO bpvo_module(config_file, K);

Where std::string::config_file points to a .cfg for Bitplane tracker parameters (one of
these is provided in the config/ folder), and cv::Mat K is a 3x3 camera instrinsic matrix.

[More information on calibrating a camera to get the intrinsic matrix](https://www.mathworks.com/help/vision/ug/camera-calibration.html)

One way to get the intrinsic matrix is by performing proper camera calibration. There are
simpler ways to construct a slightly inaccurate but sufficient intrinsic matrix by
just knowing the focal length (in pixels) of a camera, and the height and width (in pixels) of
the images returned from the camera.

The bpvo_module.solver(global_x, global_y, alt, comp_heading, I) function will
use the telemetry (global_x, global_y, alt, comp_heading) and the current
camera image (cv::Mat I) to compute a refined telemetry estimate. The estimate is
returned as a pointer to a 1D array containing refined global_x, global_y, alt, comp_heading.

The input image to bpvo_module.solver must be non-null.

Any of the telemetry inputs can be specified as INFINITY. In this case, the function will ignore these
inputs, but still use the current image I to compute a refined pose. -->

<!-- ### Simulation Test

Compilation: ./build_vo_solver_dir.sh

Running: ./vo_solver_dir config/config_cust.cfg path/to/frames/directory/ data/sm_telem.txt

Expected Output:

```
reading images ... 
reading csv ... 
BitPlanes Parameters:
MultiChannelFunction = BitPlanes
ParameterTolerance = 0.00015
FunctionTolerance = 0.0001
NumLevels = 4
sigma = 1.618
verbose = 0
subsampling = 2

Starting loop
frame 0 input telem = 
	x: -1510.65
	y: -2268.43
	alt: 436.928
	ch: 0
refined pose = 
	 x: -1510.65
	 y: -2268.43
	 z: 436.928
	 h: 0

.
.
.
.
.

frame 169 input telem = 
	x: -1509.08
	y: -461.968
	alt: 489.428
	ch: inf
refined pose = 
	 x: -1519.42
	 y: -474.938
	 z: 489.314
	 h: 96.8461

.
.
.
.
.
.

```
 -->