# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cj/code/play_opencv/KalmanFilter/KalmanTest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cj/code/play_opencv/KalmanFilter/KalmanTest

# Include any dependencies generated for this target.
include CMakeFiles/kalman.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kalman.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kalman.dir/flags.make

CMakeFiles/kalman.dir/kalman.cpp.o: CMakeFiles/kalman.dir/flags.make
CMakeFiles/kalman.dir/kalman.cpp.o: kalman.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/cj/code/play_opencv/KalmanFilter/KalmanTest/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/kalman.dir/kalman.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/kalman.dir/kalman.cpp.o -c /home/cj/code/play_opencv/KalmanFilter/KalmanTest/kalman.cpp

CMakeFiles/kalman.dir/kalman.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kalman.dir/kalman.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/cj/code/play_opencv/KalmanFilter/KalmanTest/kalman.cpp > CMakeFiles/kalman.dir/kalman.cpp.i

CMakeFiles/kalman.dir/kalman.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kalman.dir/kalman.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/cj/code/play_opencv/KalmanFilter/KalmanTest/kalman.cpp -o CMakeFiles/kalman.dir/kalman.cpp.s

CMakeFiles/kalman.dir/kalman.cpp.o.requires:
.PHONY : CMakeFiles/kalman.dir/kalman.cpp.o.requires

CMakeFiles/kalman.dir/kalman.cpp.o.provides: CMakeFiles/kalman.dir/kalman.cpp.o.requires
	$(MAKE) -f CMakeFiles/kalman.dir/build.make CMakeFiles/kalman.dir/kalman.cpp.o.provides.build
.PHONY : CMakeFiles/kalman.dir/kalman.cpp.o.provides

CMakeFiles/kalman.dir/kalman.cpp.o.provides.build: CMakeFiles/kalman.dir/kalman.cpp.o

# Object files for target kalman
kalman_OBJECTS = \
"CMakeFiles/kalman.dir/kalman.cpp.o"

# External object files for target kalman
kalman_EXTERNAL_OBJECTS =

kalman: CMakeFiles/kalman.dir/kalman.cpp.o
kalman: CMakeFiles/kalman.dir/build.make
kalman: /usr/local/lib/libopencv_videostab.so.3.0.0
kalman: /usr/local/lib/libopencv_video.so.3.0.0
kalman: /usr/local/lib/libopencv_ts.a
kalman: /usr/local/lib/libopencv_superres.so.3.0.0
kalman: /usr/local/lib/libopencv_stitching.so.3.0.0
kalman: /usr/local/lib/libopencv_softcascade.so.3.0.0
kalman: /usr/local/lib/libopencv_shape.so.3.0.0
kalman: /usr/local/lib/libopencv_photo.so.3.0.0
kalman: /usr/local/lib/libopencv_optim.so.3.0.0
kalman: /usr/local/lib/libopencv_objdetect.so.3.0.0
kalman: /usr/local/lib/libopencv_nonfree.so.3.0.0
kalman: /usr/local/lib/libopencv_ml.so.3.0.0
kalman: /usr/local/lib/libopencv_legacy.so.3.0.0
kalman: /usr/local/lib/libopencv_imgproc.so.3.0.0
kalman: /usr/local/lib/libopencv_highgui.so.3.0.0
kalman: /usr/local/lib/libopencv_flann.so.3.0.0
kalman: /usr/local/lib/libopencv_features2d.so.3.0.0
kalman: /usr/local/lib/libopencv_cudawarping.so.3.0.0
kalman: /usr/local/lib/libopencv_cudastereo.so.3.0.0
kalman: /usr/local/lib/libopencv_cudaoptflow.so.3.0.0
kalman: /usr/local/lib/libopencv_cudaimgproc.so.3.0.0
kalman: /usr/local/lib/libopencv_cudafilters.so.3.0.0
kalman: /usr/local/lib/libopencv_cudafeatures2d.so.3.0.0
kalman: /usr/local/lib/libopencv_cudacodec.so.3.0.0
kalman: /usr/local/lib/libopencv_cudabgsegm.so.3.0.0
kalman: /usr/local/lib/libopencv_cudaarithm.so.3.0.0
kalman: /usr/local/lib/libopencv_cuda.so.3.0.0
kalman: /usr/local/lib/libopencv_core.so.3.0.0
kalman: /usr/local/lib/libopencv_contrib.so.3.0.0
kalman: /usr/local/lib/libopencv_calib3d.so.3.0.0
kalman: /usr/lib/i386-linux-gnu/libGLU.so
kalman: /usr/lib/i386-linux-gnu/libGL.so
kalman: /usr/lib/i386-linux-gnu/libSM.so
kalman: /usr/lib/i386-linux-gnu/libICE.so
kalman: /usr/lib/i386-linux-gnu/libX11.so
kalman: /usr/lib/i386-linux-gnu/libXext.so
kalman: /usr/local/lib/libopencv_cudawarping.so.3.0.0
kalman: /usr/local/lib/libopencv_legacy.so.3.0.0
kalman: /usr/local/lib/libopencv_cudaimgproc.so.3.0.0
kalman: /usr/local/lib/libopencv_cudafilters.so.3.0.0
kalman: /usr/local/lib/libopencv_video.so.3.0.0
kalman: /usr/local/lib/libopencv_objdetect.so.3.0.0
kalman: /usr/local/lib/libopencv_nonfree.so.3.0.0
kalman: /usr/local/lib/libopencv_ml.so.3.0.0
kalman: /usr/local/lib/libopencv_cudaarithm.so.3.0.0
kalman: /usr/local/lib/libopencv_calib3d.so.3.0.0
kalman: /usr/local/lib/libopencv_features2d.so.3.0.0
kalman: /usr/local/lib/libopencv_highgui.so.3.0.0
kalman: /usr/local/lib/libopencv_imgproc.so.3.0.0
kalman: /usr/local/lib/libopencv_flann.so.3.0.0
kalman: /usr/local/lib/libopencv_core.so.3.0.0
kalman: CMakeFiles/kalman.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable kalman"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kalman.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kalman.dir/build: kalman
.PHONY : CMakeFiles/kalman.dir/build

CMakeFiles/kalman.dir/requires: CMakeFiles/kalman.dir/kalman.cpp.o.requires
.PHONY : CMakeFiles/kalman.dir/requires

CMakeFiles/kalman.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kalman.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kalman.dir/clean

CMakeFiles/kalman.dir/depend:
	cd /home/cj/code/play_opencv/KalmanFilter/KalmanTest && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cj/code/play_opencv/KalmanFilter/KalmanTest /home/cj/code/play_opencv/KalmanFilter/KalmanTest /home/cj/code/play_opencv/KalmanFilter/KalmanTest /home/cj/code/play_opencv/KalmanFilter/KalmanTest /home/cj/code/play_opencv/KalmanFilter/KalmanTest/CMakeFiles/kalman.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kalman.dir/depend
