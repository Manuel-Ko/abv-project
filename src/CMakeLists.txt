include_directories( ${OpenCV_INCLUDE_DIRS} )

# simple opencv test
add_executable( opencv_test opencv_test.cpp )
target_link_libraries( opencv_test ${OpenCV_LIBS} )


