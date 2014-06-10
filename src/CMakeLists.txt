include_directories( ${OpenCV_INCLUDE_DIRS} )

# simple opencv test
add_executable( targetAnalyser
#opencv_test.cpp
imageloader.h
imageloader.cpp
imageprocessor.h
imageprocessor.cpp
templatematch.h
templatematch.cpp
targetinstance.h
targetinstance.cpp
main.cpp
)

add_executable(test
imageloader.h
imageloader.cpp
targetinstance.h
targetinstance.cpp
opencv_test.cpp)

add_executable(ellipseTest
ellipseTest.cpp)

target_link_libraries( targetAnalyser ${OpenCV_LIBS} )
target_link_libraries( test ${OpenCV_LIBS} )
target_link_libraries( ellipseTest ${OpenCV_LIBS} )


