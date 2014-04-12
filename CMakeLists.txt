cmake_minimum_required(VERSION 2.8)
project( cvproject )
find_package( OpenCV REQUIRED )
add_subdirectory( src bin ) # put generated files of subdirectory 'src/' into a folder 'bin/'
