#ifndef MYIMAGEPROCESSING_H
#define MYIMAGEPROCESSING_H

#include <opencv2/opencv.hpp>
#include <iostream>

namespace myImProc
{
    void show(const char* p_windowName,cv::Mat p_image, float p_size);
    void processImage_Sobel(cv::Mat p_image, cv::Mat& p_out);
//    void CannyThreshold(cv::Mat &p_image_gray);
    void previtt(const cv::Mat& p_image, cv::Mat& dst);
    bool isPointInImage(const cv::Mat& p_image, cv::Point p_point);
    cv::Vec3f calcCircle(cv::Point2f a, cv::Point2f b, cv::Point2f c);
    std::vector<cv::Vec3f> circleRANSAC(const std::vector<cv::Point>& p_points, const cv::Mat& p_distanceImage,
                                        float p_threshold , size_t p_iterations, float p_minRad, float p_maxRad, float p_minDist);
}

#endif // MYIMAGEPROCESSING_H
