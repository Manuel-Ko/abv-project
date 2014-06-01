#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#define DEBUGWIN_WIDTH 1032
#define DEBUGWIN_HEIGHT 580

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

class ImageProcessor
{
public:
    ImageProcessor();
    void setImage(cv::Mat p_image);
    void processImage_Hough();
    void processImage_Sobel();
    void processImage_TemplateMatch();
    cv::Mat getProcessedImage();


    // only push_back into p_out
    // it is used to gather all debug images from main.cpp
    void debugOutput_Hough(std::vector<cv::Mat> &p_out);
    void debugOutput_Sobel(std::vector<cv::Mat> &p_out);
    void debugOutput_TemplateMatch(std::vector<cv::Mat> &p_out);
private:
    cv::Mat m_calcImage;
    bool m_imageProcessed;

    //used for standard houghtransform
    std::vector<cv::Vec3f> m_circles;

    //used for sobel
    cv::Mat grad;

    //used for templateMatching
    std::vector<cv::Point> matchPositions;
    cv::Point matchLoc;
    cv::Mat result;
	cv::Mat result_debug;
};

#endif // IMAGEPROCESSOR_H
