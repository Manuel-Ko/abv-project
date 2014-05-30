#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#define DEBUGWIN_WIDTH 1032
#define DEBUGWIN_HEIGHT 580

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageProcessor
{
public:
    ImageProcessor();
    void setImage(cv::Mat p_image);
    void processImage_Hough();
    void processImage_Sobel();
    void processImage_TemplateMatch();
    cv::Mat getProcessedImage();
    void debugOutput_Hough();
    void debugOutput_Sobel();
    void debugOutput_TemplateMatch();
private:
    cv::Mat m_calcImage;
    bool m_imageProcessed;

    //used for standard houghtransform
    std::vector<cv::Vec3f> m_circles;

    //used for sobel
    cv::Mat grad;

    //used for templateMatching
    cv::Point matchLoc;
};

#endif // IMAGEPROCESSOR_H
