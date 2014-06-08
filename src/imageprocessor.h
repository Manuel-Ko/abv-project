#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#define DEBUGWIN_WIDTH 1032
#define DEBUGWIN_HEIGHT 580

#include <iostream>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "templatematch.h"

class ImageProcessor
{
public:
    ImageProcessor();

    enum TemplateType
    {
        Sobel,
        Color,
        Gray
    };

    void setImage(cv::Mat p_image);
    void processImage_Hough();
    void processImage_Sobel();
    // automatically chooses thesholds, see implementation for details
    void processImage_Canny();
    void processImage_Canny(double lowThreshold, double highTreshold, int kernelSize);
    void processImage_DistTrans();
    std::vector<TemplateMatch> processImage_TemplateMatch(const cv::Mat& p_templ,
                                    TemplateType p_templType,
                                    const int p_minSize,
                                    const int p_stepSize, const float p_downScale);
    void templateMatch_Edges();
    cv::Mat getProcessedImage();


    // only push_back into p_out
    // it is used to gather all debug images in main.cpp
    void debugOutput_Hough(std::vector<cv::Mat> &p_out);
    void debugOutput_Sobel(std::vector<cv::Mat> &p_out);
    void debugOutput_TemplateMatch(std::vector<cv::Mat> &p_out);

private:
    cv::Mat m_calcImage;
    cv::Mat m_calcImage_gray;
    bool m_imageProcessed;

    //used for standard houghtransform
    std::vector<cv::Vec3f> m_circles;
    bool m_houghDebugAvailable;

    //used for sobel
    cv::Mat m_sobel_result;
    bool m_sobelDebugAvailable;

    //used for Canny
    cv::Mat m_canny_result;

    //used for DistTransform
    cv::Mat m_distanceTrans;

    //used for templateMatching
    std::vector<TemplateMatch> m_matches;
    cv::Point m_matchLoc;
    cv::Mat m_bestMatchSpace_pure;
    float m_tmplMatchDownScale;
    cv::Size m_bestTemplSize;

    // debug templ matching
    bool m_templMatchDebugAvailable;
    cv::Mat m_bestMatchSpace_blacked;

    void findMatches(cv::Mat &p_matchSpace, std::vector<TemplateMatch> &p_out, const cv::Size &p_teplSize, float p_threshold);
    void fastMatchTemplate(cv::Mat& srca,  // The reference image
                           cv::Mat& srcb,  // The template image
                           cv::Mat& dst,   // Template matching result
                           int maxlevel);
};

#endif // IMAGEPROCESSOR_H
