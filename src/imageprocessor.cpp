#include "imageprocessor.h"

ImageProcessor::ImageProcessor() :
    m_calcImage(cv::Mat()),
    m_imageProcessed(false)
{
}

//#################
//####  private ###
//#################

//#################
//####  public  ###
//#################
void ImageProcessor::setImage(cv::Mat p_image)
{
    m_calcImage = p_image;
    m_imageProcessed = false;
}

void ImageProcessor::processImage_Hough()
{
    //TODO do sth
    cv::Mat image_gray;
    cv::cvtColor( m_calcImage, image_gray, CV_BGR2GRAY );
    cv::HoughCircles( image_gray,m_circles, CV_HOUGH_GRADIENT, 1, image_gray.rows/8, 250, 100, 250 );
    m_imageProcessed = true;
}

void ImageProcessor::processImage_Sobel()
{
    cv::Mat image_gray;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::cvtColor( m_calcImage, image_gray, CV_BGR2GRAY );
    cv::GaussianBlur( image_gray, image_gray, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    cv::Sobel( image_gray, grad_x, CV_16S, 1, 0, 3 );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    cv::Sobel( image_gray, grad_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    m_imageProcessed = true;
}

void ImageProcessor::processImage_TemplateMatch()
{
    cv::Mat result;
    cv::Mat templ = cv::imread("template.jpg",0);
    processImage_Sobel();
    /// Create the result matrix
    int result_cols =  grad.cols - templ.cols + 1;
    int result_rows = grad.rows - templ.rows + 1;

    result.create( result_cols, result_rows, CV_32FC1 );

    cv::matchTemplate( grad, templ, result, CV_TM_CCOEFF_NORMED );
    cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Localizing the best match with minMaxLoc
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;

    cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
    matchLoc = maxLoc;

    m_imageProcessed = true;
}

/// @brief returns the processed Image
cv::Mat ImageProcessor::getProcessedImage()
{
    if( !m_imageProcessed )
    {
        return cv::Mat();
    }
    else
    {
        return m_calcImage;
    }
}

void ImageProcessor::debugOutput_Hough()
{
    if( !m_imageProcessed )
    {
        return;
    }
    cv::Mat debugOut = m_calcImage.clone();
    for( size_t i = 0; i < m_circles.size(); i++ )
    {
      cv::Point center(cvRound(m_circles[i][0]), cvRound(m_circles[i][1]));
      int radius = cvRound(m_circles[i][2]);
      // circle center
      cv::circle( debugOut, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      cv::circle( debugOut, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
    }
    cv::namedWindow("DebugWindow", cv::WINDOW_NORMAL);
    cv::resizeWindow("DebugWindow", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
    cv::imshow("DebugWindow", debugOut);
}

void ImageProcessor::debugOutput_Sobel()
{
    if( !m_imageProcessed )
    {
        return;
    }
    cv::namedWindow("DebugWindow", cv::WINDOW_NORMAL);
    cv::resizeWindow("DebugWindow", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
    cv::imshow("DebugWindow", grad);
    //cv::imwrite("output_sobel.jpg",grad);
}

void ImageProcessor::debugOutput_TemplateMatch()
{
    if( !m_imageProcessed )
    {
        return;
    }
    cv::Mat templ = cv::imread("template.jpg");

    cv::Mat displ_image = m_calcImage.clone();
    cv::rectangle( displ_image, matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cv::Scalar::all(0), 5, 8, 0 );
    cv::circle(displ_image,cv::Point(matchLoc.x + templ.cols/2,matchLoc.y + templ.rows/2),20,cv::Scalar(0,0,255),-1);

    cv::namedWindow("DebugWindow", cv::WINDOW_NORMAL);
    cv::resizeWindow("DebugWindow", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
    cv::imshow("DebugWindow", displ_image);
}
