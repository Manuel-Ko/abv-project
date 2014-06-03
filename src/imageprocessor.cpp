#include "imageprocessor.h"

ImageProcessor::ImageProcessor() :
    m_calcImage(cv::Mat()),
    m_imageProcessed(false),
    matchPositions(std::vector<cv::Point>())
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
	matchPositions.clear();
    cv::Mat templ = cv::imread("template.jpg",0);
    cv::Mat templ2;
    processImage_Sobel();
    double currMaxMatch = 0;
    size_t currBestScaleLevel = 0;
    std::vector<cv::Mat> resultScalePyr = std::vector<cv::Mat>();

    const float MIN_SIZE = 1.0f;
    const float STEP_SIZE = 0.05f;


    //cv::namedWindow("sizeDebug",CV_WINDOW_KEEPRATIO);
    size_t scaleLevel = 0;
    const size_t MAX_SCALE_LEVEL = unsigned int((1-MIN_SIZE)/(float)STEP_SIZE)-1;
    for(float scale = 1.00f; scale >= MIN_SIZE; scale -= STEP_SIZE)
    {
        // Resize the template to all sizes between 1 and MIN_SIZE in steps of size STEP_SIZE
        cv::resize(templ,templ2,cv::Size(0,0),scale,scale);
        //cv::imshow("sizeDebug",templ);

        int result_cols =  grad.cols - templ2.cols + 1;
        int result_rows = grad.rows - templ2.rows + 1;

        //result.create( result_cols, result_rows, CV_32FC1 );

        // create new Scale level for match results
        // TODO: use only 2 Mats and hold the "better" to optimize memory consumption
        resultScalePyr.push_back(cv::Mat(result_rows, result_cols,CV_32FC1));

        cv::matchTemplate( grad, templ2, resultScalePyr.back(), CV_TM_CCOEFF_NORMED );

        // find the scale with the best match
        double minval, maxval;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(resultScalePyr.back(), &minval, &maxval, &minloc, &maxloc);
        if(maxval > currMaxMatch)
        {
            currMaxMatch =  maxval;
            currBestScaleLevel = scaleLevel;
            tmplSize = cv::Size(templ2.cols, templ2.rows);
        }

        std::cout << "Matching at level: "  << scaleLevel << "/" << MAX_SCALE_LEVEL << std::endl;

        ++scaleLevel;
    }
    std::cout << "Matched all scales." << std::endl;
    std::cout << "Best match at: " << currBestScaleLevel << std::endl;
   // cv::resize(templ,templ2,cv::Size(0,0),MIN_SIZE,MIN_SIZE);

    cv::normalize( resultScalePyr[currBestScaleLevel], resultScalePyr[currBestScaleLevel], 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    //######	for debugging	#####
    result = resultScalePyr[currBestScaleLevel].clone();

    //cv::threshold(resultScalePyr[currBestScaleLevel], resultScalePyr[currBestScaleLevel], 0.55, 1., CV_THRESH_TOZERO);


    while (true)
    {
        // ###  $Parameter threshold
        double minval, maxval, threshold = 0.55;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(resultScalePyr[currBestScaleLevel], &minval, &maxval, &minloc, &maxloc);

        if (maxval >= threshold)
        {
            matchPositions.push_back(maxloc);
            //cv::floodFill(result, maxloc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
            cv::rectangle(resultScalePyr[currBestScaleLevel],
                          cv::Point(maxloc.x - tmplSize.width/2, maxloc.y - tmplSize.height/2),
                          cv::Point(maxloc.x + tmplSize.width/2, maxloc.y + tmplSize.height/2),
                          cv::Scalar(0,0,0),-1);
        }
        else
            break;
    }
    result_debug = resultScalePyr[currBestScaleLevel];


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

void ImageProcessor::debugOutput_Hough(std::vector<cv::Mat> &p_out)
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

    p_out.push_back(debugOut);

//    cv::namedWindow("DebugWindow", cv::WINDOW_NORMAL);
//    cv::resizeWindow("DebugWindow", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
//    cv::imshow("DebugWindow", debugOut);
}

void ImageProcessor::debugOutput_Sobel(std::vector<cv::Mat> &p_out)
{
    if( !m_imageProcessed )
    {
        return;
    }

    p_out.push_back(grad);

//    cv::namedWindow("DebugWindow_Sobel", cv::WINDOW_NORMAL);
//    cv::resizeWindow("DebugWindow_Sobel", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
//    cv::imshow("DebugWindow_Sobel", grad);
    //cv::imwrite("output_sobel.jpg",grad);
}

void ImageProcessor::debugOutput_TemplateMatch(std::vector<cv::Mat> &p_out)
{
    if( !m_imageProcessed )
    {
        return;
    }
    //cv::Mat templ = cv::imread("template.jpg");

    cv::Mat displ_image = m_calcImage.clone();
//    cv::Mat result_thresh;
//    cv::threshold(result, result_thresh, 0.55, 1., CV_THRESH_TOZERO);
    for(unsigned int i = 0; i < matchPositions.size(); ++i)
    {
        cv::rectangle( displ_image, matchPositions[i], cv::Point( matchPositions[i].x + tmplSize.width , matchPositions[i].y + tmplSize.height ), cv::Scalar::all(0), 5, 8, 0 );
        cv::circle(displ_image,cv::Point(matchPositions[i].x + tmplSize.width/2, matchPositions[i].y + tmplSize.height/2),20,cv::Scalar(0,0,255),-1);
    }


    p_out.push_back(displ_image);
	p_out.push_back(result_debug);
    p_out.push_back(result);
//    p_out.push_back(result_thresh);


//    cv::namedWindow("DebugWindow", cv::WINDOW_NORMAL);
//    cv::resizeWindow("DebugWindow", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
//    cv::imshow("DebugWindow", displ_image);

//    cv::namedWindow("matchingSpace", cv::WINDOW_NORMAL);
//    cv::resizeWindow("matchingSpace", DEBUGWIN_WIDTH, DEBUGWIN_HEIGHT );
//    cv::imshow("matchingSpace", result);
}
