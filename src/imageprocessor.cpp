#include "imageprocessor.h"

ImageProcessor::ImageProcessor() :
    m_calcImage(cv::Mat()),
    m_imageProcessed(false),
    m_matchPositions(std::vector<cv::Point>()),
    m_houghDebugAvailable(false),
    m_sobelDebugAvailable(false),
    m_templMatchDebugAvailable(false)
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
    //m_calcImage = p_image;
    p_image.convertTo(m_calcImage, CV_8U);
    m_houghDebugAvailable = false;
    m_sobelDebugAvailable = false;
    m_templMatchDebugAvailable = false;
}

void ImageProcessor::processImage_Hough()
{
    //TODO do sth
    cv::Mat image_gray;
    cv::cvtColor( m_calcImage, image_gray, CV_BGR2GRAY );
    cv::HoughCircles( image_gray,m_circles, CV_HOUGH_GRADIENT, 1, image_gray.rows/8, 250, 100, 250 );

    m_houghDebugAvailable = true;
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
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, m_sobel_result );

    // save edge image
    //cv::imwrite("template_sobel.jpg", grad);

    m_sobelDebugAvailable = true;
}

void ImageProcessor::processImage_TemplateMatch(TemplateType p_templType)
{
    // clear existing matchspaces
    m_matchPositions.clear();

    TemplateType matchOn = p_templType;
    cv::Mat origTempl;
    cv::Mat resizedTempl;
    // the image on which the template should be looked for
    cv::Mat matchOnImage;

    // set templates and imgages in regard to matching type
    switch(matchOn)
    {
    case Color:
        origTempl = cv::imread("template_color.jpg");
        matchOnImage = m_calcImage;
        break;
    case Sobel:
        origTempl = cv::imread("template_sobel.jpg");
        processImage_Sobel();
        matchOnImage = m_sobel_result;
        break;
    }

    // scale pyramid parameters
    const int MIN_SIZE = 40;
    const int STEP_SIZE = 5;
    const size_t MAX_SCALE_LEVEL = unsigned int((100-MIN_SIZE)/(float)STEP_SIZE);
    double currMaxMatch = 0;
    size_t currBestScaleLevel = 0;
    std::vector<cv::Mat> resultScalePyr = std::vector<cv::Mat>();
    size_t scaleLevel = 0;

    for(int scale = 100; scale >= MIN_SIZE; scale -= STEP_SIZE)
    {
        clock_t templStartTime = clock();
        // show progresss
        std::cout << "Matching at level: "  << scaleLevel << "/" << MAX_SCALE_LEVEL << std::endl;

        // Resize the template to all sizes between 1 and MIN_SIZE in steps of size STEP_SIZE
        cv::resize(origTempl,resizedTempl,cv::Size(0,0),scale/100.0f,scale/100.0f);

        int result_cols =  matchOnImage.cols - resizedTempl.cols + 1;
        int result_rows = matchOnImage.rows - resizedTempl.rows + 1;


        // create new Scale level for match results
        // TODO: use only 2 Mats and hold the "better" to optimize memory consumption
        resultScalePyr.push_back(cv::Mat(result_rows, result_cols,CV_32FC1));


        cv::matchTemplate( matchOnImage, resizedTempl, resultScalePyr.back(), CV_TM_CCOEFF_NORMED );

        // find the scale with the best match
        double minval, maxval;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(resultScalePyr.back(), &minval, &maxval, &minloc, &maxloc);
        if(maxval > currMaxMatch)
        {
            currMaxMatch =  maxval;
            currBestScaleLevel = scaleLevel;
            m_bestTemplSize = cv::Size(resizedTempl.cols, resizedTempl.rows);
        }

        double templElapsedTime = ((double)(clock() - templStartTime)) / (double)CLOCKS_PER_SEC;

        std::cout << "Level took " << templElapsedTime << "seconds." << std::endl;

        ++scaleLevel;
    }
    std::cout << "Matched all scales." << std::endl;
    std::cout << "Best match at: " << currBestScaleLevel << std::endl;
   // cv::resize(templ,templ2,cv::Size(0,0),MIN_SIZE,MIN_SIZE);

    cv::normalize( resultScalePyr[currBestScaleLevel], resultScalePyr[currBestScaleLevel], 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    //######	for debugging	#####
    m_bestMatchSpace_pure = resultScalePyr[currBestScaleLevel].clone();

    //cv::threshold(resultScalePyr[currBestScaleLevel], resultScalePyr[currBestScaleLevel], 0.55, 1., CV_THRESH_TOZERO);


    while (true)
    {
        // ###  $Parameter threshold
        double minval, maxval, threshold = 0.55;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(resultScalePyr[currBestScaleLevel], &minval, &maxval, &minloc, &maxloc);

        if (maxval >= threshold)
        {
            m_matchPositions.push_back(maxloc);
            //cv::floodFill(result, maxloc, cv::Scalar(0), 0, cv::Scalar(.1), cv::Scalar(1.));
            cv::rectangle(resultScalePyr[currBestScaleLevel],
                          cv::Point(maxloc.x - m_bestTemplSize.width/2, maxloc.y - m_bestTemplSize.height/2),
                          cv::Point(maxloc.x + m_bestTemplSize.width/2, maxloc.y + m_bestTemplSize.height/2),
                          cv::Scalar(0,0,0),-1);
        }
        else
            break;
    }
    m_bestMatchSpace_blacked = resultScalePyr[currBestScaleLevel];


    m_templMatchDebugAvailable = true;
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
    if( !m_houghDebugAvailable )
    {
        return;
    }

    // draw found circles on original image
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

    // ####     pass debug images   ####
    p_out.push_back(debugOut);
}

void ImageProcessor::debugOutput_Sobel(std::vector<cv::Mat> &p_out)
{
    if( !m_sobelDebugAvailable )
    {
        return;
    }

    // ####     pass debug images   ####
    p_out.push_back(m_sobel_result);
}

void ImageProcessor::debugOutput_TemplateMatch(std::vector<cv::Mat> &p_out)
{
    if( !m_templMatchDebugAvailable )
    {
        return;
    }

    // create image that shows positions of template on the orignal image
    cv::Mat displ_image = m_calcImage.clone();
    for(unsigned int i = 0; i < m_matchPositions.size(); ++i)
    {
        cv::rectangle( displ_image, m_matchPositions[i], cv::Point( m_matchPositions[i].x + m_bestTemplSize.width , m_matchPositions[i].y + m_bestTemplSize.height ), cv::Scalar::all(0), 5, 8, 0 );
        cv::circle(displ_image,cv::Point(m_matchPositions[i].x + m_bestTemplSize.width/2, m_matchPositions[i].y + m_bestTemplSize.height/2),20,cv::Scalar(0,0,255),-1);
    }

    // ####     pass debug images   ####
    p_out.push_back(displ_image);
    // match space with blacked out rectangles
    p_out.push_back(m_bestMatchSpace_blacked);
    // pure match space
    p_out.push_back(m_bestMatchSpace_pure);
//    p_out.push_back(result_thresh);

}
