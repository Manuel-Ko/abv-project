#include "imageprocessor.h"

ImageProcessor::ImageProcessor() :
    m_calcImage(cv::Mat()),
    m_calcImage_gray(cv::Mat()),
    m_canny_result(cv::Mat()),
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

void show(const char* p_windowName,cv::Mat p_image, float p_size)
{
    cv::namedWindow(p_windowName, CV_WINDOW_KEEPRATIO);
    cv::resizeWindow(p_windowName,p_image.cols * p_size, p_image.rows * p_size);
    cv::imshow(p_windowName, p_image);
}

void ImageProcessor::fastMatchTemplate(cv::Mat& srca,  // The reference image
                       cv::Mat& srcb,  // The template image
                       cv::Mat& dst,   // Template matching result
                       int maxlevel)   // Number of levels
{
    std::vector<cv::Mat> refs, tpls, results;

    // Build Gaussian pyramid
    cv::buildPyramid(srca, refs, maxlevel);
    cv::buildPyramid(srcb, tpls, maxlevel);

    cv::Mat ref, tpl, res;

    // Process each level
    for (int level = maxlevel; level >= 0; level--)
    {
        ref = refs[level];
        tpl = tpls[level];
        res = cv::Mat::zeros(ref.size() + cv::Size(1,1) - tpl.size(), CV_32FC1);

        if (level == maxlevel)
        {
            // On the smallest level, just perform regular template matching
            cv::matchTemplate(ref, tpl, res, CV_TM_CCORR_NORMED);
        }
        else
        {
            // On the next layers, template matching is performed on pre-defined
            // ROI areas.  We define the ROI using the template matching result
            // from the previous layer.

            cv::Mat mask;
            cv::pyrUp(results.back(), mask);

            cv::Mat mask8u;
            mask.convertTo(mask8u, CV_8U,10);

            // Find matches from previous layer
            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(mask8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

            // Use the contours to define region of interest and
            // perform template matching on the areas
            for (int i = 0; i < contours.size(); i++)
            {
				cv::Rect r = cv::boundingRect(contours[i]);
				cv::Rect foo = r + (tpl.size() - cv::Size(1,1));
                cv::matchTemplate(
                    ref(r + (tpl.size() - cv::Size(1,1))),
                    tpl,
                    res(r),
                    CV_TM_CCORR_NORMED
                );
            }
        }

        //## Debug
        if(false)
        {
            std::string windowname = "scale" + std::to_string(level);
            show(windowname.c_str(),res,0.6);
        }

        // Only keep good matches
        cv::threshold(res, res, 0.3, 1., CV_THRESH_TOZERO);
        results.push_back(res);

    }

    res.copyTo(dst);
}


//#################
//####  public  ###
//#################
void ImageProcessor::setImage(cv::Mat p_image)
{
    //m_calcImage = p_image;
    p_image.convertTo(m_calcImage, CV_8U);
    cv::cvtColor( m_calcImage, m_calcImage_gray, CV_BGR2GRAY );

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

void ImageProcessor::processImage_Canny()
{
    // calculate median
//    int channels = 0;
//    int histSize = 255;
//    float histRange[] = {0,255};
//    const float* ranges[] = {histRange};
//    cv::Mat hist;
//    cv::calcHist(&m_canny_result,1,&channels,cv::Mat(),hist,1,&histSize,ranges);

//    int sum = 0;
//    int median = 0;
//    while(sum < m_calcImage_gray.total()/2.0 && median < 255)
//    {
//        sum += hist.at<float>(median);
//        median++;
//    }

    //calculate mean grayvalue of input image
    cv::Scalar meanSc = cv::mean(m_calcImage_gray);
    int mean = meanSc[0];

    // mean is used to determine threshold
    processImage_Canny(0.3 * mean, 0.9 * mean, 3);

}

void ImageProcessor::processImage_Canny(double lowThreshold, double highTreshold, int kernelSize )
{
    /// Reduce noise with a kernel 3x3
    cv::blur( m_calcImage_gray, m_canny_result, cv::Size(3,3) );


    /// Canny detector
    cv::Canny( m_canny_result, m_canny_result, lowThreshold, highTreshold, kernelSize );

    //## Debug
    if(false)
    {
        cv::namedWindow("Canny", CV_WINDOW_KEEPRATIO);
        cv::resizeWindow("Canny", (int)(m_canny_result.cols * 0.3), (int)(m_canny_result.rows * 0.3));
        cv::imshow("Canny", m_canny_result);
    }
}

void ImageProcessor::processImage_DistTrans()
{
    cv::Mat invCanny = cv::Mat(m_canny_result.rows, m_canny_result.cols, m_canny_result.type() );
    cv::bitwise_not(m_canny_result,invCanny);

    cv::distanceTransform(invCanny, m_distanceTrans, CV_DIST_L2, CV_DIST_MASK_PRECISE);

    //## Debug
    if(false)
    {
        cv::Mat dist;
        cv::normalize(m_distanceTrans, dist, 0.0, 1.0, cv::NORM_MINMAX);
        cv::namedWindow("distTrans", CV_WINDOW_KEEPRATIO);
        cv::resizeWindow("distTrans", (int)(dist.cols * 0.3), (int)(dist.rows * 0.3));
        cv::imshow("distTrans", dist);
    }
}

void ImageProcessor::processImage_TemplateMatch(TemplateType p_templType)
{
    // clear existing matchspaces
    m_matchPositions.clear();

    cv::Mat origTempl;
    cv::Mat resizedTempl;
    // the image on which the template should be looked for
    cv::Mat matchOnImage;


    // set templates and imgages in regard to matching type
    switch(p_templType)
    {
    case Color:
        origTempl = cv::imread("template_color_big.jpg");
        matchOnImage = m_calcImage;
        break;
    case Sobel:
        origTempl = cv::imread("template_sobel.jpg",0);
        processImage_Sobel();
        matchOnImage = m_sobel_result;
        break;
    case Gray:
        origTempl = cv::imread("template_color_big.jpg");
        cv::cvtColor(origTempl,origTempl,CV_BGR2GRAY);
        matchOnImage = m_calcImage;
        cv::cvtColor(matchOnImage, matchOnImage, CV_BGR2GRAY);
        break;
    }

    //cv::resize(matchOnImage, matchOnImage,cv::Size(0,0), 0.3,0.3);
    //cv::resize(origTempl, origTempl,cv::Size(0,0),0.3,0.3);

    // scale pyramid parameters
    const int MIN_SIZE = 50;
    const int STEP_SIZE = 10;
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
        //fastMatchTemplate( matchOnImage, resizedTempl, resultScalePyr.back(), 2);
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

    //	pure mtchspace for debugging
    m_bestMatchSpace_pure = resultScalePyr[currBestScaleLevel].clone();

    //cv::threshold(resultScalePyr[currBestScaleLevel], resultScalePyr[currBestScaleLevel], 0.55, 1., CV_THRESH_TOZERO);

    // find the MatchPositions
    findMatches(resultScalePyr[currBestScaleLevel],m_matchPositions,m_bestTemplSize,0.55);
    m_bestMatchSpace_blacked = resultScalePyr[currBestScaleLevel];


    m_templMatchDebugAvailable = true;
}

void ImageProcessor::findMatches(cv::Mat &p_matchSpace, std::vector<cv::Point> &p_out, const cv::Size &p_teplSize, float p_threshold /*, min or max?*/ )
{
    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(p_matchSpace, &minval, &maxval, &minloc, &maxloc);

    while(maxval >= p_threshold)
    {
        p_out.push_back(maxloc);
        cv::rectangle(p_matchSpace,
                      cv::Point(maxloc.x - p_teplSize.width/2, maxloc.y - p_teplSize.height/2),
                      cv::Point(maxloc.x + p_teplSize.width/2, maxloc.y + p_teplSize.height/2),
                      cv::Scalar(0,0,0),-1);
        cv::minMaxLoc(p_matchSpace, &minval, &maxval, &minloc, &maxloc);
    }
}

void ImageProcessor::templateMatch_Edges()
{
    processImage_Canny();
    processImage_DistTrans();

    cv::Mat templ = cv::imread("template_canny3.jpg");
    if(templ.empty())
    {
        std::cerr << "No tamplate found" << std::endl;
        return;
    }
    const size_t templWidth = templ.cols;
    const size_t templHeigth = templ.rows;


//    // Create point set from Canny Output
//	std::vector<cv::Point2d> image_points;
//	for(int r = 0; r < m_canny_result.rows; r++)
//	{
//		for(int c = 0; c < m_canny_result.cols; c++)
//		{
//			if(m_canny_result.at<unsigned char>(r,c) == 255)
//			{
//				image_points.push_back(cv::Point2d(c,r));
//			}
//		}
//	}

    // Create point set from Template
    std::vector<cv::Point2d> templ_points;
    for(int r = 0; r < templHeigth; r++)
    {
        for(int c = 0; c < templWidth; c++)
        {
            if(templ.at<unsigned char>(r,c) == 255)
            {
                templ_points.push_back(cv::Point2d(c,r));
            }
        }
    }

    CV_Assert(m_canny_result.cols > templWidth && m_canny_result.rows > templHeigth);
    const cv::Size result_size = cv::Size(m_canny_result.cols - templWidth, m_canny_result.rows - templHeigth);

    //generate Array with mean distances between Template and Image for every possible template position
    cv::Mat res = cv::Mat(result_size, CV_32F);
    for(int y = 0; y < result_size.height; ++y)
    {
        for(int x = 0; x < result_size.width; ++x)
        {
//            //calculate the mean distance based on distTrans
//            float distSum = 0;
//            for(auto templPointsIter = templ_points.begin();
//                templPointsIter != templ_points.end();
//                ++templPointsIter)
//            {
//                distSum += m_distanceTrans.at<float>(*templPointsIter);
//            }
            //res.at<float>(y,x) = distSum/templ_points.size();

//            std::vector<cv::Point> imPoints = std::vector<cv::Point>();
//            for(int imY = y; imY < templHeigth; ++imY)
//            {
//                for(int imX = x; imX < templWidth; ++imX)
//                {
//                    if(m_canny_result.at<unsigned char>(imY,imX) == 255)
//                    {
//                        imPoints.push_back(cv::Point(imX,imY));
//                    }
//                }
//            }
            double max =(result_size.width * result_size.height );
            double percent = x*(y+1)/max;
            std::cout << x*(y+1) << "/" << max <<"\t" << percent << "%" << std::endl;
        }
        //std::cerr << "Y" << y << "/" << result_size.height << std::endl;
    }

    cv::namedWindow("edge distance match", CV_WINDOW_KEEPRATIO);
    cv::resizeWindow("edge distance match", res.cols * 0.3, res.rows * 0.3 );
    cv::imshow("edge distance match", res);

    //findMatches(res,m_matchPositions,0.55);
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
