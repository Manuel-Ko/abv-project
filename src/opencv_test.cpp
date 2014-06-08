#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "imageloader.h"
#include "targetinstance.h"

using namespace cv;

Mat image, image_gray;
Mat dst, detected_edges;
Mat tmp;

int testval = 0;

int edgeThresh = 1;
int lowThreshold;
const int MAX_lowThreshold = 100;
const int MAX_ratio = 100;
int ratio = 3;
int kernel_size = 3;
char* name_edgeWindow = "Edges";

int keyboard = 0;

void show(const char* p_windowName,cv::Mat p_image, float p_size)
{
    cv::namedWindow(p_windowName, CV_WINDOW_KEEPRATIO);
    cv::resizeWindow(p_windowName,p_image.cols * p_size, p_image.rows * p_size);
    cv::imshow(p_windowName, p_image);
}

void on_trackbar(int, void*)
{
    Mat modified;
    image_gray.convertTo(modified,-1,testval/100.0f,0);

	
    imshow("Display Image", modified);

}

void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( image_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, ratio, kernel_size );

  cvtColor(detected_edges,detected_edges,CV_GRAY2BGR);
  bitwise_not(detected_edges,tmp);
 }

void previtt(const cv::Mat& p_image, cv::Mat& dst)
{
    cv::Mat horkernel = (cv::Mat_<float>(3,3) << 1,1,1,0,0,0,-1,-1,-1);
    cv::Mat vertkernel = (cv::Mat_<float>(3,3) << 1,0,-1,1,0,-1,1,0,-1);
	cv::Mat horTmp = cv::Mat(cv::Size(p_image.cols, p_image.rows), CV_32FC1);
	cv::Mat vertTmp = cv::Mat(cv::Size(p_image.cols, p_image.rows), CV_32FC1);

	cv::filter2D(p_image,horTmp,CV_32FC1,horkernel);
    cv::filter2D(p_image,vertTmp,CV_32FC1,vertkernel);

	cv::Mat abs_horTmp;
	cv::Mat abs_vertTmp;
	cv::convertScaleAbs( horTmp, abs_horTmp);
	cv::convertScaleAbs( vertTmp, abs_vertTmp);

	/*double horMin = 0;
	double horMax = 0;
	cv::minMaxIdx(horTmp,&horMin, &horMax);*/

    // ## Debug show the previtt results
    if(false)
    {
        show("hortmp", abs_horTmp, 1);
        show("verttmp", abs_vertTmp, 1);
    }

    cv::addWeighted( abs_horTmp, 0.5, abs_vertTmp, 0.5, 0, dst );
	cv::threshold(dst,dst,120,255,CV_THRESH_TOZERO);

}

bool evaluateFirstCondition_Point(cv::Mat p_image, cv::Point p_point)
{
    int up = std::max(p_point.y -1, 0);
    int down = std::min(p_point.y +2, p_image.rows);
    int left = std::max(p_point.x -1, 0);
    int right = std::min(p_point.x +2, p_image.cols);

    cv::Rect roi = cv::Rect(cv::Point(left,up), cv::Point(right, down));

    cv::Mat roiMat = p_image(roi);
    roiMat.convertTo(roiMat,CV_32F);

    roiMat -= p_image.at<unsigned char>(p_point.y,p_point.x);
    roiMat = abs(roiMat);

    double result = 0;
    double garbage = 0;
    cv::minMaxIdx(roiMat,&garbage, &result);
    if(result < p_image.at<unsigned char>(p_point.y,p_point.x))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool evaluateFirstCondition(cv::Mat p_image, std::vector<cv::Point> p_contour)
{
    int firstConditionCount = 0;
    for(int i = 0; i < p_contour.size(); ++i)
    {
        if(evaluateFirstCondition_Point(p_image,p_contour[i]))
        {
            firstConditionCount++;
        }
    }
    if(firstConditionCount >= 0.9 * p_contour.size())
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool evaluateSecondCondition(cv::Rect p_rect)
{
	double a = std::max(p_rect.width,p_rect.height);
	double b = std::min(p_rect.width,p_rect.height);

	double res = a/b;
	return (res < 2);
}

void refineBullsEyes(cv::Mat& p_out, std::vector<cv::Point> p_contour)
{
    if(p_out.dims < 2)
	{
        p_out = cv::Mat::zeros(image.size(), CV_8UC1);
	}
    std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
    contours.push_back(p_contour);
    cv::Rect boundingRec = cv::boundingRect(contours[0]);
    cv::Point contourCenter = cv::Point(boundingRec.tl().x + boundingRec.width/2, boundingRec.tl().y + boundingRec.height/2 );
    for(int i = 0; i < contours[0].size(); ++i)
    {
        contours[0][i] -= contourCenter;
        contours[0][i] *= 1.1;
        contours[0][i] += contourCenter;
		
		contours[0][i].x = std::max(0,contours[0][i].x);
		contours[0][i].y = std::max(0,contours[0][i].y);
		contours[0][i].x = std::min(image.cols,contours[0][i].x);
		contours[0][i].y = std::min(image.rows,contours[0][i].y);

    }
    boundingRec = cv::boundingRect(contours[0]);
    contourCenter = cv::Point(boundingRec.tl().x + boundingRec.width/2, boundingRec.tl().y + boundingRec.height/2 );
    for(int i = 0; i < contours[0].size(); ++i)
    {
        contours[0][i] -= boundingRec.tl();
    }

    cv::Mat roiImage = image(boundingRec);
    // create a mask based on coarse contour
    cv::Mat mask = cv::Mat::zeros(roiImage.size(), CV_8UC1);
    cv::drawContours(mask,contours,0,cv::Scalar(255),-1);

    // mask out roughly detected bullsEye
    cv::Mat thresholdImage;
//    cv::Mat element = getStructuringElement( MORPH_RECT,
//        cv::Size( 2*5 + 1, 2*5 + 1 ),
//        cv::Point( 5, 5 ) );
    // enlarge mask to avoid cutting of the bullsEye
    //cv::dilate(mask,mask,element);
    //TODO: use bounds to cut out ROI


    roiImage.copyTo(thresholdImage,mask);

    // discard bright pixels as they are not part of the bullsEye
	// ## parameter thresh for bullsEye
    cv::threshold(thresholdImage,thresholdImage, 60,255,CV_THRESH_BINARY_INV);

	cv::Mat tmp;
	thresholdImage.copyTo(tmp,mask);


	// closing the refined mask to maintain a circular shape
    cv::Mat element2 = getStructuringElement( MORPH_RECT, cv::Size( 2*11 + 1, 2*11 + 1 ),
                            cv::Point( 11, 11 ) );
	cv::Mat element3 = getStructuringElement( MORPH_RECT, cv::Size( 2*2 + 1, 2*2 + 1 ),
                            cv::Point( 2, 2 ) );
    clock_t start = clock();
	cv::erode(tmp,tmp,element3);
    cv::dilate(tmp,tmp,element3);

    cv::dilate(tmp,tmp,element2);
    cv::erode(tmp,tmp,element2);
	
    

    // end of profiling
    double elapsed = (double)(clock() - start);
    double elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
    std::cout << "postprocessing took  " << elapsed_sec << "seconds." << std::endl;



    cv::Mat submat = p_out.colRange(boundingRec.x, boundingRec.x + boundingRec.width)
                     .rowRange(boundingRec.y, boundingRec.y + boundingRec.height);
    tmp.copyTo(submat,mask);

}

cv::Point findRingPoint(cv::Point p_p1, cv::Point p_p2)
{
    // ## parameter threshold to fit point to ring
    const double alpha = 0.7;
    uchar vMin = 255;
    uchar vMax = 0;
    float vAvg = 0;
    int maxDiff = 0;
    cv::Point pMin;
    cv::Point pMax;
    cv::Point pDiff;
    uchar vPrev;
    uchar vNext;

    cv::LineIterator lineIter = cv::LineIterator(image,p_p1, p_p2);
    int i = 0;
    for( i = 0; i < lineIter.count; i++)
    {
        uchar val = **lineIter;
        vAvg += val;
        cv::Point pos = lineIter.pos();

        if(val > vMax)
        {
            vMax = val;
            pMax = pos;
        }
        if(val < vMin)
        {
            vMin = val;
            pMin = pos;
        }

        ++lineIter;
        vNext = **lineIter;

        if(i == 0)
        {
            vPrev = val;
        }

        int diff1 = std::abs((int) val - (int)vPrev);
        int diff2 = std::abs((int) val - (int)vNext);
        int diff3 = diff1 + diff2;

        if(diff3 > maxDiff)
        {
            maxDiff = diff3;
            pDiff = pos;
        }

        vPrev = val;
    }

    vAvg /= i;

    float kappa = (vAvg - vMin) / (vMax - vMin);
    if(kappa > 1 - alpha)
    {
//        if(kappa > alpha)
//        {
//            return pDiff;
//        }
        return pMin;
    }
    else //if(kappa < alpha)
    {
        return pMax;
    }
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        cv::waitKey();
        return -1;
    }

    ImageLoader imageLoader = ImageLoader(argv[1]);

    image = imageLoader.getCurrentImage();
            //imread( argv[1], 1 );

    while((char)keyboard != 27)
    {
        char k = (char)keyboard;
        switch(keyboard)
        {
        // right arrow
        case 2555904:
            image = imageLoader.getNextImage();
            break;
        // left arrow
        case 2424832:
            image = imageLoader.getPreviousImage();
            break;
        case 0:
            break;
        default:
            keyboard = cv::waitKey();
            continue;
            break;
        }

        if ( !imageLoader.getCurrentImage().data )
        {
            printf("No image data \n");
            return EXIT_FAILURE;
        }

        std::vector<TargetInstance> detectedTargets = std::vector<TargetInstance>();

        // ######   COARSE      ##############
		const double downSampleFac = 0.05;

        cv::Mat prewitt;
        cv::Mat downSampledImage;
		cv::cvtColor(image,image,CV_BGR2GRAY);
		cv::resize(image,downSampledImage,cv::Size(0,0), downSampleFac, downSampleFac);
        previtt(downSampledImage, prewitt);

		//cv::Mat morphKern = getStructuringElement( MORPH_RECT, cv::Size( 2*1 , 2*1  ),
  //                          cv::Point( -1, -1 ) );
		//cv::erode(prewitt,prewitt,morphKern);
		//cv::dilate(prewitt,prewitt,morphKern);
		//cv::Mat shiftkernerl = Mat::zeros( 3, 3, CV_32F );
		//shiftkernerl.at<float>(2,2) = 1.0f; // Indices are zero-based, not relative

		//  /// Apply filter
		//  filter2D(prewitt, prewitt, -1 , shiftkernerl);
		
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(prewitt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        
		// ## Debug for contours at small resolution
		cv::Mat conts = downSampledImage.clone();
		
		// ## Debug for contours resized to full resolution
        cv::Mat resizedConts = image.clone();
        cv::cvtColor(resizedConts,resizedConts,CV_GRAY2BGR);

        cv::Mat tmp;

        // ####     prepare refining    ####

		for(int i = 0; i < contours.size(); ++i)
		{
			cv::Rect r = cv::boundingRect(contours[i]);

			// ## parameter discard small contours
			bool bigEnough = r.width >= 20 && r.height >= 20;
			bool biggerCont = contours[i].size() >= 70;
			if(bigEnough && biggerCont )
			{
                //cv::drawContours(conts,contours,i,cv::Scalar(0,0,255),1);

				if(evaluateFirstCondition(downSampledImage,contours[i]) && evaluateSecondCondition(r))
                {
                    for(int j = 0; j < contours[i].size(); ++j)
                    {
                        // ## Debug show coarse contours on coarse image
                        cv::circle(conts,contours[i][j],1,cv::Scalar(0,0,255),-1);

						contours[i][j].x *= 1/downSampleFac;
						contours[i][j].y *= 1/downSampleFac;



                        // ## Debug show coarse contours on highres image
                        cv::circle(resizedConts, contours[i][j],10,cv::Scalar(0,255,0),-1);
                    }

                    clock_t start = clock();
                    refineBullsEyes(tmp, contours[i]);
					// end of profiling
                    double elapsed = (double)(clock() - start);
                    double elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
                    std::cout << "refinement took  " << elapsed_sec << "seconds." << std::endl;
                }
			}
		}
		// ## Debug show the generated mask
        show("mask", tmp,0.3);

        // #########        REFINE      ##########
		contours.clear();
		cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		for(int i = 0; i < contours.size(); ++i)
		{
            cv::Rect bounds = cv::boundingRect(contours[i]);
			cv::Rect b = cv::boundingRect(contours[i]);
			if(b.width > 150 || b.height > 150)
			{
                std::vector<std::vector<cv::Point>> targetRings = std::vector<std::vector<cv::Point>>();
                targetRings.push_back(contours[i]);
                detectedTargets.push_back(TargetInstance(targetRings));

                // center of contour
                cv::Point center = detectedTargets.back().getCenter();
                // ## Debug draw targetCenter
                cv::circle(resizedConts, center, 10, cv::Scalar(255,0,0),-1);

				
				//std::vector<std::vector<cv::Point>> targetRings = std::vector<std::vector<cv::Point>>();
				targetRings.clear();
				// loop through points of bullsEye
                for(int j = 0; j < contours[i].size(); ++j)
				{
                    // ## Debug draw bullsEye
					cv::circle(resizedConts, contours[i][j],1,cv::Scalar(0,0,255),-1);

                    // calc and draw ring estimates
                    if(j%8 == 0)
                    {
                        double distToCenter = cv::norm(contours[i][j] - center);
                        double distOfRings = distToCenter/6;
                        for(int ring = 1; ring < 10; ++ring)
                        {
							std::vector<cv::Point> targetRing = std::vector<cv::Point>();
                            if(ring == 6)
                            {
								targetRing.push_back(contours[i][j]);
                                continue;
                            }
                            double scale7R1 = (ring * distOfRings - distOfRings/5)/ distToCenter;
                            double scale7R2 = (ring * distOfRings + distOfRings/5)/ distToCenter;
                            cv::Point ring7_1 = contours[i][j] - center;
                            cv::Point ring7_2;
                            ring7_2 = scale7R2 * ring7_1;
                            ring7_1 *= scale7R1;
                            ring7_1 += center;
                            ring7_2 += center;
                            cv::line(resizedConts,ring7_1, ring7_2, cv::Scalar(150,150,150),1);
                            cv::Point finRingPoint = findRingPoint(ring7_1,ring7_2);
							targetRing.push_back(finRingPoint);
							targetRings.push_back(targetRing);
                            cv::circle(resizedConts, finRingPoint,2,cv::Scalar(255,0,255),-1);
                        }

                    }
				}
				detectedTargets.clear();
				detectedTargets.push_back(TargetInstance(targetRings));
			}  
		}
		
		

        //show("prewittC", conts, 1);
        show("conts", resizedConts,0.3);

//        dst.create(image.size(), image.type());
//        tmp.create(image.size(), image.type());
//        dst = Scalar(0,0,255);

//        cvtColor(image,image_gray,CV_BGR2GRAY);
//        GaussianBlur(image_gray, image_gray, Size(9,9),2,2);

//        namedWindow(name_edgeWindow, WINDOW_NORMAL );
//        createTrackbar("myTrackbar",name_edgeWindow,&lowThreshold,MAX_lowThreshold,CannyThreshold);
//        createTrackbar("myTrackbar2",name_edgeWindow,&ratio,MAX_ratio,CannyThreshold);

//        CannyThreshold(0,0);

//        dst.copyTo( detected_edges, detected_edges);
//        image.copyTo(detected_edges, tmp);
//        imshow( name_edgeWindow, detected_edges );

        //vector<Vec3f> circles;
        //HoughCircles(image_gray,circles,CV_HOUGH_GRADIENT,1,50,20,100,0);

        //for( size_t i = 0; i < circles.size(); i++ )
     // {
     //     Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
     //     int radius = cvRound(circles[i][2]);
     //     // circle center
     //     circle( image_gray, center, 3, Scalar(0,255,0), -1, 8, 0 );
     //     // circle outline
     //     circle( image_gray, center, radius, Scalar(0,0,255), 3, 8, 0 );
     //  }

        //namedWindow("Display Image", WINDOW_AUTOSIZE);
     //   imshow("Display Image", image_gray);

        keyboard = waitKey();

    }

    return 0;
}
