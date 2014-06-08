#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "imageloader.h"

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


    show("hortmp", abs_horTmp, 1);
    show("verttmp", abs_vertTmp, 1);

    cv::addWeighted( abs_horTmp, 0.5, abs_vertTmp, 0.5, 0, dst );
	cv::threshold(dst,dst,120,255,CV_THRESH_TOZERO);

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


        cv::Mat prewitt;
		cv::cvtColor(image,image,CV_BGR2GRAY);
        cv::resize(image,image,cv::Size(0,0), 0.05,0.05);
        previtt(image, prewitt);

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(prewitt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		cv::Mat conts = cv::Mat(cv::Size(prewitt.cols, prewitt.rows),CV_8UC3);
		for(int i = 0; i < contours.size(); ++i)
		{
			cv::Rect r = cv::boundingRect(contours[i]);
			if(r.width >= 30, r.height >= 30)
			{
				cv::drawContours(conts,contours,i,cv::Scalar(0,0,255),1);
				int firstConditionCount = 0;
			for(int j = 0; j < contours[i].size(); ++j)
			{
				int up = std::max(contours[i][j].y -1, 0);
				int down = std::min(contours[i][j].y +2, image.rows);
				int left = std::max(contours[i][j].x -1, 0);
				int right = std::min(contours[i][j].x +2, image.cols);
				/*cv::Point upP = cv::Point(contours[i][j].x, up);
				cv::Point downP = cv::Point(contours[i][j].x, down);
				cv::Point leftP = cv::Point(left, contours[i][j].y);
				cv::Point rightP = cv::Point(right, contours[i][j].y);*/

				cv::Rect roi = cv::Rect(cv::Point(left,up), cv::Point(right, down));
				cv::Point lolo = roi.br();
				
				if(roi.x <0 || roi.y < 0)
				{
					std::cout << "here";
				}
				if(roi.x+roi.width > image.cols)
				{
					std::cout << "or here";
				}
				if(roi.y + roi.height > image.rows)
				{
					std::cout << "maybe here";
				}
				cv::Mat roiMat = image(roi);
				roiMat.convertTo(roiMat,CV_32F);

				roiMat -= image.at<unsigned char>(contours[i][j].y,contours[i][j].x);
				roiMat = abs(roiMat);
				
				double result = 0;
				double garbage = 0;
				cv::minMaxIdx(roiMat,&garbage, &result);
				if(result < image.at<unsigned char>(contours[i][j].y,contours[i][j].x))
				{
					firstConditionCount++;
				}
			}
			firstConditionCount--;
			}
			
		}
		

        show("prewittC", conts, 1);

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
