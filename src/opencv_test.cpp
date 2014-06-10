#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "imageloader.h"
#include "targetinstance.h"

using namespace cv;

Mat image;
Mat image_gr, image_gray;
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
    if(p_image.cols <= 0 && p_image.rows <= 0)
    {
        std::cerr << "cant show an empty image" << std::endl;
        return;
    }
    cv::namedWindow(p_windowName, CV_WINDOW_KEEPRATIO);
    cv::resizeWindow(p_windowName,p_image.cols * p_size, p_image.rows * p_size);
    cv::imshow(p_windowName, p_image);
}

void drawImageNr(cv::Mat p_image, int p_index, int p_maxindex, cv::Point p_pos = cv::Point(30,30), int p_width = 170)
{
	std::string text = cv::format("%d/%d",p_index,p_maxindex);
    cv::rectangle(p_image,p_pos,cv::Point(p_pos.x + p_width,130),cv::Scalar(255,255,255),-1);
    cv::putText(p_image,text,cv::Point(p_pos.x + 10,110),CV_FONT_HERSHEY_DUPLEX,2,cv::Scalar(0,0,0),5);
}

void on_trackbar(int, void*)
{
    Mat modified;
    image_gray.convertTo(modified,-1,testval/100.0f,0);

	
    imshow("Display Image", modified);

}

cv::Vec3f calcCircle(cv::Point2f a, cv::Point2f b, cv::Point2f c)
{
	cv::Matx33f first (a.x, a.y, 1,
						b.x, b.y, 1,
						c.x, c.y, 1);
	cv::Matx33f second (a.x*a.x + a.y*a.y, a.y, 1,
						b.x*b.x + b.y*b.y, b.y, 1,
						c.x*c.x + c.y*c.y, c.y, 1);
	cv::Matx33f third (a.x*a.x + a.y*a.y, a.x, 1,
						b.x*b.x + b.y*b.y, b.x, 1,
						c.x*c.x + c.y*c.y, c.x, 1);
	cv::Matx33f fourth (a.x*a.x + a.y*a.y, a.x, a.y,
						b.x*b.x + b.y*b.y, b.x, b.y,
						c.x*c.x + c.y*c.y, c.x, c.y);

	float firstDet = cv::determinant(first);
	float secondDet = -1 * cv::determinant(second);
	float thirdDet = cv::determinant(third);
	float fourthDet = -1 * cv::determinant(fourth);

	float centerX = -1 * secondDet/(2*firstDet);
	float centerY = -1 * thirdDet/(2*firstDet);
	float radius = sqrt(secondDet*secondDet + thirdDet*thirdDet - 4 * firstDet * fourthDet)/(2*abs(firstDet));

	return cv::Vec3f(centerX, centerY, radius);
}

std::vector<cv::Vec3f> circleRANSAC(const std::vector<cv::Point>& p_points, const cv::Mat& p_distanceImage,
									float p_threshold , int p_iterations, float p_minRad, float p_maxRad, float p_minDist)
{
	const bool DEBUG = true;
	std::vector<cv::Vec3f> out = std::vector<cv::Vec3f>();
	
	int index[3];
	for(int i = 0; i < 3;i++)
	{
		bool match = false;
		do {
			  match = false;
			  index[i] = rand()%p_points.size();
			  for(int j=0; j<i ;j++)
			  {
					if(index[i] == index[j])
					{
						match=true;
					}
			  }
		}while(match);
	}

	for(size_t iteration = 0; iteration < p_iterations; ++iteration)
	{
		cv::Vec3f circ = calcCircle(p_points[index[0]], p_points[index[1]], p_points[index[2]]);
		int radius = cvRound(circ[2]);
		cv::Point circleCenter(cvRound(circ[0]), cvRound(circ[1]));

		bool tooSmall = radius < p_minRad;
		bool tooBig = radius > p_maxRad;
		if(tooSmall || tooBig)
		{
			continue;
		}

		//int oddplus = 1 - radius%2;

		// sample Circle on Mat with minimum size
		cv::Mat circSamples = cv::Mat::zeros(cv::Size(radius * 2 + 1,radius * 2 + 1), CV_8UC1);
		cv::Point sampleCircleCenter(radius, radius);
		cv::circle(circSamples, sampleCircleCenter, radius, cv::Scalar(255));

		std::vector<cv::Point> circlePoints = std::vector<cv::Point>();
		//collect circle points
		for(int row = 0; row < circSamples.rows; ++row)
		{
			uchar* p = circSamples.ptr(row);
			for(int col = 0; col < circSamples.cols; ++col) 
			{
				if(*p == 255)
				{
					circlePoints.push_back(cv::Point(col,row));
				}
				 *p++;
			}
		}

		// ## Debug
		cv::Mat debugImg;
		if(DEBUG)
		{
			cv::threshold(p_distanceImage,debugImg,0,255,CV_THRESH_BINARY_INV);
			cv::cvtColor(debugImg,debugImg, CV_GRAY2BGR);
			cv::circle(debugImg,circleCenter, radius, cv::Scalar(255,0,0),1);
			cv::circle(debugImg,circleCenter, 2, cv::Scalar(255,0,0),-1);
		}

		float distSum = 0;
		for(size_t i = 0; i < circlePoints.size(); ++i)
		{
			// translate points back on their original Position as detected by calcCircle
			circlePoints[i] -= sampleCircleCenter;
			circlePoints[i] += circleCenter;
			
			bool upLeftOutside = circlePoints[i].x < 0 || circlePoints[i].y < 0;
			bool bottomRightOutside = circlePoints[i].x >= p_distanceImage.cols || circlePoints[i].y >= p_distanceImage.rows;
			if(upLeftOutside || bottomRightOutside)
			{
				continue;
			}

			float curVal = p_distanceImage.at<float>(circlePoints[i].y, circlePoints[i].x );
			distSum += curVal;
			if(DEBUG)
			{
				cv::line(debugImg,circlePoints[i],circlePoints[i],cv::Scalar(0,255,255));
				if(curVal <= 0.0001)
				{
					cv::line(debugImg, circlePoints[i], circlePoints[i], cv::Scalar(255,0,255));
				}
			}
		}

		// use mean Distance
		distSum /= circlePoints.size();

		if(distSum <= p_threshold)
		{
			bool tooClose = false;
			// check for detected circles close to current
			for(int circIdx = 0; circIdx < out.size(); ++circIdx)
			{
				cv::Point oldCirc = cv::Point(out[circIdx][0], out[circIdx][1]);
				float circDist = cv::norm(oldCirc - sampleCircleCenter);

				if(circDist < p_minDist)
				{
					tooClose = true;
				}
			}
			if(!tooClose)
			{
				out.push_back(circ);
			}
		}
	}
	return out;
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
        p_out = cv::Mat::zeros(image_gr.size(), CV_8UC1);
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
        contours[0][i].x = std::min(image_gr.cols,contours[0][i].x);
        contours[0][i].y = std::min(image_gr.rows,contours[0][i].y);

    }
    boundingRec = cv::boundingRect(contours[0]);
    contourCenter = cv::Point(boundingRec.tl().x + boundingRec.width/2, boundingRec.tl().y + boundingRec.height/2 );
    for(int i = 0; i < contours[0].size(); ++i)
    {
        contours[0][i] -= boundingRec.tl();
    }

    cv::Mat roiImage = image_gr(boundingRec);
    // create a mask based on coarse contour
    cv::Mat mask = cv::Mat::zeros(roiImage.size(), CV_8UC1);
    cv::drawContours(mask,contours,0,cv::Scalar(255),-1);

    // mask out roughly detected bullsEye
    cv::Mat thresholdImage;

    roiImage.copyTo(thresholdImage,mask);

    // discard bright pixels as they are not part of the bullsEye
    // ## parameter URGENT thresh for bullsEye
    cv::threshold(thresholdImage,thresholdImage, 100,255,CV_THRESH_BINARY_INV);

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

    cv::LineIterator lineIter = cv::LineIterator(image_gr,p_p1, p_p2);
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
        cv::Point checkPos = lineIter.pos();
        bool posXInImage = checkPos.x >= 0 && checkPos.x < image_gr.cols;
        bool posYInImage = checkPos.y >= 0 && checkPos.y < image_gr.rows;
        if(i < lineIter.count && posXInImage && posYInImage)
        {
            vNext = **lineIter;
        }

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

//TODO: refactor
void findOnAllRings(int i, std::vector<std::vector<cv::Point>> contours, int j, cv::Point center, std::vector<std::vector<cv::Point>>& targetRings)
{
    double distToCenter = cv::norm(contours[i][j] - center);
    double distOfRings = distToCenter/6;
    for(int ring = 1; ring < 10; ++ring)
    {
        if(ring == 6)
        {
            targetRings[ring-1].push_back(contours[i][j]);
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
        //cv::line(resizedConts,ring7_1, ring7_2, cv::Scalar(150,150,150),1);
        cv::Point finRingPoint = findRingPoint(ring7_1,ring7_2);
        targetRings[ring-1].push_back(finRingPoint);
        //cv::circle(resizedConts, finRingPoint,2,cv::Scalar(255,0,255),-1);
    }
}

void drawTargets(cv::Mat& p_drawOn, std::vector<TargetInstance> p_targets)
{
    for(int i = 0; i < p_targets.size(); ++i)
    {

        cv::circle(p_drawOn, p_targets[i].getCenter(), 5, cv::Scalar(255,0,0),-1);
        std::vector<std::vector<cv::Point>> rings = p_targets[i].getRings();
       /* for(int j = 0; j < rings.size(); ++j)
        {
            for( int k = 0; k < rings[j].size(); ++k )
            {
                cv::Scalar color = cv::Scalar(255,255,0);
                if(j == 3)
                {
                    color = cv::Scalar(255,255,255);
                }
                cv::circle(p_drawOn, rings[j][k],1,color,-1);
            }
        }*/
        std::vector<cv::RotatedRect> ringEllipses = p_targets[i].getRingEllipses();
        for(int j = 0; j < ringEllipses.size(); ++j)
        {
            cv::Scalar color = cv::Scalar(180,0,0);
            /*switch(j)
            {
            case 0:
                color = cv::Scalar(255,0,255);
                break;
            case 2:
                color = cv::Scalar(0,255,255);
                break;
            case 5:
                color = cv::Scalar(0,0,255);
                break;
            case 7:
                color = cv::Scalar(0,255,0);
                break;
            }*/

            cv::ellipse(p_drawOn, ringEllipses[j],color,3);
        }

		std::vector<cv::Vec4f> holes = p_targets[i].getBulletHoles();
		for(int j = 0; j < holes.size(); ++j)
		{
			cv::Vec4f hole = holes[j]; 
			cv::Point2f dist = cv::Point2f(hole[0], hole[1]) - cv::Point2f(p_targets[i].getCenter());
			float distLength = cv::norm(dist);
			dist.x /= distLength;
			dist.y /= distLength;
			float debugDistL = cv::norm(dist);
			float sclaeToCenter = std::min(distLength, hole[2]);
			cv::Point closestToCenter = cv::Point(hole[0], hole[1]) - cv::Point(dist * sclaeToCenter);


			
			cv::Point center = cv::Point(holes[j][0], holes[j][1]);
			cv::Scalar color(255,0,255);
			cv::circle(p_drawOn,center,holes[j][2],color,-1);
			cv::circle(p_drawOn,closestToCenter,1,cv::Scalar(255,255,0),-1);
			float sc = holes[j][3];
			std::string score = cv::format("%.1f",sc);
			cv::putText(p_drawOn,score,center /*- cv::Point(holes[j][2],holes[j][2])*/,CV_FONT_HERSHEY_DUPLEX,0.9,cv::Scalar(0,0,0),2);
		}
    }
}

void processImage_Sobel(cv::Mat p_image, cv::Mat& p_out)
{
    cv::Mat image_grau = p_image;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::GaussianBlur( image_grau, image_grau, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

    cv::Sobel( image_grau, grad_x, CV_16S, 1, 0, 3 );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    cv::Sobel( image_grau, grad_y, CV_16S, 0, 1, 3);
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, p_out );


}

void detectBulletHoles(cv::Mat p_image, TargetInstance& p_target)
{
    cv::Rect roi = p_target.getBoundingRect();
    cv::Mat roiImage = p_image.clone()(roi);

	cv::Mat roi_gray;
	cv::cvtColor(roiImage, roi_gray, CV_BGR2GRAY);

    cv::Mat thresholdImage;
    cv::cvtColor(roiImage, roiImage, CV_BGR2HSV);

	// ## parameter HSV threshold for bulletholes
    cv::Scalar lowerBounds = cv::Scalar(0,0,130);
    cv::Scalar upperBounds = cv::Scalar(255,50,255);
    cv::inRange(roiImage,lowerBounds,upperBounds,thresholdImage);

	cv::Mat holeMask = cv::Mat::zeros(roiImage.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
    cv::findContours(thresholdImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    std::vector<std::vector<cv::Point>> bulletHoles = std::vector<std::vector<cv::Point>>();

    // ## parameter bullethole
	double maxSideLength = std::max(roi.width, roi.height);
	double minDist = maxSideLength/22;
	double minRad = maxSideLength/37;
	double maxRad = 2*minRad;
	
	for(int i = 0; i < contours.size(); ++i)
    {
        cv::Rect bounds = cv::boundingRect(contours[i]);
		//overlapping holes can form a bigger region than maxRad
		bool tooBig = bounds.width > roi.width/2 || bounds.height > roi.height/2;
		bool tooSmall = bounds.width < minRad && bounds.height < minRad;
		if(!tooSmall && !tooBig)
        {
            double area = cv::contourArea(contours[i]);
            if(area >= 0.5 * bounds.area())
            {
                bulletHoles.push_back(contours[i]);
            }
       }
    }


	std::vector<cv::Vec3f> circles = std::vector<cv::Vec3f>();
    for(int i = 0; i < bulletHoles.size(); ++i)
    {
		cv::Mat holeContours = cv::Mat::zeros(holeMask.size(),CV_8UC1);
        cv::drawContours(holeMask, bulletHoles,i,cv::Scalar(255),-1);
		cv::drawContours(holeContours, bulletHoles,i,cv::Scalar(255), 1);

		cv::bitwise_not(holeContours,holeContours);
		cv::Mat holeDistTrans;
		cv::distanceTransform(holeContours, holeDistTrans, CV_DIST_L2, CV_DIST_MASK_PRECISE);

		// TODO: use Lists instead of vectors
		std::vector<cv::Vec3f> holeCircles = circleRANSAC(bulletHoles[i],holeDistTrans,10,30,minRad,maxRad,minDist);
		for(auto circIter = holeCircles.begin(); circIter != holeCircles.end(); ++circIter)
		{
			circles.push_back(*circIter);
		}
	}

    
    //cv::HoughCircles(holeMask,circles,CV_HOUGH_GRADIENT,1,minDist,100,8,minRad,maxRad);
	//30,100,8,15,60
	
	// ## Debug
	cv::Mat debugHoleImage(holeMask.size(), CV_8UC1);
	cv::Canny(holeMask,debugHoleImage,50,100);


    for( size_t i = 0; i < circles.size(); i++ )
    {
		cv::Point samplePoint = cv::Point(circles[i][0], circles[i][1]); 
		//samplePoint -= roi.tl();
		unsigned char sampleVal = holeMask.at<unsigned char>(samplePoint.y, samplePoint.x);
		if( sampleVal == 255 )
		{
			//holeMask.at<unsigned char>(samplePoint.x, samplePoint.y) = 120;
			p_target.addBulletHole(circles[i]);

			// ## Debug
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			cv::circle( debugHoleImage, center, 3, cv::Scalar(180,255,0), -1, 8, 0 );
			// circle outline
			cv::circle( debugHoleImage, center, radius, cv::Scalar(180,0,255), 1, 8, 0 );
		}
        
    }





//	cv::Mat previttImage;
//	previtt(roi_gray, previttImage);
//    processImage_Sobel(roi_gray,previttImage);

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
        switch(keyboard)
        {
        // right arrow
        case 2555904:
            image = imageLoader.getNextImage();
            cv::destroyWindow("frame");
            break;
        // left arrow
        case 2424832:
            image = imageLoader.getPreviousImage();
            cv::destroyWindow("frame");
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
        const double downSampleFac = 0.1;

        cv::Mat prewitt;
        cv::Mat downSampledImage;
        cv::cvtColor(image,image_gr,CV_BGR2GRAY);
        cv::resize(image_gr,downSampledImage,cv::Size(0,0), downSampleFac, downSampleFac);
        previtt(downSampledImage, prewitt);
		
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(prewitt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        
		// ## Debug for contours at small resolution
		cv::Mat conts = downSampledImage.clone();
		
		// ## Debug for contours resized to full resolution
        cv::Mat resizedConts = image_gr.clone();
        cv::cvtColor(resizedConts,resizedConts,CV_GRAY2BGR);

        //TODO: replace with counter
        std::vector<std::vector<cv::Point>> coarseBullsEyes = std::vector<std::vector<cv::Point>>();

        cv::Mat tmp;

        // ####     COARSE POSTPOCESS    ####

		for(int i = 0; i < contours.size(); ++i)
		{
			cv::Rect r = cv::boundingRect(contours[i]);

			// ## parameter discard small contours
			bool bigEnough = r.width >= 20 && r.height >= 20;
			bool biggerCont = contours[i].size() >= 70;
			if(bigEnough && biggerCont )
			{
                // test if contour is a BullsEye
				if(evaluateFirstCondition(downSampledImage,contours[i]) && evaluateSecondCondition(r))
                {
                    for(int j = 0; j < contours[i].size(); ++j)
                    {
                        // ## Debug draw coarse contours on coarse image
                        cv::circle(conts,contours[i][j],1,cv::Scalar(0,0,255),-1);

						contours[i][j].x *= 1/downSampleFac;
						contours[i][j].y *= 1/downSampleFac;



                        // ## Debug draw coarse contours on highres image
                        cv::circle(resizedConts, contours[i][j],10,cv::Scalar(0,255,0),-1);
                    }
                    coarseBullsEyes.push_back(contours[i]);

                    clock_t start = clock();
                    refineBullsEyes(tmp, contours[i]);
					// end of profiling
                    double elapsed = (double)(clock() - start);
                    double elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
                    std::cout << "refinement took  " << elapsed_sec << " seconds." << std::endl;
                }
			}
		}

        if(coarseBullsEyes.empty())
        {
            std::cout << "no bullsEyes detected" << std::endl;
            show("frame", image_gr, 0.3);
            keyboard = cv::waitKey();
            continue;
        }

        if(false)
        {
            // ## Debug show the generated mask
            show("mask", tmp,0.3);
        }


        // #########    REFINE POSTPROCESS      ##########
		contours.clear();
		cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        // loop through detected bullsEyes
		for(int i = 0; i < contours.size(); ++i)
		{
            // bounding Rect of possible BullsEye
            cv::Rect bounds = cv::boundingRect(contours[i]);

            // ## parameter reject too small contours (generated by refinement)
            if(bounds.width > 150 || bounds.height > 150)
			{
                cv::Point center = bounds.tl() + cv::Point(bounds.width/2, bounds.height/2);

                std::vector<std::vector<cv::Point>> targetRings = std::vector<std::vector<cv::Point>>();
                //create 9 rings
                for(int ringIdx = 0; ringIdx < 9; ++ringIdx)
                {
                    targetRings.push_back(std::vector<cv::Point>());
                }

				// loop through points of bullsEye
                for(int j = 0; j < contours[i].size(); ++j)
				{
                    // ## Debug draw bullsEye
					cv::circle(resizedConts, contours[i][j],1,cv::Scalar(0,0,255),-1);


                    if(j%8 == 0)
                    {
                        // calc ring estimates
                        findOnAllRings(i, contours, j, center, targetRings);
                    }
				}
                // store target with all rings
                detectedTargets.push_back(TargetInstance(targetRings));
			}  
		}


        for(int i = 0; i < detectedTargets.size(); ++i)
        {
            detectBulletHoles(image,detectedTargets[i]);
        }

		drawTargets(resizedConts, detectedTargets);
		
        //show("prewittC", conts, 1);
        drawImageNr(resizedConts, imageLoader.getIndex(), imageLoader.getMaxIndex());
        show("conts", resizedConts,0.3);

        keyboard = waitKey();

    }

    return 0;
}
