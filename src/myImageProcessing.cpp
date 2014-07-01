#include "myImageProcessing.h"

namespace myImProc
{
    void show(const char* p_windowName,cv::Mat p_image, float p_size)
    {
        if(p_image.cols <= 0 && p_image.rows <= 0)
        {
            std::cerr << "cant show an empty image" << std::endl;
            return;
        }
        cv::namedWindow(p_windowName, CV_WINDOW_KEEPRATIO);
        cv::resizeWindow(p_windowName,cvRound(p_image.cols * p_size), cvRound(p_image.rows * p_size));
        cv::imshow(p_windowName, p_image);
    }

    void show(const char* p_windowName,cv::Mat p_image, cv::Size p_size)
    {
        if(p_image.cols <= 0 && p_image.rows <= 0)
        {
            std::cerr << "cant show an empty image" << std::endl;
            return;
        }
        cv::namedWindow(p_windowName, CV_WINDOW_KEEPRATIO);
        cv::resizeWindow(p_windowName,p_size.width, p_size.height);
        cv::imshow(p_windowName, p_image);
    }


	bool pointsAreCollinear(cv::Point a, cv::Point b, cv::Point c)
	{
		cv::Point ab = a-b;
		cv::Point bc = b-c;
		double det = ab.x * bc.y - bc.x * ab.y;
		return det < 0.0001 && det > -0.0001;
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

    void CannyThreshold(const cv::Mat& p_image_gray, cv::Mat& p_out, int p_lowThreshold, int p_highThreshold, int p_kernel_size)
    {
      /// Reduce noise with a kernel 3x3
      blur( p_image_gray, p_image_gray, cv::Size(3,3) );

      /// Canny detector
	  cv::Canny( p_image_gray, p_out, p_lowThreshold, p_highThreshold, p_kernel_size );

      //cv::cvtColor(detected_edges,detected_edges,CV_GRAY2BGR);
     // cv::bitwise_not(detected_edges,tmp);
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
        cv::threshold(dst,dst,100,255,CV_THRESH_TOZERO);

    }

    bool isPointInImage(const cv::Mat& p_image, cv::Point p_point)
    {
        bool upLeftOutside = p_point.x < 0 || p_point.y < 0;
        bool bottomRightOutside = p_point.x >= p_image.cols || p_point.y >= p_image.rows;
        return !(upLeftOutside || bottomRightOutside);
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

		if(cvIsNaN(radius))
		{
			std::cout << "dont get here!";
		}

        return cv::Vec3f(centerX, centerY, radius);
    }


    std::vector<cv::Vec3f> circleRANSAC(const std::vector<cv::Point>& p_points, const cv::Mat& p_distanceImage,
                                        float p_threshold , size_t p_iterations, float p_minRad, float p_maxRad, float p_minDist)
    {
        const bool DEBUG = false;
        std::vector<cv::Vec3f> out = std::vector<cv::Vec3f>();
        std::vector<float> meanDists = std::vector<float>();

        for(size_t iteration = 0; iteration < p_iterations; ++iteration)
        {
            // find 3 diffetent random Points
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

			cv::Point a = p_points[index[0]];
			cv::Point b = p_points[index[1]];
			cv::Point c = p_points[index[2]];

			// ## Debug
            cv::Mat debugImg;
			if(DEBUG)
			{
				cv::threshold(p_distanceImage,debugImg,0,255,CV_THRESH_BINARY_INV);
				cv::cvtColor(debugImg,debugImg, CV_GRAY2BGR);
				// Points to generate the circle
				cv::Scalar rndColor = cv::Scalar(0,0,255);
				cv::line(debugImg,p_points[index[0]], p_points[index[0]], rndColor);
				cv::line(debugImg,p_points[index[1]], p_points[index[1]], rndColor);
				cv::line(debugImg,p_points[index[2]], p_points[index[2]], rndColor);
			}
			

            cv::Vec3f circ = calcCircle(a, b, c);
            int radius = cvRound(circ[2]);
			cv::Point circleCenter(cvRound(circ[0]), cvRound(circ[1]));

			//TODO: avoid negative radi
			if(radius < 0)
			{
				continue;
			}
			
			if(DEBUG)
            {
                
                cv::circle(debugImg,circleCenter, radius, cv::Scalar(255,0,0),1);
                cv::circle(debugImg,circleCenter, 2, cv::Scalar(255,0,0),-1);
                cv::Scalar rndColor = cv::Scalar(0,0,255);

                // Points to generate the circle
                cv::line(debugImg,p_points[index[0]], p_points[index[0]], rndColor);
                cv::line(debugImg,p_points[index[1]], p_points[index[1]], rndColor);
                cv::line(debugImg,p_points[index[2]], p_points[index[2]], rndColor);
            }

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

            float distSum = 0;
            for(size_t i = 0; i < circlePoints.size(); ++i)
            {
                // translate points back on their original Position as detected by calcCircle
                circlePoints[i] -= sampleCircleCenter;
                circlePoints[i] += circleCenter;


                if(!isPointInImage(p_distanceImage, circlePoints[i]))
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
                for(size_t circIdx = 0; circIdx < out.size(); ++circIdx)
                {
                    cv::Point oldCirc = cv::Point(cvRound(out[circIdx][0]), cvRound(out[circIdx][1]));
                    float circDist = cv::norm(oldCirc - circleCenter);

                    if(circDist < p_minDist)
                    {
                        tooClose = true;
                        float oldMean = meanDists[circIdx];
                        if(oldMean > distSum)
                        {
                            // replace old circle if it's too close and has worse score
                            // TODO: check other circles if they now hurt minDist criterion
                            out[circIdx] = circ;
                            meanDists[circIdx] = distSum;
                        }
                        break;
                    }
                }
                if(!tooClose)
                {
                    out.push_back(circ);
                    meanDists.push_back(distSum);
                }
            }
        }
        return out;
    }

}
