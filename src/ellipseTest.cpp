#include <iostream>
#include <opencv2/opencv.hpp>

#define _USE_MATH_DEFINES
#include <math.h>


cv::Mat image = cv::Mat::zeros(cv::Size(800,800), CV_8UC3);
cv::RotatedRect rotRec = cv::RotatedRect(cv::Point(255,255),cv::Size(46,51),40);
std::vector<cv::Point> ellipse;


void something()
{
    float majAxLen = 0;
    float minAxLen = 0;
    if(rotRec.size.width >= rotRec.size.height)
    {
        majAxLen = rotRec.size.width;
        minAxLen = rotRec.size.height;
    }
    else
    {
        majAxLen = rotRec.size.height;
        minAxLen = rotRec.size.width;
    }

	float fLength = sqrt(majAxLen/2 * majAxLen/2 - minAxLen/2 * minAxLen/2);
    float alpha = rotRec.angle;
	alpha *= M_PI/180 * -1;
    cv::Matx22f rotate (cos(alpha),sin(alpha),
                        -sin(alpha), cos(alpha));
    cv::Matx21f vec(1,0);
    vec = rotate * vec;
	vec *= fLength;
    cv::Point vecP = cv::Point(vec(0,0),vec(1,0));
    cv::Point f1 = cv::Point(rotRec.center) + vecP;
    cv::Point f2 = cv::Point(rotRec.center) - vecP;

    /*float dist1 = cv::norm(f1 - closestToCenter);
    float dist2 = cv::norm(f2 - closestToCenter);*/

	cv::Rect debugRoi = rotRec.boundingRect();
	cv::ellipse2Poly(rotRec.center,cv::Size(rotRec.size.width/2, rotRec.size.height/2), rotRec.angle,0,360,5,ellipse);
    for(int i = 0; i < ellipse.size(); ++i)
    {
        ellipse[i] -= debugRoi.tl();
    }
    cv::Mat debug = cv::Mat(debugRoi.size(), CV_8UC3);
    std::vector<std::vector<cv::Point>> ellipse2 = std::vector<std::vector<cv::Point>>();
    ellipse2.push_back(ellipse);
    cv::polylines(debug,ellipse2,true,cv::Scalar(255,0,255),1);
    cv::circle(debug,f1 - debugRoi.tl(),2,cv::Scalar(0,0,255),-1);
    cv::circle(debug,f2 - debugRoi.tl(),2,cv::Scalar(0,255,255),-1);
    //cv::circle(debug,closestToCenter - debugRoi.tl(),2,cv::Scalar(255,0,255),-1);

}

int main(int argc, char** argv )
{
    something();
	std::vector<std::vector<cv::Point>> ellipse2 = std::vector<std::vector<cv::Point>>();
    ellipse2.push_back(ellipse);
	cv::polylines(image,ellipse2,true,cv::Scalar(255,0,255),1);
	cv::rectangle(image,rotRec.boundingRect() - rotRec.boundingRect().tl() ,cv::Scalar(0,255,0),1);
    cv::ellipse(image, rotRec, cv::Scalar(255,0,0));
}
