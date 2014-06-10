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

cv::Vec3f calcCircle(cv::Point2f a, cv::Point2f b, cv::Point2f c)
{
	cv::Matx22f left (2*b.x - 2*a.x, 2*b.y - 2*a.x,
						2*c.x - 2*b.x, 2*c.y - 2*b.y);
	cv::Matx21f right (b.x*b.x + b.y*b.y - a.x*a.x -a.y*a.y,
						c.x*c.x + c.y*c.y - b.x*b.x - b.y*b.y);
	cv::Matx21f center;
	cv::solve(left,right,center,cv::DECOMP_LU);

	float aMcX = a.x - center(0,0);
	float aMcY = a.y - center(1,0);
	float radius = sqrt(aMcX*aMcX + aMcY*aMcY);
	return cv::Vec3f(center(0,0),center(1,0), radius);
}
cv::Vec3f calcCircle2(cv::Point2f a, cv::Point2f b, cv::Point2f c)
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

int main(int argc, char** argv )
{
    /*something();
	std::vector<std::vector<cv::Point>> ellipse2 = std::vector<std::vector<cv::Point>>();
    ellipse2.push_back(ellipse);
	cv::polylines(image,ellipse2,true,cv::Scalar(255,0,255),1);
	cv::rectangle(image,rotRec.boundingRect() - rotRec.boundingRect().tl() ,cv::Scalar(0,255,0),1);
    cv::ellipse(image, rotRec, cv::Scalar(255,0,0));*/

	cv::Point p1(230,310);
	cv::Point p2(600,450);
	cv::Point p3(790,266);

	cv::Vec3f circ = calcCircle2(cv::Point2f(p1),cv::Point2f(p2),cv::Point2f(p3));
	cv::circle(image, cv::Point(circ[0], circ[1]), circ[2], cv::Scalar(255,0,255),2);

	cv::circle(image, p1, 2, cv::Scalar(255,0,0),-1);
	cv::circle(image, p2, 2, cv::Scalar(0,255,0),-1);
	cv::circle(image, p3, 2, cv::Scalar(0,0,255),-1);

	

}
