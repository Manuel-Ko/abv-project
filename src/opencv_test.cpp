#include <stdio.h>
#include <opencv2/opencv.hpp>

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

  dst.copyTo( detected_edges, detected_edges);
  image.copyTo(detected_edges, tmp);
  imshow( name_edgeWindow, detected_edges );
 }

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

	dst.create(image.size(), image.type());
	tmp.create(image.size(), image.type());
	dst = Scalar(0,0,255);

	cvtColor(image,image_gray,CV_BGR2GRAY);
	GaussianBlur(image_gray, image_gray, Size(9,9),2,2);

	namedWindow(name_edgeWindow, WINDOW_NORMAL );
	createTrackbar("myTrackbar",name_edgeWindow,&lowThreshold,MAX_lowThreshold,CannyThreshold);
	createTrackbar("myTrackbar2",name_edgeWindow,&ratio,MAX_ratio,CannyThreshold);

	CannyThreshold(0,0);
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

    waitKey();

    return 0;
}
