#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <time.h>
#include "imageloader.h"
#include "imageprocessor.h"

//const char* ERROR_MSG = "Error: ";
const char* ERROR_INVALID_ARGUMENTS = "Error: The arguments are not valid \n";
const char* HELP_MSG = "usage: <progName> <path_with_images>\nthe images bust be named 000.<ext> 001.<ext> etc";

const char* WINDOWNAME_FRAME = "Image";

int keyboard = 0;
ImageLoader imageLoader;
ImageProcessor imageProcesor;

void drawImageNr(cv::Mat p_image, cv::Point p_pos = cv::Point(30,30), int p_width = 170)
{
    std::string text = cv::format("%d/%d",imageLoader.getIndex(),imageLoader.getMaxIndex());
    cv::rectangle(p_image,p_pos,cv::Point(p_pos.x + p_width,130),cv::Scalar(255,255,255),-1);
    cv::putText(p_image,text,cv::Point(p_pos.x + 10,110),CV_FONT_HERSHEY_DUPLEX,2,cv::Scalar(0,0,0),5);
}

void drawDebug()
{
    const float RESIZE = 0.3f;

    std::vector<cv::Mat> debugImages = std::vector<cv::Mat>();
    imageProcesor.debugOutput_Sobel(debugImages);
    imageProcesor.debugOutput_TemplateMatch(debugImages);
    for(uint8_t i = 0; i < debugImages.size(); ++i)
    {
        std::string windowName = cv::format( "Debugimage%d", i );
        cv::namedWindow( windowName, CV_WINDOW_KEEPRATIO );
        cv::resizeWindow( windowName, int(RESIZE*debugImages[i].cols), int(RESIZE*debugImages[i].rows) );
        drawImageNr(debugImages[i]);
        cv::imshow(windowName,debugImages[i]);
    }
}

int main(int argc, char** argv )
{
    if(argc <= 1)
    {
        std::cerr << ERROR_INVALID_ARGUMENTS;
        std::cout << HELP_MSG << std::endl;
        return EXIT_FAILURE;
    }

    imageLoader = ImageLoader(argv[1]);
    imageProcesor = ImageProcessor();

    cv::Mat calc_image = imageLoader.getCurrentImage();
    cv::Mat disp_image;

    while((char)keyboard != 27)
    {
        switch(keyboard)
        {
        // right arrow
        case 2555904:
            calc_image = imageLoader.getNextImage();
            break;
        // left arrow
        case 2424832:
            calc_image = imageLoader.getPreviousImage();
            break;
        }

        if ( !calc_image.data )
        {
            printf("No image data \n");
            return -1;
        }

        ImageProcessor::TemplateType matchOn = ImageProcessor::Sobel;
        imageProcesor.setImage(calc_image);



        clock_t templStart = clock();
        imageProcesor.processImage_TemplateMatch(matchOn);
        double templElapsed = ((double)(clock() - templStart)) / (double)CLOCKS_PER_SEC;
        std::cout << "Template Matching took " << templElapsed << "seconds." << std::endl;

        drawDebug();

        cv::namedWindow(WINDOWNAME_FRAME, cv::WINDOW_NORMAL);
        cv::resizeWindow(WINDOWNAME_FRAME, 1032, 580);
        disp_image = calc_image.clone();
        drawImageNr(disp_image);
        cv::imshow(WINDOWNAME_FRAME, disp_image);

        //wait until user presses a key
        keyboard = cv::waitKey();
    }
}
