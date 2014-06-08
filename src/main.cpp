#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <time.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "imageloader.h"
#include "imageprocessor.h"
#include "targetinstance.h"



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
    const float RESIZE = 0.6f;

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

void chooseTemplate(ImageProcessor::TemplateType p_templType, cv::Mat& p_templ)
{
    switch(p_templType)
    {
    case ImageProcessor::Color:
        p_templ = cv::imread("template_color_big.jpg");

        break;
    case ImageProcessor::Sobel:
        p_templ = cv::imread("template_sobel.jpg",0);
        break;
    case ImageProcessor::Gray:
        p_templ = cv::imread("template_color_big.jpg");
        cv::cvtColor(p_templ,p_templ,CV_BGR2GRAY);
        break;
    }
}

std::vector<cv::Matx21f> calcRadialRays(int p_amount)
{
    std::vector<cv::Matx21f> out = std::vector<cv::Matx21f>();
    cv::Matx21f start(1,0);
    float stepSize = 2 * M_PI / (float)p_amount;
    for(float alpha = 0; alpha < 2 * M_PI; alpha += stepSize)
    {
        cv::Matx22f rotate (cos(alpha),sin(alpha),
                            -sin(alpha), cos(alpha));
        out.push_back(rotate * start);
    }
    return out;
}

void drawRays(const std::vector<cv::Matx21f>& rays, cv::Mat drawOn, cv::Point2f drawAt)
{
    float size = 100;
    for(auto rayIter = rays.cbegin(); rayIter != rays.cend(); ++rayIter )
    {
        cv::line(drawOn,drawAt,cv::Point(drawAt.x + size * (*rayIter)(0,0),
                 drawAt.y + size *(*rayIter)(1,0)),cv::Scalar(0,255,0),1);
    }

    cv::imshow("rays", drawOn);
}

int main(int argc, char** argv )
{
    if(argc <= 1)
    {
        std::cerr << ERROR_INVALID_ARGUMENTS;
        std::cout << HELP_MSG << std::endl;
        return EXIT_FAILURE;
    }

    // prepare helper classes
    imageLoader = ImageLoader(argv[1]);
    imageProcesor = ImageProcessor();
    imageProcesor.setImage(imageLoader.getCurrentImage());

    std::vector<cv::Matx21f> rays = calcRadialRays(20);
    cv::Mat draw = cv::Mat::zeros(cv::Size(600,600),CV_8UC3);
    drawRays(rays,draw,cv::Point(200,200));
    cv::waitKey();
    return 0;


    while((char)keyboard != 27)
    {
        switch(keyboard)
        {
        // right arrow
        case 2555904:
            imageProcesor.setImage(imageLoader.getNextImage());
            break;
        // left arrow
        case 2424832:
            imageProcesor.setImage(imageLoader.getPreviousImage());
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

        ImageProcessor::TemplateType templateType = ImageProcessor::Gray;
        imageProcesor.setImage(imageLoader.getCurrentImage());
        std::vector<TargetInstance> targets = std::vector<TargetInstance>();
        std::vector<TemplateMatch> targetPositions = std::vector<TemplateMatch>();

        const int MIN_TEMPL_SIZE = 60;
        const int TEMPL_STEP_SIZE = 10;
        // scale image and template to speed up matching
        float imageDonwnScale = 0.5;

        cv::Mat templ = cv::Mat();
        //choose template
        chooseTemplate(templateType, templ);

        // check if template was loaded
        if(templ.empty())
        {
            std::cerr << "temlate not loaded" << std::endl;
            return EXIT_FAILURE;
        }

        // profile template matching
        clock_t templStart = clock();

        // find positions of targets with template matching
        targetPositions = imageProcesor.processImage_TemplateMatch(templ, templateType, MIN_TEMPL_SIZE, TEMPL_STEP_SIZE, imageDonwnScale);

        // end of profiling
        double templElapsed = (double)(clock() - templStart);
        double templElapsed_sec =  templElapsed / (double)CLOCKS_PER_SEC;
        std::cout << "Template Matching took " << templElapsed_sec << "seconds." << std::endl;

        // some debug Output
        drawDebug();

        cv::namedWindow(WINDOWNAME_FRAME, cv::WINDOW_NORMAL);
        cv::resizeWindow(WINDOWNAME_FRAME, 1032, 580);
        cv::Mat disp_image;
        disp_image = imageLoader.getCurrentImage();
        drawImageNr(disp_image);
        cv::imshow(WINDOWNAME_FRAME, disp_image);

        //wait until user presses a key
        keyboard = cv::waitKey();
    }
}
