#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "imageloader.h"
#include "targetinstance.h"
#include "targetfinder.h"

int keyboard = 0;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        cv::waitKey();
        return -1;
    }

    ImageLoader imageLoader = ImageLoader(argv[1]);

    cv::Mat image = imageLoader.getCurrentImage();

    TargetFinder targetFinder = TargetFinder();
    targetFinder.setImage(image.clone());

    while((char)keyboard != 27)
    {
        switch(keyboard)
        {
        // right arrow
        case 2555904:
            image = imageLoader.getNextImage();
            targetFinder.setImage(image.clone());
            cv::destroyWindow("frame");
            break;
        // left arrow
        case 2424832:
            image = imageLoader.getPreviousImage();
            targetFinder.setImage(image.clone());
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

        std::vector<std::vector<cv::Point>> coarseBullsEyes = std::vector<std::vector<cv::Point>>();
        std::vector<TargetInstance> detectedTargets = std::vector<TargetInstance>();
        targetFinder.findCoarseBullsEyes(coarseBullsEyes,image.clone());
        targetFinder.extractAllTargetRings(coarseBullsEyes, image.clone());
        targetFinder.detectAllBulletHoles();
        detectedTargets = targetFinder.getTargets();
        targetFinder.drawTargets(image);

        myImProc::show("result", image, 0.3);


        keyboard = cv::waitKey();
    }
}
