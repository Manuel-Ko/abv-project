#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "imageloader.h"
#include "targetinstance.h"
#include "targetfinder.h"

int keyboard = 0;

float SHOW_SCALE = 1;
const cv::Size SHOW_SIZE = cv::Size(1920,1080);

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

    SHOW_SCALE = SHOW_SIZE.width/(float)image.cols;

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

        clock_t startTime = clock();
		clock_t startAll = startTime;

        targetFinder.findCoarseBullsEyes(coarseBullsEyes,image.clone());

        double elapsed = (double)(clock() - startTime);
        double elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
        std::cout << "finding coarse bullsEyes took " << elapsed_sec << " seconds." << std::endl;

        startTime = clock();

        targetFinder.extractAllTargetRings(coarseBullsEyes, image);

        elapsed = (double)(clock() - startTime);
        elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
        std::cout << "extracting targetrings took " << elapsed_sec << " seconds." << std::endl;

        startTime = clock();

        targetFinder.detectAllBulletHoles();

        double endTime = clock();
		elapsed = (double)(endTime - startTime);
        elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
        std::cout << "detecting bulletholes took " << elapsed_sec << " seconds." << std::endl;

		double elapsedAll = (double)(endTime - startAll);
        elapsed_sec =  elapsedAll / (double)CLOCKS_PER_SEC;
        std::cout << "Processing Image took " << elapsed_sec << " seconds." << std::endl;
		

        detectedTargets = targetFinder.getTargets();
        targetFinder.drawTargets(image);

        myImProc::show("result", image, SHOW_SCALE);
		//cv::imwrite("Ringdetection.jpg", image);


        keyboard = cv::waitKey();
    }
}
