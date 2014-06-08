#ifndef FILELOADER_H
#define FILELOADER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

class ImageLoader
{
public:
    ImageLoader();
    ImageLoader(char* dir);

    void setDir(char* dir);
    int getIndex();
    int getMaxIndex();
    cv::Mat getCurrentImage();
    void goToNextImage();
    cv::Mat getNextImage();
    void goTOPrevImage();
    cv::Mat getPreviousImage();
    cv::Mat getImageAt(int m_index);

private:
    char* m_directoryPath;
    int m_index;
    int MAX_INDEX;
    bool m_isValid;
    cv::Mat m_currentImage;

    void lookInFolder();
    bool loadImage(int i);

};

#endif // FILELOADER_H
