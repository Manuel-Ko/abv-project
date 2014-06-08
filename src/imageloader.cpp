#include "imageloader.h"

ImageLoader::ImageLoader() :
    m_directoryPath(""),
    m_index(-1),
    m_isValid(false)
{
}

void ImageLoader::lookInFolder()
{
    std::stringstream ss;
    ss << m_directoryPath;
    cv::VideoCapture sequence( ss.str() + "\\%03d.jpg" );
    if( !sequence.isOpened() )
    {
        std::cerr << "Failed to open image sequence!\n" << std::endl;
        m_isValid = false;
    }

    // go thorough all images to count them
    int count = -1;
    for( ; ; )
    {
        cv::Mat image;
        sequence >> image;

        if( image.empty() )
        {
            std::cout << "End of Sequence - found " << count + 1 << " files" << std::endl;
            break;
        }
        ++count;

    }
    MAX_INDEX = count;
}

ImageLoader::ImageLoader(char *dir) :
    m_directoryPath(dir),
    m_index(0),
    m_isValid(false)
{
    lookInFolder();
    loadImage(m_index);
}

bool ImageLoader::loadImage(int i)
{
    std::string fullFileName = cv::format( "%s/%03d.jpg", m_directoryPath, i );
    m_currentImage = cv::imread(fullFileName, CV_LOAD_IMAGE_ANYCOLOR );
    m_isValid = m_currentImage.data != NULL;
    return m_isValid;
}

void ImageLoader::setDir(char* dir)
{
    m_directoryPath = dir;
    lookInFolder();
}

int ImageLoader::getIndex()
{
    return m_index;
}

int ImageLoader::getMaxIndex()
{
    return MAX_INDEX;
}

cv::Mat ImageLoader::getCurrentImage()
{
    loadImage(m_index);
    return m_currentImage.clone();
}

void ImageLoader::goToNextImage()
{
    ++m_index;
    if(m_index > MAX_INDEX)
    {
        m_index = 0;
    }
    loadImage(m_index);
}

cv::Mat ImageLoader::getNextImage()
{
    goToNextImage();
    return m_currentImage.clone();
}

void ImageLoader::goTOPrevImage()
{
    --m_index;
    if(m_index < 0)
    {
        m_index = MAX_INDEX;
    }
    loadImage(m_index);
}

cv::Mat ImageLoader::getPreviousImage()
{
    goTOPrevImage();
    return m_currentImage.clone();
}

cv::Mat ImageLoader::getImageAt(int index)
{
    loadImage(index);
    return m_currentImage.clone();
}

