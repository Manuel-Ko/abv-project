#ifndef TEMPLATEMATCH_H
#define TEMPLATEMATCH_H
#include <opencv2/core/core.hpp>

class TemplateMatch
{
public:
    TemplateMatch();
    TemplateMatch(cv::Point p_topLeft, cv::Size p_size);

    cv::Point getCenter();
    cv::Point getTopLeft();
    cv::Point getBottomRight();
    cv::Size getSize();

private:
    cv::Point m_topLeft;
    cv::Point m_center;
    cv::Size m_size;
};

#endif // TEMPLATEMATCH_H
