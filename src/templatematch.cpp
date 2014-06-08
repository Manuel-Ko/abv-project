#include <math.h>
#include "templatematch.h"

TemplateMatch::TemplateMatch() :
    m_topLeft(cv::Point()),
    m_size(cv::Size())    
{
	//TODO get more accurate
    m_center = cv::Point( m_topLeft.x+m_size.width/2, m_topLeft.y + m_size.height/2);
}

TemplateMatch::TemplateMatch(cv::Point p_topLeft, cv::Size p_size) :
    m_topLeft(p_topLeft),
    m_size(p_size)
{
	//TODO get more accurate
    m_center = cv::Point( m_topLeft.x+m_size.width/2, m_topLeft.y + m_size.height/2);
}
cv::Point TemplateMatch::getCenter()
{
    return m_center;
}

cv::Point TemplateMatch::getTopLeft()
{
    return m_topLeft;
}
cv::Point TemplateMatch::getBottomRight()
{
    return cv::Point(m_topLeft.x + m_size.width - 1, m_topLeft.y + m_size.height -1);
}

cv::Size TemplateMatch::getSize()
{
    return m_size;
}
