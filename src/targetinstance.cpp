#include "targetinstance.h"

// ####     constructors    ####
TargetInstance::TargetInstance() :
    m_center(cv::Point(0,0)),
    m_bounds(cv::Rect()),
    m_rings(std::vector<std::vector<cv::Point>>())
{
}

TargetInstance::TargetInstance(std::vector<std::vector<cv::Point>> p_rings) :
    m_rings(p_rings)
{
    m_bounds = cv::boundingRect(m_rings[0]);
    m_center = m_bounds.tl() + cv::Point(m_bounds.width/2, m_bounds.height/2);
}

// ####     public methods      ####


void TargetInstance::setRings(std::vector<std::vector<cv::Point>> p_rings)
{
    if(p_rings.empty())
    {
        std::cerr << "no rings supplied" << std::endl;
        return;
    }
    m_rings = p_rings;
    m_bounds = cv::boundingRect(m_rings[0]);
}

cv::Point TargetInstance::getCenter()
{
    return m_center;
}

std::vector<std::vector<cv::Point>> TargetInstance::getRings()
{
    return m_rings;
}

std::vector<cv::Point> TargetInstance::getRing(size_t p_nr)
{
    if(p_nr >= m_rings.size())
    {
        std::cerr << "tried to get invalid ring" << std::endl;
        return std::vector<cv::Point>();
    }
    return m_rings[p_nr];
}

bool TargetInstance::isComplete()
{
    return m_rings.size() == 9;
}

// ####     private methods     ####

