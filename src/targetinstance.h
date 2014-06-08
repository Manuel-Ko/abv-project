#ifndef TARGETINSTANCE_H
#define TARGETINSTANCE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "templatematch.h"

class TargetInstance
{
public:
    TargetInstance();
    TargetInstance(std::vector<std::vector<cv::Point>> p_rings);

    void setCenter(cv::Point p_center);
    void setRings(std::vector<std::vector<cv::Point>> p_rings);
    cv::Point getCenter();
    std::vector<std::vector<cv::Point>> getRings();
    std::vector<cv::Point> getRing(size_t p_nr);

    // checks if there are all rings
    bool isComplete();

private:
    cv::Point m_center;
    cv::Rect m_bounds;

    // Ringpoints stored relative to the center
    std::vector<std::vector<cv::Point>> m_rings;

};

#endif // TARGETINSTANCE_H
