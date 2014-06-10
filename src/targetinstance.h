#ifndef TARGETINSTANCE_H
#define TARGETINSTANCE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "templatematch.h"

class TargetInstance
{
public:
    TargetInstance();
    TargetInstance(std::vector<std::vector<cv::Point>> p_rings);

    static const unsigned int MAX_RANSAC_ITERATIONS = 9999;

    void setCenter(cv::Point p_center);
    void setRings(std::vector<std::vector<cv::Point>> p_rings);
    void addBulletHole(cv::Vec3f p_hole);
    cv::Point getCenter();
    cv::Rect getBoundingRect();
    std::vector<std::vector<cv::Point>> getRings();
    std::vector<cv::Point> getRing(size_t p_nr);
    std::vector<cv::RotatedRect> getRingEllipses();
    std::vector<cv::Vec4f> getBulletHoles();
    float getScoreMean();
    float getScoreSum();
    float getScoreMax();

    // checks if there are all rings
    bool isComplete();

private:    
    cv::Point m_center;
    cv::Rect m_bounds;

    std::vector<std::vector<cv::Point>> m_rings;
    std::vector<cv::RotatedRect> m_ringEllipses;
    std::vector<cv::Vec4f> m_bulletHoles;

};

#endif // TARGETINSTANCE_H
