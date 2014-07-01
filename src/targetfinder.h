#ifndef TARGETFINDER_H
#define TARGETFINDER_H

#include <opencv2/core/core.hpp>

// Debug/Profile
#include <time.h>

#include "targetinstance.h"
#include "myImageProcessing.h"


class TargetFinder
{
public:
    TargetFinder();

    cv::Mat getImage() const;
    void setImage(const cv::Mat &image);

    cv::Mat getImage_gray() const;
    std::vector<TargetInstance> getTargets() const;

    void findCoarseBullsEyes(std::vector<std::vector<cv::Point>>& coarseBullsEyes, cv::Mat& p_debugImage = cv::Mat());
    void extractAllTargetRings(const std::vector<std::vector<cv::Point>>& coarseBullsEyes, cv::Mat& p_debugMat = cv::Mat());
    void detectAllBulletHoles();

    void drawTargets(cv::Mat& p_drawOn);


private:

    static const bool DRAW_RINGLINES = true;
	static const bool DRAW_RINGPOINTS = true;
	static const bool DRAW_BULLSEYES = false;
    static const bool DRAW_BULLERHOLES = false;
    static const bool SHOW_HOLESEGMENTATION = false;
    static const bool SHOW_COARSEDETECTION = true;
    static const cv::Size SHOW_WINDOWSIZE;
    float SHOW_SCALE;


    cv::Mat m_image;
    cv::Mat m_image_gray;

    std::vector<TargetInstance> m_targets;

    bool evaluateFirstCondition_Point(cv::Mat p_image, cv::Point p_point);
    bool evaluateFirstCondition(cv::Mat p_image, std::vector<cv::Point> p_contour);
    bool evaluateSecondCondition(cv::Rect p_rect);
    cv::Point findRingPoint(cv::Point p_p1, cv::Point p_p2);
    void findOnAllRings(cv::Point p_point, cv::Point p_targetCenter,
                        std::vector<std::vector<cv::Point>>& p_targetRings,
                        cv::Mat &p_debugImage = cv::Mat());
    void createRefinedBullsEyeMask(cv::Mat& p_out, std::vector<cv::Point> p_contour);
    void detectBulletHolesOnTarget(TargetInstance& p_target, cv::Mat segMentationImage = cv::Mat());
};

#endif // TARGETFINDER_H
