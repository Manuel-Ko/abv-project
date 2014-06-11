#include "targetfinder.h"

TargetFinder::TargetFinder()
{
}
cv::Mat TargetFinder::getImage() const
{
    return m_image;
}

void TargetFinder::setImage(const cv::Mat &image)
{
    m_image = image;
    cv::cvtColor(m_image,m_image_gray, CV_BGR2GRAY);

	m_targets.clear();
}
cv::Mat TargetFinder::getImage_gray() const
{
    return m_image_gray;
}

std::vector<TargetInstance> TargetFinder::getTargets() const
{
    return m_targets;
}

bool TargetFinder::evaluateFirstCondition_Point(cv::Mat p_image, cv::Point p_point)
{
    int up = std::max(p_point.y -1, 0);
    int down = std::min(p_point.y +2, p_image.rows);
    int left = std::max(p_point.x -1, 0);
    int right = std::min(p_point.x +2, p_image.cols);

    cv::Rect roi = cv::Rect(cv::Point(left,up), cv::Point(right, down));

    cv::Mat roiMat = p_image(roi);
    roiMat.convertTo(roiMat,CV_32F);

    roiMat -= p_image.at<unsigned char>(p_point.y,p_point.x);
    roiMat = abs(roiMat);

    double result = 0;
    double garbage = 0;
    cv::minMaxIdx(roiMat,&garbage, &result);
    if(result < p_image.at<unsigned char>(p_point.y,p_point.x))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool TargetFinder::evaluateFirstCondition(cv::Mat p_image, std::vector<cv::Point> p_contour)
{
    int firstConditionCount = 0;
    for(size_t i = 0; i < p_contour.size(); ++i)
    {
        if(evaluateFirstCondition_Point(p_image,p_contour[i]))
        {
            firstConditionCount++;
        }
    }
    if(firstConditionCount >= 0.9 * p_contour.size())
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool TargetFinder::evaluateSecondCondition(cv::Rect p_rect)
{
    double a = std::max(p_rect.width,p_rect.height);
    double b = std::min(p_rect.width,p_rect.height);

    double res = a/b;
    return (res < 2);
}

void TargetFinder::findCoarseBullsEyes(std::vector<std::vector<cv::Point>>& coarseBullsEyes, cv::Mat p_debugImage)
{
    bool DEBUG = SHOW_COARSEDETECTION && p_debugImage.data;

    // ## parameter corse Resolution
    const double downSampleFac = 0.1;

    cv::Mat prewitt;
    cv::Mat downSampledImage;
    cv::resize(m_image_gray,downSampledImage,cv::Size(0,0), downSampleFac, downSampleFac);
    myImProc::previtt(downSampledImage, prewitt);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(prewitt, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // ## Debug for contours at small resolution
    cv::Mat conts = downSampledImage.clone();

    // ## Debug for contours resized to full resolution
//    cv::Mat resizedConts = image_gr.clone();
//    cv::cvtColor(resizedConts,resizedConts,CV_GRAY2BGR);

    for(size_t i = 0; i < contours.size(); ++i)
    {
        cv::Rect r = cv::boundingRect(contours[i]);

        // ## parameter discard small contours
        bool bigEnough = r.width >= 20 && r.height >= 20;
        bool biggerCont = contours[i].size() >= 70;
        if(bigEnough && biggerCont )
        {
            if(DEBUG)
            {
                std::vector<std::vector<cv::Point>> debugContours = std::vector<std::vector<cv::Point>>();
                std::vector<cv::Point> debugContour = std::vector<cv::Point>();

                for(size_t contourIdx = 0; contourIdx < contours[i].size(); ++ contourIdx)
                {
                    debugContour.push_back(contours[i][contourIdx] * cvRound(1/downSampleFac));
                }
                debugContours.push_back(debugContour);
                cv::drawContours(p_debugImage,debugContours,0,cv::Scalar(0,0,195),3);
                cv::Rect debugBounds = cv::boundingRect(debugContours[0]);
                cv::rectangle(p_debugImage,debugBounds,cv::Scalar(255,0,0));
            }

            // test if contour is a BullsEye
            if(evaluateFirstCondition(downSampledImage,contours[i]) && evaluateSecondCondition(r))
            {
                for(size_t j = 0; j < contours[i].size(); ++j)
                {
                    // ## Debug draw coarse contours on coarse image
                    cv::circle(conts,contours[i][j],1,cv::Scalar(0,0,255),-1);

                    contours[i][j].x *= cvRound(1/downSampleFac);
                    contours[i][j].y *= cvRound(1/downSampleFac);

                    // ## Debug draw coarse contours on highres image
//                    cv::circle(resizedConts, contours[i][j],10,cv::Scalar(0,255,0),-1);
                }
                coarseBullsEyes.push_back(contours[i]);

                if(DEBUG)
                {
                    cv::Rect debugBounds = cv::boundingRect(contours[i]);
                    cv::rectangle(p_debugImage,debugBounds,cv::Scalar(255,255,0));
                }
            }
        }
    }
    if(DEBUG)
    {
        myImProc::show("coarseDEtection",p_debugImage,0.3);
    }
}

cv::Point TargetFinder::findRingPoint(cv::Point p_p1, cv::Point p_p2)
{
    // ## parameter threshold to fit point to ring
    const double alpha = 0.7;
    uchar vMin = 255;
    uchar vMax = 0;
    float vAvg = 0;
    int maxDiff = 0;
    cv::Point pMin;
    cv::Point pMax;
    cv::Point pDiff;
    uchar vPrev;
    uchar vNext;

    cv::LineIterator lineIter = cv::LineIterator(m_image_gray,p_p1, p_p2);
    int i = 0;
    for( i = 0; i < lineIter.count; i++)
    {
        uchar val = **lineIter;
        vAvg += val;
        cv::Point pos = lineIter.pos();

        if(val > vMax)
        {
            vMax = val;
            pMax = pos;
        }
        if(val < vMin)
        {
            vMin = val;
            pMin = pos;
        }

        ++lineIter;
        cv::Point checkPos = lineIter.pos();
        bool posXInImage = checkPos.x >= 0 && checkPos.x < m_image_gray.cols;
        bool posYInImage = checkPos.y >= 0 && checkPos.y < m_image_gray.rows;
        if(i < lineIter.count && posXInImage && posYInImage)
        {
            vNext = **lineIter;
        }

        if(i == 0)
        {
            vPrev = val;
        }

        int diff1 = std::abs((int) val - (int)vPrev);
        int diff2 = std::abs((int) val - (int)vNext);
        int diff3 = diff1 + diff2;

        if(diff3 > maxDiff)
        {
            maxDiff = diff3;
            pDiff = pos;
        }

        vPrev = val;
    }

    vAvg /= i;

    float kappa = (vAvg - vMin) / (vMax - vMin);
    if(kappa > 1 - alpha)
    {
//        if(kappa > alpha)
//        {
//            return pDiff;
//        }
        return pMin;
    }
    else //if(kappa < alpha)
    {
        return pMax;
    }
}

void TargetFinder::findOnAllRings(cv::Point p_point, cv::Point p_targetCenter,
                                  std::vector<std::vector<cv::Point>>& p_targetRings,
                                  cv::Mat& p_debugImage)
{
    bool DEBUG = (DRAW_RINGLINES || DRAW_RINGPOINTS) && p_debugImage.data;
    cv::Point distToCenterVec = p_point - p_targetCenter;
    double distToCenter = cv::norm(distToCenterVec);
    double distOfRings = distToCenter/6;
    for(int ring = 1; ring < 10; ++ring)
    {
        // point of bullsEye is already detected
        if(ring == 6)
        {
            p_targetRings[ring-1].push_back(p_point);
            continue;
        }
        double scaleInner = (ring * distOfRings - distOfRings/3)/ distToCenter;
        double scaleOuter = (ring * distOfRings + distOfRings/3)/ distToCenter;
        cv::Point ringInner = scaleInner * distToCenterVec + p_targetCenter;
        cv::Point ringOuter = scaleOuter * distToCenterVec + p_targetCenter;
        if(!cv::clipLine(m_image.size(),ringInner, ringOuter))
        {
            continue;
        }
        cv::Point finalRingPoint = findRingPoint(ringInner,ringOuter);
        if(DEBUG)
        {
            if(DRAW_RINGLINES)
            {
                cv::Scalar lineColor = cv::Scalar(150,150,150);
                cv::line(p_debugImage,ringInner, ringOuter,lineColor);
            }
            if(DRAW_RINGPOINTS)
            {
                cv::Scalar ringPointColor = cv::Scalar(0,255,0);
                cv::line(p_debugImage,finalRingPoint, finalRingPoint,ringPointColor);
            }
        }
        if(myImProc::isPointInImage(m_image,finalRingPoint))
        {
            p_targetRings[ring-1].push_back(finalRingPoint);
        }
    }
}

void TargetFinder::extractAllTargetRings(const std::vector<std::vector<cv::Point>>& coarseBullsEyes, cv::Mat& p_debugMat)
{
    bool DEBUG = true && p_debugMat.data;
    cv::Mat bullsEyeMask = cv::Mat::zeros(m_image.size(),CV_8UC1);
    for(size_t coarseIter = 0; coarseIter < coarseBullsEyes.size(); ++coarseIter)
    {
        createRefinedBullsEyeMask(bullsEyeMask,coarseBullsEyes[coarseIter]);
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bullsEyeMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // loop through detected bullsEyes
    for(size_t i = 0; i < contours.size(); ++i)
    {
        // bounding Rect of possible BullsEye
        cv::Rect bounds = cv::boundingRect(contours[i]);

        // ## parameter reject too small contours (generated by refinement)
        if(bounds.width > 150 || bounds.height > 150)
        {
            cv::Point center = bounds.tl() + cv::Point(bounds.width/2, bounds.height/2);

            std::vector<std::vector<cv::Point>> targetRings = std::vector<std::vector<cv::Point>>();
            //create 9 rings
            for(int ringIdx = 0; ringIdx < 9; ++ringIdx)
            {
                targetRings.push_back(std::vector<cv::Point>());
            }

            clock_t startTime = clock();

            // loop through points of bullsEye
            for(size_t j = 0; j < contours[i].size(); ++j)
            {
                // ## Debug draw bullsEye
                if(DEBUG)
                {
                    cv::circle(p_debugMat, contours[i][j],1,cv::Scalar(0,0,255),-1);
                }

                // take ony evry 8th point of the bullsEye Ring
                if(j%8 == 0)
                {
                    // calc ring estimates
                    findOnAllRings(contours[i][j], center, targetRings, p_debugMat);
                }
            }
            // store target with all rings
            m_targets.push_back(TargetInstance(targetRings));

            // end of profiling rings per Target
            double elapsed = (double)(clock() - startTime);
            double elapsed_sec =  elapsed / (double)CLOCKS_PER_SEC;
            std::cout << "setting rings for this taget took " << elapsed_sec << " seconds." << std::endl;
        }
    }
}

void TargetFinder::createRefinedBullsEyeMask(cv::Mat& p_out, std::vector<cv::Point> p_contour)
{
    if(p_out.dims < 2)
    {
        p_out = cv::Mat::zeros(m_image_gray.size(), CV_8UC1);
    }
    std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
    contours.push_back(p_contour);
    cv::Rect boundingRec = cv::boundingRect(contours[0]);
    cv::Point contourCenter = cv::Point(boundingRec.tl().x + boundingRec.width/2, boundingRec.tl().y + boundingRec.height/2 );
    for(size_t i = 0; i < contours[0].size(); ++i)
    {
        contours[0][i] -= contourCenter;
        contours[0][i] *= 1.1;
        contours[0][i] += contourCenter;

        contours[0][i].x = std::max(0,contours[0][i].x);
        contours[0][i].y = std::max(0,contours[0][i].y);
        contours[0][i].x = std::min(m_image_gray.cols,contours[0][i].x);
        contours[0][i].y = std::min(m_image_gray.rows,contours[0][i].y);

    }
    boundingRec = cv::boundingRect(contours[0]);
    contourCenter = cv::Point(boundingRec.tl().x + boundingRec.width/2, boundingRec.tl().y + boundingRec.height/2 );
    for(size_t i = 0; i < contours[0].size(); ++i)
    {
        contours[0][i] -= boundingRec.tl();
    }

    cv::Mat roiImage = m_image_gray(boundingRec);
    // create a mask based on coarse contour
    cv::Mat mask = cv::Mat::zeros(roiImage.size(), CV_8UC1);
    cv::drawContours(mask,contours,0,cv::Scalar(255),-1);

    // mask out roughly detected bullsEye
    cv::Mat thresholdImage;

    roiImage.copyTo(thresholdImage,mask);

    // discard bright pixels as they are not part of the bullsEye
    // ## parameter URGENT thresh for bullsEye
    cv::threshold(thresholdImage,thresholdImage, 100,255,CV_THRESH_BINARY_INV);

    cv::Mat tmp;
    thresholdImage.copyTo(tmp,mask);


    // morphologic operations to get rid of clutter and to define a good circular shape
    cv::Mat element2 = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*11 + 1, 2*11 + 1 ),
                            cv::Point( 11, 11 ) );
    cv::Mat element3 = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*2 + 1, 2*2 + 1 ),
                            cv::Point( 2, 2 ) );
    cv::erode(tmp,tmp,element3);
    cv::dilate(tmp,tmp,element3);
    cv::dilate(tmp,tmp,element2);
    cv::erode(tmp,tmp,element2);

    cv::Mat submat = p_out.colRange(boundingRec.x, boundingRec.x + boundingRec.width)
                     .rowRange(boundingRec.y, boundingRec.y + boundingRec.height);
    tmp.copyTo(submat,mask);

}

void TargetFinder::detectBulletHolesOnTarget(TargetInstance& p_target, cv::Mat segMentationImage)
{
    // create a roi on m_image where the target is in
    cv::Rect roi = p_target.getBoundingRect();
    cv::Mat roiImage = m_image.clone()(roi);

    // grayScaleImage for Edgedetection
//    cv::Mat roi_gray;
//    cv::cvtColor(roiImage, roi_gray, CV_BGR2GRAY);

    cv::Mat thresholdImage;
    cv::cvtColor(roiImage, roiImage, CV_BGR2HSV);

    // ## parameter HSV threshold for bulletholes
    cv::Scalar lowerBounds = cv::Scalar(0,0,130);
    cv::Scalar upperBounds = cv::Scalar(255,50,255);
    cv::inRange(roiImage,lowerBounds,upperBounds,thresholdImage);

    cv::Mat holeMask = cv::Mat::zeros(roiImage.size(), CV_8UC1);

    std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
    cv::findContours(thresholdImage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    std::vector<std::vector<cv::Point>> bulletHoles = std::vector<std::vector<cv::Point>>();

    // ## parameter bullethole shape
    double maxSideLength = std::max(roi.width, roi.height);
    double minDist = maxSideLength/22;
    double minRad = maxSideLength/37;
    double maxRad = 2*minRad;

    // filter the detected contour for possible bulletholes
    for(size_t i = 0; i < contours.size(); ++i)
    {
        cv::Rect bounds = cv::boundingRect(contours[i]);

		if(SHOW_HOLESEGMENTATION)
		{
			cv::Mat subMat = segMentationImage.colRange(roi.tl().x, roi.tl().x + roi.width).
                         rowRange(roi.tl().y, roi.tl().y + roi.height);
			cv::Scalar rectColor = cv::Scalar(255,0,0);
			cv::rectangle(subMat, bounds,rectColor);
		}

        //overlapping holes can form a bigger region than maxRad
        bool tooBig = bounds.width > roi.width/2 || bounds.height > roi.height/2;
        bool tooSmall = bounds.width < minRad && bounds.height < minRad;
        if(!tooSmall && !tooBig)
        {
            double area = cv::contourArea(contours[i]);
            if(area >= 0.5 * bounds.area())
            {
                bulletHoles.push_back(contours[i]);
				if(SHOW_HOLESEGMENTATION)
				{
					cv::Mat subMat = segMentationImage.colRange(roi.tl().x, roi.tl().x + roi.width).
                         rowRange(roi.tl().y, roi.tl().y + roi.height);
					cv::Scalar selectedRectColor = cv::Scalar(255,255,0);
					cv::rectangle(subMat, bounds,selectedRectColor);
				}
            }
       }
    }


    std::vector<cv::Vec3f> circles = std::vector<cv::Vec3f>();
    for(size_t i = 0; i < bulletHoles.size(); ++i)
    {
        cv::Mat holeContours = cv::Mat::zeros(holeMask.size(),CV_8UC1);
        cv::drawContours(holeMask, bulletHoles,i,cv::Scalar(255),-1);
        cv::drawContours(holeContours, bulletHoles,i,cv::Scalar(255), 1);

        cv::bitwise_not(holeContours,holeContours);
        cv::Mat holeDistTrans;
        cv::distanceTransform(holeContours, holeDistTrans, CV_DIST_L2, CV_DIST_MASK_PRECISE);

        // TODO: use Lists instead of vectors
        std::vector<cv::Vec3f> holeCircles = myImProc::circleRANSAC(bulletHoles[i],holeDistTrans,8,30,minRad,maxRad,minDist);
        for(auto circIter = holeCircles.begin(); circIter != holeCircles.end(); ++circIter)
        {
            circles.push_back(*circIter);
        }
    }

    if(SHOW_HOLESEGMENTATION)
    {
        cv::Mat subMat = segMentationImage.colRange(roi.tl().x, roi.tl().x + roi.width).
                         rowRange(roi.tl().y, roi.tl().y + roi.height);
        cv::Mat colorMat = cv::Mat(holeMask.size(), CV_8UC3, cv::Scalar(0,0,255));
        cv::Mat colorHoleMask = cv::Mat::zeros(holeMask.size(), CV_8UC3);
        colorMat.copyTo(subMat,thresholdImage);
        //cv::cvtColor(holeMask, colorHoleMask, CV_GRAY2BGR);
        //cv::bitwise_or(colorHoleMask, subMat,subMat);
    }


    //cv::HoughCircles(holeMask,circles,CV_HOUGH_GRADIENT,1,minDist,100,8,minRad,maxRad);
    //30,100,8,15,60

    // ## Debug
    cv::Mat debugHoleImage(holeMask.size(), CV_8UC1);
    cv::Canny(holeMask,debugHoleImage,50,100);


    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Point samplePoint = cv::Point(cvRound(circles[i][0]), cvRound(circles[i][1]));

        // discard circles with center outside of the holemask
        if(!myImProc::isPointInImage(holeMask,samplePoint))
        {
            continue;
        }

        unsigned char sampleVal = holeMask.at<unsigned char>(samplePoint.y, samplePoint.x);
        if( sampleVal == 255 )
        {
            p_target.addBulletHole(circles[i]);

            // ## Debug
            cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            cv::circle( debugHoleImage, center, 3, cv::Scalar(180,255,0), -1, 8, 0 );
            // circle outline
            cv::circle( debugHoleImage, center, radius, cv::Scalar(180,0,255), 1, 8, 0 );
        }

    }


//	cv::Mat previttImage;
//	previtt(roi_gray, previttImage);
//    processImage_Sobel(roi_gray,previttImage);

}

void TargetFinder::detectAllBulletHoles()
{
    if(SHOW_HOLESEGMENTATION)
    {
        cv::Mat segmentationImgage = m_image.clone();
        for(size_t targetIdx = 0; targetIdx < m_targets.size(); ++targetIdx)
        {
            detectBulletHolesOnTarget(m_targets[targetIdx],segmentationImgage);
        }
        myImProc::show("holeSegmentation", segmentationImgage, 0.3);
    }
    else
    {
        for(size_t targetIdx = 0; targetIdx < m_targets.size(); ++targetIdx)
        {
            detectBulletHolesOnTarget(m_targets[targetIdx]);
        }
    }
}

void TargetFinder::drawTargets(cv::Mat& p_drawOn)
{
    for(size_t i = 0; i < m_targets.size(); ++i)
    {

        // Draw center
        cv::circle(p_drawOn, m_targets[i].getCenter(), 5, cv::Scalar(255,0,0),-1);
        std::vector<std::vector<cv::Point>> rings = m_targets[i].getRings();
       /* for(int j = 0; j < rings.size(); ++j)
        {
            for( int k = 0; k < rings[j].size(); ++k )
            {
                cv::Scalar color = cv::Scalar(255,255,0);
                if(j == 3)
                {
                    color = cv::Scalar(255,255,255);
                }
                cv::circle(p_drawOn, rings[j][k],1,color,-1);
            }
        }*/

        // Draw rings
        std::vector<cv::RotatedRect> ringEllipses = m_targets[i].getRingEllipses();
        for(size_t j = 0; j < ringEllipses.size(); ++j)
        {
            cv::Scalar color = cv::Scalar(180,0,0);
            /*switch(j)
            {
            case 0:
                color = cv::Scalar(255,0,255);
                break;
            case 2:
                color = cv::Scalar(0,255,255);
                break;
            case 5:
                color = cv::Scalar(0,0,255);
                break;
            case 7:
                color = cv::Scalar(0,255,0);
                break;
            }*/

            cv::ellipse(p_drawOn, ringEllipses[j],color,1);
        }

        // Draw bulletHoles
        if(DRAW_BULLERHOLES)
        {
            std::vector<cv::Vec4f> holes = m_targets[i].getBulletHoles();
            for(size_t j = 0; j < holes.size(); ++j)
            {
                cv::Vec4f hole = holes[j];
                // vector from center to holeCenter
                cv::Point2f dist = cv::Point2f(hole[0], hole[1]) - cv::Point2f(m_targets[i].getCenter());
                float distLength = cv::norm(dist);
                dist.x /= distLength;
                dist.y /= distLength;
                float debugDistL = cv::norm(dist);
                float sclaeToCenter = std::min(distLength, hole[2]);
                cv::Point holeCenter = cv::Point(cvRound(holes[j][0]), cvRound(holes[j][1]));
                cv::Point closestToCenter = holeCenter - cv::Point(dist * sclaeToCenter);

                cv::Scalar color(255,0,255);
                cv::circle(p_drawOn,holeCenter,cvRound(holes[j][2]),color,-1);
                cv::circle(p_drawOn,closestToCenter,1,cv::Scalar(255,255,0),-1);
                float sc = holes[j][3];
                std::string score = cv::format("%.1f",sc);
                cv::putText(p_drawOn,score,holeCenter /*- cv::Point(holes[j][2],holes[j][2])*/,CV_FONT_HERSHEY_DUPLEX,0.9,cv::Scalar(0,0,0),2);
            }
        }
    }
}





