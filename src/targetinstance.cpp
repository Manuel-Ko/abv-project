#include "targetinstance.h"

// ####     constructors    ####
TargetInstance::TargetInstance() :
    m_center(cv::Point(0,0)),
    m_bounds(cv::Rect()),
    m_rings(std::vector<std::vector<cv::Point>>()),
    m_ringEllipses(std::vector<cv::RotatedRect>())
{
}

TargetInstance::TargetInstance(std::vector<std::vector<cv::Point>> p_rings) :
    m_rings(p_rings)
{
    setRings(p_rings);
}

typedef struct{
  double x,y,majorAxis,minorAxis,angle;
  double f1_x,f1_y,f2_x,f2_y;
} Ellipse;

double acot_d(double val);
void getEllipseParam(double a,double b,double c,double d,double f,double g,Ellipse& ellipse);
bool pointInEllipse(cv::Point point, Ellipse ellipse);
Ellipse fitEllipseRANSAC(std::vector<cv::Point> points,int &count);


// ####     public methods      ####


void TargetInstance::setRings(std::vector<std::vector<cv::Point>> p_rings)
{
    if(p_rings.empty())
    {
        std::cerr << "TargetInstance (set Rings): no rings supplied" << std::endl;
        return;
    }
    else if(p_rings[0].empty())
    {
        std::cerr << "TargetInstance (set Rings): firsr ring must not be empty" << std::endl;
        return;
    }
    // rigs have changed so bullet holes aren't valid anymore
    m_bulletHoles.clear();

    m_rings = p_rings;
    m_bounds = cv::boundingRect(m_rings.back());
    cv::Rect smallestRect = cv::boundingRect(m_rings[0]);
    m_center = smallestRect.tl() + cv::Point(smallestRect.width/2, smallestRect.height/2);

    m_ringEllipses.clear();

    for(int i = 0; i < m_rings.size(); ++i)
    {
        //fit Ellipse with OpenCV (least squares)
        m_ringEllipses.push_back(cv::fitEllipse(m_rings[i]));

        //TODO: use RANSAC to fit Ellipse to weaken influence of multiple outliers

        //fit Ellipse with RANSAC
//        int maxVotes = 0;
//        Ellipse finalEllipse;
//        for(int iteration = 0; iteration < MAX_RANSAC_ITERATIONS; ++iteration)
//        {
//            Ellipse currentEllipse;
//            int votes = 0;
//            currentEllipse = fitEllipseRANSAC(m_rings[i],votes);
//            if(votes > maxVotes)
//            {
//                maxVotes = votes;
//                finalEllipse = currentEllipse;
//            }
//        }
//        cv::Size2f rectSize = cv::Size2f(finalEllipse.majorAxis, finalEllipse.minorAxis);
//        cv::RotatedRect rotRec = cv::RotatedRect(cv::Point2f(finalEllipse.x,finalEllipse.y),rectSize, finalEllipse.angle);
//        m_ringEllipses.push_back(rotRec);
    }
}

void TargetInstance::addBulletHole(cv::Vec3f p_hole)
{
	p_hole += cv::Vec3f(m_bounds.tl().x, m_bounds.tl().y);  
	cv::Point2f dist = cv::Point2f(p_hole[0], p_hole[1]) - cv::Point2f(m_center);
	float distLength = cv::norm(dist);
	dist.x /= distLength;
	dist.y /= distLength;
	float debugDistL = cv::norm(dist);
	float sclaeToCenter = std::min(distLength, p_hole[2]);
	cv::Point closestToCenter = cv::Point(p_hole[0], p_hole[1]) - cv::Point(dist * sclaeToCenter);

    float score = 0;
    for(int i = 0; i < m_ringEllipses.size(); ++i)
    {
        cv::RotatedRect rotRec = m_ringEllipses[i];
        float majAxLen = 0;
        float minAxLen = 0;
        if(rotRec.size.width >= rotRec.size.height)
        {
            majAxLen = rotRec.size.width;
            minAxLen = rotRec.size.height;
        }
        else
        {
            majAxLen = rotRec.size.height;
            minAxLen = rotRec.size.width;
        }

        // check for 10
        if(i == 0)
        {
            float distToCenter = cv::norm(closestToCenter - m_center);
            float meanSz = (minAxLen + majAxLen)/2;

            // ## parameter distance to center to be recognized as 10
            if(distToCenter < 0.2 * meanSz)
            {
                score = 10;
                break;
            }
        }

		float fLength = sqrt(majAxLen/2 * majAxLen/2 - minAxLen/2 * minAxLen/2);
		float alpha = rotRec.angle;
		alpha *= M_PI/180 * -1;
		cv::Matx22f rotate (cos(alpha),sin(alpha),
							-sin(alpha), cos(alpha));
		cv::Matx21f vec(1,0);
		vec = rotate * vec;
		vec *= fLength;
		cv::Point vecP = cv::Point(vec(0,0),vec(1,0));
		cv::Point f1 = cv::Point(rotRec.center) + vecP;
		cv::Point f2 = cv::Point(rotRec.center) - vecP;

        float dist1 = cv::norm(f1 - closestToCenter);
        float dist2 = cv::norm(f2 - closestToCenter);

        // ## Debug
		std::vector<cv::Point> ellipse;
		cv::Rect debugRoi = rotRec.boundingRect() + cv::Size(200,200);
		cv::Point offset = cv::Point(100,100);
		cv::ellipse2Poly(rotRec.center,cv::Size(rotRec.size.width/2, rotRec.size.height/2), rotRec.angle,0,360,5,ellipse);
		for(int j = 0; j < ellipse.size(); ++j)
		{
			ellipse[j] = ellipse[j] - debugRoi.tl() + offset;
		}
		cv::Mat debug = cv::Mat(debugRoi.size(), CV_8UC3);
		std::vector<std::vector<cv::Point>> ellipse2 = std::vector<std::vector<cv::Point>>();
		ellipse2.push_back(ellipse);
		cv::polylines(debug,ellipse2,true,cv::Scalar(255,0,255),1);
		cv::circle(debug,f1 - debugRoi.tl() + offset,2,cv::Scalar(0,0,255),-1);
		cv::circle(debug,f2 - debugRoi.tl() + offset,2,cv::Scalar(0,255,255),-1);
		cv::Point holeCenter = cv::Point(p_hole[0],p_hole[1]) - debugRoi.tl() + offset;
		cv::circle(debug, holeCenter ,p_hole[2],cv::Scalar(0,120,255));
		cv::Point debugPoint = closestToCenter - debugRoi.tl() + offset;
        cv::circle(debug,debugPoint,2,cv::Scalar(255,0,255),-1);

        if(dist1 + dist2 <= majAxLen)
        {
            score = 9 - i;
            break;
        }

    }

    m_bulletHoles.push_back(cv::Vec4f(p_hole[0], p_hole[1], p_hole[2], score));
}

cv::Point TargetInstance::getCenter()
{
    return m_center;
}

cv::Rect TargetInstance::getBoundingRect()
{
    return m_bounds;
}

float TargetInstance::getScoreMean()
{
    float mean = getScoreSum();
    mean /= m_bulletHoles.size();
    return mean;
}

float TargetInstance::getScoreSum()
{
    float sum = 0;
    for(auto holeIter = m_bulletHoles.begin(); holeIter != m_bulletHoles.end(); ++holeIter)
    {
        sum += (*holeIter)[3];
    }
    return sum;
}
float TargetInstance::getScoreMax()
{
    float max = 0;
    for(auto holeIter = m_bulletHoles.begin(); holeIter != m_bulletHoles.end(); ++holeIter)
    {
        max = std::max(max,(*holeIter)[3]);
    }
    return max;
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

std::vector<cv::RotatedRect> TargetInstance::getRingEllipses()
{
    return m_ringEllipses;
}

std::vector<cv::Vec4f> TargetInstance::getBulletHoles()
{
	return m_bulletHoles;
}

bool TargetInstance::isComplete()
{
    return m_rings.size() == 9;
}

// ####     private methods     ####


// RANSAC ELLIPSE Fitting from http://vgl-ait.org/cvwiki/doku.php?id=opencv:tutorial:fitting_ellipse_using_ransac_algorithm
// author   Worawit Panpanyatep
//          st111453
//          Certified of Advanced Study (CAS)
//          Asian Institute of Technology



double acot_d(double val){
  double acot = atan(1/val);
  return acot*180/M_PI;
}

void getEllipseParam(double a,double b,double c,double d,double f,double g,Ellipse& ellipse){
  ellipse.x = (c * d - b * f)/(b * b - a * c);
  ellipse.y = (a * f - b * d)/(b * b - a * c);

  ellipse.majorAxis = sqrt( (2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g))/((b*b-a*c)*(sqrt((a-c)*(a-c)+4*b*b)-(a+c))));
  ellipse.minorAxis = sqrt( (2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g))/((b*b-a*c)*(sqrt((a-c)*(a-c)+4*b*b)+(a+c))));

  ellipse.angle=0;
  if(b == 0 && a < c){
    ellipse.angle = 0;
  }
  else if(b == 0 && a > c){
    ellipse.angle = 90;
  }
  else if(b != 0 && a < c){
    ellipse.angle = 0.5 * acot_d( (a-c)/(2*b) );
  }
  else if(b != 0 && a > c){
    ellipse.angle = 90 + 0.5 * acot_d( (a-c)/(2*b) );
  }
  if(ellipse.minorAxis > ellipse.majorAxis){
    double temp = ellipse.majorAxis;
    ellipse.majorAxis = ellipse.minorAxis;
    ellipse.minorAxis = temp;
    ellipse.angle += 90;
  }

  double temp_c;
  if(ellipse.majorAxis > ellipse.minorAxis)
    temp_c = sqrt(ellipse.majorAxis * ellipse.majorAxis - ellipse.minorAxis * ellipse.minorAxis);
  else
    temp_c = sqrt(ellipse.minorAxis * ellipse.minorAxis - ellipse.majorAxis * ellipse.majorAxis);
  ellipse.f1_x = ellipse.x - temp_c * cos(ellipse.angle*M_PI/180);
  ellipse.f1_y = ellipse.y - temp_c * sin(ellipse.angle*M_PI/180);
  ellipse.f2_x = ellipse.x + temp_c * cos(ellipse.angle*M_PI/180);
  ellipse.f2_y = ellipse.y + temp_c * sin(ellipse.angle*M_PI/180);
}

bool pointInEllipse(cv::Point point,Ellipse ellipse){
  double dist1 = sqrt((point.x - ellipse.f1_x) * (point.x - ellipse.f1_x) +
              (point.y - ellipse.f1_y) * (point.y - ellipse.f1_y));
  double dist2 = sqrt((point.x - ellipse.f2_x) * (point.x - ellipse.f2_x) +
              (point.y - ellipse.f2_y) * (point.y - ellipse.f2_y));
  double max;
  if(ellipse.majorAxis > ellipse.minorAxis)
    max = ellipse.majorAxis;
  else
    max = ellipse.minorAxis;
  if(dist1+dist2 <= 2*max)
    return true;
  else
    return false;
}

Ellipse fitEllipseRANSAC(std::vector<cv::Point> points,int &count){
  Ellipse ellipse;
  count = 0;
  int index[5];
  bool match = false;
  for(int i = 0; i < 5;i++){
    do {
      match = false;
      index[i] = rand()%points.size();
      for(int j=0; j<i ;j++){
        if(index[i] == index[j]){
          match=true;
        }
      }
    }
    while(match);
  }
  double aData[] = {
    points[index[0]].x * points[index[0]].x, 2 * points[index[0]].x * points[index[0]].y, points[index[0]].
    y * points[index[0]].y, 2 * points[index[0]].x, 2 * points[index[0]].y,

    points[index[1]].x * points[index[1]].x, 2 * points[index[1]].x * points[index[1]].y, points[index[1]].
    y * points[index[1]].y, 2 * points[index[1]].x, 2 * points[index[1]].y,

    points[index[2]].x * points[index[2]].x, 2 * points[index[2]].x * points[index[2]].y, points[index[2]].
    y * points[index[2]].y, 2 * points[index[2]].x, 2 * points[index[2]].y,

    points[index[3]].x * points[index[3]].x, 2 * points[index[3]].x * points[index[3]].y, points[index[3]].
    y * points[index[3]].y, 2 * points[index[3]].x, 2 * points[index[3]].y,

    points[index[4]].x * points[index[4]].x, 2 * points[index[4]].x * points[index[4]].y, points[index[4]].
    y * points[index[4]].y, 2 * points[index[4]].x, 2 * points[index[4]].y };
  cv::Mat matA = cv::Mat(5,5,CV_64F,aData);
  cv::Mat D = cv::Mat(5,5,CV_64F);
  cv::Mat U = cv::Mat(5,5,CV_64F);
  cv::Mat V = cv::Mat(5,5,CV_64F);

  //cvSVD(&matA,D,U,V,CV_SVD_MODIFY_A);
  cv::SVD::compute(matA,D,U,V,CV_SVD_MODIFY_A);

  double a,b,c,d,f,g;
  a = V.at<double>(0,4);
  b = V.at<double>(1,4);
  c = V.at<double>(2,4);
  d = V.at<double>(3,4);
  f = V.at<double>(4,4);
  g = 1;

  getEllipseParam(a,b,c,d,f,g,ellipse);

  std::vector<cv::Point>::iterator point_iter;
  if(ellipse.majorAxis > 0 && ellipse.minorAxis > 0){
    for(point_iter=points.begin();point_iter!=points.end();point_iter++){
      cv::Point point = *point_iter;
      if(pointInEllipse(point,ellipse)){
    count++;
      }
    }
  }

  return ellipse;
}

// End of RANSAC ELLIPE
