#include "Tracking.h"

Tracking::Tracking() {
    // std::cout << cv::getBuildInformation() << std::endl;
}

#define UNKNOWN_FLOW_THRESH 1e9

const int N = 1500;
int flag[N][N];

cv::Mat prevgray, gray, flow, cflow, frame, pre_frame, img_scale, img_temp, mask = cv::Mat(cv::Size(1, 300), CV_8UC1); // Image containers
cv::Size dsize;
std::vector < cv::Point2f > prepoint, nextpoint; // corner vectors
std::vector < cv::Point2f > F_prepoint, F_nextpoint;
std::vector < uchar > state;
std::vector < float > err;
double dis[N];
int cal = 0;
int width = 1200, height = 900;
int rec_width = 60;
int Harris_num = 0;

std::string file_name = "D:/computer_vision/d.mp4";

double vehicle_speed = 1;
double limit_of_check = 2400; //2120, 6000 is too high but for slow objects it is kind of neccessary
double limit_of_check_min = 400;
double scale = 1;
int margin = 2;
double limit_dis_epi = 2;

std::string itos(int i) {
    std::stringstream s;
    s << i;
    return s.str();
}

bool ROI_mod(int x1, int y1) {
    if (x1 >= width / 16 && x1 <= width - width / 16 && y1 >= height / 3 && y1 <= height - height / 6) return 1;
    return 0;
}

/*
 *
 * Doing necessary variable updates and resettings
 * the part where the frame is read and equalizeHist is done
 *
 */

void ready() {
    //qDebug() << "ready()";
    Harris_num = 0;

    // resetting these vectors is a must
    F_prepoint.clear();
    F_nextpoint.clear();

    // frame holds the frame that is currenly read and processed
    height = frame.rows * scale;
    width = frame.cols * scale;
    dsize = cv::Size(frame.cols * scale, frame.rows * scale);

    // creating 2 empty photos with the same size as the frame
    img_scale = cv::Mat(dsize, CV_32SC3);
    img_temp = cv::Mat(dsize, CV_32SC3);

    resize(frame, img_scale, dsize);
    resize(frame, img_temp, dsize);

    cvtColor(img_scale, gray, cv::COLOR_BGR2GRAY);

    rec_width = frame.cols / 60;

    equalizeHist(gray, gray);

    // not good practice for FPS, ~55 -> ~38
    //equalizeHist(gray, gray);
    //equalizeHist(gray, gray);

    //ret, gray = threshold(gray, 150, 255, THRESH_BINARY);

    // threshold(gray, gray, 150, 255, THRESH_BINARY);

    //visualize the binary image

    return;
}

void optical_flow_check() {
    // corner means the shift from borders of the original frame
    int limit_edge_corner = 15;
    for (int i = 0; i < (int)state.size(); i++)
        // meaning we succesfully tracked a previous feature
        if (state[i] == 1) {
            int dx[10] = {
                -1,
                0,
                1,
                -1,
                0,
                1,
                -1,
                0,
                1
            };
            int dy[10] = {
                -1,
                -1,
                -1,
                0,
                0,
                0,
                1,
                1,
                1
            };
            int x1 = prepoint[i].x, y1 = prepoint[i].y;
            int x2 = nextpoint[i].x, y2 = nextpoint[i].y;

            // eliminating the frame borders and corners
            if ((x1 < limit_edge_corner || x1 >= gray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= gray.cols - limit_edge_corner ||
                y1 < limit_edge_corner || y1 >= gray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= gray.rows - limit_edge_corner)) {
                state[i] = 0;
                continue;
            }

            // a type of check for the pixel value between feature points kernel
            double sum_check = 0;
            for (int j = 0; j < 9; j++)
                sum_check += abs(prevgray.at < uchar >(y1 + dy[j], x1 + dx[j]) - gray.at < uchar >(y2 + dy[j], x2 + dx[j]));

            if (sum_check > limit_of_check || sum_check < limit_of_check_min) {
                state[i] = 0;
            }

            if (state[i] == 1) {
                Harris_num++;
                F_prepoint.push_back(prepoint[i]);
                F_nextpoint.push_back(nextpoint[i]);
            }
        }
    return;
}

bool stable_judge() {
    int stable_num = 0;
    double limit_stable = 0.5;
    for (int i = 0; i < (int)state.size(); i++) {
        if (state[i]) {
            if (sqrt((prepoint[i].x - nextpoint[i].x) * (prepoint[i].x - nextpoint[i].x) + (prepoint[i].y - nextpoint[i].y) * (prepoint[i].y - nextpoint[i].y)) < limit_stable)
                stable_num++;
        }
    }
    if (stable_num * 1.0 / Harris_num > 0.2) return 1;
    return 0;
}

double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

void Tracking::ADAS() {
    cv::VideoCapture cap;
    cap.open(file_name);

    if (!cap.isOpened()) {
        return;
    }

    for (;;) {
        double t = (double)cv::getTickCount();
        cv::TickMeter tm;

        cv::Mat raw_frame;

        cap >> raw_frame;

        // I crop the image, so it only looks for the center of the camera
        frame = raw_frame(cv::Range(0, raw_frame.rows ), cv::Range(0, raw_frame.cols));
        tm.start();

        if (frame.empty()) break;
        cal++;

        ready();

        if (cal % margin != 0) {
            // continue;
        }

        if (prevgray.data) {

            /*
             *  ------  goodFeaturesToTrack -> strong corner detection
             * 	InputArray 	image,
             *  OutputArray 	corners,
             *  int 	maxCorners,
             *  double 	qualityLevel,
             *  double 	minDistance,
             *  InputArray 	mask = noArray(),
             *  int 	blockSize = 3,
             *  bool 	useHarrisDetector = false. Harris corner detector gives many false results
             *  double 	k = 0.04
             *
             *  NOTS:
             *      qualityLevel 0.03 is too high for small movements, 0.0001 is relatively good but FPS drops is a concern
             *
             * */
            goodFeaturesToTrack(prevgray, prepoint, 150, 0.003, 4, cv::Mat(), 3, true, 0.04);

            /*
             *  ------  cornerSubPix -> Refines the corner locations.
             *          InputArray 	image,
             *          InputOutputArray 	corners,
             *          Size 	winSize,
             *          Size 	zeroZone,
             *          TermCriteria 	criteria
             *
             *          NOTS:
             *              Tried TermCritera Types COUNT + COUNT
             *                                      COUNT + EPS
             *                                      EPS + EPS
             *
             * */

            if (prepoint.size() <= 0)
                continue;

            cornerSubPix(prevgray, prepoint, cv::Size(5, 5), cv::Size(3, 3), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.01));

            calcOpticalFlowPyrLK(prevgray, gray, prepoint, nextpoint, state, err, cv::Size(15, 15), 5, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.01));

            //calcOpticalFlowFarneback(prevgray, gray, prepoint, nextpoint, state, err,), -> another method for calculating optic flow

            optical_flow_check();

            
            for (int i = 0; i < (int)state.size(); i++) {
                double x1 = prepoint[i].x, y1 = prepoint[i].y;
                double x2 = nextpoint[i].x, y2 = nextpoint[i].y;
                if (state[i] != 0) {
                    //circle(img_scale, nextpoint[i], 3, cv::Scalar(255, 0, 255));
                    // circle(pre_frame, prepoint[i], 2, cv::Scalar(255, 0, 255));
                }
            }

            //28/08 addition
            cv::Mat F = cv::Mat(3, 3, CV_32FC1);

           //  std::cout << "c \n";
            double ppp = 110;
            cv::Mat L = cv::Mat(1000, 3, CV_32FC1);
            
            int rc = 0;
            while (ppp > 5) {
                std::vector < cv::Point2f > F2_prepoint, F2_nextpoint;
                F2_prepoint.clear();
                F2_nextpoint.clear();
                ppp = 0;

                //qDebug() << "F_prepoint.size(): " <<F_prepoint.size();
                //qDebug() << "F_nextpoint.size(): " <<F_nextpoint.size();

                if (F_prepoint.size() < 2 || F_nextpoint.size() < 2)
                    continue;

                F = findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);
                //cout << F << endl;

                // gives some errors when the size of state changes, need to be inspected
                computeCorrespondEpilines(F_prepoint, 1, F, L);
                for (int i = 0; i < mask.rows; i++) {
                    if (mask.at < uchar >(i, 0) == 0);
                    else {
                        circle(pre_frame, F_prepoint[i], 6, cv::Scalar(255, 255, 0), 3);
                        double A = F.at < double >(0, 0) * F_prepoint[i].x + F.at < double >(0, 1) * F_prepoint[i].y + F.at < double >(0, 2);
                        double B = F.at < double >(1, 0) * F_prepoint[i].x + F.at < double >(1, 1) * F_prepoint[i].y + F.at < double >(1, 2);
                        double C = F.at < double >(2, 0) * F_prepoint[i].x + F.at < double >(2, 1) * F_prepoint[i].y + F.at < double >(2, 2);
                        double dd = fabs(A * F_nextpoint[i].x + B * F_nextpoint[i].y + C) / sqrt(A * A + B * B);

                        ppp += dd;
                        if (dd > 0.1) {
                            circle(pre_frame, F_prepoint[i], 6, cv::Scalar(255, 0, 0), 3);
                        }
                        else {
                            F2_prepoint.push_back(F_prepoint[i]);
                            F2_nextpoint.push_back(F_nextpoint[i]);
                        }
                    }
                }

                F_prepoint = F2_prepoint;
                F_nextpoint = F2_nextpoint;
                if (rc > 500) {
                    break;
                }
                rc++;
            }

            std::vector < cv::Point2f > T;

            std::vector<cv::Point2f> refinedStates;
            //std::cout << "dd \n";

            for (int i = 0; i < (int)prepoint.size(); i++) {
                if (state[i] != 0) {
                    double A = F.at < double >(0, 0) * prepoint[i].x + F.at < double >(0, 1) * prepoint[i].y + F.at < double >(0, 2);
                    double B = F.at < double >(1, 0) * prepoint[i].x + F.at < double >(1, 1) * prepoint[i].y + F.at < double >(1, 2);
                    double C = F.at < double >(2, 0) * prepoint[i].x + F.at < double >(2, 1) * prepoint[i].y + F.at < double >(2, 2);
                    double dd = fabs(A * nextpoint[i].x + B * nextpoint[i].y + C) / sqrt(A * A + B * B);

                    //int x1 = (int)prepoint[i].x, y1 = (int)prepoint[i].y;
                    //int x2 = (int)nextpoint[i].x, y2 = (int)nextpoint[i].y;
                    //if (sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)) < limit_flow) continue;
                    //line(img_scale, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar{ 255, 255, 0 }, 2);
                    //line(pre_frame, Point((int)prepoint[i].x, (int)prepoint[i].y), Point((int)nextpoint[i].x, (int)nextpoint[i].y), Scalar{ 0, 255, 0 }, 1);

                    if (dd <= limit_dis_epi) {
                        continue;
                    }
                    //cout << "dis: " << dd << endl;
                    dis[T.size()] = dd;
                    T.push_back(nextpoint[i]);

                    // cv::circle(img_scale, nextpoint[i], 7, cv::Scalar(255, 255, 255), 3);
                    // cv::circle(pre_frame, prepoint[i], 3, cv::Scalar(255, 255, 255), 2);
                    refinedStates.push_back(nextpoint[i]);

                    if (fabs(B) < 0.0001) {
                        double xx = C / A, yy = 0;
                        double xxx = C / A, yyy = gray.cols;
                        cv::line(pre_frame, cv::Point(xx, yy), cv::Point(xxx, yyy), cv::Scalar::all(-1), 1);
                        continue;
                    }

                    double xx = 0, yy = -C / B;
                    double xxx = gray.cols, yyy = -(C + A * gray.cols) / B;
                    if (fabs(yy) > 12345 || fabs(yyy) > 12345) {
                        yy = 0;
                        xx = -C / A;
                        yyy = gray.rows;
                        xxx = -(C + B * yyy) / A;
                    }
                    //cv::line(img_scale, cv::Point(xx, yy), cv::Point(xxx, yyy), cv::Scalar::all(-1), 1);
                    //cv::line(pre_frame, cv::Point(xx, yy), cv::Point(xxx, yyy), cv::Scalar::all(-1), 1);

                }
            }
            std::vector<cv::Point2f> mergedCenters;

            for (size_t i = 0; i < refinedStates.size(); ++i) {
                bool merged = false;
                for (size_t j = 0; j < mergedCenters.size(); ++j) {
              //      std::cout << calculateDistance(refinedStates[i], mergedCenters[j]) << "\n";
                    if (calculateDistance(refinedStates[i], mergedCenters[j]) < 50.0) {
                        mergedCenters[j] = (refinedStates[i] + mergedCenters[j]) * 0.5;
                        merged = true;
                        break;
                    }
                }
                if (!merged) {

                    mergedCenters.push_back(refinedStates[i]);
                }
            }

            if (mergedCenters.size() == 0) {
                goto a;
            }


            try {
                std::vector<cv::Point2f> circleCenters = mergedCenters; // Fill this with your circle centers

                int k = std::min(10, (int)circleCenters.size());// Number of clusters
                
                cv::Mat points(circleCenters.size(), 2, CV_32F);
                for (size_t i = 0; i < circleCenters.size(); ++i) {
                    points.at<float>(i, 0) = circleCenters[i].x;
                    points.at<float>(i, 1) = circleCenters[i].y;
                }

                cv::Mat labels, centers;
                cv::kmeans(points, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);
                

                
                    // Visualize clustered centers
                    for (int i = 0; i < k; ++i) {
                        cv::Point2f center(centers.at<float>(i, 0), centers.at<float>(i, 1));
                        int radius = 10 * sqrt(cv::countNonZero(labels == i)); // Adjust this for radius
                        cv::circle(img_scale, center, radius, cv::Scalar(255, 255, 255), 4);
                    }
                }
                catch (...) {
                    std::cout << "Error! \n ";
                }
            
                

            /*if (1) {
                int tt = 10;
                double flag_meiju[100][100];
                memset(flag_meiju, 0, sizeof(flag_meiju));
                for (int i = 0; i < gray.rows / tt; i++)
                    for (int j = 0; j < gray.cols / tt; j++) {
                        double x1 = i * tt + tt / 2;
                        double y1 = j * tt + tt / 2;
                        for (int k = 0; k < T.size(); k++)
                            if (ROI_mod(T[k].x, T[k].y) && sqrt((T[k].x - y1) * (T[k].x - y1) + (T[k].y - x1) * (T[k].y - x1)) < tt * sqrt(2)) flag_meiju[i][j]++; //flag_meiju[i][j] += dis[k];
                    }
                double mm = 0;
                int mark_i = 0, mark_j = 0;
                for (int i = 0; i < gray.rows / tt; i++)
                    for (int j = 0; j < gray.cols / tt; j++)
                        if (ROI_mod(j * tt, i * tt) && flag_meiju[i][j] > mm) {
                            mark_i = i;
                            mark_j = j;
                            mm = flag_meiju[i][j];
                            if (mm < 2) continue;
                            rectangle(frame, cv::Point(mark_j * tt / scale - rec_width, mark_i * tt / scale + rec_width), cv::Point(mark_j * tt / scale + rec_width, mark_i * tt / scale - rec_width), cv::Scalar(0, 255, 255), 3);

                        }
                if (mm > 1111) 
                    cv::rectangle(frame, cv::Point(mark_j * tt / scale - rec_width, mark_i * tt / scale + rec_width), cv::Point(mark_j * tt / scale + rec_width, mark_i * tt / scale - rec_width), cv::Scalar(0, 255, 255), 3);
                
                else {
                    // found on CSDN. Need more investigation
                    /*
                    memset(flag_meiju, 0, sizeof(flag_meiju));
                    for (int i = 0; i < gray.rows / tt; i++)
                    for (int j = 0; j < gray.cols / tt; j++)
                    {
                    double x1 = i*tt + tt / 2;
                    double y1 = j*tt + tt / 2;
                    for (int k = 0; k < T.size(); k++)
                    if (ROI_obscale(T[k].x, T[k].y) && sqrt((T[k].x - y1)*(T[k].x - y1) + (T[k].y - x1)*(T[k].y - x1)) < tt*sqrt(2)) flag_meiju[i][j] ++;
                    }
                    mm = 0;
                    mark_i = 0, mark_j = 0;
                    for (int i = 0; i < gray.rows / tt; i++)
                    for (int j = 0; j < gray.cols / tt; j++)
                    if (flag_meiju[i][j] > mm)
                    {
                    mark_i = i;
                    mark_j = j;
                    mm = flag_meiju[i][j];
                    }
                    //rectangle(frame, Point(mark_j*tt / scale - rec_width, mark_i*tt / scale + rec_width), Point(mark_j*tt / scale + rec_width, mark_i*tt / scale - rec_width), Scalar(255, 0, 0), 3);
                    
                }
            }*/

            //rectangle(frame, Point(width / 16 / scale, height * 5 / 6 / scale), Point((width - width / 16) / scale, height / 3 / scale), Scalar(255, 0, 0), 1, 0);
            //rectangle(frame, Point(100 , 180), Point(width - 100, height - 180), Scalar(255, 0, 0), 1, 0);

            tm.stop();
            std::cout << "FPS: " << 1000 / tm.getTimeMilli() << "\n";
            tm.reset();
            a:
            cv::namedWindow("img_scale", 0);
            cv::imshow("img_scale", img_scale);
            cv::resizeWindow("img_scale", img_scale.cols, img_scale.rows);
        }


        // cv::waitKeywaitKey(1000/30);
        if (cv::waitKey(27) >= 0)
            break;
        std::swap(prevgray, gray);
        resize(img_temp, pre_frame, dsize);
        t = (double)cv::getTickCount() - t;

    }
    return;
}