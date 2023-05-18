
#include "cv_method.h"

using namespace std;
using namespace cv;

cv::Mat imshow_parallel(const cv::Mat &image1, const cv::Mat &image2)
{
    Mat image1_resized, image2_resized;
    int Maxrow = MAX(image1.rows, image2.rows);
    double image1_resize_scale = Maxrow/(double)image1.rows;
    cv::resize(image1,image1_resized,cv::Size((int)(image1.cols*image1_resize_scale),Maxrow),0,0,cv::INTER_LINEAR);
    double image2_resize_scale = Maxrow/(double)image2.rows;
    cv::resize(image2,image2_resized,cv::Size((int)(image2.cols*image2_resize_scale),Maxrow),0,0,cv::INTER_LINEAR);

    Mat Image_parallel;
    cv::hconcat(image1_resized,image2_resized,Image_parallel);
    return Image_parallel;
}


cv::Mat calculate_H(const cv::Mat &image1, const cv::Mat &image2 , double thres)
{
    Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(image1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(image2, Mat(), keypoints2, descriptors2);

    // 匹配描述符
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Mat img1_withkp, img2_withkp;
    // drawKeypoints( image1, keypoints1, img1_withkp );
    // drawKeypoints( image2, keypoints2, img2_withkp );

    //imshow("pic_show",imshow_parallel(img1_withkp,img2_withkp));

    //waitKey(0);


    // 筛选最佳匹配
    double minDist = 100.0;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) {
            minDist = dist;
        }
    }
    //娱乐写法 minDist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    vector<DMatch> bestMatches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance < max(3 * minDist, thres)) {
            bestMatches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    // drawMatches( image1, keypoints1, image2, keypoints2, matches, img_matches );
    // imshow("Matches", img_matches );
    // waitKey(0);

    drawMatches( image1, keypoints1, image2, keypoints2, bestMatches, img_matches );
    imshow("Matches", img_matches );
    waitKey(0);

    // 计算变换矩阵
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < bestMatches.size(); i++) {
        points1.push_back(keypoints1[bestMatches[i].queryIdx].pt);
        points2.push_back(keypoints2[bestMatches[i].trainIdx].pt);
    }
    Mat H = findHomography(points2, points1, RANSAC);

    return H;
}

cv::Mat calculate_H(const cv::Mat &image1, const cv::Mat &image2, const string &strSettingPath)
{ 
    mImGray = image1;

    // Step 1 ：将彩色图像转为灰度图像
    //若图片是3、4通道的，还需要转化成灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }




    // Step 1 从配置文件中加载相机参数
    // const string strSettingPath = "config.yaml";
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    using namespace ORB_SLAM2;
    ORBextractor* mpORBextractorLeft;
    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(
        nFeatures,      //参数的含义还是看上面的注释吧
        fScaleFactor,
        nLevels,
        fIniThFAST,
        fMinThFAST);
    


    std::vector<cv::KeyPoint> mvKeys_1, mvKeys_2;
    cv::Mat mDescriptors_1, mDescriptors_2;

    (*mpORBextractorLeft)(  image1,				//待提取特征点的图像
                            cv::Mat(),		//掩摸图像, 实际没有用到
                            mvKeys_1,			//输出变量，用于保存提取后的特征点
                            mDescriptors_1);	//输出变量，用于保存特征点的描述子

    (*mpORBextractorLeft)(  image2,				//待提取特征点的图像
                            cv::Mat(),		//掩摸图像, 实际没有用到
                            mvKeys_2,			//输出变量，用于保存提取后的特征点
                            mDescriptors_2);	//输出变量，用于保存特征点的描述子

    // 匹配描述符
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> matches;
    matcher->match(mDescriptors_1, mDescriptors_2, matches);


    // 筛选最佳匹配
    double minDist = 100.0;
    for (int i = 0; i < mDescriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) {
            minDist = dist;
        }
    }

    double thres = fSettings["ORBextractor.matchthres"];
    vector<DMatch> bestMatches;
    for (int i = 0; i < mDescriptors_1.rows; i++) {
        if (matches[i].distance < max(3 * minDist, thres)) {
            bestMatches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    // drawMatches( image1, keypoints1, image2, keypoints2, matches, img_matches );
    // imshow("Matches", img_matches );
    // waitKey(0);

    drawMatches( image1, mvKeys_1, image2, mvKeys_2, bestMatches, img_matches );
    imshow("Matches", img_matches );
    waitKey(0);

    // 计算变换矩阵
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < bestMatches.size(); i++) {
        points1.push_back(mvKeys_1[bestMatches[i].queryIdx].pt);
        points2.push_back(mvKeys_2[bestMatches[i].trainIdx].pt);
    }
    Mat H = findHomography(points2, points1, RANSAC);

    return H;


}

cv::Mat picmerge(const cv::Mat &image1, const cv::Mat &image2, cv::Mat H ,cv::Mat &Trans)
{
    // 融合两幅图片
    Mat blendedImage;
    int x_offset=(int)H.at<double>(0,2);
    int y_offset=(int)H.at<double>(1,2);
    int x_merged, y_merged;
    int roix_start, roiy_start; //纠正量
        
    Mat T = Mat::eye(3, 3, H.type()); // img1 在 blendedImage 中的变换
    
    if(x_offset > 0)
    {
        x_merged = MAX(image1.cols, image2.cols + x_offset);
        roix_start = 0;
    }
    else 
    {
        x_merged = MAX(image2.cols, image1.cols - x_offset);
        roix_start = -x_offset;
        T.at<double>(0,2) = -H.at<double>(0,2);
    }
    if(y_offset > 0)
    {
        y_merged = MAX(image1.rows, image2.rows + y_offset);
        roiy_start = 0;
    }
    else
    {
        y_merged = MAX(image2.rows, image1.rows - y_offset);
        roiy_start = -y_offset;
        T.at<double>(1,2) = -H.at<double>(1,2);
    }
    
    // H.at<double>(0,2) = H.at<double>(0,2) + (double)roix_start; 
    // H.at<double>(1,2) = H.at<double>(1,2) + (double)roiy_start;
    //cout << H.type() <<endl;
    H = T*H; // img2 在 blendedImage 中的变换
    Trans = T;

    warpPerspective(image2, blendedImage, H, Size(x_merged, y_merged),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    Mat roi(blendedImage, Rect(roix_start, roiy_start, image1.cols, image1.rows));
    image1.copyTo(roi);

    return blendedImage;
}


