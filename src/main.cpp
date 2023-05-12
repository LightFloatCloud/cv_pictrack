
#include <iostream>
#include <string>

#include "fileload.h"

//#include "opencv2/opencv.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/xfeatures2d.hpp"


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

cv::Mat calculate_H(const cv::Mat &image1, const cv::Mat &image2 , double thres = 20)
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

    Mat img1_withkp, img2_withkp;
    drawKeypoints( image1, keypoints1, img1_withkp );
    drawKeypoints( image2, keypoints2, img2_withkp );

    //imshow("pic_show",imshow_parallel(img1_withkp,img2_withkp));

    waitKey(0);


    // 筛选最佳匹配
    double minDist = 100.0;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) {
            minDist = dist;
        }
    }
    vector<DMatch> bestMatches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance < 3 * minDist || matches[i].distance < thres) {
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

int main(int argc, const char * argv[])
{
    CommandLineParser parser( argc, argv, 
        "{ help h usage ?    |                | usage: *.exe <Input image1> <Input image2>}"
        "{ @input1           | ../kurisu1.jpg | input image       }" 
        "{ @input2           | ../kurisu2.jpg | input image       }"
        "{ thres             | 1              | threshold of describtor}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    const char *file_path = argv[0];
    cout<<file_path<<endl;

    std::vector<std::string> imgpath_list;
    LoadFileList("../data", ".jpg", imgpath_list);
    
    Mat img_merged = cv::imread("../data/" + imgpath_list.front(),  cv::IMREAD_COLOR);
    Mat image_last = img_merged;
    Mat T = cv::Mat::eye(3,3, CV_64F); // 新 img 相对于 blendedImage 的变换
    Mat T_offset = cv::Mat::eye(3,3, CV_64F); //img1 在 blendedImage 中的变换
    for(auto imgpath=imgpath_list.begin()+1; imgpath!=imgpath_list.end() ;imgpath++)
    {
        Mat image = cv::imread("../data/" + *imgpath, cv::IMREAD_COLOR);
        Mat H = calculate_H(image_last, image);
        //T = T*H;
        T = T_offset * H; 
        cout << "H:" <<endl<< H <<endl;
        //T_offset = cv::Mat::eye(3,3, CV_64F)
        img_merged = picmerge(image, img_merged, T.inv() ,T_offset);
        image_last = image;
        imshow("pic_show",img_merged);
        waitKey(500);
    }


    //Mat H = calculate_H()
    //
    cout << imwrite("../MERGED.jpg",img_merged);
    
    waitKey(0);
/*
    cv::String image1_path = parser.get<cv::String>("@input1");
    cv::String image2_path = parser.get<cv::String>("@input2");
    double thres = parser.get<double>("thres");

    Mat image1 = cv::imread(image1_path, 1); // 忽略alpha通道，仅RGB
    Mat image2 = cv::imread(image2_path, 1);

    if ( image1.empty() || image2.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    
    imshow("pic_show",imshow_parallel(image1,image2));
    waitKey(0);
    //Mat gray1, gray2;
    //cvtColor(image1, gray1, COLOR_BGR2GRAY);
    //cvtColor(image2, gray2, COLOR_BGR2GRAY);


    cout << "H:" <<endl<< H <<endl;


    

    // 显示融合后的图片
    namedWindow("Blended Image", WINDOW_NORMAL);
    imshow("Blended Image", blendedImage);
    waitKey(0);
    imwrite("../MERGED.jpg",blendedImage);
*/

    return 0;
}
