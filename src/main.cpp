
#include <iostream>
#include <string>

#include "fileload.h"
#include "cv_method.h"

//#include "opencv2/opencv.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/xfeatures2d.hpp"


using namespace std;
using namespace cv;


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
