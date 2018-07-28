#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>


using namespace std;
using namespace cv;

	Scalar colors[] =
	{
		// 红橙黄绿青蓝紫
		CV_RGB(255, 0, 0),
		CV_RGB(255, 97, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 0, 255),
		CV_RGB(160, 32, 240)
	};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("camera/image", 1);
  Mat image, image_gray;      //定义两个Mat变量，用于存储每一帧的图像
 /* cv::Mat image = cv::imread("/home/topeet/test.jpg", CV_LOAD_IMAGE_COLOR);
  if(image.empty()){
   printf("open error\n");
   }*/
   cv::VideoCapture cap;
   cap.open(0);
   if(!cap.isOpened())
   {
    ROS_INFO("Open camera err!");
    return -1;
   }
   ROS_INFO("Open camera ok!");
  
  
   //cv::namedWindow("Video",0);
  ros::Rate loop_rate(30);
  while (nh.ok()) {
     
     image=imread("/home/wander/test/lena.jpg");
     cap>>image;
    if(image.empty())
    {
      ROS_INFO("No picture");
    }
    cvtColor(image, image_gray, CV_BGR2GRAY);//转为灰度图
   // equalizeHist(image_gray, image_gray);//直方图均衡化，增加对比度方便处理
    CascadeClassifier eye_Classifier;  //载入分类器
    CascadeClassifier face_cascade;    //载入分类器

    //加载分类训练器，OpenCv官方文档提供的xml文档，可以直接调用
    //xml文档路径  opencv\sources\data\haarcascades 
    if (!eye_Classifier.load("/home/wander/test/haarcascade_profileface.xml"))  //需要将xml文档放在自己指定的路径下
    {  
        cout << "Load haarcascade_eye.xml failed!" << endl;
        return 0;
    }

    if (!face_cascade.load("/home/wander/test/haarcascade_frontalface_alt.xml"))
    {
        cout << "Load haarcascade_frontalface_alt failed!" << endl;
        return 0;
    }

    //vector 是个类模板 需要提供明确的模板实参 vector<Rect>则是个确定的类 模板的实例化
    vector<Rect> eyeRect;
    vector<Rect> faceRect;

    //检测关于眼睛部位位置
    eye_Classifier.detectMultiScale(image_gray, eyeRect, 1.2, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++)
    {   
		Point  center;
		int radius;
		center.x = cvRound((eyeRect[eyeIdx].x + eyeRect[eyeIdx].width * 0.5));
		center.y = cvRound((eyeRect[eyeIdx].y + eyeRect[eyeIdx].height * 0.5));
 
		radius = cvRound((eyeRect[eyeIdx].width + eyeRect[eyeIdx].height) * 0.25);
		circle(image, center, radius, colors[eyeIdx% 7], 2);

       // rectangle(image, eyeRect[eyeIdx], Scalar(0, 0, 255));   //用矩形画出检测到的位置
    }

    //检测关于脸部位置
    face_cascade.detectMultiScale(image_gray, faceRect, 1.2, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faceRect.size(); i++)
    {   
        rectangle(image, faceRect[i], Scalar(0, 0, 255));      //用矩形画出检测到的位置
    }


    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}
