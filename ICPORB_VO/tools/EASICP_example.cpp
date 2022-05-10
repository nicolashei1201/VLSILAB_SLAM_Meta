/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : EASICP_example.cpp

* Purpose :

* Creation Date : 2020-08-29

* Last Modified : 廿廿一年六月十九日 (週六) 十六時卅分十四秒

* Created By :  

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

// #include <EAS_ICP.h>
#include <ICP_VO.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <ElasticFusion.h> 

int main(int argc,  char *argv[])
{
  if (argc != 6) {
    std::cout << "./build/rgbd_tum path_to_source_depth_file path_to_target_depth_file path_to_setting " << std::endl;
    return 1;
  }
  ICP_VO icp_vo(argv[5]);
  
  cv::Mat source_depth = cv::imread(argv[1], -1);
  cv::Mat target_depth = cv::imread(argv[2], -1);
  cv::Mat source_rgb = cv::imread(argv[3], -1);
  cv::Mat target_rgb = cv::imread(argv[4], -1);

  cv::Mat dIdx, dIdy;
  float gsx3x3[9] = {-0.52201,  0.00000, 0.52201,
                   -0.79451, -0.00000, 0.79451,
                   -0.52201,  0.00000, 0.52201};

  float gsy3x3[9] = {-0.52201, -0.79451, -0.52201,
                     0.00000, 0.00000, 0.00000,
                    0.52201, 0.79451, 0.52201};


  cv::Mat intensity;
  cv::cvtColor(source_rgb, intensity, cv::COLOR_BGR2GRAY);
  cv::Mat kernelX(3, 3, CV_32F, gsx3x3);
  cv::Mat kernelY(3, 3, CV_32F, gsy3x3);
  cv::filter2D( intensity, dIdx, CV_16S , kernelX, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  cv::filter2D( intensity, dIdy, CV_16S , kernelY, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  //dIdx = dIdx * (-1);
  std::cout<<(int)intensity.at<char>(1,1)<<"\n";
  std::cout<<(int)intensity.at<char>(2,1)<<"\n";
  std::cout<<(int)intensity.at<char>(3,1)<<"\n";
  std::cout<<(int)intensity.at<char>(1,3)<<"\n";
  std::cout<<(int)intensity.at<char>(2,3)<<"\n";
  std::cout<<(int)intensity.at<char>(3,3)<<"\n";
  std::cout<<"answer:\n";
  //std::cout<<dIdx.at<short>(2,2);
  std::cout<<dIdx;
  std::cout<<dIdy;
  //std::cout<<source_depth<<"\n";
  //icp_vo.TrackJoint(source_depth,source_rgb, 0);
 // icp_vo.TrackJoint(target_depth,target_rgb, 1);
 /*
  Resolution::getInstance(640, 480);
  Intrinsics::getInstance(528, 528, 320, 240);
  std::cout<<"why??\n";

  pangolin::Params windowParams;
  windowParams.Set("SAMPLE_BUFFERS", 0);
  windowParams.Set("SAMPLES", 0);
  pangolin::CreateWindowAndBind("Main", 1280, 800, windowParams);
  ElasticFusion eFusion;
  std::cout<<"why??\n";
  Eigen::Matrix4f * currentPose = 0;
  currentPose = new Eigen::Matrix4f;
  currentPose->setIdentity();
  int64_t timestamp = 0;
  std::cout<<eFusion.getTick()<<"\n";
  eFusion.setEASICPORB(false);
  eFusion.processFrame((uchar*)source_rgb.data, (unsigned short*)source_depth.data, timestamp);//, currentPose);
  timestamp++;
  eFusion.processFrame((uchar*)target_rgb.data, (unsigned short*)target_depth.data, timestamp);
  std::cout<<eFusion.getCurrPose()<<"\n";
  std::cout<<eFusion.getTick()<<"\n";
  */
  //std::cout << icp_vo.GetPoses().back() << std::endl;
  return 0; 
}