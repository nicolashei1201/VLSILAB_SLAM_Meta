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
  cv::Mat dDdx, dDdy, dDAll;
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
  
  cv::Mat m2;
  cv::Mat RGB_ALL;
  cv::normalize(target_depth, m2, 0, 255, cv::NORM_MINMAX);
  m2.convertTo(m2, CV_8U);
  cv::Canny(m2, dDdx, 150, 100);
  cv::Mat RGBD_edge;
  RGB_ALL = cv::abs(dIdx | dIdy);
  RGB_ALL.setTo(0, cv::abs(RGB_ALL) < 60);
  RGB_ALL.setTo(255, cv::abs(RGB_ALL) > 60);
  RGB_ALL.convertTo(RGB_ALL, CV_8U);
  RGBD_edge = cv::abs( RGB_ALL & dDdx);

  std::cout<<"\nDepth Canny Mean: "<<cv::mean(dDdx).val[0]<<"\n";
  cv::Mat RGBCanny;
  cv::Canny(target_rgb, RGBCanny, 250, 200);
  /*
  cv:: Sobel(m2, dDdx, CV_16S, 1, 0, 1);
  cv:: Sobel(m2, dDdy, CV_16S, 0, 1, 1);
  dDdx = cv::abs(dDdx);
  dDdy = cv::abs(dDdy);
  dDAll = cv::abs(dDdy | dDdx);
  dDAll.setTo(0, cv::abs(dDAll) < 5);
  dDAll.setTo(255, cv::abs(dDAll) > 5);
  */
 
  cv::imwrite("depth_Canny.png", dDdx);
  cv::imwrite("RGBD_Edge.png", RGBD_edge);
  cv::imwrite("RGB_CAnny.png", RGBCanny);
  //cv::imwrite("depth_Ylook.png", dDdy);
  //cv::imwrite("depth_Alllook.png", dDAll);
  
  
  //cv::filter2D( source_depth, dDdx, CV_16S , kernelX, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  
  //cv::filter2D( source_depth, dDdy, CV_16S , kernelY, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  //std::cout<<source_depth<<"\n";
  
  //icp_vo.Track(source_depth,source_rgb, 0);
  //icp_vo.Track(target_depth,target_rgb, 1);
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