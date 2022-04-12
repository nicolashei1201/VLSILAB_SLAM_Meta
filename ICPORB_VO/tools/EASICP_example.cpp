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
  //std::cout<<source_depth<<"\n";
  icp_vo.TrackJoint(source_depth,source_rgb, 0);
  icp_vo.TrackJoint(target_depth,target_rgb, 1);
  
  std::cout << icp_vo.GetPoses().back() << std::endl;
  return 0; 
}
