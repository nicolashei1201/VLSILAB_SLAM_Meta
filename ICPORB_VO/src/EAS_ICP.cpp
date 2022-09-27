/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : EAS_ICP.cpp

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿一年六月十九日 (週六) 廿一時58分53秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <EAS_ICP.h>
#include "pcl_normal.hpp"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <ctime>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
EAS_ICP::EAS_ICP( const std::string& strSettings){
  cv::FileStorage FsSettings(strSettings.c_str(), cv::FileStorage::READ);

  //TODO initial with fsettings
  //basic
  width  = FsSettings["Camera.width"];
  height = FsSettings["Camera.height"];
  pixelSize=width*height;
  mDepthMapFactor = FsSettings["DepthMapFactor"];
  fx = FsSettings["Camera.fx"];
  fy = FsSettings["Camera.fy"];
  cx = FsSettings["Camera.cx"];
  cy = FsSettings["Camera.cy"];
  max_depth = FsSettings["Sampling.max_depth"];
  min_depth = FsSettings["Sampling.min_depth"];
  icp_converged_threshold_rot   = FsSettings["ICP.icp_converged_threshold_rot"]; 
  icp_converged_threshold_trans = FsSettings["ICP.icp_converged_threshold_trans"]; 
  max_iters = FsSettings["ICP.max_iters"];
  thresAccSlidingExtent= FsSettings["ICP.thresAccSlidingExtent"]; 

  //sampling
  random_seed    = FsSettings["Sampling.random_seed"];
  if (random_seed == -1) {
    random_seed = 0;
  }
  stride         = FsSettings["Sampling.stride"]; 
  edge_threshold = FsSettings["Sampling.edge_threshold"];
  sampling_size  = FsSettings["Sampling.number_of_sampling"];

  //matching
  search_step                 = FsSettings["DataAssociating.search_step"]; 
  search_range                = FsSettings["DataAssociating.search_range"]; 
  dynamic_threshold_rejection = FsSettings["DataAssociating.dynamic_threshold_rejection"]; 
  fixed_threshold_rejection   = FsSettings["DataAssociating.fixed_threshold_rejection"]; 
  top_bound                   = FsSettings["DataAssociating.top_bound"]; 
  bottom_bound                = FsSettings["DataAssociating.bottom_bound"]; 
  left_bound                  = FsSettings["DataAssociating.left_bound"]; 
  right_bound                 = FsSettings["DataAssociating.right_bound"]; 
                                            
  //Minimizing
  thresEvalRatio              = FsSettings["TransformSolver.thresEvalRatio"]; 
  
  iteration_loop2 = 0;
  iterations = 0;
}

EAS_ICP::~EAS_ICP(){

}

const EAS_ICP::Transform& EAS_ICP::RegisterPure(const SourceCloud& srcCloud, const cv::Mat& depth, const cv::Mat& rgb, const TargetCloud& tgtCloud, const Transform& initialGuess, const cv::Mat& Last_intens) {

  //initial parameters
  rtSE3 = initialGuess;
  rtSE3 = Transform::Identity();
  iterations = 0;
  accSlidingExtent = 0;
  computeDerivativeImages(last_rgb, dIdx, dIdy);
  
  Transform initialGuess_I = Transform::Identity();
  Transform SO3Pose = SO3_prealign(initialGuess,rgb, 4);
  SO3Pose = SO3_prealign(SO3Pose,rgb, 1);
  //rtSE3 = rtSE3;
  
  //cv::namedWindow("rgb_edge", cv::WINDOW_AUTOSIZE);
  
  //cv::Mat rgb_cpy;
  //cv::cvtColor(rgb, rgb_cpy, cv::COLOR_BGR2RGB);
  //cv::imshow("rgb_edge", rgb_cpy);
  //cv::waitKey(30);
  
  //cv::Mat final_edges = cv::abs(dIdx | dIdy);
  //final_edges.setTo(0, cv::abs(final_edges) < 60);
  //final_edges.setTo(255, cv::abs(final_edges) > 60);


  std::cout<<"\nRGB quality: "<<rgb_quality;
  std::cout<<"\nDepth quality: "<<depth_quality<<"\n";

  while (true) {
    iterations+=1;

    //transform source cloud by inital guess
    SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    clock_t t_m1, t_m2;
	TargetCloud LastCloud = ComputeCurrentCloud(last_depth);
    t_m1 = clock();
    //match correspondence
    if (!MatchingByProject2DAndWalk(transformedCloud, tgtCloud)) {
      break; // when correspondence size less than 6
    }
	/*
	if(iterations == 1){
		RANSAC_strategy(transformedCloud,tgtCloud);
	}
	*/
    t_m2 = clock();
    //std::cout<<"matching corres"<<std::endl;
    //std::cout<<double(t_m2-t_m1)/CLOCKS_PER_SEC<<std::endl;
    //get iteration transformation by minimizing p2pl error metric
    clock_t t_p1, t_p2;
    t_p1 = clock();
    Eigen::Vector<Scalar, 6> rt6D;
	Transform iterRtSE3;
	if(iterations < 25){
		//rt6D = MinimizingP2PLErrorMetric(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		rt6D = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3, LastCloud);
		//rt6D =MinimizingP2PLErrorMetricGaussianNewton(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		iterRtSE3 = ConstructSE3_GN(rt6D);
	}
	else{
		//rt6D =MinimizingP2PLErrorMetricGaussianNewton(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		rt6D = MinimizingP2PLErrorMetric(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		
		//convert 6D vector to SE3
		//rt6D = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3, LastCloud);
		iterRtSE3 = ConstructSE3(rt6D);
	}
    //chain iterRtSE3 to rtSE3
    rtSE3 = iterRtSE3 * rtSE3;
    t_p2 = clock();
    //std::cout<<"pose esti"<<std::endl;
    //std::cout<<double(t_p2-t_p1)/CLOCKS_PER_SEC<<std::endl;

    //check termination
    if (iterations > 50) {
      break;
    }
  }
  //justify valid by sliding extent
  if (accSlidingExtent < thresAccSlidingExtent) {
    valid = true;
  } else {
    valid = false;
  }
  return rtSE3;
}
const EAS_ICP::Transform& EAS_ICP::Register(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Transform& initialGuess) {

  //initial parameters
  rtSE3 = initialGuess;
  int iterations = 0;
  accSlidingExtent = 0;
  while (true) {
    iterations+=1;

    //transform source cloud by inital guess
    SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    clock_t t_m1, t_m2;
    t_m1 = clock();
    //match correspondence
    if (!MatchingByProject2DAndWalk(transformedCloud, tgtCloud)) {
      break; // when correspondence size less than 6
    }
    t_m2 = clock();
    //std::cout<<"matching corres"<<std::endl;
    //std::cout<<double(t_m2-t_m1)/CLOCKS_PER_SEC<<std::endl;
    //get iteration transformation by minimizing p2pl error metric
    clock_t t_p1, t_p2;
    t_p1 = clock();
    Eigen::Vector<Scalar, 6> rt6D;

	//rt6D =MinimizingP2PLErrorMetricGaussianNewton(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
    rt6D = MinimizingP2PLErrorMetric(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
    
    //convert 6D vector to SE3
    Transform iterRtSE3;
    iterRtSE3 = ConstructSE3(rt6D);

    //chain iterRtSE3 to rtSE3
    rtSE3 = iterRtSE3 * rtSE3;
    t_p2 = clock();
    //std::cout<<"pose esti"<<std::endl;
    //std::cout<<double(t_p2-t_p1)/CLOCKS_PER_SEC<<std::endl;

    //check termination
    if (iterations > 50) {
      break;
    }
  }
  //justify valid by sliding extent
  if (accSlidingExtent < thresAccSlidingExtent) {
    valid = true;
  } else {
    valid = false;
  }
  return rtSE3;
}



const EAS_ICP::Transform& EAS_ICP::Register(const SourceCloud& srcCloud, const cv::Mat& depth, const cv::Mat& rgb, const TargetCloud& tgtCloud, const Transform& initialGuess, const cv::Mat& Last_intens) {

  //initial parameters
  std::cout<<"Meta ICP\n";
  long t1, t2;
  t1 = clock();
  //RANSAC_strategy(srcCloud,tgtCloud);
  Transform initialGuess_I = Transform::Identity();
  Transform SO3Pose = SO3_prealign(initialGuess,rgb, 4);
  SO3Pose = SO3_prealign(SO3Pose,rgb, 1);
  //SO3Pose = SO3_prealign(SO3Pose,rgb, 1);
  
  rtSE3 = initialGuess;
  rtSE3_1 = initialGuess;
  rtSE3_2 = initialGuess;
  rtSE3_3 = initialGuess;
  rtSE3_4 = initialGuess;
  
  SourceCloud transformedCloudOri0(srcCloud.rows(), 6) ;
  transformedCloudOri0.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
  transformedCloudOri0.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
  //kinectNoiseWeights = KinectNoiseWighting(transformedCloudOri0);
  t2 = clock();
  std::cout<<"\nSO3 time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";

  MatchingByProject2DAndWalk(transformedCloudOri0, tgtCloud);
  /*
  t1 = clock();
  RANSAC_strategy(transformedCloudOri0, tgtCloud);
  Eigen::Vector<Scalar, 6> rt6D0;
  rt6D0 = MinimizingP2PLErrorMetric(transformedCloudOri0(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
  Transform iterRtSE30;
  iterRtSE30 = ConstructSE3(rt6D0);
  rtSE3 = rtSE3 * iterRtSE30;
  rtSE3_1 = rtSE3_1 * iterRtSE30;
  rtSE3_2 = rtSE3_2 * iterRtSE30;
  rtSE3_3 = rtSE3_3 * iterRtSE30;
  rtSE3_4 = rtSE3_4 * iterRtSE30;
  t2 = clock();
  std::cout<<"\nRANSAC time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
  */
 /*
  rtSE3 = Transform::Identity();
  rtSE3_1 = Transform::Identity();
  rtSE3_2 = Transform::Identity();
  rtSE3_3 = Transform::Identity();
  rtSE3_4 = Transform::Identity();
  */
  int cnt_task1_large =0;
  int cnt_task2_large =0;
  int cnt_task3_large =0;
  int cnt_task4_large =0;
  iterations = 0;
  accSlidingExtent = 0;
  Useonlyforeground = 1;
  Useweigtedrandomsampling = 1;
  Useedgeaware = 0;
  srand( time(NULL) );
  int iteration_divide = 20;

  last_inten = Last_intens;
  t1 = clock();
  computeDerivativeImages(last_rgb, dIdx, dIdy);
  while (true) {
  iterations+=1;
  //meta training
    /*double randomRGB = (double) rand() / (RAND_MAX + 1.0);
	double randomdepth = (double) rand() / (RAND_MAX + 1.0);
	std::cout<<"randomRGB"<<std::endl;
	std::cout<<randomRGB<<std::endl;
	std::cout<<"randomdepth"<<std::endl;
	std::cout<<randomdepth<<std::endl;*/
  //RGB ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  cv::Mat nan_map;
  cv::Mat rej_map;

  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  //cv::Mat inten;
  //cv::cvtColor(rgb, inten, cv::COLOR_BGR2GRAY);

  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::Mat edge_map_inner;
  cv::Mat edge_map_outer;
  cv::resize(src, out_resized, cv::Size(320, 240), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  //cv::blur(src, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  //dilation
  cv::Mat dilatemat1111 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11));
  cv::Mat dilatemat99 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
  cv::Mat dilatemat77 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat dilatemat55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::Mat dilatemat33 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

  cv::dilate(edge_map2, edge_map_inner, dilatemat33);
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner;
  inv_edge_map_inner = ~edge_map_inner;
  cv::Mat edge_distance_map_inner;
  cv::distanceTransform(inv_edge_map_inner, edge_distance_map_inner, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  
  /////////////////////////dilate outer

 
  cv::dilate(edge_map2, edge_map_outer, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer, edge_map_inner, edge_map_outer);
  //cv::imwrite("ROI_detection_rgb_dilateouter.png", edge_map_outer);
  cv::Mat inv_edge_map_outer;
  inv_edge_map_outer = ~edge_map_outer;
  cv::Mat edge_distance_map_outer;
  //cv::distanceTransform(inv_edge_map_outer, edge_distance_map_outer, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2;
  std::vector<double> weights2;
  std::vector<int> EASInds2;

  
  //*********************************************
  //Depth ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  keycloud_depth = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  cv::Mat edge_map_depth;
  cv::Mat nan_map_depth;
  cv::Mat rej_map_depth;
  cv::Mat edge_map_inner_depth;
  EdgeDetection(*keycloud_depth, edge_map_depth, nan_map_depth, rej_map_depth);

  cv::dilate(edge_map_depth, edge_map_inner_depth, dilatemat33);
  //cv::imwrite("ROI_detection_depth_inner.png", edge_map_inner_depth);
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner_depth;
  inv_edge_map_inner_depth = ~edge_map_inner_depth;
  cv::Mat edge_distance_map_inner_depth;
  cv::distanceTransform(inv_edge_map_inner_depth, edge_distance_map_inner_depth, cv::DIST_L2, 5);
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds_depth;
  std::vector<double> weights_depth;
  std::vector<int> EASInds_depth;

  
  /////////////////////////dilate outer
  //cv::Mat dilatemat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat edge_map_outer_depth;

  cv::dilate(edge_map_depth, edge_map_outer_depth, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer_depth, edge_map_inner_depth, edge_map_outer_depth);
  
  cv::Mat inv_edge_map_outer_depth;
  inv_edge_map_outer_depth = ~edge_map_outer_depth;
  //cv::Mat edge_distance_map_outer_depth;
  //cv::distanceTransform(inv_edge_map_outer_depth, edge_distance_map_outer_depth, cv::DIST_L2, 5);
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2_depth;
  std::vector<double> weights2_depth;
  std::vector<int> EASInds2_depth;
  cv::Mat mapR1;
  cv::Mat mapR2;
  cv::Mat mapR3;
  cv::Mat mapR4;
  mapR1 = edge_map_inner_depth;
  mapR2 = edge_map_outer_depth;
  mapR3 = edge_map_inner;
  mapR4 = edge_map_outer;
  for(int i =0;i<mapR1.rows;i++)
  {
	for(int j =0;j<mapR1.cols;j++)
	{
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR1.at<uchar>(i,j) = 0;
		}		
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR2.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR3.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR4.at<uchar>(i,j) = 0;
		}
	}		
  }
  
	  
  //Before iteration 20
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_inner, nan_map_depth| rej_map_depth, remindPointInds, weights);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR3, nan_map_depth| rej_map_depth, remindPointInds, weights);
	
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1_before20);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_outer, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR4, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2_before20);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_inner_depth, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR1, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3_before20);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_outer_depth, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR2, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4_before20);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	
	
  //After iteration 20
  //4 ROImap pixelwise OR 
  cv::Mat ROI4OR;
  //cv::bitwise_or(edge_map_inner_depth, edge_map_outer_depth, ROI4OR);
  cv::bitwise_or(edge_map_inner, edge_map_outer, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_inner_depth, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_outer_depth, ROI4OR);
  if(Useedgeaware==0)
	{	  
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth| rej_map_depth, remindPointInds, weights);
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
  else
  {
	// 1st 
	cv::Mat InvEdgeMap;
	cv::Mat EdgeDistanceMap;
	InvEdgeMap = ~edge_map_depth;
	cv::distanceTransform(InvEdgeMap, EdgeDistanceMap, cv::DIST_L2, 5);
 
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth| rej_map_depth, remindPointInds, weights);
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
    CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	
	// 2nd 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
		
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
		
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
		
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	
	
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
	kinectNoiseWeightsdepth_before20 = KinectNoiseWighting(mSrcCloud3_before20);
	kinectNoiseWeightsdepth2_before20 = KinectNoiseWighting(mSrcCloud4_before20);
	kinectNoiseWeightsrgb_before20 = KinectNoiseWighting(mSrcCloud1_before20);
	kinectNoiseWeightsrgb2_before20 = KinectNoiseWighting(mSrcCloud2_before20);
	kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	/*kinectNoiseWeightsdepth = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(srcCloud);*/
	//
	
	
	
	//sampling end
	
	
    //transform depth source cloud by inital guess
    //SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    //transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    //transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    SourceCloud transformedCloudrgb(mSrcCloud1.rows(), 6) ;
	SourceCloud transformedCloudrgb2(mSrcCloud2.rows(), 6) ;
	SourceCloud transformedClouddepth(mSrcCloud3.rows(), 6) ;
	SourceCloud transformedClouddepth2(mSrcCloud4.rows(), 6) ;
	SourceCloud transformedCloudrgb_before20(mSrcCloud1_before20.rows(), 6) ;
	SourceCloud transformedCloudrgb2_before20(mSrcCloud2_before20.rows(), 6) ;
	SourceCloud transformedClouddepth_before20(mSrcCloud3_before20.rows(), 6) ;
	SourceCloud transformedClouddepth2_before20(mSrcCloud4_before20.rows(), 6) ;
	/*SourceCloud transformedCloudrgb(srcCloud.rows(), 6) ;
	SourceCloud transformedCloudrgb2(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth2(srcCloud.rows(), 6) ;*/
	if(Useedgeaware==0)
	{
		if(iterations<=iteration_divide)
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1_before20.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2_before20.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3_before20.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4_before20.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4_before20.rightCols<3>().transpose()).transpose();
		}	
		
		else
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
		}
	}
	else
	{
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
		
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
    }	
	
	//TargetCloud LastCloud_1 = ((rtSE3_1.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
	//TargetCloud LastCloud_2 = ((rtSE3_2.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
	//TargetCloud LastCloud_3 = ((rtSE3_3.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
	//TargetCloud LastCloud_4 = ((rtSE3_4.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
	
	TargetCloud LastCloud_1 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_2 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_3 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_4 = ComputeCurrentCloud(last_depth);
	//std::cout<<pLastCloud->leftCols(3);
	int cnttrans1, cnttrans2, cnttrans3, cnttrans4;
	cnttrans1 = 0;
	cnttrans2 = 0;
	cnttrans3 = 0;
	cnttrans4 = 0;
    int meanweight1, meanweight2, meanweight3, meanweight4;
    double randomx_thres = 0.1;
	if(Useweigtedrandomsampling==1)
	{
	  randomx_thres = 100;
	}
    //get iteration transformation by minimizing p2pl error metric

    Eigen::Vector<Scalar, 6> rt6D;
	
    if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)) {
			//std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all));
			//rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all));
			//std::cout<<"\ntransformedClouddepth_before20:\n"<<pLastCloud.rows()<<"\n\n";
			//std::cout<<"\nCorres:\n"<<corrs<<"\n\n";
			std::cout<<"hello~~~";
			//RANSAC_strategy(transformedClouddepth_before20, tgtCloud);
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_1, LastCloud_1);
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)) {
			//std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
			//rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsdepth(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_1, LastCloud_1);
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
		}
		}
		//std::cout<<"cloudnumber"<<std::endl;
		//std::cout<<mSrcCloud3.rows()<<std::endl;

		meanweight1 = 1;
	}
    
	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all));
			//rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all));
			//RANSAC_strategy(transformedClouddepth2_before20, tgtCloud);
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_2, LastCloud_2);
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
			//rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_2, LastCloud_2);
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
		}
		}	

		meanweight2 = 1;
	}


	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
		//std::cout<<"cnttrans3"<<std::endl;
		//std::cout<<cnttrans3<<std::endl;
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all));
			//rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all));
			//RANSAC_strategy(transformedCloudrgb_before20, tgtCloud);
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_3, LastCloud_3);
		}	
		else
		{
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {

		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
			//rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_3, LastCloud_3);
		}	
		else 
		{
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	}
	
	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all));
			//rt6D4 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_4, LastCloud_4);
			//RANSAC_strategy(transformedCloudrgb2_before20, tgtCloud);
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_4, LastCloud_4);
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}

		meanweight4 = 1;
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
			//rt6D4 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_4, LastCloud_4);
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
			
		}
		//std::cout<<"cnttrans4"<<std::endl;
		//std::cout<<cnttrans4<<std::endl;
		meanweight4 = 1;
		}
	}	
	//std::cout<<"corrs4num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	// ++ meta task loop end
	// ++ pose fusion = mean(all tasks' rt6D)
	
	if((meanweight1+meanweight2+meanweight3+meanweight4)==0)
	{
		break;
	}
	/*else{

    rt6D(0) = (rt6D1(0)*meanweight1+ rt6D2(0)*meanweight2 +rt6D3(0)*meanweight3 +rt6D4(0)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(1) = (rt6D1(1)*meanweight1+ rt6D2(1)*meanweight2 +rt6D3(1)*meanweight3 +rt6D4(1)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
    rt6D(2) = (rt6D1(2)*meanweight1+ rt6D2(2)*meanweight2 +rt6D3(2)*meanweight3 +rt6D4(2)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(3) = (rt6D1(3)*meanweight1+ rt6D2(3)*meanweight2 +rt6D3(3)*meanweight3 +rt6D4(3)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(4) = (rt6D1(4)*meanweight1+ rt6D2(4)*meanweight2 +rt6D3(4)*meanweight3 +rt6D4(4)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(5) = (rt6D1(5)*meanweight1+ rt6D2(5)*meanweight2 +rt6D3(5)*meanweight3 +rt6D4(5)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	}*/
    //convert 6D vector to SE3
    Eigen::Vector<Scalar, 6> rt6D_L1_largest;
	double s1, s2, s3, s4;
	
	s1 = fabs(rt6D1(0))+fabs(rt6D1(1))+fabs(rt6D1(2));
	s2 = fabs(rt6D2(0))+fabs(rt6D2(1))+fabs(rt6D2(2));
	s3 = fabs(rt6D3(0))+fabs(rt6D3(1))+fabs(rt6D3(2));
	s4 = fabs(rt6D4(0))+fabs(rt6D4(1))+fabs(rt6D4(2));
	
	/*
	s1 = fabs(rt6D1(3))+fabs(rt6D1(4))+fabs(rt6D1(5));
	s2 = fabs(rt6D2(4))+fabs(rt6D2(4))+fabs(rt6D2(5));
	s3 = fabs(rt6D3(4))+fabs(rt6D3(4))+fabs(rt6D3(5));
	s4 = fabs(rt6D4(4))+fabs(rt6D4(4))+fabs(rt6D4(5));
	*/
	if((s1>=s2)&&(s1>=s3)&&(s1>=s4))
	{
		std::cout<<"\n\nHAHSAHAH\n\n";
		rt6D_L1_largest=rt6D1;
		cnt_task1_large++;
	}
	if((s2>=s1)&&(s2>=s3)&&(s2>=s4))
	{
		rt6D_L1_largest=rt6D2;
		cnt_task2_large++;
	}
	if((s3>=s2)&&(s3>=s1)&&(s3>=s4))
	{
		rt6D_L1_largest=rt6D3;
		cnt_task3_large++;
	}
	if((s4>=s2)&&(s4>=s3)&&(s4>=s1))
	{
		rt6D_L1_largest=rt6D4;
		cnt_task4_large;
	}
	
	Transform iterRtSE3;
	Transform Pre_RtSE3;
	//Eigen::Vector3f Pre_trans(rt6D_L1_largest(0),rt6D_L1_largest(1),rt6D_L1_largest(2));
	Eigen::Vector3f Pre_trans(rt6D_L1_largest(3),rt6D_L1_largest(4),rt6D_L1_largest(5));
    iterRtSE3 = ConstructSE3_GN(rt6D_L1_largest);
	//iterRtSE3 = ConstructSE3(rt6D_L1_largest);
    //chain iterRtSE3 to rtSE3
	std::cout<<iterRtSE3.col(3).head<3>().transpose()<<"\n";
	std::cout<<"\npre: "<<Pre_trans;
	std::cout<<"\nnorm: "<<Pre_trans.norm();
	if(Pre_trans.norm() > 0.3){
		rtSE3_1 = rtSE3_1;
		rtSE3_2 = rtSE3_2;
		rtSE3_3 = rtSE3_3;
		rtSE3_4 = rtSE3_4;
	}
	else{
    	rtSE3_1 = iterRtSE3 * rtSE3_1;
		rtSE3_2 = iterRtSE3 * rtSE3_2;
		rtSE3_3 = iterRtSE3 * rtSE3_3;
		rtSE3_4 = iterRtSE3 * rtSE3_4;
		rtSE3 = rtSE3 * iterRtSE3;
	}
    //chain iterRtSE3 to rtSE3
    

    //chain iterRtSE3 to rtSE3
    

    //chain iterRtSE3 to rtSE3
    
    //check termination
    /*if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }*/
	max_iters = 30;
	if (iterations > max_iters)
	{
		std::cout<<"4 tasks count"<<std::endl;
		std::cout<<cnt_task1_large<<std::endl;
		std::cout<<cnt_task2_large<<std::endl;
		std::cout<<cnt_task3_large<<std::endl;
		std::cout<<cnt_task4_large<<std::endl;		
		break;
	}
  }
  t2 = clock();
  std::cout<<"\nMeta Training time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
  
  
  
  //meta testing
  //icp loop 2 START
  iteration_loop2 = 0;
  t1 = clock();
  while (true) {
  iteration_loop2+=1;
  std::cout<<"\niter2: "<<iteration_loop2;
  //RGB ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  cv::Mat nan_map;
  cv::Mat rej_map;
  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::Mat edge_map_inner;
  cv::Mat edge_map_outer;
  cv::resize(src, out_resized, cv::Size(320, 240), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  //cv::blur(src, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  //dilation
  cv::Mat dilatemat55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::Mat dilatemat33 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  cv::dilate(edge_map2, edge_map_inner, dilatemat33);

  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner;
  inv_edge_map_inner = ~edge_map_inner;
  cv::Mat edge_distance_map_inner;
  cv::distanceTransform(inv_edge_map_inner, edge_distance_map_inner, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  
  /////////////////////////dilate outer
  cv::Mat dilatemat99 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
  cv::Mat dilatemat77 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::dilate(edge_map2, edge_map_outer, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer, edge_map_inner, edge_map_outer);
  
  cv::Mat inv_edge_map_outer;
  inv_edge_map_outer = ~edge_map_outer;
  cv::Mat edge_distance_map_outer;
  //cv::distanceTransform(inv_edge_map_outer, edge_distance_map_outer, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2;
  std::vector<double> weights2;
  std::vector<int> EASInds2;

  
  //*********************************************
  //Depth ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  keycloud_depth = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  cv::Mat edge_map_depth;
  cv::Mat nan_map_depth;
  cv::Mat rej_map_depth;
  cv::Mat edge_map_inner_depth;
  EdgeDetection(*keycloud_depth, edge_map_depth, nan_map_depth, rej_map_depth);
  
  cv::dilate(edge_map_depth, edge_map_inner_depth, dilatemat33);
  
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner_depth;
  inv_edge_map_inner_depth = ~edge_map_inner_depth;
  cv::Mat edge_distance_map_inner_depth;
  cv::distanceTransform(inv_edge_map_inner_depth, edge_distance_map_inner_depth, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds_depth;
  std::vector<double> weights_depth;
  std::vector<int> EASInds_depth;

  
  /////////////////////////dilate outer
  //cv::Mat dilatemat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat edge_map_outer_depth;
  cv::dilate(edge_map_depth, edge_map_outer_depth, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer_depth, edge_map_inner_depth, edge_map_outer_depth);
  cv::Mat inv_edge_map_outer_depth;
  inv_edge_map_outer_depth = ~edge_map_outer_depth;
  //cv::Mat edge_distance_map_outer_depth;
  //cv::distanceTransform(inv_edge_map_outer_depth, edge_distance_map_outer_depth, cv::DIST_L2, 5);
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2_depth;
  std::vector<double> weights2_depth;
  std::vector<int> EASInds2_depth;
  cv::Mat mapR1;
  cv::Mat mapR2;
  cv::Mat mapR3;
  cv::Mat mapR4;
  mapR1 = edge_map_inner_depth;
  mapR2 = edge_map_outer_depth;
  mapR3 = edge_map_inner;
  mapR4 = edge_map_outer;
  for(int i =0;i<mapR1.rows;i++)
  {
	for(int j =0;j<mapR1.cols;j++)
	{
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR1.at<uchar>(i,j) = 0;
		}		
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR2.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR3.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR4.at<uchar>(i,j) = 0;
		}
	}		
  }
  
  //Before iteration 20
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_inner, nan_map_depth| rej_map_depth, remindPointInds, weights);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR3, nan_map_depth| rej_map_depth, remindPointInds, weights);
	
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1_before20);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_outer, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR4, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2_before20);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_inner_depth, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR1, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3_before20);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_outer_depth, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR2, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4_before20);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	
	
  //After iteration 20
  //4 ROImap pixelwise OR 
  cv::Mat ROI4OR;
  //cv::bitwise_or(edge_map_inner_depth, edge_map_outer_depth, ROI4OR);
  cv::bitwise_or(edge_map_inner, edge_map_outer, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_inner_depth, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_outer_depth, ROI4OR);
  if(Useedgeaware==0)
	{	  
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth| rej_map_depth, remindPointInds, weights);
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
  else
  {
	// 1st 
	cv::Mat InvEdgeMap;
	cv::Mat EdgeDistanceMap;
	InvEdgeMap = ~edge_map_depth;
	cv::distanceTransform(InvEdgeMap, EdgeDistanceMap, cv::DIST_L2, 5);
 
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth| rej_map_depth, remindPointInds, weights);
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
    CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	
	// 2nd 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
		
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
		
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
		
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	
	
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
	kinectNoiseWeightsdepth_before20 = KinectNoiseWighting(mSrcCloud3_before20);
	kinectNoiseWeightsdepth2_before20 = KinectNoiseWighting(mSrcCloud4_before20);
	kinectNoiseWeightsrgb_before20 = KinectNoiseWighting(mSrcCloud1_before20);
	kinectNoiseWeightsrgb2_before20 = KinectNoiseWighting(mSrcCloud2_before20);
	kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	/*kinectNoiseWeightsdepth = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(srcCloud);*/
	//
	
	
	
	//sampling end
	
	
    //transform depth source cloud by inital guess
    //SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    //transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    //transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    SourceCloud transformedCloudrgb(mSrcCloud1.rows(), 6) ;
	SourceCloud transformedCloudrgb2(mSrcCloud2.rows(), 6) ;
	SourceCloud transformedClouddepth(mSrcCloud3.rows(), 6) ;
	SourceCloud transformedClouddepth2(mSrcCloud4.rows(), 6) ;
	SourceCloud transformedCloudrgb_before20(mSrcCloud1_before20.rows(), 6) ;
	SourceCloud transformedCloudrgb2_before20(mSrcCloud2_before20.rows(), 6) ;
	SourceCloud transformedClouddepth_before20(mSrcCloud3_before20.rows(), 6) ;
	SourceCloud transformedClouddepth2_before20(mSrcCloud4_before20.rows(), 6) ;
	/*SourceCloud transformedCloudrgb(srcCloud.rows(), 6) ;
	SourceCloud transformedCloudrgb2(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth2(srcCloud.rows(), 6) ;*/
	if(Useedgeaware==0)
	{
		if(iteration_loop2<=iteration_divide)
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud1_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud1_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud2_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud3_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud3_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud4_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud4_before20.rightCols<3>().transpose()).transpose();
		
		/*//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb_before20.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2_before20.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth_before20.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2_before20.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2_before20.rightCols<3>().transpose()).transpose();
		*/}	
		
		else
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
		
		/*//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2.rightCols<3>().transpose()).transpose();
		*/}
	}
	else
	{
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
		
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
    }

	TargetCloud LastCloud_1 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_2 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_3 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_4 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloudOri = ComputeCurrentCloud(last_depth);

	int cnttrans1, cnttrans2, cnttrans3, cnttrans4;
	cnttrans1 = 0;
	cnttrans2 = 0;
	cnttrans3 = 0;
	cnttrans4 = 0;
    int meanweight1, meanweight2, meanweight3, meanweight4;
    double randomx_thres = 0.1;
	if(Useweigtedrandomsampling==1)
	{
	  randomx_thres = 100;
	}
   
    Eigen::Vector<Scalar, 6> rt6D;

    
	SourceCloud transformedCloudOri(srcCloud.rows(), 6) ;
    transformedCloudOri.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    transformedCloudOri.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
	if (!MatchingByProject2DAndWalk(transformedCloudOri, tgtCloud)) {
		break; // when correspondence size less than 6
		}
	/*
	if(iteration_loop2 == 1){
		RANSAC_strategy(transformedCloudOri, tgtCloud);
	}
	*/
	std::cout<<"\ncorrs: "<<corrs.rows();
	//rt6D = rt6D1;
	rt6D = MinimizingP2PLErrorMetric(transformedCloudOri(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
	//rt6D = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudOri(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeights(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3, LastCloudOri);
    Transform iterRtSE3;
	Eigen::Vector3f Pre_trans(rt6D(0),rt6D(1),rt6D(2));

    iterRtSE3 = ConstructSE3(rt6D);

    //chain iterRtSE3 to rtSE3
    //rtSE3 = iterRtSE3 * rtSE3;
	std::cout<<iterRtSE3.col(3).head<3>().transpose()<<"\n";
	std::cout<<"\npre: "<<Pre_trans;
	std::cout<<"\nnorm: "<<Pre_trans.norm();
	if(Pre_trans.norm() > 0.3){
		rtSE3 = rtSE3;
	}
	else{
		rtSE3 = rtSE3 * iterRtSE3;
	}
    //check termination
    /*if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }*/
	max_iters = 20;
	if (iteration_loop2 > max_iters)
	{
		break;
	}
  }
  t2 = clock();
  std::cout<<"\nMeta Testing time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
  valid = true;
  return rtSE3;
}
void EAS_ICP::EdgeDetection(const Cloud& cloud, cv::Mat& edge_mat, cv::Mat& nan_mat, cv::Mat& rej_mat) {
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> (cloud.data()+2, cloud.rows());
  edge_mat= cv::Mat::zeros(height, width, CV_8UC1);
  nan_mat = cv::Mat::zeros(height, width, CV_8UC1);
  rej_mat = cv::Mat::zeros(height, width, CV_8UC1);
  uchar* edge_map = edge_mat.data,
           *nan_map = nan_mat.data,
           *rej_map = rej_mat.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map[i*width + j] = 255;
	    }
    }
  }
  for (int i = 0; i < height; i+=stride) {
    int last_pixel_index = -1;
    double last_pixel_z = -1;
    for (int j = 0; j < width; j+=stride) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        continue;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
          continue;
	    }
      if (last_pixel_index >= 0) {
        double pixel_min = std::min(last_pixel_z, point_z);
        double threshold = edge_threshold* pixel_min * (abs(j - last_pixel_index));
        if (fabs(point_z - last_pixel_z) > threshold)
        {
			if(Useonlyforeground==1)
			{
				if(edge_map[i*width + last_pixel_index]>edge_map[i*width + j])
				{
					edge_map[i*width + j] = 255;
				}
				else
				{
					edge_map[i*width + last_pixel_index] = 255;
				}
			}
			else
			{
				edge_map[i*width + j] = 255;
				edge_map[i*width + last_pixel_index] = 255;
			}
        }
      }
      last_pixel_index = j;
      last_pixel_z = point_z;
    }
  }

  for (int j = 0; j < width; j+=stride) {
    int last_pixel_index = -1;
    double last_pixel_z = -1;
    for (int i = 0; i < height; i+=stride) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        continue;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
          continue;
	    }
      if (last_pixel_index >= 0) {
        double pixel_min = std::min(last_pixel_z, point_z);
        double threshold = edge_threshold* pixel_min * (abs(i - last_pixel_index));
        if (fabs(point_z - last_pixel_z) > threshold)
        {
			if(Useonlyforeground==1)
			{
				if(edge_map[last_pixel_index*width + j]>edge_map[i*width + j])
				{
					edge_map[i*width + j] = 255;
				}
				else
				{
					edge_map[last_pixel_index*width + j] = 255;
				}
			}		
			else
			{
				edge_map[i*width + j] = 255;
				edge_map[last_pixel_index*width + j] = 255;
			}
        }
      }
      last_pixel_index = i;
      last_pixel_z = point_z;
    }
  }
  
}
void EAS_ICP::GeometryWeightingFunction(const cv::Mat& x, cv::Mat& out) {
  std::vector<double> geometry_weighting_coeff{4.55439848e-03, 7.06670913e+00, 1.08456189e+00};
  cv::Mat powmat;
  cv::pow(geometry_weighting_coeff[1] - x, 2.0, powmat);
  out = 1/(geometry_weighting_coeff[0] * powmat + geometry_weighting_coeff[2]);
}

void EAS_ICP::PointRejectionByDepthRangeAndGeometryWeight( const Cloud& cloud, const cv::Mat& edgeDistanceMat, const cv::Mat& rej_mat, std::vector<int>& remindPointIndexes, std::vector<double>& weights){
  cv::Mat geo_weight;
  GeometryWeightingFunction(edgeDistanceMat, geo_weight);
  float* fw_ptr = (float*)geo_weight.data;
  uchar* rej_ptr = (uchar*)rej_mat.data;
  int size = edgeDistanceMat.rows*edgeDistanceMat.cols*edgeDistanceMat.channels();
  remindPointIndexes.resize(size);
  weights.resize(size);
  int cnt = 0;
  for (int i = 0; i < size; ++i) {
    if ((rej_ptr[i] == 0)) {
        remindPointIndexes[cnt] = i;
        weights[cnt] = (fw_ptr[i]);
        ++cnt;
    }
  }
  weights.resize(cnt);
  remindPointIndexes.resize(cnt);
}

void EAS_ICP::PointRejectionByDepthRangeAndGeometryWeight2( const Cloud& cloud, const cv::Mat& edgeDistanceMat, const cv::Mat& rej_mat, std::vector<int>& remindPointIndexes, std::vector<double>& weights){
   cv::Mat geo_weight;
   //GeometryWeightingFunction(edgeDistanceMat, geo_weight);
   //float* fw_ptr = (float*)geo_weight.data;
   uchar* rej_ptr = (uchar*)rej_mat.data;
   int size = edgeDistanceMat.rows*edgeDistanceMat.cols*edgeDistanceMat.channels();
   uchar edgemap [size];
   int edge_cnt = 0;
   for(int u=0;u<edgeDistanceMat.rows;++u) 
   {
       for(int i=0;i<edgeDistanceMat.cols;++i)
       {
	        edgemap[edge_cnt]=edgeDistanceMat.at<uchar>(u, i);
			edge_cnt++;
	   }
   }
   remindPointIndexes.resize(size);
   weights.resize(size);
   int cnt = 0;
   /*cv::Mat ROIrejectmap = cv::Mat::ones(480, 640, CV_8UC1);
   	for(int u=0;u<edgeDistanceMat.rows;u++) 
    {
       for(int i=0;i<edgeDistanceMat.cols;i++)
       {
		   	//std::cout<<"debug"<<std::endl;
	        ROIrejectmap.at<uchar>(u, i)=edgeDistanceMat.at<uchar>(u, i);
		
	   }
    }*/
	
   for (int i = 0; i < size; ++i) {
     if ((rej_ptr[i] == 0) && (edgemap[i] > 0)) 
	 {
          remindPointIndexes[cnt] = i;
          weights[cnt] = 1;
          ++cnt;
     }
	 /*else
	 {
		int x = i%(640);
		int y = i/(640);
		ROIrejectmap.at<uchar>(y, x) = 0;
	 }*/
    }
    weights.resize(cnt);
    remindPointIndexes.resize(cnt);
	
}

void EAS_ICP::WeightedRandomSampling(int sampling_size, const std::vector<int>& roiInds, const std::vector<double>& weights, std::vector<int>& samplingInds) {
  //std::cout<<"\nHELLLO\n";
  //std::cout<<"\nN:"<<roiInds.size();
  size_t N = roiInds.size();  
  if (sampling_size >= N)
  {
    samplingInds = roiInds;
  } else {
    samplingInds.resize(sampling_size);
    
    std::default_random_engine gen; 
    gen.seed(random_seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Algorithm S
    size_t i = 0;
    size_t index = 0;
    std::vector<bool> added;
    size_t n = sampling_size;
    double sum = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(weights.data(), N).sum();
    while (n > 0)
    {
      // Step 1: [Generate U.] Generate a random variate U that is uniformly distributed between 0 and 1.
      //const float U = rand()/ double(RAND_MAX);
      const float U = dis(gen);
      double prob = weights[index] /sum;
	  double randomx = (double) rand() / (RAND_MAX + 1.0);
      // std::csamplingInds << norm_test << std::endl;
      // Step 2: [Test.] If N * U > n, go to Step 4. 
      //if (N <= n) {
      //  samplingInds[i++] = roiInds[index];
      //  --n;
      //} else
       if ((randomx) <= n*(prob)) {
        samplingInds[i++] = roiInds[index];
        --n;
      }
      --N;
      sum = sum - weights[index];
      ++index;
    }
  }
}

void EAS_ICP::CalculateNormal( const Cloud& cloud,  const std::vector<int>& samplingInds, SourceCloud& srcCloud){

  const int normal_step = 2;
  const int normal_range = 3;
  int size = samplingInds.size();
  int low_bound= - normal_range;
  int up_bound= normal_range + 1;
  std::vector<int> validNormalInds;
  validNormalInds.reserve(size);
  Cloud normals(size, 3);
  typedef PointXYZ PointT;
  for (int i = 0; i < size; ++i) {
    int x = samplingInds[i] % width;
    int y = samplingInds[i] / width;
    std::vector<PointT> points;
    //reserve range size memory
    points.reserve(std::pow(2*normal_range+1, 2));
    //center epoint
    PointT p (
            cloud(samplingInds[i], 0),
            cloud(samplingInds[i], 1),
            cloud(samplingInds[i], 2)
        );
    for (int ix = low_bound; ix < up_bound; ++ix){
      for (int iy = low_bound; iy < up_bound; ++iy)
      {
        int x_ = x + ix*normal_step;
        int y_ = y + iy*normal_step;
        if ( (x_ >= width)
          || (y_ >= height)
          || (x_ < 0)
          || (y_ < 0) )
        {
          continue;
        }

        // use non-sampled point cloud points
        const PointT  original_p(
            cloud(y_ * width + x_, 0),
            cloud(y_ * width + x_, 1),
            cloud(y_ * width + x_, 2)
            );

        //check nan
        if(original_p.z != original_p.z){
          continue;
        }

        // skip the larger normal (cloud be noise)
        if (fabs(original_p.z - p.z) > normal_step*3*p.z/ fx )
        {
          continue;
        }
        // pick the points
        points.push_back(original_p);
      }
    }
    // compute the normal of ref points
    Eigen::Vector4f plane_param_out;
    float curvature_out;
    computePointNormal(points, plane_param_out, curvature_out);

    // correct the normal, set the view point at (0,0,0)
    flipNormalTowardsViewpoint<PointT, float>(p, 0, 0, 0, plane_param_out);
    normals(i, 0) = plane_param_out[0];
    normals(i, 1) = plane_param_out[1];
    normals(i, 2) = plane_param_out[2];
    if (!plane_param_out.hasNaN()) {
      validNormalInds.push_back(i);
    }
  }
  //std::cout<<"\nvalid size: :"<<validNormalInds.size();
  if(validNormalInds.size() == 0){
	  SourceCloud tmp(1, 6);
	  tmp.row(0)<<0,0,0,0,0,0;
	  tmp.swap(srcCloud); 
	  return;
  }
  //construct Source Cloud
  std::vector<int> validSrcCloudInds(validNormalInds.size());
  
  for (int i = 0; i < validNormalInds.size(); ++i) {
    validSrcCloudInds[i] = samplingInds[validNormalInds[i]];
	
  }
 
  SourceCloud tmp(validSrcCloudInds.size(), 6); 
  tmp.leftCols(3) = cloud(validSrcCloudInds, Eigen::all);
  tmp.rightCols(3) = normals(validNormalInds, Eigen::all);
  
  tmp.swap(srcCloud);
}
const EAS_ICP::SourceCloud& EAS_ICP::EdgeAwareSampling(const Cloud& cloud) {
  //sampling
  //edge detection
  cv::Mat edge_map;
  cv::Mat nan_map;
  cv::Mat rej_map;
  clock_t t_e1, t_e2;
  t_e1 = clock();
  EdgeDetection(cloud, edge_map, nan_map, rej_map);
  t_e2 = clock();
  std::cout<<"edge detection cloud"<<std::endl;
  //std::cout<<"\nedge MEAN: "<<cv::mean(edge_map).val[0]<<"\n";
  if(cv::mean(edge_map).val[0] == 0){
  	edge_map = (~nan_map) & (~rej_map);
  }
  std::cout<<double(t_e2-t_e1)/CLOCKS_PER_SEC<<std::endl;
  
  //calculate edge distance map
  cv::Mat inv_edge_map;
  inv_edge_map = ~edge_map;
  
  cv::Mat edge_distance_map;
  clock_t t_d1, t_d2;
  t_d1 = clock();
  cv::distanceTransform(inv_edge_map, edge_distance_map, cv::DIST_L2, 5);
  t_d2 = clock();
  std::cout<<"distance_transform cloud"<<std::endl;
  std::cout<<double(t_d2-t_d1)/CLOCKS_PER_SEC<<std::endl;
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::cout<<"Before Point Reject\n";
  PointRejectionByDepthRangeAndGeometryWeight(cloud, edge_distance_map, nan_map | rej_map, remindPointInds, weights);
  //input:edge_distance_map, output:weights set as 1
  std::cout<<"After Point Reject\n";
  //PointRejectionByDepthRangeAndGeometryWeight2(cloud, edge_map, nan_map | rej_map, remindPointInds, weights);

  //cv::imwrite("edge.jpg", edge_map);
  //sample depend on edge distance
  std::vector<int> EASInds;
  std::cout<<"Before Weight Random Sampling\n";
  WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
  std::cout<<"After Weight Random Sampling\n";
  //calculate normal
  std::cout<<"Before Calculate Normal\n";
  CalculateNormal(cloud, EASInds, mSrcCloud);
  std::cout<<"After Calculate Normal\n";
  //weighting
  kinectNoiseWeights = KinectNoiseWighting(mSrcCloud);
  
  return mSrcCloud;
}
const EAS_ICP::SourceCloud& EAS_ICP::JustSampling(const Cloud& cloud) {
  //sampling
  //edge detection
  cv::Mat edge_map;
  cv::Mat nan_map;
  cv::Mat rej_map;
  clock_t t_e1, t_e2;
  t_e1 = clock();
  EdgeDetection(cloud, edge_map, nan_map, rej_map);
  //edge_map.setTo(255, cv::abs(edge_map) > 0);
  //edge_map = cv::Mat::zeros(480, 640, CV_8UC1);
  //edge_map.setTo(255, cv::abs(edge_map) > 0);
  edge_map = (~nan_map) & (~rej_map);
  t_e2 = clock();
  //cv::imwrite("just_depth.jpg", edge_map);
  std::cout<<"\nedge MEAN: "<<cv::mean(edge_map).val[0];
  std::cout<<"edge detection cloud"<<std::endl;
  std::cout<<double(t_e2-t_e1)/CLOCKS_PER_SEC<<std::endl;
  
  //calculate edge distance map
  cv::Mat inv_edge_map;
  inv_edge_map = ~edge_map;
  
  cv::Mat edge_distance_map;
  clock_t t_d1, t_d2;
  t_d1 = clock();
  cv::distanceTransform(inv_edge_map, edge_distance_map, cv::DIST_L2, 5);
  t_d2 = clock();
  std::cout<<"distance_transform cloud"<<std::endl;
  std::cout<<double(t_d2-t_d1)/CLOCKS_PER_SEC<<std::endl;
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::cout<<"Before Point Reject\n";
  PointRejectionByDepthRangeAndGeometryWeight(cloud, edge_distance_map, nan_map | rej_map, remindPointInds, weights);
  //input:edge_distance_map, output:weights set as 1
  std::cout<<"After Point Reject\n";
  //PointRejectionByDepthRangeAndGeometryWeight2(cloud, edge_map, nan_map | rej_map, remindPointInds, weights);

  //cv::imwrite("edge.jpg", edge_map);
  //sample depend on edge distance
  std::vector<int> EASInds;
  std::cout<<"Before Weight Random Sampling\n";
  WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
  std::cout<<"After Weight Random Sampling\n";
  //calculate normal
  std::cout<<"Before Calculate Normal\n";
  CalculateNormal(cloud, EASInds, mSrcCloud);
  std::cout<<"After Calculate Normal\n";
  //weighting
  kinectNoiseWeights = KinectNoiseWighting(mSrcCloud);
  
  return mSrcCloud;
}

//EdgeAwareSampling  cloud& rgb 
const EAS_ICP::SourceCloud& EAS_ICP::EdgeAwareSampling(const Cloud& cloud, const cv::Mat& rgb) {
  //sampling
  //edge detection
  cv::Mat edge_map;
  cv::Mat nan_map;
  cv::Mat rej_map;
  clock_t t_e1, t_e2;
  t_e1 = clock();
  EdgeDetection(cloud, edge_map, nan_map, rej_map);
  t_e2 = clock();
  std::cout<<"edge detection"<<std::endl;
  std::cout<<double(t_e2-t_e1)/CLOCKS_PER_SEC<<std::endl;
  //cv::imwrite("edge_map2021.png", edge_map);  
  //canny edge
  //read rgb files
  
  cv::Mat src;
  src = rgb;
  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::resize(src, out_resized, cv::Size(80, 60), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  clock_t t_ca1, t_ca2;
  t_ca1 = clock();
  cv::Canny(blurred, canny_edge_map, 100, 150);
  t_ca2 = clock();
  std::cout<<"canny_edge_detection"<<std::endl;
  std::cout<<double(t_ca2-t_ca1)/CLOCKS_PER_SEC<<std::endl;
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  
  //
  int count_edge_points = 0;
  for(int r =0;r<480;r++){
    for(int c=0;c<640;c++){
      if(edge_map.ptr<uchar>(r)[c]!=0){
        count_edge_points++;
      }
    
    }
  
  }
  std::cout<<"points"<<std::endl;
  std::cout<<count_edge_points<<std::endl;
  
  int count_canny_edge_points = 0;
  for(int r =0;r<480;r++){
   for(int c=0;c<640;c++){
     if(edge_map2.ptr<uchar>(r)[c]!=0){
        count_canny_edge_points++;
      }
         
      }
       
   }
  std::cout<<"canny_points"<<std::endl;
  std::cout<<count_canny_edge_points<<std::endl;
  //dilation
  
  cv::dilate(edge_map2, edge_map2, cv::Mat());
  
  //cv::imwrite("before_add.png", edge_map);
  //cv::bitwise_or(edge_map, edge_map2, edge_map);
  //cv::imwrite("after_add.png", edge_map);
  //calculate edge distance map
  cv::Mat inv_edge_map;
  inv_edge_map = ~edge_map;
  
  cv::Mat edge_distance_map;
  clock_t t_d1, t_d2;
  t_d1 = clock();
  cv::distanceTransform(inv_edge_map, edge_distance_map, cv::DIST_L2, 5);
  t_d2 = clock();
  std::cout<<"distance_transform"<<std::endl;
  std::cout<<double(t_d2-t_d1)/CLOCKS_PER_SEC<<std::endl;
  
   cv::Mat inv_edge_map2;
   inv_edge_map2 = ~edge_map2;
   cv::Mat edge_distance_map2;
   clock_t t_d11, t_d21;
   t_d11 = clock();
   cv::distanceTransform(inv_edge_map2, edge_distance_map2, cv::DIST_L2, 5);
   t_d21 = clock();
   std::cout<<"canny_distance_transform"<<std::endl;
   std::cout<<double(t_d21-t_d11)/CLOCKS_PER_SEC<<std::endl;
   
  
  
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  //PointRejectionByDepthRangeAndGeometryWeight(cloud, edge_distance_map, nan_map | rej_map, remindPointInds, weights);
  //input:edge_distance_map, output:weights set as 1
  PointRejectionByDepthRangeAndGeometryWeight2(cloud, edge_map, nan_map | rej_map, remindPointInds, weights);
  //sample depend on edge distance
  WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
  //calculate normal
  CalculateNormal(cloud, EASInds, mSrcCloud);
  //weighting
  kinectNoiseWeights = KinectNoiseWighting(mSrcCloud);
  
  return mSrcCloud;
}
//EdgeAwareSampling rgb 
const EAS_ICP::SourceCloud& EAS_ICP::EdgeAwareSampling(const cv::Mat& rgb) {
  //sampling
  //edge detection
  //cv::Mat edge_map;
  cv::Mat nan_map;
  cv::Mat rej_map;

  //EdgeDetection(cloud, edge_map, nan_map, rej_map);
  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(rgb));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::resize(src, out_resized, cv::Size(80, 60), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  std::cout<<"canny_edge_detection"<<std::endl;
  
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  
  

  
  //count canny edge points
  int count_canny_edge_points = 0;
  for(int r =0;r<480;r++){
   for(int c=0;c<640;c++){
     if(edge_map2.ptr<uchar>(r)[c]!=0){
        count_canny_edge_points++;
      }
         
      }
       
   }
  std::cout<<"canny_points"<<std::endl;
  std::cout<<count_canny_edge_points<<std::endl;
  //dilation
  
  cv::dilate(edge_map2, edge_map2, cv::Mat());
  
  //calculate edge distance map
  
   cv::Mat inv_edge_map2;
   inv_edge_map2 = ~edge_map2;
   cv::Mat edge_distance_map2;
   cv::distanceTransform(inv_edge_map2, edge_distance_map2, cv::DIST_L2, 5);
   std::cout<<"canny_distance_transform"<<std::endl;
  
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;

  PointRejectionByDepthRangeAndGeometryWeight(*keycloud, edge_distance_map2, nan_map | rej_map, remindPointInds, weights);
  //input:edge_distance_map, output:weights set as 1
  //PointRejectionByDepthRangeAndGeometryWeight2(cloud, edge_distance_map2, nan_map | rej_map, remindPointInds, weights);
  //sample depend on edge distance
  WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
  //calculate normal
  CalculateNormal(*keycloud, EASInds, mSrcCloud1);
  //weighting
  kinectNoiseWeights = KinectNoiseWighting(mSrcCloud1);
  return mSrcCloud1;
}
EAS_ICP::CurrentCloud EAS_ICP::ComputeCurrentCloud(const cv::Mat& rgb) {
  
  CurrentCloud ret(pixelSize, 3);
  for (int n = 0; n < height; ++n) {
    for (int m = 0; m < width; ++m) {
      int i = n*width +m;
      Scalar z=0;
      if (rgb.type() == CV_32FC1) {
         // z = ((float*)depth.data)[i]/ mDepthMapFactor;
         z = 5000.0;
      } else {
         z = ((int16_t*)rgb.data)[i]/ mDepthMapFactor;
      }
      if (z == 0) {
        for (int j = 0; j < 3; ++j) {
          ret(i, j) = std::numeric_limits<Scalar>::quiet_NaN();
        }
      } else {
        ret(i, 2) = z;
        Scalar x = (Scalar)m;
        Scalar y = (Scalar)n;
        ret(i, 0) = z*(x-cx) / fx;
        ret(i, 1) = z*(y-cy) / fy;
      }
    }
  }
  return ret;
}
bool EAS_ICP::MatchingByProject2DAndWalk(const SourceCloud& srcCloud, const TargetCloud& tgtCloud) {
  int size = srcCloud.rows();
  //correspondence declare
  std::vector<std::tuple<Scalar, int, int>> stdCorrs; 
  std::vector<double> residual_vector;

  int corr_cnt= 0;

  //for all correspondence
  for (int i = 0; i < size; ++i) {
    //a source point
    const Scalar& src_px = srcCloud(i, 0);
    const Scalar& src_py = srcCloud(i, 1);
    const Scalar& src_pz = srcCloud(i, 2);

    // project to 2D target frame
    int x_warp = fx / src_pz*src_px +cx;
    int y_warp = fy / src_pz*src_py +cy;
    //declare and initial variables
    Scalar min_distance = std::numeric_limits<Scalar>::max();
    int target_index = -1;

    //check the 2D point in target frame range
    if (x_warp >= width || y_warp >= height || x_warp < 0 || y_warp < 0)
    {
      continue;
    }

    //search range
    for (int ix = -search_range; ix < search_range + 1 ; ++ix)
    {
      for (int iy = -search_range; iy < search_range + 1 ; ++iy)
      {
        // search a circle range
        int grid_distance2 = ix*ix + iy*iy;
        if (grid_distance2 > search_range* search_range)
        {
          continue;
        }
        // x index and y index of target frame
        int x_idx = x_warp + ix * search_step;
        int y_idx = y_warp + iy * search_step;

        // avoid index out of target frame
        if (x_idx >= (width)
          || x_idx < 0
          || y_idx >=height
          || y_idx < 0)
        {
          continue;
        }

        //calculate 1D target frame index
        int tmp_index = (y_idx * width + x_idx);

        // get x,y,z of target point
        double tgt_px = tgtCloud(tmp_index,0);
        double tgt_py = tgtCloud(tmp_index,1);
        double tgt_pz = tgtCloud(tmp_index,2);

        //check nan
        if(
             (tgt_px!= tgt_px)||
             (tgt_py!= tgt_py)||
             (tgt_pz!= tgt_pz)
          ) continue; 


        //calculate the distance between source point and target point
        double distance = sqrt((src_px - tgt_px)*(src_px - tgt_px)
          + (src_py - tgt_py)*(src_py - tgt_py)
          + (src_pz - tgt_pz)*(src_pz - tgt_pz));

        // if new distance is less than min distance => record this index and distance
        if (distance < min_distance)
        {
          min_distance = distance;
          target_index = tmp_index;// target index: height x width x pointsize //pointsize is 6
        }
      }
    }
    //image boundary rejection
    //check closet point whether in the margin of boundary
    int target_x = target_index % width;
    int target_y = target_index / width;
    if (target_x > right_bound|| target_x < left_bound || target_y > bottom_bound || target_y < top_bound) {
      continue;
    }

    //check closet point existed and smaller fix threshold of rejection ==> if true, this pair is correspondence, and store in vector stdCorrs
    if ( min_distance !=  std::numeric_limits<Scalar>::max() && min_distance < fixed_threshold_rejection)
    {
      stdCorrs.push_back(std::make_tuple(min_distance, i, target_index));
      residual_vector.push_back(min_distance);
      ++corr_cnt;
    }
  }
  //dynamic rejction
  //calculate the real index from the ratio of dynamic threshold
  int dynamic_threshold_index = dynamic_threshold_rejection * residual_vector.size();

  //check the index is not over the vector size
  if (dynamic_threshold_index < residual_vector.size()) {
      //rejection theshold(unit:m)
      //calculate the real value corresponded the index
      std::nth_element(residual_vector.begin(), residual_vector.begin() + dynamic_threshold_index, residual_vector.end());
      float reject_distance_threshold = 0; 
      //check the vector is no empty, or segmentation fault would occur.
      if (residual_vector.size() > 0)
        reject_distance_threshold = residual_vector[dynamic_threshold_index];
      
      //erase the correspondence over the dynamic threshold
      stdCorrs.erase(std::remove_if(stdCorrs.begin(), stdCorrs.end(), [reject_distance_threshold](const std::tuple<double, int,int>& elem){ return std::get<0>(elem)>reject_distance_threshold;}), stdCorrs.end());
  }

  //change the type for meeting the function output requirement
  int final_size = stdCorrs.size();
  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> correspondences(final_size, 2);
  for (int i = 0; i < stdCorrs.size(); ++i) {
    correspondences(i, 0) = std::get<1>(stdCorrs[i]);
    correspondences(i, 1) = std::get<2>(stdCorrs[i]);
  }
  correspondences.swap(corrs);

  //check the correspondence size over 6 for solving 6 rt valuables
  return corrs.rows() >=6;
}

Eigen::Vector<EAS_ICP::Scalar, 6> EAS_ICP::MinimizingP2PLErrorMetric(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weights) {
  //solve Ax=b
  //calculate b
  auto b = (tgtCloud - srcCloud.leftCols(3)).transpose().cwiseProduct(srcCloud.rightCols(3).transpose()).colwise().sum().transpose();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_weighted = b.array() * weights.array();

  //calculate A
  Eigen::Matrix<Scalar, Eigen::Dynamic, 6, Eigen::RowMajor> A(tgtCloud.rows(), 6);
  for (int i = 0; i < tgtCloud.rows(); ++i) {
    Eigen::Vector3<Scalar> s = srcCloud.row(i).tail(3);
    A.row(i).head(3) = tgtCloud.row(i).cross(s);
  }
  A.rightCols(3) = srcCloud.rightCols(3);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_weighted = A.array().colwise() * weights.array();

  //solve x by svd
  auto _bdcSVD = A_weighted.bdcSvd(Eigen::ComputeThinU |Eigen:: ComputeThinV);
  Eigen::Vector<EAS_ICP::Scalar, 6> ret;
  ret = _bdcSVD.solve(b_weighted);

  //Accumulate sliding extent 
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 6, 6> > evalsolver (A_weighted.transpose()*A_weighted);
  auto norm_eval = evalsolver.eigenvalues()/evalsolver.eigenvalues().maxCoeff();
  std::vector<int> inds;
  for (int i = 0; i < norm_eval.size(); ++i) {
    if (norm_eval(i) < 1.0/thresEvalRatio) {
      inds.push_back(i);
    }
  }
  if (inds.size() > 0) {
    auto less_eval_vecs = evalsolver.eigenvectors()(Eigen::all, inds);
    double sliding_dist = (less_eval_vecs.transpose() * ret).norm();
    accSlidingExtent += sliding_dist;
  }
  //std::cout<<"result:\n"<<ret<<"\n";
  return ret;
}
Eigen::Vector<EAS_ICP::Scalar, 6> EAS_ICP::MinimizingP2PLErrorMetricGaussianNewton(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weights) {
  //solve Ax=b
  //calculate b
  //auto b = (tgtCloud - srcCloud.leftCols(3)).transpose().cwiseProduct(srcCloud.rightCols(3).transpose()).colwise().sum().transpose();
  	//std::cout<<"\nrows: "<<weights.rows();
	Eigen::Matrix<EAS_ICP::Scalar,Eigen::Dynamic, 7> JacoPre(tgtCloud.rows(), 7);
  	JacoPre.leftCols(3) = srcCloud.rightCols(3);
	//std::cout<<"\ntgt num: "<<tgtCloud.rows();
	//std::cout<<"\nsrc num: "<<srcCloud.rows();
	for (int i = 0; i < tgtCloud.rows(); ++i) {
    Eigen::Vector3<Scalar> s = srcCloud.row(i).head(3);
	Eigen::Vector3<Scalar> n = srcCloud.row(i).tail(3);
	Eigen::Vector3<Scalar> d = tgtCloud.row(i).tail(3);
	Eigen::Vector3<Scalar> dist = s-d;
    JacoPre.row(i).segment(3,3) = s.cross(n);
	JacoPre.row(i).tail(1)(0) = dist.dot(n);

	//std::cout<<" \ns:"<<s;
	//std::cout<<" \nd:"<<d;
	//std::cout<<" \nn:"<<n;
	//std::cout<<"\nJacoPre: "<<JacoPre.row(i);
  	}
	Eigen::Matrix<EAS_ICP::Scalar,6, 6,Eigen::RowMajor> A_icp = Eigen::MatrixXd::Zero(6, 6);
	Eigen::Matrix<EAS_ICP::Scalar, 6, 1> b_icp = Eigen::MatrixXd::Zero(6, 1);

	Eigen::Matrix<EAS_ICP::Scalar,6, 6,Eigen::RowMajor> Temp;
	Eigen::Matrix<EAS_ICP::Scalar, 6, 1> Temp_b;
	float sum = 0;
	int count = 0;
	for (int i = 0; i < tgtCloud.rows(); ++i) {
		double a,b,c,d,e,f,g;
		a = JacoPre.row(i)(0);
		b = JacoPre.row(i)(1);
		c = JacoPre.row(i)(2);
		d = JacoPre.row(i)(3);
		e = JacoPre.row(i)(4);
		f = JacoPre.row(i)(5);
		g = JacoPre.row(i)(6);

		Temp(0,0) = a * a;
		Temp(0,1) = Temp(1,0) = a * b;
		Temp(0,2) = Temp(2,0) = a * c;
		Temp(0,3) = Temp(3,0) = a * d;
		Temp(0,4) = Temp(4,0) = a * e;
		Temp(0,5) = Temp(5,0) = a * f;

		Temp(1,1) = b * b;
		Temp(1,2) = Temp(2,1) = b * c;
		Temp(1,3) = Temp(3,1) = b * d;
		Temp(1,4) = Temp(4,1) = b * e;
		Temp(1,5) = Temp(5,1) = b * f;

		Temp(2,2) = c * c;
		Temp(2,3) = Temp(3,2) = c * d;
		Temp(2,4) = Temp(4,2) = c * e;
		Temp(2,5) = Temp(5,2) = c * f;

		Temp(3,3) = d * d;
		Temp(3,4) = Temp(4,3) = d * e;
		Temp(3,5) = Temp(5,3) = d * f;

		Temp(4,4) = e * e;
		Temp(4,5) = Temp(5,4) = e * f;

		Temp(5,5) = f * f;

		Temp_b(0,0) = a * g;
		Temp_b(1,0) = b * g;
		Temp_b(2,0) = c * g;
		Temp_b(3,0) = d * g;
		Temp_b(4,0) = e * g;
		Temp_b(5,0) = f * g;
		float w = weights.row(i)(0);
		//w = 1; 
		A_icp += Temp * w * w;
		b_icp += Temp_b * w;
		sum += g * g;
		count++;
  	}
	Eigen::Vector<EAS_ICP::Scalar, 6> ret;
	ret = A_icp.ldlt().solve(b_icp);
	std::cout<<"sum:\n"<<sqrt(sum)/count<<"\n";
	//std::cout<<"b_icp:\n"<<b_icp<<"\n";
	/*
	std::cout<<"\n-------------------------\n"; 
	std::cout<<"src:\n"<<srcCloud.row(0)<<"\n";
	std::cout<<"tgt:\n"<<tgtCloud.row(0)<<"\n";
	std::cout<<"Jaco:\n"<<JacoPre.row(0)<<"\n";
	std::cout<<"A_icp:\n"<<A_icp<<"\n";
	std::cout<<"b_icp:\n"<<b_icp<<"\n";
	std::cout<<"result:\n"<<ret<<"\n";
	std::cout<<"\n-------------------------\n";
	*/
	//std::cout<<"A_icp:\n"<<A_icp<<"\n";
	//std::cout<<"resultGN:\n"<<ret<<"\n";
  	return ret;
}
Eigen::Vector<EAS_ICP::Scalar, 6> EAS_ICP::MinimizingP2PLErrorMetricGaussianNewtonRGB(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weights, const cv::Mat depth, const cv::Mat rgb, const cv::Mat rgb_last, Transform resultRt, TargetCloud& LastCloud){
  //solve Ax=b
  //calculate b
  //auto b = (tgtCloud - srcCloud.leftCols(3)).transpose().cwiseProduct(srcCloud.rightCols(3).transpose()).colwise().sum().transpose();
	std::cout<<"\niter:  "<<iterations<<"\n";
	//std::cout<<"\niter2:  "<<iteration_loop2<<"\n";
  	Eigen::Matrix<EAS_ICP::Scalar,Eigen::Dynamic, 7> JacoPre(tgtCloud.rows(), 7);
  	JacoPre.leftCols(3) = srcCloud.rightCols(3);

	for (int i = 0; i < tgtCloud.rows(); ++i) {
    Eigen::Vector3<Scalar> s = srcCloud.row(i).head(3);
	Eigen::Vector3<Scalar> n = srcCloud.row(i).tail(3);
	Eigen::Vector3<Scalar> d = tgtCloud.row(i).tail(3);
	Eigen::Vector3<Scalar> dist = s-d;
    JacoPre.row(i).segment(3,3) = s.cross(n);
	JacoPre.row(i).tail(1)(0) = dist.dot(n);
  	}
	Eigen::Matrix<EAS_ICP::Scalar,6, 6,Eigen::RowMajor> A_icp = Eigen::MatrixXd::Zero(6, 6);
	Eigen::Matrix<EAS_ICP::Scalar, 6, 1> b_icp = Eigen::MatrixXd::Zero(6, 1);

	Eigen::Matrix<EAS_ICP::Scalar,6, 6,Eigen::RowMajor> Temp;
	Eigen::Matrix<EAS_ICP::Scalar, 6, 1> Temp_b;

	for (int i = 0; i < tgtCloud.rows(); ++i) {
		double a,b,c,d,e,f,g;
		a = JacoPre.row(i)(0);
		b = JacoPre.row(i)(1);
		c = JacoPre.row(i)(2);
		d = JacoPre.row(i)(3);
		e = JacoPre.row(i)(4);
		f = JacoPre.row(i)(5);
		g = JacoPre.row(i)(6);

		Temp(0,0) = a * a;
		Temp(0,1) = Temp(1,0) = a * b;
		Temp(0,2) = Temp(2,0) = a * c;
		Temp(0,3) = Temp(3,0) = a * d;
		Temp(0,4) = Temp(4,0) = a * e;
		Temp(0,5) = Temp(5,0) = a * f;

		Temp(1,1) = b * b;
		Temp(1,2) = Temp(2,1) = b * c;
		Temp(1,3) = Temp(3,1) = b * d;
		Temp(1,4) = Temp(4,1) = b * e;
		Temp(1,5) = Temp(5,1) = b * f;

		Temp(2,2) = c * c;
		Temp(2,3) = Temp(3,2) = c * d;
		Temp(2,4) = Temp(4,2) = c * e;
		Temp(2,5) = Temp(5,2) = c * f;

		Temp(3,3) = d * d;
		Temp(3,4) = Temp(4,3) = d * e;
		Temp(3,5) = Temp(5,3) = d * f;

		Temp(4,4) = e * e;
		Temp(4,5) = Temp(5,4) = e * f;

		Temp(5,5) = f * f;

		Temp_b(0,0) = a * g;
		Temp_b(1,0) = b * g;
		Temp_b(2,0) = c * g;
		Temp_b(3,0) = d * g;
		Temp_b(4,0) = e * g;
		Temp_b(5,0) = f * g;

		A_icp += Temp;
		b_icp += Temp_b;
  	}
	//cv::Mat dIdx;
	//cv::Mat dIdy;
	A_term A_rgb;
	b_term b_rgb;
	//EAS_ICP::computeDerivativeImages(rgb, dIdx, dIdy);
	//std::cout<<"\nDerivativeImages time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
	EAS_ICP::RGBJacobianGet(dIdx, dIdy, depth, rgb, rgb_last, tgtCloud, resultRt ,A_rgb, b_rgb, LastCloud);
	

	Eigen::Vector<EAS_ICP::Scalar, 6> ret;
	Eigen::Vector<EAS_ICP::Scalar, 6> retRGB;
	Eigen::Vector<EAS_ICP::Scalar, 6> retAll;
	A_term A_all;
	b_term b_all;
	int wt = 10;

	A_rgb = A_rgb;
	b_rgb = b_rgb;
	float iterations_final = iterations/100;
	//if(iterations < 20){

	if(iterations < 20){
		iterations_final = 0;
		wt = 1;
	}
	else{
		//iterations_final = 1;
			//double quality_rate = depth_quality/rgb_quality;
			double quality_rate = depth_quality/rgb_quality;
			iterations_final = 0.0001 * rgb_quality;
		
		//wt = 1000;

			
			//wt = 10 * quality_rate;
			wt = 10;
			std::cout<<"\nquality rate: "<< depth_quality/rgb_quality<<"\n";
			if(depth_quality == 0 || rgb_quality == 0){
				wt = 10;
			}
			//wt = 10;
	}

	A_all = A_rgb * iterations_final * iterations_final + A_icp*wt*wt;
	b_all = b_rgb * iterations_final + b_icp*wt;
	//ret = A_icp.ldlt().solve(b_icp);
	retAll = A_all.ldlt().solve(b_all);
	//retAll = A_all.ldlt().solve(b_all);
	/*
	std::cout<<"\n-------------------------\n"; 
	std::cout<<"src:\n"<<srcCloud.row(0)<<"\n";
	std::cout<<"tgt:\n"<<tgtCloud.row(0)<<"\n";
	std::cout<<"Jaco:\n"<<JacoPre.row(0)<<"\n";
	std::cout<<"A_icp:\n"<<A_icp<<"\n";
	std::cout<<"b_icp:\n"<<b_icp<<"\n";
	std::cout<<"result:\n"<<ret<<"\n";
	std::cout<<"\n-------------------------\n";
	*/
	std::cout<<"b_icp:\n"<<b_icp * wt<<"\n";
	std::cout<<"b_rgb:\n"<<b_rgb * iterations_final<<"\n";
	//std::cout<<"resultICP:\n"<<ret<<"\n";
	//std::cout<<"resultRGB:\n"<<retRGB<<"\n";
	std::cout<<"resultAll:\n"<<retAll<<"\n";
  	return retAll;
}
//reference linear least-square Optimization of p2pl ICP
EAS_ICP::Transform EAS_ICP::ConstructSE3(const Eigen::Vector<Scalar, 6> rt){
  Transform ret = Transform::Identity();
  const Scalar & alpha = rt(0); const Scalar & beta = rt(1); const Scalar & gamma = rt(2);
  const Scalar & tx = rt(3);    const Scalar & ty = rt(4);   const Scalar & tz = rt(5);
  ret(0,0)= static_cast<Scalar> ( cos (gamma) * cos (beta));
  ret(0,1)= static_cast<Scalar> (-sin (gamma) * cos (alpha) + cos (gamma) * sin (beta) * sin (alpha));
  ret(0,2)= static_cast<Scalar> ( sin (gamma) * sin (alpha) + cos (gamma) * sin (beta) * cos (alpha));
  ret(1,0)= static_cast<Scalar> ( sin (gamma) * cos (beta));
  ret(1,1)= static_cast<Scalar> ( cos (gamma) * cos (alpha) + sin (gamma) * sin (beta) * sin (alpha));
  ret(1,2)= static_cast<Scalar> (-cos (gamma) * sin (alpha) + sin (gamma) * sin (beta) * cos (alpha));
  ret(2,0)= static_cast<Scalar> (-sin (beta));
  ret(2,1)= static_cast<Scalar> ( cos (beta) * sin (alpha));
  ret(2,2) = static_cast<Scalar> ( cos (beta) * cos (alpha));
  ret(0,3)= static_cast<Scalar> (tx);
  ret(1,3)= static_cast<Scalar> (ty);
  ret(2,3) = static_cast<Scalar> (tz);
  return ret;
}
EAS_ICP::Transform EAS_ICP::ConstructSE3_GN(const Eigen::Vector<Scalar, 6> rt){
  Transform ret = Transform::Identity();
  const Scalar & alpha = rt(3); const Scalar & beta = rt(4); const Scalar & gamma = rt(5);
  const Scalar & tx = rt(0);    const Scalar & ty = rt(1);   const Scalar & tz = rt(2);
  ret(0,0)= static_cast<Scalar> ( cos (gamma) * cos (beta));
  ret(0,1)= static_cast<Scalar> (-sin (gamma) * cos (alpha) + cos (gamma) * sin (beta) * sin (alpha));
  ret(0,2)= static_cast<Scalar> ( sin (gamma) * sin (alpha) + cos (gamma) * sin (beta) * cos (alpha));
  ret(1,0)= static_cast<Scalar> ( sin (gamma) * cos (beta));
  ret(1,1)= static_cast<Scalar> ( cos (gamma) * cos (alpha) + sin (gamma) * sin (beta) * sin (alpha));
  ret(1,2)= static_cast<Scalar> (-cos (gamma) * sin (alpha) + sin (gamma) * sin (beta) * cos (alpha));
  ret(2,0)= static_cast<Scalar> (-sin (beta));
  ret(2,1)= static_cast<Scalar> ( cos (beta) * sin (alpha));
  ret(2,2) = static_cast<Scalar> ( cos (beta) * cos (alpha));
  ret(0,3)= static_cast<Scalar> (tx);
  ret(1,3)= static_cast<Scalar> (ty);
  ret(2,3) = static_cast<Scalar> (tz);
  return ret.inverse();
}
bool EAS_ICP::CheckConverged(const Eigen::Vector<Scalar, 6>& rt6D) {
    return (sqrt(rt6D.head(3).array().pow(2).sum())  < icp_converged_threshold_rot )&&(sqrt(rt6D.tail(3).array().pow(2).sum())  < icp_converged_threshold_trans );
  
}
bool EAS_ICP::CheckConverged_GN(const Eigen::Vector<Scalar, 6>& rt6D) {
    return (sqrt(rt6D.tail(3).array().pow(2).sum())  < icp_converged_threshold_rot )&&(sqrt(rt6D.head(3).array().pow(2).sum())  < icp_converged_threshold_trans );
  
}

void EAS_ICP::RGBJacobianGet(const cv::Mat& dIdx, const cv::Mat& dIdy, const cv::Mat depth, const cv::Mat rgb, const cv::Mat rgb_last, const TargetCloud& tgtCloud, const Transform resultRt, A_term& A_rgb, b_term& b_rgb, TargetCloud& LastCloud){
	A_rgb = Eigen::MatrixXd::Zero(6, 6);
	b_rgb = Eigen::MatrixXd::Zero(6, 1);
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
	cv::Mat inten;
	cv::Mat inten_last;
	cv::Mat hsv;
	cv::Mat hsv_last;
	cv::cvtColor(rgb, inten_last, cv::COLOR_BGR2GRAY);
	cv::cvtColor(rgb_last, inten, cv::COLOR_BGR2GRAY);
	cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(rgb_last, hsv_last, cv::COLOR_BGR2HSV);
	K(0, 0) = fx;//fx
	K(1, 1) = fy;//fy
	K(0, 2) = cx;//cx
	K(1, 2) = cy;//cy
	K(2, 2) = 1;
	//Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt;

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();


	Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
	Kt = K * Kt;

	//std::cout<<"\nt:\n"<<K.inverse() * Kt;
	//std::cout<<"\nKRKinv:\n"<<KRK_inv;
	//std::cout<<"\nKRKinv(1,0):\n"<<KRK_inv(1,0);
	//std::cout<<"\nRt:\n"<<Rt;
	//std::cout<<"\nR:\n"<<R;
	float total = 0;
	A_term Temp;
	b_term Temp_b;
	int count = 0;
	for (int y = 0; y < 480; ++y){
		for (int x = 0; x<640; ++x){
			//float d1 = tgtCloud.row(640*y + x)(2) * 5000;
			float d1 = (float)next_depth.at<short>(y, x) / 1000;

			for(int u = std::max(y - 2, 0); u < std::min(x + 2, 480); u++)
			{
				for(int v = std::max(x - 2, 0); v < std::min(x + 2, 640); v++)
				{
					if((float)inten.at<uchar>(y, x) <= 0){
						continue;
					}
				}
			}

			if(!isnan(d1) && d1 > 0){
				float transformed_d1 = (float)(d1 * (KRK_inv(2,0) * x + KRK_inv(2,1) * y + KRK_inv(2,2)) + Kt(2));
				int u0 = round((d1 * (KRK_inv(0,0) * x + KRK_inv(0,1) * y + KRK_inv(0,2)) + Kt(0)) / transformed_d1);
				int v0 = round((d1 * (KRK_inv(1,0) * x + KRK_inv(1,1) * y + KRK_inv(1,2)) + Kt(1)) / transformed_d1);
				//std::cout<<"\nd1:  "<<d1;
				//std::cout<<"\ntrans d1:  "<<transformed_d1;
				
				
				//std::cout<<"\nd2_trans:  "<<transformed_d2;
				//float d0 = LastCloud.row(640*v0 + u0)(2) * 1000;
				//float d0 = (float)last_depth.at<short>(v0, u0) / 1000;

				short valX = dIdx.at<short>(y, x);
				short valY = dIdy.at<short>(y, x);
				float mTwo = valX * valX + valY * valY;
				
				//std::cout<<"\nValx: "<<valX<<" Valy: "<<valY;
				//std::cout<<"\nmTwo: "<<mTwo;
				
				//if(u0>0 && u0<640 && v0<480 && v0>0 && !isnan(d1) && d0>0 && std::abs(transformed_d1-d0)<40 && mTwo>1600){
				if(u0 >= 0 && v0 >= 0 && u0 < 640 && v0 < 480){

					float d0 = (float)last_depth.at<short>(v0, u0) / 1000;

					if(d0>0 && std::abs(transformed_d1-d0)<=0.07 && (inten_last.at<uchar>(v0, u0) != 0) && (mTwo > 1600) && (inten.at<uchar>(y, x) > 0)){
						//std::cout<<"\nd0:  "<<d0;
						//std::cout<<"\nd1_trans:  "<<transformed_d1;
						//std::cout<<"\n(x, y): "<<"("<<x<<", "<<y<<")";
						//std::cout<<"\n(u0, v0): "<<"("<<u0<<", "<<v0<<")";
						//std::cout<<"\nd1:  "<<d1;
						//std::cout<<"\nd0:  "<<d0;
						
						float t = (float)inten.at<uchar>(y, x);
						float s = (float)inten_last.at<uchar>(v0, u0);
						//std::cout<<"\nintens target: "<<t;
						//std::cout<<"\nintens source: "<<s;
						int alfa = (int)hsv.at<cv::Vec3b>(y, x)[0];
						int beta = (int)hsv_last.at<cv::Vec3b>(v0, u0)[0];

						float diff = t-s;
						//int diff = alfa - beta;
						total += (diff*diff);
						count++;
					}
				}
			}
		}
	}
	float delta = std::sqrt((float)total/count == 0 ? 1 : count);
	
	std::cout<<"\ncount: "<<count;
	std::cout<<"\ntotal: "<<total;
	std::cout<<"\ndelta: "<<delta;
	
	total = 0;
	count = 0;
	for (int y = 0; y < 480; ++y){
		for (int x = 0; x<640; ++x){
			//float d1 = tgtCloud.row(640*y + x)(2) * 5000;

			//condition check


			float d1 = (float)next_depth.at<short>(y, x) / 1000;
			if(!isnan(d1) && d1 > 0){
				float transformed_d1 = (float)(d1 * (KRK_inv(2,0) * x + KRK_inv(2,1) * y + KRK_inv(2,2)) + Kt(2));
				int u0 = round((d1 * (KRK_inv(0,0) * x + KRK_inv(0,1) * y + KRK_inv(0,2)) + Kt(0)) / transformed_d1);
				int v0 = round((d1 * (KRK_inv(1,0) * x + KRK_inv(1,1) * y + KRK_inv(1,2)) + Kt(1)) / transformed_d1);
				//std::cout<<"\nd1:  "<<d1;
				//std::cout<<"\nd2:  "<<d2;
				
				short valX = dIdx.at<short>(y, x);
				short valY = dIdy.at<short>(y, x);
				float mTwo = valX * valX + valY * valY;
				//std::cout<<"\nd2_trans:  "<<transformed_d2;
				//float d0 = LastCloud.row(640*v0 + u0)(2) * 1000;
				//float d0 = (float)last_depth.at<short>(v0, u0);
				
				//if(u0>0 && u0<640 && v0<480 && v0>0 && !isnan(d1) && d0>0 && std::abs(transformed_d1-d0)<70 && mTwo > 1600){
				if(u0 >= 0 && v0 >= 0 && u0 < 640 && v0 < 480){
					float d0 = (float)last_depth.at<short>(v0, u0) / 1000;
					if(d0>0 && std::abs(transformed_d1-d0)<=0.07 && (inten_last.at<uchar>(v0, u0) != 0) && (mTwo > 1600) && (inten.at<uchar>(y, x) > 0)){
						//std::cout<<"\nd0:  "<<d0;
						//std::cout<<"\nd1_trans:  "<<transformed_d1;
						//std::cout<<"\n(x, y): "<<"("<<x<<", "<<y<<")";
						//std::cout<<"\n(u0, v0): "<<"("<<u0<<", "<<v0<<")";
						//std::cout<<"\nd1:  "<<d1;
						//std::cout<<"\nd0:  "<<d0;
						for(int u = std::max(y - 2, 0); u < std::min(x + 2, 480); u++)
						{
							for(int v = std::max(x - 2, 0); v < std::min(x + 2, 640); v++)
							{
								if((float)inten.at<uchar>(y, x) <= 0){
									continue;
								}
							}
						}

						float t = (float)inten.at<uchar>(y, x);
						float s = (float)inten_last.at<uchar>(v0, u0);

						int alfa = (int)hsv.at<cv::Vec3b>(y, x)[0];
						int beta = (int)hsv_last.at<cv::Vec3b>(v0, u0)[0];

						float diff = t-s;
						//int diff = alfa-beta;
						//std::cout<<"\nintens: "<<diff<<"   hsv: "<<diffhsv;
						float w =  std::abs(diff) + delta;
						w = w > FLT_EPSILON ? 1.0 / w : 1.0;
						//std::cout<<"\ndiff:  "<<diff;
						float cloud_x = (float)LastCloud.row(640*v0 + u0)(0);
						float cloud_y = (float)LastCloud.row(640*v0 + u0)(1);
						float cloud_z = (float)LastCloud.row(640*v0 + u0)(2);
						//std::cout<<"\nd0:  "<<d0;
						//std::cout<<"\nz:  "<<cloud_z;
						float invz = 1/cloud_z;
						float dI_dx_val = w * dIdx.at<short>(y, x) * 0.125;
						float dI_dy_val = w * dIdy.at<short>(y, x) * 0.125;
						float v0 = dI_dx_val * K(0, 0) * invz;
						float v1 = dI_dy_val * K(1, 1) * invz;
						float v2 = -(v0 * cloud_x + v1 * cloud_y) * invz;
						//std::cout<<"\ndI_dx_val:  "<<dI_dx_val;
						//std::cout<<"\ndI_dx:  "<<dIdx.at<short>(y, x);
						double a,b,c,d,e,f,g;

						a = v0;
						b = v1;
						c = v2;
						d = (-cloud_z * v1) + cloud_y * v2;
						e = cloud_z * v0 - cloud_x * v2;
						f = (-cloud_y * v0) + cloud_x * v1;
						g = (-w) * diff;
						//std::cout<<"\npoint: ("<<cloud_x<<", "<<cloud_y<<", "<<cloud_z<<")";
						//std::cout<<"\na :"<<a;
						//std::cout<<"\nb :"<<b;
						//std::cout<<"\nc :"<<c;
						//std::cout<<"\nd :"<<d;
						//std::cout<<"\ne :"<<e;
						//std::cout<<"\nf :"<<f;
						//std::cout<<"\ng :"<<g;

						Temp = Eigen::MatrixXd::Zero(6, 6);
						Temp_b = Eigen::MatrixXd::Zero(6, 1);

						Temp(0,0) = a * a;
						Temp(0,1) = Temp(1,0) = a * b;
						Temp(0,2) = Temp(2,0) = a * c;
						Temp(0,3) = Temp(3,0) = a * d;
						Temp(0,4) = Temp(4,0) = a * e;
						Temp(0,5) = Temp(5,0) = a * f;

						Temp(1,1) = b * b;
						Temp(1,2) = Temp(2,1) = b * c;
						Temp(1,3) = Temp(3,1) = b * d;
						Temp(1,4) = Temp(4,1) = b * e;
						Temp(1,5) = Temp(5,1) = b * f;

						Temp(2,2) = c * c;
						Temp(2,3) = Temp(3,2) = c * d;
						Temp(2,4) = Temp(4,2) = c * e;
						Temp(2,5) = Temp(5,2) = c * f;

						Temp(3,3) = d * d;
						Temp(3,4) = Temp(4,3) = d * e;
						Temp(3,5) = Temp(5,3) = d * f;

						Temp(4,4) = e * e;
						Temp(4,5) = Temp(5,4) = e * f;

						Temp(5,5) = f * f;

						Temp_b(0,0) = a * g;
						Temp_b(1,0) = b * g;
						Temp_b(2,0) = c * g;
						Temp_b(3,0) = d * g;
						Temp_b(4,0) = e * g;
						Temp_b(5,0) = f * g;

						//std::cout<<"\n\nrgb_b: "<<b_rgb;
						A_rgb += Temp;
						b_rgb += Temp_b;

						count++;
						total += (diff * diff);
						
						//std::cout<<"\ntemp_b: "<<Temp_b;
						//std::cout<<"\ndI_dx_val: "<<dI_dx_val;
						//std::cout<<"\ndI_dx: "<<(int)dIdx.at<char>(y, x);
						//std::cout<<"\ncoord:  "<<"("<<u0<<","<<v0<<")";
					}
				}
			}
		}
	}
	//Eigen::Vector<EAS_ICP::Scalar, 6> ret;
	//ret = A_rgb.ldlt().solve(b_rgb);
	//std::cout<<"\nA_rgb: "<<A_rgb<<"\n";
	//std::cout<<"\nb_rgb: "<<b_rgb<<"\n";
	//std::cout<<"\nret: "<<ret<<"\n";
	float average = sqrt((float)total/(float)count);
	/*
	std::cout<<"\ncount: "<<count<<"\n";
	std::cout<<"\ntotal loss: "<<total<<"\n";
	std::cout<<"\naverage: "<<average<<"\n";
	*/
}
void EAS_ICP::RGBJacobianGetCorres(const cv::Mat& dIdx, const cv::Mat& dIdy, const cv::Mat depth, const cv::Mat rgb, const cv::Mat rgb_last, const TargetCloud& tgtCloud, const Transform resultRt, A_term& A_rgb, b_term& b_rgb, TargetCloud& LastCloud){
	A_rgb = Eigen::MatrixXd::Zero(6, 6);
	b_rgb = Eigen::MatrixXd::Zero(6, 1);
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
	cv::Mat inten;
	cv::Mat inten_last;
	cv::Mat hsv;
	cv::Mat hsv_last;
	cv::cvtColor(rgb, inten, cv::COLOR_BGR2GRAY);
	cv::cvtColor(rgb_last, inten_last, cv::COLOR_BGR2GRAY);

	cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
	cv::cvtColor(rgb_last, hsv_last, cv::COLOR_BGR2HSV);
	K(0, 0) = 528;//fx;
	K(1, 1) = 528;//fy;
	K(0, 2) = 320;//cx;
	K(1, 2) = 240;//cy;
	K(2, 2) = 1;

	//Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt;

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();

	Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
	//std::cout<<inten;
	Kt = K * Kt;
	float total = 0;
	A_term Temp = A_term::Zero();
	b_term Temp_b = b_term::Zero();
	int count = 0;
	//std::cout<<"corrs:"<<corrs;
	//int coord = corrs.row(0)(1);
	//std::cout<<"\n("<<coord%640<<", "<<coord/640<<")\n";
			//float d1 = tgtCloud.row(640*y + x)(2) * 5000;
	for(int i = 0; i < corrs.rows(); i++){
		int coord = corrs.row(i)(1);
		int x = coord%640;
		int y = coord/640;

		float d1 = (float)depth.at<short>(y, x);

		float transformed_d1 = (float)(d1 * (KRK_inv(2,0) * x + KRK_inv(2,1) * y + KRK_inv(2,2)) + Kt(2));
		int u0 = (d1 * (KRK_inv(0,0) * x + KRK_inv(0,1) * y + KRK_inv(0,2)) + Kt(0)) / transformed_d1;
		int v0 = (d1 * (KRK_inv(1,0) * x + KRK_inv(1,1) * y + KRK_inv(1,2)) + Kt(1)) / transformed_d1;
		//std::cout<<"\nd1:  "<<d1;
		//std::cout<<"\nd2:  "<<d2;
		
		
		//std::cout<<"\nd2_trans:  "<<transformed_d2;
		//float d0 = LastCloud.row(640*v0 + u0)(2) * 1000;
		if(u0 >= 0 && v0 >= 0 && u0 < 640 && v0 < 480){
			float d0 = (float)last_depth.at<short>(v0, u0);
			if(d0>0 && std::abs(transformed_d1-d0)<70 && inten_last.at<uchar>(v0, u0) != 0){

				//std::cout<<"\nd0:  "<<d0;
				//std::cout<<"\nd1_trans:  "<<transformed_d1;
				//std::cout<<"\n(x, y): "<<"("<<x<<", "<<y<<")";
				//std::cout<<"\n(u0, v0): "<<"("<<u0<<", "<<v0<<")";
				//std::cout<<"\nd1:  "<<d1;
				//std::cout<<"\nd0:  "<<d0;
				
				int t = (int)inten.at<uchar>(y, x);
				int s = (int)inten_last.at<uchar>(v0, u0);
				//std::cout<<"\n\nt:  "<<t;
				//std::cout<<"\nt2:  "<<t2<<"\n";
				float diff = (float)t-(float)s;
				//std::cout<<"\ndiff: "<<diff;
				total += (diff*diff);
				count++;

			}
		}
	}
	float delta = sqrt((float)total/(float)count);
	/*
	std::cout<<"\ncount: "<<count;
	std::cout<<"\ntotal: "<<total;
	std::cout<<"\ndelta: "<<delta;
	*/
	total = 0;
	count = 0;
	for(int i = 0; i < corrs.rows(); i++){

		int coord = corrs.row(i)(1);
		int x = coord%640;
		int y = coord/640;

		float d1 = (float)next_depth.at<short>(y, x);

		float transformed_d1 = (float)(d1 * (KRK_inv(2,0) * x + KRK_inv(2,1) * y + KRK_inv(2,2)) + Kt(2));
		int u0 = (d1 * (KRK_inv(0,0) * x + KRK_inv(0,1) * y + KRK_inv(0,2)) + Kt(0)) / transformed_d1;
		int v0 = (d1 * (KRK_inv(1,0) * x + KRK_inv(1,1) * y + KRK_inv(1,2)) + Kt(1)) / transformed_d1;
		//std::cout<<"\nd1:  "<<d1;
		//std::cout<<"\nd2:  "<<d2;

		float valX = dIdx.at<short>(y, x);
		float valY = dIdy.at<short>(y, x);
		float mTwo = valX * valX + valY * valY;
		//std::cout<<"\nmTwo:  "<<mTwo;
		//std::cout<<"\nd2_trans:  "<<transformed_d2;
		//float d0 = LastCloud.row(640*v0 + u0)(2) * 1000;
	if(u0 >= 0 && v0 >= 0 && u0 < 640 && v0 < 480 && mTwo >1600){

		float d0 = (float)last_depth.at<short>(v0, u0);
		if(d0>0 && std::abs(transformed_d1-d0)<70 && inten_last.at<uchar>(v0, u0) != 0){
			
			std::cout<<"\nmTwo:  "<<mTwo;
			//std::cout<<"\nd0:  "<<d0;
			//std::cout<<"\nd1_trans:  "<<transformed_d1;
			//std::cout<<"\n(x, y): "<<"("<<x<<", "<<y<<")";
			//std::cout<<"\n(u0, v0): "<<"("<<u0<<", "<<v0<<")";
			//std::cout<<"\nd1:  "<<d1;
			//std::cout<<"\nd0:  "<<d0;
			
			int t = inten.at<uchar>(y, x);
			int s = inten_last.at<uchar>(v0, u0);

			int alfa = (int)hsv.at<cv::Vec3b>(y, x)[0];
			int beta = (int)hsv_last.at<cv::Vec3b>(v0, u0)[0];

			float diff = (float)t-(float)s;
			int diffhsv = alfa-beta;
			//std::cout<<"\nintens: "<<diff<<"   hsv: "<<diffhsv;
			float w =  std::abs(diff) + delta;
			w = w > FLT_EPSILON ? 1.0 / w : 1.0;
			float cloud_x = LastCloud.row(640*v0 + u0)(0);
			float cloud_y = LastCloud.row(640*v0 + u0)(1);
			float cloud_z = LastCloud.row(640*v0 + u0)(2);

			float invz = 1/cloud_z;
			float dI_dx_val = w * dIdx.at<float>(y, x) * 0.125;
			float dI_dy_val = w * dIdy.at<float>(y, x) * 0.125;
			float v0 = dI_dx_val * fx * invz;
			float v1 = dI_dy_val * fy * invz;
			float v2 = -(v0 * cloud_x + v1 * cloud_y) * invz;
			//std::cout<<"\ndI_dx_val:  "<<dI_dx_val;
			std::cout<<"\ncloud z :"<<cloud_z;
			std::cout<<"\nd0 :"<<d0;
			double a,b,c,d,e,f,g;

			a = v0;
			b = v1;
			c = v2;
			d = -cloud_z * v1 + cloud_y * v2;
			e = cloud_z * v0 - cloud_x * v2;
			f = -cloud_y * v0 + cloud_x * v1;
			g = -w * diff;
			//std::cout<<"\npoint: ("<<cloud_x<<", "<<cloud_y<<", "<<cloud_z<<")";
			//std::cout<<"\na :"<<a;
			//std::cout<<"\nb :"<<b;
			//std::cout<<"\nc :"<<c;
			//std::cout<<"\nd :"<<d;
			//std::cout<<"\ne :"<<e;
			//std::cout<<"\nf :"<<f;
			//std::cout<<"\ng :"<<g;
			Temp(0,0) = a * a;
			Temp(0,1) = Temp(1,0) = a * b;
			Temp(0,2) = Temp(2,0) = a * c;
			Temp(0,3) = Temp(3,0) = a * d;
			Temp(0,4) = Temp(4,0) = a * e;
			Temp(0,5) = Temp(5,0) = a * f;

			Temp(1,1) = b * b;
			Temp(1,2) = Temp(2,1) = b * c;
			Temp(1,3) = Temp(3,1) = b * d;
			Temp(1,4) = Temp(4,1) = b * e;
			Temp(1,5) = Temp(5,1) = b * f;

			Temp(2,2) = c * c;
			Temp(2,3) = Temp(3,2) = c * d;
			Temp(2,4) = Temp(4,2) = c * e;
			Temp(2,5) = Temp(5,2) = c * f;

			Temp(3,3) = d * d;
			Temp(3,4) = Temp(4,3) = d * e;
			Temp(3,5) = Temp(5,3) = d * f;

			Temp(4,4) = e * e;
			Temp(4,5) = Temp(5,4) = e * f;

			Temp(5,5) = f * f;

			Temp_b(0,0) = a * g;
			Temp_b(1,0) = b * g;
			Temp_b(2,0) = c * g;
			Temp_b(3,0) = d * g;
			Temp_b(4,0) = e * g;
			Temp_b(5,0) = f * g;

			A_rgb += Temp;
			b_rgb += Temp_b;

			count++;
			total += (diff * diff);
			//std::cout<<"\ndI_dx_val: "<<dI_dx_val;
			//std::cout<<"\ndI_dx: "<<(int)dIdx.at<char>(y, x);
			//std::cout<<"\ncoord:  "<<"("<<u0<<","<<v0<<")";
			}
		}
	}
	//Eigen::Vector<EAS_ICP::Scalar, 6> ret;
	//ret = A_rgb.ldlt().solve(b_rgb);
	//std::cout<<"\nA_rgb: "<<A_rgb<<"\n";
	//std::cout<<"\nb_rgb: "<<b_rgb<<"\n";
	//std::cout<<"\nret: "<<ret<<"\n";
	float average = sqrt((float)total/(float)count);
	/*
	std::cout<<"\ncount: "<<count<<"\n";
	std::cout<<"\ntotal loss: "<<total<<"\n";
	std::cout<<"\naverage: "<<average<<"\n";
	*/
}
void EAS_ICP::computeDerivativeImages(const cv::Mat& rgb, cv::Mat& dIdx, cv::Mat& dIdy){
  float gsx3x3[9] = {-0.52201,  0.00000, 0.52201,
                   -0.79451, -0.00000, 0.79451,
                   -0.52201,  0.00000, 0.52201};

  float gsy3x3[9] = {-0.52201, -0.79451, -0.52201,
                     0.00000, 0.00000, 0.00000,
                    0.52201, 0.79451, 0.52201};


  cv::Mat intensity;
  cv::cvtColor(rgb, intensity, cv::COLOR_BGR2GRAY);
  cv::Mat kernelX(3, 3, CV_32F, gsx3x3);
  cv::Mat kernelY(3, 3, CV_32F, gsy3x3);
  cv::filter2D( intensity, dIdx, CV_16S , kernelX, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  cv::filter2D( intensity, dIdy, CV_16S , kernelY, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  //dIdx = dIdx * (-1);
  //dIdy = dIdy * (-1);
  //cv:: Sobel(intensity, dIdx, CV_16S, 1, 0, 1);
  //cv:: Sobel(intensity, dIdy, CV_16S, 0, 1, 1);
  //std::cout<<dIdx;
  //std::cout<<dIdx.at<float>(10,10);

}
void EAS_ICP::computeDerivativeImagesHSV(const cv::Mat& rgb, cv::Mat& dIdx, cv::Mat& dIdy){
  float gsx3x3[9] = {-0.52201,  0.00000, 0.52201,
                   -0.79451, -0.00000, 0.79451,
                   -0.52201,  0.00000, 0.52201};

  float gsy3x3[9] = {-0.52201, -0.79451, -0.52201,
                     0.00000, 0.00000, 0.00000,
                    0.52201, 0.79451, 0.52201};

  cv::Mat HSV;
  cv::Mat Hue;
  cv::cvtColor(rgb, HSV, cv::COLOR_BGR2HSV);
  
  cv::extractChannel(HSV, Hue, 0);

  cv::Mat kernelX(3, 3, CV_32F, gsx3x3);
  cv::Mat kernelY(3, 3, CV_32F, gsy3x3);
  cv::filter2D( Hue, dIdx, CV_32F , kernelX, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  cv::filter2D( Hue, dIdy, CV_32F , kernelY, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  //cv:: Sobel(Hue, dIdx, CV_32F, 1, 0, 1);
  //cv:: Sobel(Hue, dIdy, CV_32F, 0, 1, 1);

}
const EAS_ICP::Transform& EAS_ICP::Register_Ori(const SourceCloud& srcCloud, const cv::Mat& depth, const cv::Mat& rgb, const TargetCloud& tgtCloud, const Transform& initialGuess) {

  //initial parameters
  std::cout<<"Meta ICP\n";
  rtSE3 = initialGuess;
  rtSE3_1 = initialGuess;
  rtSE3_2 = initialGuess;
  rtSE3_3 = initialGuess;
  rtSE3_4 = initialGuess;
  int cnt_task1_large =0;
  int cnt_task2_large =0;
  int cnt_task3_large =0;
  int cnt_task4_large =0;
  int iterations = 0;
  accSlidingExtent = 0;
  Useonlyforeground = 1;
  Useweigtedrandomsampling = 1;
  Useedgeaware = 0;
  srand( time(NULL) );
  int iteration_divide = 20;

  while (true) {
  iterations+=1;
  //meta training
    /*double randomRGB = (double) rand() / (RAND_MAX + 1.0);
	double randomdepth = (double) rand() / (RAND_MAX + 1.0);
	std::cout<<"randomRGB"<<std::endl;
	std::cout<<randomRGB<<std::endl;
	std::cout<<"randomdepth"<<std::endl;
	std::cout<<randomdepth<<std::endl;*/
  //RGB ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  cv::Mat nan_map;
  cv::Mat rej_map;
  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::Mat edge_map_inner;
  cv::Mat edge_map_outer;
  cv::resize(src, out_resized, cv::Size(320, 240), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  //cv::blur(src, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  //dilation
  cv::Mat dilatemat1111 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11));
  cv::Mat dilatemat99 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
  cv::Mat dilatemat77 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat dilatemat55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::Mat dilatemat33 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

  cv::dilate(edge_map2, edge_map_inner, dilatemat33);
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner;
  inv_edge_map_inner = ~edge_map_inner;
  cv::Mat edge_distance_map_inner;
  cv::distanceTransform(inv_edge_map_inner, edge_distance_map_inner, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  
  /////////////////////////dilate outer

 
  cv::dilate(edge_map2, edge_map_outer, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer, edge_map_inner, edge_map_outer);
  //cv::imwrite("ROI_detection_rgb_dilateouter.png", edge_map_outer);
  cv::Mat inv_edge_map_outer;
  inv_edge_map_outer = ~edge_map_outer;
  cv::Mat edge_distance_map_outer;
  //cv::distanceTransform(inv_edge_map_outer, edge_distance_map_outer, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2;
  std::vector<double> weights2;
  std::vector<int> EASInds2;

  
  //*********************************************
  //Depth ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  keycloud_depth = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  cv::Mat edge_map_depth;
  cv::Mat nan_map_depth;
  cv::Mat rej_map_depth;
  cv::Mat edge_map_inner_depth;
  EdgeDetection(*keycloud_depth, edge_map_depth, nan_map_depth, rej_map_depth);

  cv::dilate(edge_map_depth, edge_map_inner_depth, dilatemat33);
  //cv::imwrite("ROI_detection_depth_inner.png", edge_map_inner_depth);
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner_depth;
  inv_edge_map_inner_depth = ~edge_map_inner_depth;
  cv::Mat edge_distance_map_inner_depth;
  cv::distanceTransform(inv_edge_map_inner_depth, edge_distance_map_inner_depth, cv::DIST_L2, 5);
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds_depth;
  std::vector<double> weights_depth;
  std::vector<int> EASInds_depth;

  
  /////////////////////////dilate outer
  //cv::Mat dilatemat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat edge_map_outer_depth;

  cv::dilate(edge_map_depth, edge_map_outer_depth, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer_depth, edge_map_inner_depth, edge_map_outer_depth);
  
  cv::Mat inv_edge_map_outer_depth;
  inv_edge_map_outer_depth = ~edge_map_outer_depth;
  //cv::Mat edge_distance_map_outer_depth;
  //cv::distanceTransform(inv_edge_map_outer_depth, edge_distance_map_outer_depth, cv::DIST_L2, 5);
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2_depth;
  std::vector<double> weights2_depth;
  std::vector<int> EASInds2_depth;
  cv::Mat mapR1;
  cv::Mat mapR2;
  cv::Mat mapR3;
  cv::Mat mapR4;
  mapR1 = edge_map_inner_depth;
  mapR2 = edge_map_outer_depth;
  mapR3 = edge_map_inner;
  mapR4 = edge_map_outer;
  for(int i =0;i<mapR1.rows;i++)
  {
	for(int j =0;j<mapR1.cols;j++)
	{
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR1.at<uchar>(i,j) = 0;
		}		
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR2.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR3.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR4.at<uchar>(i,j) = 0;
		}
	}		
  }
  
	  
  //Before iteration 20
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_inner, nan_map_depth| rej_map_depth, remindPointInds, weights);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR3, nan_map_depth| rej_map_depth, remindPointInds, weights);
	
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1_before20);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_outer, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR4, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2_before20);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_inner_depth, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR1, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3_before20);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_outer_depth, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR2, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4_before20);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	
	
  //After iteration 20
  //4 ROImap pixelwise OR 
  cv::Mat ROI4OR;
  //cv::bitwise_or(edge_map_inner_depth, edge_map_outer_depth, ROI4OR);
  cv::bitwise_or(edge_map_inner, edge_map_outer, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_inner_depth, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_outer_depth, ROI4OR);
  if(Useedgeaware==0)
	{	  
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth| rej_map_depth, remindPointInds, weights);
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
  else
  {
	// 1st 
	cv::Mat InvEdgeMap;
	cv::Mat EdgeDistanceMap;
	InvEdgeMap = ~edge_map_depth;
	cv::distanceTransform(InvEdgeMap, EdgeDistanceMap, cv::DIST_L2, 5);
 
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth| rej_map_depth, remindPointInds, weights);
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
    CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	
	// 2nd 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
		
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
		
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
		
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	
	
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
	kinectNoiseWeightsdepth_before20 = KinectNoiseWighting(mSrcCloud3_before20);
	kinectNoiseWeightsdepth2_before20 = KinectNoiseWighting(mSrcCloud4_before20);
	kinectNoiseWeightsrgb_before20 = KinectNoiseWighting(mSrcCloud1_before20);
	kinectNoiseWeightsrgb2_before20 = KinectNoiseWighting(mSrcCloud2_before20);
	kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	/*kinectNoiseWeightsdepth = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(srcCloud);*/
	//
	
	
	
	//sampling end
	
	
    //transform depth source cloud by inital guess
    //SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    //transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    //transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    SourceCloud transformedCloudrgb(mSrcCloud1.rows(), 6) ;
	SourceCloud transformedCloudrgb2(mSrcCloud2.rows(), 6) ;
	SourceCloud transformedClouddepth(mSrcCloud3.rows(), 6) ;
	SourceCloud transformedClouddepth2(mSrcCloud4.rows(), 6) ;
	SourceCloud transformedCloudrgb_before20(mSrcCloud1_before20.rows(), 6) ;
	SourceCloud transformedCloudrgb2_before20(mSrcCloud2_before20.rows(), 6) ;
	SourceCloud transformedClouddepth_before20(mSrcCloud3_before20.rows(), 6) ;
	SourceCloud transformedClouddepth2_before20(mSrcCloud4_before20.rows(), 6) ;
	/*SourceCloud transformedCloudrgb(srcCloud.rows(), 6) ;
	SourceCloud transformedCloudrgb2(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth2(srcCloud.rows(), 6) ;*/
	if(Useedgeaware==0)
	{
		if(iterations<=iteration_divide)
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1_before20.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2_before20.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3_before20.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4_before20.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4_before20.rightCols<3>().transpose()).transpose();
		}	
		
		else
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
		}
	}
	else
	{
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
		
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
    }	
	
	int cnttrans1, cnttrans2, cnttrans3, cnttrans4;
	cnttrans1 = 0;
	cnttrans2 = 0;
	cnttrans3 = 0;
	cnttrans4 = 0;
    int meanweight1, meanweight2, meanweight3, meanweight4;
    double randomx_thres = 0.1;
	if(Useweigtedrandomsampling==1)
	{
	  randomx_thres = 100;
	}
    //get iteration transformation by minimizing p2pl error metric
	//srand( time(NULL) );
	//1st set of correspondences
	/*int cnt1[mSrcCloud3.rows()];
    for(int i=0; i<mSrcCloud3.rows(); i++)
	{
		//random
		
		double randomx = (double) rand() / (RAND_MAX + 1.0);
			if(randomx<randomx_thres)
			{
				cnt1[i] = 1;
				cnttrans1++;
			}
			else
			{
				cnt1[i] = 0;
			}
	}
	SourceCloud transformedCloud1(cnttrans1, 6) ;
	
	cnttrans1 = 0;
	for(int i =0;i<mSrcCloud3.rows(); i++)
	{
		if(cnt1[i]==1)
		{
			transformedCloud1(cnttrans1, 0) = transformedClouddepth(i, 0);
			transformedCloud1(cnttrans1, 1) = transformedClouddepth(i, 1);
			transformedCloud1(cnttrans1, 2) = transformedClouddepth(i, 2);
			transformedCloud1(cnttrans1, 3) = transformedClouddepth(i, 3);
			transformedCloud1(cnttrans1, 4) = transformedClouddepth(i, 4);
			transformedCloud1(cnttrans1, 5) = transformedClouddepth(i, 5);
			cnttrans1++;
		}
		
	}*/
    Eigen::Vector<Scalar, 6> rt6D;
	
    if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)) {
			//std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)) {
			//std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
		}
		}
		//std::cout<<"cloudnumber"<<std::endl;
		//std::cout<<mSrcCloud3.rows()<<std::endl;

		meanweight1 = 1;
	}
	//std::cout<<"corrs1num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	//2nd set of correspondences
	/*int cnt2[mSrcCloud4.rows()];
    for(int i=0; i<mSrcCloud4.rows(); i++)
	{
		//random
		
		double randomx = (double) rand() / (RAND_MAX + 1.0);
			if(randomx<randomx_thres)
			{
				cnt2[i] = 1;
				cnttrans2++;
			}
			else
			{
				cnt2[i] = 0;
			}
	}
	SourceCloud transformedCloud2(cnttrans2, 6) ;
	
	cnttrans2 = 0;
	for(int i =0;i<mSrcCloud4.rows(); i++)
	{
		if(cnt2[i]==1)
		{
			transformedCloud2(cnttrans2, 0) = transformedClouddepth2(i, 0);
			transformedCloud2(cnttrans2, 1) = transformedClouddepth2(i, 1);
			transformedCloud2(cnttrans2, 2) = transformedClouddepth2(i, 2);
			transformedCloud2(cnttrans2, 3) = transformedClouddepth2(i, 3);
			transformedCloud2(cnttrans2, 4) = transformedClouddepth2(i, 4);
			transformedCloud2(cnttrans2, 5) = transformedClouddepth2(i, 5);
			cnttrans2++;
		}
		
	}*/
    
	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
		}
		}	

		meanweight2 = 1;
	}
	//std::cout<<"corrs2num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	//3rd set of correspondences
	/*int cnt3[mSrcCloud1.rows()];
    for(int i=0; i<mSrcCloud1.rows(); i++)
	{
		//random
		
		double randomx = (double) rand() / (RAND_MAX + 1.0);
			if(randomx<randomx_thres)
			{
				cnt3[i] = 1;
				cnttrans3++;
			}
			else
			{
				cnt3[i] = 0;
			}
	}
	SourceCloud transformedCloud3(cnttrans3, 6) ;
	
	cnttrans3 = 0;
	for(int i =0;i<mSrcCloud1.rows(); i++)
	{
		if(cnt3[i]==1)
		{
			transformedCloud3(cnttrans3, 0) = transformedCloudrgb(i, 0);
			transformedCloud3(cnttrans3, 1) = transformedCloudrgb(i, 1);
			transformedCloud3(cnttrans3, 2) = transformedCloudrgb(i, 2);
			transformedCloud3(cnttrans3, 3) = transformedCloudrgb(i, 3);
			transformedCloud3(cnttrans3, 4) = transformedCloudrgb(i, 4);
			transformedCloud3(cnttrans3, 5) = transformedCloudrgb(i, 5);
			cnttrans3++;
		}
		
	}*/

	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
		//std::cout<<"cnttrans3"<<std::endl;
		//std::cout<<cnttrans3<<std::endl;
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all));
		}	
		else
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {

		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
		}	
		else 
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	}
	//std::cout<<"corrs3num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	//4th set of correspondences
	/*int cnt4[mSrcCloud2.rows()];
    for(int i=0; i<mSrcCloud2.rows(); i++)
	{
		//random
		
		double randomx = (double) rand() / (RAND_MAX + 1.0);
			if(randomx<randomx_thres)
			{
				cnt4[i] = 1;
				cnttrans4++;
			}
			else
			{
				cnt4[i] = 0;
			}
	}
	SourceCloud transformedCloud4(cnttrans4, 6) ;
	
	cnttrans4 = 0;
	for(int i =0;i<mSrcCloud2.rows(); i++)
	{
		if(cnt4[i]==1)
		{
			transformedCloud4(cnttrans4, 0) = transformedCloudrgb2(i, 0);
			transformedCloud4(cnttrans4, 1) = transformedCloudrgb2(i, 1);
			transformedCloud4(cnttrans4, 2) = transformedCloudrgb2(i, 2);
			transformedCloud4(cnttrans4, 3) = transformedCloudrgb2(i, 3);
			transformedCloud4(cnttrans4, 4) = transformedCloudrgb2(i, 4);
			transformedCloud4(cnttrans4, 5) = transformedCloudrgb2(i, 5);
			cnttrans4++;
		}
		
	}*/
	
	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}

		meanweight4 = 1;
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
		}
		//std::cout<<"cnttrans4"<<std::endl;
		//std::cout<<cnttrans4<<std::endl;
		meanweight4 = 1;
		}
	}	
	//std::cout<<"corrs4num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	// ++ meta task loop end
	// ++ pose fusion = mean(all tasks' rt6D)
	
	if((meanweight1+meanweight2+meanweight3+meanweight4)==0)
	{
		break;
	}
	/*else{
    rt6D(0) = (rt6D1(0)*meanweight1+ rt6D2(0)*meanweight2 +rt6D3(0)*meanweight3 +rt6D4(0)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(1) = (rt6D1(1)*meanweight1+ rt6D2(1)*meanweight2 +rt6D3(1)*meanweight3 +rt6D4(1)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
    rt6D(2) = (rt6D1(2)*meanweight1+ rt6D2(2)*meanweight2 +rt6D3(2)*meanweight3 +rt6D4(2)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(3) = (rt6D1(3)*meanweight1+ rt6D2(3)*meanweight2 +rt6D3(3)*meanweight3 +rt6D4(3)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(4) = (rt6D1(4)*meanweight1+ rt6D2(4)*meanweight2 +rt6D3(4)*meanweight3 +rt6D4(4)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(5) = (rt6D1(5)*meanweight1+ rt6D2(5)*meanweight2 +rt6D3(5)*meanweight3 +rt6D4(5)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	}*/
    //convert 6D vector to SE3
    Eigen::Vector<Scalar, 6> rt6D_L1_largest;
	double s1, s2, s3, s4;
	s1 = fabs(rt6D1(3))+fabs(rt6D1(4))+fabs(rt6D1(5));
	s2 = fabs(rt6D2(3))+fabs(rt6D2(4))+fabs(rt6D2(5));
	s3 = fabs(rt6D3(3))+fabs(rt6D3(4))+fabs(rt6D3(5));
	s4 = fabs(rt6D4(3))+fabs(rt6D4(4))+fabs(rt6D4(5));
	if((s1>=s2)&&(s1>=s3)&&(s1>=s4))
	{
		rt6D_L1_largest=rt6D1;
		cnt_task1_large++;
	}
	if((s2>=s1)&&(s2>=s3)&&(s2>=s4))
	{
		rt6D_L1_largest=rt6D2;
		cnt_task2_large++;
	}
	if((s3>=s2)&&(s3>=s1)&&(s3>=s4))
	{
		rt6D_L1_largest=rt6D3;
		cnt_task3_large++;
	}
	if((s4>=s2)&&(s4>=s3)&&(s4>=s1))
	{
		rt6D_L1_largest=rt6D4;
		cnt_task4_large;
	}
	
	Transform iterRtSE3_1;
    iterRtSE3_1 = ConstructSE3(rt6D_L1_largest);
    //chain iterRtSE3 to rtSE3
    rtSE3_1 = iterRtSE3_1 * rtSE3_1;
	Transform iterRtSE3_2;
    iterRtSE3_2 = ConstructSE3(rt6D_L1_largest);
    //chain iterRtSE3 to rtSE3
    rtSE3_2 = iterRtSE3_2 * rtSE3_2;
	Transform iterRtSE3_3;
    iterRtSE3_3 = ConstructSE3(rt6D_L1_largest);
    //chain iterRtSE3 to rtSE3
    rtSE3_3 = iterRtSE3_3 * rtSE3_3;
	Transform iterRtSE3_4;
    iterRtSE3_4 = ConstructSE3(rt6D_L1_largest);
    //chain iterRtSE3 to rtSE3
    rtSE3_4 = iterRtSE3_4 * rtSE3_4;
    //check termination
    /*if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }*/
	max_iters = 30;
	if (iterations > max_iters)
	{
		std::cout<<"4 tasks count"<<std::endl;
		std::cout<<cnt_task1_large<<std::endl;
		std::cout<<cnt_task2_large<<std::endl;
		std::cout<<cnt_task3_large<<std::endl;
		std::cout<<cnt_task4_large<<std::endl;		
		break;
	}
  }
  
  
  
  
  //meta testing
  //icp loop 2 START
  int iteration_loop2 = 0;
  while (true) {
  iteration_loop2+=1;
  //RGB ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  cv::Mat nan_map;
  cv::Mat rej_map;
  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::Mat edge_map_inner;
  cv::Mat edge_map_outer;
  cv::resize(src, out_resized, cv::Size(320, 240), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  //cv::blur(src, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  //dilation
  cv::Mat dilatemat55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::Mat dilatemat33 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  cv::dilate(edge_map2, edge_map_inner, dilatemat33);

  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner;
  inv_edge_map_inner = ~edge_map_inner;
  cv::Mat edge_distance_map_inner;
  cv::distanceTransform(inv_edge_map_inner, edge_distance_map_inner, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  
  /////////////////////////dilate outer
  cv::Mat dilatemat99 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
  cv::Mat dilatemat77 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::dilate(edge_map2, edge_map_outer, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer, edge_map_inner, edge_map_outer);
  
  cv::Mat inv_edge_map_outer;
  inv_edge_map_outer = ~edge_map_outer;
  cv::Mat edge_distance_map_outer;
  //cv::distanceTransform(inv_edge_map_outer, edge_distance_map_outer, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2;
  std::vector<double> weights2;
  std::vector<int> EASInds2;

  
  //*********************************************
  //Depth ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  keycloud_depth = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  cv::Mat edge_map_depth;
  cv::Mat nan_map_depth;
  cv::Mat rej_map_depth;
  cv::Mat edge_map_inner_depth;
  EdgeDetection(*keycloud_depth, edge_map_depth, nan_map_depth, rej_map_depth);
  
  cv::dilate(edge_map_depth, edge_map_inner_depth, dilatemat33);
  
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner_depth;
  inv_edge_map_inner_depth = ~edge_map_inner_depth;
  cv::Mat edge_distance_map_inner_depth;
  cv::distanceTransform(inv_edge_map_inner_depth, edge_distance_map_inner_depth, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds_depth;
  std::vector<double> weights_depth;
  std::vector<int> EASInds_depth;

  
  /////////////////////////dilate outer
  //cv::Mat dilatemat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat edge_map_outer_depth;
  cv::dilate(edge_map_depth, edge_map_outer_depth, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer_depth, edge_map_inner_depth, edge_map_outer_depth);
  cv::Mat inv_edge_map_outer_depth;
  inv_edge_map_outer_depth = ~edge_map_outer_depth;
  //cv::Mat edge_distance_map_outer_depth;
  //cv::distanceTransform(inv_edge_map_outer_depth, edge_distance_map_outer_depth, cv::DIST_L2, 5);
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2_depth;
  std::vector<double> weights2_depth;
  std::vector<int> EASInds2_depth;
  cv::Mat mapR1;
  cv::Mat mapR2;
  cv::Mat mapR3;
  cv::Mat mapR4;
  mapR1 = edge_map_inner_depth;
  mapR2 = edge_map_outer_depth;
  mapR3 = edge_map_inner;
  mapR4 = edge_map_outer;
  for(int i =0;i<mapR1.rows;i++)
  {
	for(int j =0;j<mapR1.cols;j++)
	{
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR1.at<uchar>(i,j) = 0;
		}		
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR2.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR3.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR4.at<uchar>(i,j) = 0;
		}
	}		
  }
  
  //Before iteration 20
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_inner, nan_map_depth| rej_map_depth, remindPointInds, weights);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR3, nan_map_depth| rej_map_depth, remindPointInds, weights);
	
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1_before20);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_outer, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR4, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2_before20);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_inner_depth, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR1, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3_before20);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_outer_depth, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR2, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4_before20);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	
	
  //After iteration 20
  //4 ROImap pixelwise OR 
  cv::Mat ROI4OR;
  //cv::bitwise_or(edge_map_inner_depth, edge_map_outer_depth, ROI4OR);
  cv::bitwise_or(edge_map_inner, edge_map_outer, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_inner_depth, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_outer_depth, ROI4OR);
  if(Useedgeaware==0)
	{	  
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth| rej_map_depth, remindPointInds, weights);
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
  else
  {
	// 1st 
	cv::Mat InvEdgeMap;
	cv::Mat EdgeDistanceMap;
	InvEdgeMap = ~edge_map_depth;
	cv::distanceTransform(InvEdgeMap, EdgeDistanceMap, cv::DIST_L2, 5);
 
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth| rej_map_depth, remindPointInds, weights);
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
    CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	
	// 2nd 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
		
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
		
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
		
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	
	
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
	kinectNoiseWeightsdepth_before20 = KinectNoiseWighting(mSrcCloud3_before20);
	kinectNoiseWeightsdepth2_before20 = KinectNoiseWighting(mSrcCloud4_before20);
	kinectNoiseWeightsrgb_before20 = KinectNoiseWighting(mSrcCloud1_before20);
	kinectNoiseWeightsrgb2_before20 = KinectNoiseWighting(mSrcCloud2_before20);
	kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	/*kinectNoiseWeightsdepth = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(srcCloud);*/
	//
	
	
	
	//sampling end
	
	
    //transform depth source cloud by inital guess
    //SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    //transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    //transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    SourceCloud transformedCloudrgb(mSrcCloud1.rows(), 6) ;
	SourceCloud transformedCloudrgb2(mSrcCloud2.rows(), 6) ;
	SourceCloud transformedClouddepth(mSrcCloud3.rows(), 6) ;
	SourceCloud transformedClouddepth2(mSrcCloud4.rows(), 6) ;
	SourceCloud transformedCloudrgb_before20(mSrcCloud1_before20.rows(), 6) ;
	SourceCloud transformedCloudrgb2_before20(mSrcCloud2_before20.rows(), 6) ;
	SourceCloud transformedClouddepth_before20(mSrcCloud3_before20.rows(), 6) ;
	SourceCloud transformedClouddepth2_before20(mSrcCloud4_before20.rows(), 6) ;
	/*SourceCloud transformedCloudrgb(srcCloud.rows(), 6) ;
	SourceCloud transformedCloudrgb2(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth2(srcCloud.rows(), 6) ;*/
	if(Useedgeaware==0)
	{
		if(iteration_loop2<=iteration_divide)
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud1_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud1_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud2_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud3_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud3_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud4_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud4_before20.rightCols<3>().transpose()).transpose();
		
		/*//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb_before20.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2_before20.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth_before20.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2_before20.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2_before20.rightCols<3>().transpose()).transpose();
		*/}	
		
		else
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
		
		/*//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2.rightCols<3>().transpose()).transpose();
		*/}
	}
	else
	{
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
		
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
    }	
	
	int cnttrans1, cnttrans2, cnttrans3, cnttrans4;
	cnttrans1 = 0;
	cnttrans2 = 0;
	cnttrans3 = 0;
	cnttrans4 = 0;
    int meanweight1, meanweight2, meanweight3, meanweight4;
    double randomx_thres = 0.1;
	if(Useweigtedrandomsampling==1)
	{
	  randomx_thres = 100;
	}
   
    Eigen::Vector<Scalar, 6> rt6D;

    if(iteration_loop2<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)) {
			std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)) {
			std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
		}
		}
		//std::cout<<"cloudnumber"<<std::endl;
		//std::cout<<mSrcCloud3.rows()<<std::endl;

		meanweight1 = 1;
	}
	//std::cout<<"corrs1num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	
   
	if(iteration_loop2<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
		}
		}	

		meanweight2 = 1;
	}
	//std::cout<<"corrs2num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	//3rd set of correspondences

    
	if(iteration_loop2<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
		//std::cout<<"cnttrans3"<<std::endl;
		//std::cout<<cnttrans3<<std::endl;
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all));
		}	
		else
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {

		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
		}	
		else 
		{
			rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	}
	//std::cout<<"corrs3num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	//4th set of correspondences
	
	
	if(iteration_loop2<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}

		meanweight4 = 1;
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
		}
		//std::cout<<"cnttrans4"<<std::endl;
		//std::cout<<cnttrans4<<std::endl;
		meanweight4 = 1;
		}
	}	
	//std::cout<<"corrs4num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	// ++ meta task loop end
	// ++ pose fusion = mean(all tasks' rt6D)
	
	if((meanweight1+meanweight2+meanweight3+meanweight4)==0)
	{
		break;
	}
	else{

    rt6D(0) = (rt6D1(0)*meanweight1+ rt6D2(0)*meanweight2 +rt6D3(0)*meanweight3 +rt6D4(0)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(1) = (rt6D1(1)*meanweight1+ rt6D2(1)*meanweight2 +rt6D3(1)*meanweight3 +rt6D4(1)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
    rt6D(2) = (rt6D1(2)*meanweight1+ rt6D2(2)*meanweight2 +rt6D3(2)*meanweight3 +rt6D4(2)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(3) = (rt6D1(3)*meanweight1+ rt6D2(3)*meanweight2 +rt6D3(3)*meanweight3 +rt6D4(3)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(4) = (rt6D1(4)*meanweight1+ rt6D2(4)*meanweight2 +rt6D3(4)*meanweight3 +rt6D4(4)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(5) = (rt6D1(5)*meanweight1+ rt6D2(5)*meanweight2 +rt6D3(5)*meanweight3 +rt6D4(5)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	}
	Transform iterRtSE3_1;
    iterRtSE3_1 = ConstructSE3(rt6D1);
    //chain iterRtSE3 to rtSE3
    rtSE3_1 = iterRtSE3_1 * rtSE3_1;
	Transform iterRtSE3_2;
    iterRtSE3_2 = ConstructSE3(rt6D2);
    //chain iterRtSE3 to rtSE3
    rtSE3_2 = iterRtSE3_2 * rtSE3_2;
	Transform iterRtSE3_3;
    iterRtSE3_3 = ConstructSE3(rt6D3);
    //chain iterRtSE3 to rtSE3
    rtSE3_3 = iterRtSE3_3 * rtSE3_3;
	Transform iterRtSE3_4;
    iterRtSE3_4 = ConstructSE3(rt6D4);
    //chain iterRtSE3 to rtSE3
    rtSE3_4 = iterRtSE3_4 * rtSE3_4;
    //convert 6D vector to SE3
    Transform iterRtSE3;
    iterRtSE3 = ConstructSE3(rt6D);

    //chain iterRtSE3 to rtSE3
    rtSE3 = iterRtSE3 * rtSE3;

    //check termination
    /*if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }*/
	max_iters = 20;
	if (iteration_loop2 > max_iters)
	{
		break;
	}
  }
  
  //icp loop 2 END
  //justify valid by sliding extent
  //cancel it if meta perform
  /*if (accSlidingExtent < thresAccSlidingExtent) {
    valid = true;
  } else {
    valid = false;
  }*/
  valid = true;

  return rtSE3;
}
EAS_ICP::Transform EAS_ICP::SO3_prealign(EAS_ICP::Transform Pose_in, const cv::Mat rgb, int scale){

	cv::Mat intensity;
	cv::Mat last_intensity;
  	cv::cvtColor(last_rgb, intensity, cv::COLOR_RGB2GRAY);
	cv::cvtColor(rgb, last_intensity, cv::COLOR_RGB2GRAY);

	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
	cv::Mat rgb_in;
	//int scale = 1;
	cv::resize(rgb, rgb_in, cv::Size(640/scale, 480/scale), cv::INTER_CUBIC);

	int cols = rgb_in.cols;
	int rows = rgb_in.rows;

	K(0, 0) = fx/scale;//fx
	K(1, 1) = fy/scale;//fy
	K(0, 2) = cx/scale;//cx
	K(1, 2) = cy/scale;//cy
	K(2, 2) = 1;
	//Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = Pose_in.inverse();
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = Transform::Identity();
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Rt.topLeftCorner(3, 3);
	//Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Pose_in.topLeftCorner(3, 3).inverse();
	Eigen::Vector3d resultDelta;
	//Eigen::Matrix<double, 3, 3, Eigen::RowMajor> last_resultR = Rt.topLeftCorner(3, 3);
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> last_resultR = resultR;
	double last_count, last_error, count, error, SO3error;
	last_count = 0;
	SO3error = std::numeric_limits<double>::max() / 2;
	last_error = std::numeric_limits<double>::max() / 2;
	for (int iter = 0; iter<10; iter++){


		
		count = 0;
		error = 0;
		//std::cout<<"\n"<<resultR;
		double row[4];
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> jtj;
		Eigen::Matrix<double, 3, 1> jtr;
		
		//Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();
		

		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;

		double a = K_R_lr(0,0);
		double b = K_R_lr(0,1);
		double c = K_R_lr(0,2);

		double d = K_R_lr(1,0);
		double e = K_R_lr(1,1);
		double f = K_R_lr(1,2);

		double g = K_R_lr(2,0);
		double h = K_R_lr(2,1);
		double i = K_R_lr(2,2);

		for (int y = 1; y < rgb_in.rows-1; y++){
			for(int x = 1; x < rgb_in.cols-1; x++){

				Eigen::Matrix<double, 3, 3, Eigen::RowMajor> jtj_temp;
				Eigen::Matrix<double, 3, 1> jtr_temp;

				Eigen::Vector3d unwarpedReferencePoint(x, y, 1.0);
				Eigen::Vector3d warpedReferencePoint = homography * unwarpedReferencePoint;
				int warped_x = (int)(warpedReferencePoint(0)/warpedReferencePoint(2));
				int warped_y = (int)(warpedReferencePoint(1)/warpedReferencePoint(2));
				if(	warped_x >= 1 && 
					warped_x < cols - 1 &&
					warped_y >= 1 &&
					warped_y < rows){

					double gradNext_x = ((double)intensity.at<uchar>(warped_y, warped_x - 1) - (double)intensity.at<uchar>(warped_y, warped_x + 1))/2;
					double gradNext_y = ((double)intensity.at<uchar>(warped_y - 1, warped_x) - (double)intensity.at<uchar>(warped_y + 1, warped_x))/2;
					double gradLast_x = ((double)last_intensity.at<uchar>(y, x - 1) - (double)last_intensity.at<uchar>(y, x + 1))/2;
					double gradLast_y = ((double)last_intensity.at<uchar>(y - 1, x) - (double)last_intensity.at<uchar>(y + 1, x))/2;
					
					double gx = (gradNext_x + gradLast_x)/2;
					double gy = (gradNext_y + gradLast_y)/2;

					Eigen::Vector3d point = K_inv * warpedReferencePoint;

					double z2 = point(2) * point(2);

					Eigen::Vector3d leftProduct(((point(2) * (d * gy + a * gx)) - (gy * g * y) - (gx * g * x)) / z2,
												((point(2) * (e * gy + b * gx)) - (gy * h * y) - (gx * h * x)) / z2,
												((point(2) * (f * gy + c * gx)) - (gy * i * y) - (gx * i * x)) / z2);
					Eigen::Vector3d jacRow = leftProduct.cross(point);

					row[0] = jacRow(0);
					row[1] = jacRow(1);
					row[2] = jacRow(2);
					row[3] = -((double)intensity.at<uchar>(warped_y, warped_x) - (double)last_intensity.at<uchar>(y, x));

					jtj_temp(0,0) = row[0] * row[0];
					jtj_temp(0,1) = jtj_temp(1,0) = row[0] * row[1];
					jtj_temp(0,2) = jtj_temp(2,0) = row[0] * row[2];

					jtj_temp(1,1) = row[1] * row[1];
					jtj_temp(1,2) = jtj_temp(2,1) = row[1] * row[2];

					jtj_temp(2,2) = row[2] * row[2];

					jtr_temp(0,0) = row[0] * row[3];
					jtr_temp(1,0) = row[1] * row[3];
					jtr_temp(2,0) = row[2] * row[3];

					jtj = jtj + jtj_temp;
					jtr = jtr + jtr_temp;

					count++;
					error += row[3] * row[3];
				}
				
			}
		
		}
		SO3error = sqrt(error) / count;
		std::cout<<"\nSO3 error: "<<SO3error<<"\n";
		std::cout<<"\nlast SO3 error: "<<last_error<<"\n";
		if(SO3error < last_error && last_count == count)
        {
            break;
        }
		else if(SO3error > last_error + 0.001) //Diverging
        {
			SO3error = last_error;
			count = last_count;
			resultR = last_resultR;
			break;
        }

		last_error = SO3error;
		last_count = count;
		last_resultR = resultR;

		Eigen::Vector3d delta = jtj.ldlt().solve(jtr);
		resultDelta += delta;
		double n_norm = delta.norm();
		std::cout<<"\ndelta: "<<n_norm;
		Eigen::AngleAxisd rotation_vector (n_norm, delta/n_norm);
		Eigen::Matrix3d rotm;
		rotm = rotation_vector.toRotationMatrix();
		resultR = resultR * rotm;
		//Rt.topLeftCorner(3, 3) = rotm;
		//std::cout<<"\n"<<jtr<<"\n";

	}
	Eigen::Vector3d Trans_in = Pose_in.topRightCorner(3, 1);
	std::cout<<"\ndelta All: "<<resultDelta.norm();
	std::cout<<"\nin Trans: "<<Trans_in.norm();
	Rt.topLeftCorner(3, 3) = resultR.inverse();

	//if(resultDelta.norm()>0.05 || Trans_in.norm() > 0.02){
	
	if(resultDelta.norm() > 0.05|| Trans_in.norm() > 0.05){
		//Rt = Transform::Identity();
		Rt = Pose_in;
	}
	
	//std::cout<<"\n"<<Rt<<"\n";
	//Rt = Transform::Identity();
	//Rt.topRightCorner(3, 1) = Trans_in;
	return Rt;

}

bool EAS_ICP::RANSAC_strategy(const SourceCloud& srcCloud, const TargetCloud& tgtCloud){
	
	int samp_times = 50;
	float samp_rate = 0.4;
	int all_num = srcCloud.rows();
	int samp_point = (int)(all_num * samp_rate);

	int max_corrs_num = 0;
	Eigen::Vector<Scalar, 6> rt6D_ransac;
	Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> RANSAC_corrs_final;
	srand( time(NULL) );
	for (int i = 0; i<samp_times; i++){
		

		char src_subset[srcCloud.rows()] = {0};
		//Sampling 70% of point from source
		int total = 0;
		SourceCloud srcTemp(samp_point,6);
		SourceCloud srcCpy = srcCloud;
		while (total < samp_point){   
				
			int idx_rand = rand()%all_num;
			//std::cout<<"idx rand:"<<idx_rand<<"\n";
			if(src_subset[idx_rand] == 0){
				src_subset[idx_rand] = 1;
				total++;
			}

			//std::cout<<"final idx: "<<final_idx<<"\n";
		}

		int idx_tmp = 0;
		for (int n = 0; n < all_num; n++){
			if(src_subset[n] == 1){
				srcTemp.row(idx_tmp) = srcCloud.row(n);
				idx_tmp++;
			}
		}
		std::cout<<"\nidx_tmp: "<<idx_tmp;
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1> kinectNoiseWeights;
		kinectNoiseWeights = KinectNoiseWighting(srcTemp);
		MatchingByProject2DAndWalk(srcTemp, tgtCloud);

		//std::cout<<corrs;

		std::cout<<"\ncorr:"<<corrs.rows();
		//std::cout<<"\ncorr:"<<corrs;
		Transform SE3_ransac;
		rt6D_ransac = MinimizingP2PLErrorMetric(srcTemp(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		SE3_ransac = ConstructSE3(rt6D_ransac);

		srcCpy.leftCols(3) = ((SE3_ransac.topLeftCorner<3,3>()* srcCpy.leftCols<3>().transpose()).colwise() + SE3_ransac.col(3).head<3>()).transpose();
		srcCpy.rightCols(3) = (SE3_ransac.topLeftCorner<3,3>()* srcCpy.rightCols<3>().transpose()).transpose();
		MatchingByProject2DAndWalkRANSAC(srcCpy, tgtCloud);
		std::cout<<"\ncorr after:"<<corrs.rows();
		
		if(corrs.rows() > max_corrs_num){

			//Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> RANSAC_final;
			max_corrs_num = corrs.rows();
			RANSAC_corrs_final = corrs;
		}

		for (int z = 0; z < samp_point; z++){

			srcTemp.row(z)<<0, 0, 0, 0 ,0, 0;
		}

		std::cout<<"\nPose:\n"<<SE3_ransac;
	}
	//corrs = RANSAC_corrs_final;
	//std::cout<<"\nfinal map:"<<RANSAC_corrs_final.rows();
	//SourceCloud srcFinal(RANSAC_corrs_final.rows(),6);

	MatchingByProject2DAndWalk(srcCloud, tgtCloud);
	//char final_map[corrs.rows()] = {0};
	int replace_count = 0;
	for(int k = 0; k < RANSAC_corrs_final.rows(); k++){
		int replace_valid = 0;
		//final_map[RANSAC_corrs_final(k, 0)] = 1;
		
		for(int j= 0; j < corrs.rows(); j++){
			if(RANSAC_corrs_final(k,0) == corrs(j,0)){
				
				RANSAC_corrs_final.row(k) = corrs.row(j);
				//std::cout<<"\nk: "<<k;
				replace_valid = 1;
				replace_count++;
			}
		}
		if(replace_valid == 0){
			//std::cout<<"\nGot it: "<<k;
			RANSAC_corrs_final.row(k)<<9999,9999;
			
		}
	}
	Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> final_RANSAC(replace_count, 2);
	int idx_rans = 0;
	for(int k = 0; k < RANSAC_corrs_final.rows(); k++){
		if(RANSAC_corrs_final(k,0) != 9999 && RANSAC_corrs_final(k,1) != 9999){
			final_RANSAC.row(idx_rans) = RANSAC_corrs_final.row(k);
			idx_rans++;
		}
	}
	//corrs = RANSAC_corrs_final;
	//std::cout<<"\nRANSAC Final: "<<RANSAC_corrs_final;
	//srcFinal = srcCloud(RANSAC_corrs_final.col(0), Eigen::all);
	std::cout<<"\nidx_rans:"<<idx_rans;
	
	//std::cout<<"\nOrigin"<<corrs;
	//MatchingByProject2DAndWalk(srcFinal, tgtCloud);
	//std::cout<<"\nAfter RANSAC"<<corrs;
	std::cout<<"\nfinal_RANSAC: "<<final_RANSAC.rows();
	//std::cout<<"\nfinal_RANSAC: "<<final_RANSAC;
	if(final_RANSAC.rows() > 10){
		corrs = final_RANSAC;
	}


}
bool EAS_ICP::MatchingByProject2DAndWalkRANSAC(const SourceCloud& srcCloud, const TargetCloud& tgtCloud) {
  int size = srcCloud.rows();
  //correspondence declare
  std::vector<std::tuple<Scalar, int, int>> stdCorrs; 
  std::vector<double> residual_vector;

  int corr_cnt= 0;

  //for all correspondence
  for (int i = 0; i < size; ++i) {
    //a source point
    const Scalar& src_px = srcCloud(i, 0);
    const Scalar& src_py = srcCloud(i, 1);
    const Scalar& src_pz = srcCloud(i, 2);

    // project to 2D target frame
    int x_warp = fx / src_pz*src_px +cx;
    int y_warp = fy / src_pz*src_py +cy;
    //declare and initial variables
    Scalar min_distance = std::numeric_limits<Scalar>::max();
    int target_index = -1;

    //check the 2D point in target frame range
    if (x_warp >= width || y_warp >= height || x_warp < 0 || y_warp < 0)
    {
      continue;
    }

    //search range
    for (int ix = -search_range; ix < search_range + 1 ; ++ix)
    {
      for (int iy = -search_range; iy < search_range + 1 ; ++iy)
      {
        // search a circle range
        int grid_distance2 = ix*ix + iy*iy;
        if (grid_distance2 > search_range* search_range)
        {
          continue;
        }
        // x index and y index of target frame
        int x_idx = x_warp + ix * search_step;
        int y_idx = y_warp + iy * search_step;

        // avoid index out of target frame
        if (x_idx >= (width)
          || x_idx < 0
          || y_idx >=height
          || y_idx < 0)
        {
          continue;
        }

        //calculate 1D target frame index
        int tmp_index = (y_idx * width + x_idx);

        // get x,y,z of target point
        double tgt_px = tgtCloud(tmp_index,0);
        double tgt_py = tgtCloud(tmp_index,1);
        double tgt_pz = tgtCloud(tmp_index,2);

        //check nan
        if(
             (tgt_px!= tgt_px)||
             (tgt_py!= tgt_py)||
             (tgt_pz!= tgt_pz)
          ) continue; 


        //calculate the distance between source point and target point
        double distance = sqrt((src_px - tgt_px)*(src_px - tgt_px)
          + (src_py - tgt_py)*(src_py - tgt_py)
          + (src_pz - tgt_pz)*(src_pz - tgt_pz));

        // if new distance is less than min distance => record this index and distance
        if (distance < min_distance)
        {
          min_distance = distance;
          target_index = tmp_index;// target index: height x width x pointsize //pointsize is 6
        }
      }
    }
    //image boundary rejection
    //check closet point whether in the margin of boundary
    int target_x = target_index % width;
    int target_y = target_index / width;
    if (target_x > right_bound|| target_x < left_bound || target_y > bottom_bound || target_y < top_bound) {
      continue;
    }
    //check closet point existed and smaller fix threshold of rejection ==> if true, this pair is correspondence, and store in vector stdCorrs
    Eigen::Vector3d src_n = srcCloud.row(i).rightCols(3);
	SourceCloud Tgt_normal;
	std::vector<int> tgt_idx{ target_index };
	CalculateNormal(tgtCloud, tgt_idx, Tgt_normal);
	Eigen::Vector3d tgt_n2 = Tgt_normal.row(0).rightCols(3);
	double sine = tgt_n2.cross(src_n).norm();
	if ( min_distance !=  std::numeric_limits<Scalar>::max() && min_distance < 0.02 && sine < 0.2 && tgt_n2.norm() != 0)
    {
      stdCorrs.push_back(std::make_tuple(min_distance, i, target_index));
      residual_vector.push_back(min_distance);
      ++corr_cnt;
    }
  }
  //dynamic rejction
  //calculate the real index from the ratio of dynamic threshold
  int dynamic_threshold_index = dynamic_threshold_rejection * residual_vector.size();

  //check the index is not over the vector size
  if (dynamic_threshold_index < residual_vector.size()) {
      //rejection theshold(unit:m)
      //calculate the real value corresponded the index
      std::nth_element(residual_vector.begin(), residual_vector.begin() + dynamic_threshold_index, residual_vector.end());
      float reject_distance_threshold = 0; 
      //check the vector is no empty, or segmentation fault would occur.
      if (residual_vector.size() > 0)
        reject_distance_threshold = residual_vector[dynamic_threshold_index];
      
      //erase the correspondence over the dynamic threshold
      stdCorrs.erase(std::remove_if(stdCorrs.begin(), stdCorrs.end(), [reject_distance_threshold](const std::tuple<double, int,int>& elem){ return std::get<0>(elem)>reject_distance_threshold;}), stdCorrs.end());
  }

  //change the type for meeting the function output requirement
  int final_size = stdCorrs.size();
  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> correspondences(final_size, 2);
  for (int i = 0; i < stdCorrs.size(); ++i) {
    correspondences(i, 0) = std::get<1>(stdCorrs[i]);
    correspondences(i, 1) = std::get<2>(stdCorrs[i]);
  }
  correspondences.swap(corrs);

  //check the correspondence size over 6 for solving 6 rt valuables
  return corrs.rows() >=6;
}
const EAS_ICP::Transform& EAS_ICP::Register_EdgeAdd(const SourceCloud& srcCloud, const cv::Mat& depth, const cv::Mat& rgb, const TargetCloud& tgtCloud, const Transform& initialGuess, const cv::Mat& Last_intens) {

  //initial parameters
  std::cout<<"Meta Edge Add ICP\n";
  long t1, t2;
  t1 = clock();
  //RANSAC_strategy(srcCloud,tgtCloud);
  Transform initialGuess_I = Transform::Identity();
  Transform SO3Pose = SO3_prealign(initialGuess,rgb, 4);
  SO3Pose = SO3_prealign(SO3Pose,rgb, 1);
  //SO3Pose = SO3_prealign(SO3Pose,rgb, 1);
  
  rtSE3 = initialGuess;
  rtSE3_1 = initialGuess;
  rtSE3_2 = initialGuess;
  rtSE3_3 = initialGuess;
  rtSE3_4 = initialGuess;
  
  SourceCloud transformedCloudOri0(srcCloud.rows(), 6) ;
  transformedCloudOri0.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
  transformedCloudOri0.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
  //kinectNoiseWeights = KinectNoiseWighting(transformedCloudOri0);
  t2 = clock();
  std::cout<<"\nSO3 time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";

  MatchingByProject2DAndWalk(transformedCloudOri0, tgtCloud);
  /*
  t1 = clock();
  RANSAC_strategy(transformedCloudOri0, tgtCloud);
  Eigen::Vector<Scalar, 6> rt6D0;
  rt6D0 = MinimizingP2PLErrorMetric(transformedCloudOri0(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
  Transform iterRtSE30;
  iterRtSE30 = ConstructSE3(rt6D0);
  rtSE3 = rtSE3 * iterRtSE30;
  rtSE3_1 = rtSE3_1 * iterRtSE30;
  rtSE3_2 = rtSE3_2 * iterRtSE30;
  rtSE3_3 = rtSE3_3 * iterRtSE30;
  rtSE3_4 = rtSE3_4 * iterRtSE30;
  t2 = clock();
  std::cout<<"\nRANSAC time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
  */
 /*
  rtSE3 = Transform::Identity();
  rtSE3_1 = Transform::Identity();
  rtSE3_2 = Transform::Identity();
  rtSE3_3 = Transform::Identity();
  rtSE3_4 = Transform::Identity();
  */
  int cnt_task1_large =0;
  int cnt_task2_large =0;
  int cnt_task3_large =0;
  int cnt_task4_large =0;
  iterations = 0;
  accSlidingExtent = 0;
  Useonlyforeground = 1;
  Useweigtedrandomsampling = 1;
  Useedgeaware = 0;
  srand( time(NULL) );
  int iteration_divide = 20;

  last_inten = Last_intens;
  t1 = clock();
  computeDerivativeImages(last_rgb, dIdx, dIdy);
  while (true) {
  iterations+=1;
  //meta training
    /*double randomRGB = (double) rand() / (RAND_MAX + 1.0);
	double randomdepth = (double) rand() / (RAND_MAX + 1.0);
	std::cout<<"randomRGB"<<std::endl;
	std::cout<<randomRGB<<std::endl;
	std::cout<<"randomdepth"<<std::endl;
	std::cout<<randomdepth<<std::endl;*/
  //RGB ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  cv::Mat nan_map;
  cv::Mat rej_map;

  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  //cv::Mat inten;
  //cv::cvtColor(rgb, inten, cv::COLOR_BGR2GRAY);

  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::Mat edge_map_inner;
  cv::Mat edge_map_outer;
  cv::resize(src, out_resized, cv::Size(320, 240), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  //cv::blur(src, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  //dilation
  cv::Mat dilatemat1111 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11));
  cv::Mat dilatemat99 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
  cv::Mat dilatemat77 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat dilatemat55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::Mat dilatemat33 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));

  cv::dilate(edge_map2, edge_map_inner, dilatemat33);
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner;
  inv_edge_map_inner = ~edge_map_inner;
  cv::Mat edge_distance_map_inner;
  cv::distanceTransform(inv_edge_map_inner, edge_distance_map_inner, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  
  /////////////////////////dilate outer

 
  cv::dilate(edge_map2, edge_map_outer, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer, edge_map_inner, edge_map_outer);
  //cv::imwrite("ROI_detection_rgb_dilateouter.png", edge_map_outer);
  cv::Mat inv_edge_map_outer;
  inv_edge_map_outer = ~edge_map_outer;
  cv::Mat edge_distance_map_outer;
  //cv::distanceTransform(inv_edge_map_outer, edge_distance_map_outer, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2;
  std::vector<double> weights2;
  std::vector<int> EASInds2;

  
  //*********************************************
  //Depth ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  keycloud_depth = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  cv::Mat edge_map_depth;
  cv::Mat nan_map_depth;
  cv::Mat rej_map_depth;
  cv::Mat edge_map_inner_depth;
  EdgeDetection(*keycloud_depth, edge_map_depth, nan_map_depth, rej_map_depth);

  cv::dilate(edge_map_depth, edge_map_inner_depth, dilatemat33);
  //cv::imwrite("ROI_detection_depth_inner.png", edge_map_inner_depth);
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner_depth;
  inv_edge_map_inner_depth = ~edge_map_inner_depth;
  cv::Mat edge_distance_map_inner_depth;
  cv::distanceTransform(inv_edge_map_inner_depth, edge_distance_map_inner_depth, cv::DIST_L2, 5);
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds_depth;
  std::vector<double> weights_depth;
  std::vector<int> EASInds_depth;

  
  /////////////////////////dilate outer
  //cv::Mat dilatemat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat edge_map_outer_depth;

  cv::dilate(edge_map_depth, edge_map_outer_depth, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer_depth, edge_map_inner_depth, edge_map_outer_depth);
  
  cv::Mat inv_edge_map_outer_depth;
  inv_edge_map_outer_depth = ~edge_map_outer_depth;
  //cv::Mat edge_distance_map_outer_depth;
  //cv::distanceTransform(inv_edge_map_outer_depth, edge_distance_map_outer_depth, cv::DIST_L2, 5);
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2_depth;
  std::vector<double> weights2_depth;
  std::vector<int> EASInds2_depth;
  cv::Mat mapR1;
  cv::Mat mapR2;
  cv::Mat mapR3;
  cv::Mat mapR4;
  mapR1 = edge_map_inner_depth;
  mapR2 = edge_map_outer_depth;
  mapR3 = edge_map_inner;
  mapR4 = edge_map_outer;
  for(int i =0;i<mapR1.rows;i++)
  {
	for(int j =0;j<mapR1.cols;j++)
	{
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR1.at<uchar>(i,j) = 0;
		}		
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR2.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR3.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR4.at<uchar>(i,j) = 0;
		}
	}		
  }
  
	  
  //Before iteration 20
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_inner, nan_map_depth| rej_map_depth, remindPointInds, weights);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR3, nan_map_depth| rej_map_depth, remindPointInds, weights);
	
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1_before20);
	SourceCloud normal_samp;
	normal_samp = JustSampling(*keycloud);
	SourceCloud tmp(mSrcCloud1_before20.rows() + normal_samp.rows(), 6);
	tmp << mSrcCloud1_before20, normal_samp;
	mSrcCloud1_before20 = tmp;
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_outer, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR4, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2_before20);
	normal_samp = JustSampling(*keycloud);
	tmp.resize(mSrcCloud2_before20.rows() + normal_samp.rows(), 6);
	tmp << mSrcCloud2_before20, normal_samp;
	mSrcCloud2_before20 = tmp;
	
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_inner_depth, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR1, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3_before20);
	normal_samp = JustSampling(*keycloud);
	tmp.resize(mSrcCloud3_before20.rows() + normal_samp.rows(), 6);
	tmp << mSrcCloud3_before20, normal_samp;
	mSrcCloud3_before20 = tmp;
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_outer_depth, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR2, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4_before20);
	normal_samp = JustSampling(*keycloud);
	tmp.resize(mSrcCloud4_before20.rows() + normal_samp.rows(), 6);
	tmp << mSrcCloud4_before20, normal_samp;
	mSrcCloud4_before20 = tmp;
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	
	
  //After iteration 20
  //4 ROImap pixelwise OR 
  cv::Mat ROI4OR;
  //cv::bitwise_or(edge_map_inner_depth, edge_map_outer_depth, ROI4OR);
  cv::bitwise_or(edge_map_inner, edge_map_outer, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_inner_depth, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_outer_depth, ROI4OR);
  if(Useedgeaware==0)
	{	  
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth| rej_map_depth, remindPointInds, weights);
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
  else
  {
	// 1st 
	cv::Mat InvEdgeMap;
	cv::Mat EdgeDistanceMap;
	InvEdgeMap = ~edge_map_depth;
	cv::distanceTransform(InvEdgeMap, EdgeDistanceMap, cv::DIST_L2, 5);
 
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth| rej_map_depth, remindPointInds, weights);
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
    CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	
	// 2nd 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
		
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
		
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
		
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	
	
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
	kinectNoiseWeightsdepth_before20 = KinectNoiseWighting(mSrcCloud3_before20);
	kinectNoiseWeightsdepth2_before20 = KinectNoiseWighting(mSrcCloud4_before20);
	kinectNoiseWeightsrgb_before20 = KinectNoiseWighting(mSrcCloud1_before20);
	kinectNoiseWeightsrgb2_before20 = KinectNoiseWighting(mSrcCloud2_before20);
	kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	/*kinectNoiseWeightsdepth = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(srcCloud);*/
	//
	
	
	
	//sampling end
	
	
    //transform depth source cloud by inital guess
    //SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    //transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    //transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    SourceCloud transformedCloudrgb(mSrcCloud1.rows(), 6) ;
	SourceCloud transformedCloudrgb2(mSrcCloud2.rows(), 6) ;
	SourceCloud transformedClouddepth(mSrcCloud3.rows(), 6) ;
	SourceCloud transformedClouddepth2(mSrcCloud4.rows(), 6) ;
	SourceCloud transformedCloudrgb_before20(mSrcCloud1_before20.rows(), 6) ;
	SourceCloud transformedCloudrgb2_before20(mSrcCloud2_before20.rows(), 6) ;
	SourceCloud transformedClouddepth_before20(mSrcCloud3_before20.rows(), 6) ;
	SourceCloud transformedClouddepth2_before20(mSrcCloud4_before20.rows(), 6) ;
	/*SourceCloud transformedCloudrgb(srcCloud.rows(), 6) ;
	SourceCloud transformedCloudrgb2(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth2(srcCloud.rows(), 6) ;*/
	if(Useedgeaware==0)
	{
		if(iterations<=iteration_divide)
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1_before20.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2_before20.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3_before20.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4_before20.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4_before20.rightCols<3>().transpose()).transpose();
		}	
		
		else
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
		}
	}
	else
	{
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
		
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
    }	
	
	//TargetCloud LastCloud_1 = ((rtSE3_1.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
	//TargetCloud LastCloud_2 = ((rtSE3_2.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
	//TargetCloud LastCloud_3 = ((rtSE3_3.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
	//TargetCloud LastCloud_4 = ((rtSE3_4.topLeftCorner<3,3>()* pLastCloud->leftCols(3).transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
	
	TargetCloud LastCloud_1 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_2 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_3 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_4 = ComputeCurrentCloud(last_depth);
	//std::cout<<pLastCloud->leftCols(3);
	int cnttrans1, cnttrans2, cnttrans3, cnttrans4;
	cnttrans1 = 0;
	cnttrans2 = 0;
	cnttrans3 = 0;
	cnttrans4 = 0;
    int meanweight1, meanweight2, meanweight3, meanweight4;
    double randomx_thres = 0.1;
	if(Useweigtedrandomsampling==1)
	{
	  randomx_thres = 100;
	}
    //get iteration transformation by minimizing p2pl error metric

    Eigen::Vector<Scalar, 6> rt6D;
	
    if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)) {
			//std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all));
			//rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all));
			//std::cout<<"\ntransformedClouddepth_before20:\n"<<pLastCloud.rows()<<"\n\n";
			//std::cout<<"\nCorres:\n"<<corrs<<"\n\n";
			std::cout<<"hello~~~";
			//RANSAC_strategy(transformedClouddepth_before20, tgtCloud);
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_1, LastCloud_1);
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)) {
			//std::cout<<"jump1"<<std::endl;
			meanweight1 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D1 = MinimizingP2PLErrorMetric(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
			//rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsdepth(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_1, LastCloud_1);
		}
		else
		{
			rt6D1 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth(corrs.col(0), Eigen::all));
		}
		}
		//std::cout<<"cloudnumber"<<std::endl;
		//std::cout<<mSrcCloud3.rows()<<std::endl;

		meanweight1 = 1;
	}
    
	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2_before20, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all));
			//rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all));
			//RANSAC_strategy(transformedClouddepth2_before20, tgtCloud);
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsdepth2_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_2, LastCloud_2);
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)) {
			//std::cout<<"jump2"<<std::endl;
			meanweight2 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedClouddepth2, tgtCloud)){
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D2 = MinimizingP2PLErrorMetric(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
			//rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_2, LastCloud_2);
		}
		else
		{
			rt6D2 = MinimizingP2PLErrorMetricGaussianNewton(transformedClouddepth2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsdepth2(corrs.col(0), Eigen::all));
		}
		}	

		meanweight2 = 1;
	}


	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb_before20, tgtCloud)) {
		//std::cout<<"cnttrans3"<<std::endl;
		//std::cout<<cnttrans3<<std::endl;
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all));
			//rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all));
			//RANSAC_strategy(transformedCloudrgb_before20, tgtCloud);
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_3, LastCloud_3);
		}	
		else
		{
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {
			//std::cout<<"jump3"<<std::endl;
			meanweight3 =0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb, tgtCloud)) {

		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D3 = MinimizingP2PLErrorMetric(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
			//rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_3, LastCloud_3);
		}	
		else 
		{
			rt6D3 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb(corrs.col(0), Eigen::all));
		}
		meanweight3 = 1;
		}
	}
	
	if(iterations<=iteration_divide)
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2_before20, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all));
			//rt6D4 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_4, LastCloud_4);
			//RANSAC_strategy(transformedCloudrgb2_before20, tgtCloud);
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb2_before20(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb2_before20(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_4, LastCloud_4);
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
		}

		meanweight4 = 1;
		}
	}
	else
	{
		//match correspondence
		if (!MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
			//std::cout<<"jump4"<<std::endl;
			meanweight4 = 0;
		//break; // when correspondence size less than 6
		}
		if (MatchingByProject2DAndWalk(transformedCloudrgb2, tgtCloud)) {
		//get iteration transformation by minimizing p2pl error metric
		if(Useedgeaware==0)
		{
			//rt6D4 = MinimizingP2PLErrorMetric(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
			//rt6D4 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3_4, LastCloud_4);
		}
		else
		{
			rt6D4 = MinimizingP2PLErrorMetricGaussianNewton(transformedCloudrgb2(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeightsrgb2(corrs.col(0), Eigen::all));
			
		}
		//std::cout<<"cnttrans4"<<std::endl;
		//std::cout<<cnttrans4<<std::endl;
		meanweight4 = 1;
		}
	}	
	//std::cout<<"corrs4num"<<std::endl;
	//std::cout<<corrs.rows()<<std::endl;
	// ++ meta task loop end
	// ++ pose fusion = mean(all tasks' rt6D)
	
	if((meanweight1+meanweight2+meanweight3+meanweight4)==0)
	{
		break;
	}
	/*else{

    rt6D(0) = (rt6D1(0)*meanweight1+ rt6D2(0)*meanweight2 +rt6D3(0)*meanweight3 +rt6D4(0)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(1) = (rt6D1(1)*meanweight1+ rt6D2(1)*meanweight2 +rt6D3(1)*meanweight3 +rt6D4(1)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
    rt6D(2) = (rt6D1(2)*meanweight1+ rt6D2(2)*meanweight2 +rt6D3(2)*meanweight3 +rt6D4(2)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(3) = (rt6D1(3)*meanweight1+ rt6D2(3)*meanweight2 +rt6D3(3)*meanweight3 +rt6D4(3)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(4) = (rt6D1(4)*meanweight1+ rt6D2(4)*meanweight2 +rt6D3(4)*meanweight3 +rt6D4(4)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	rt6D(5) = (rt6D1(5)*meanweight1+ rt6D2(5)*meanweight2 +rt6D3(5)*meanweight3 +rt6D4(5)*meanweight4)/(meanweight1+meanweight2+meanweight3+meanweight4);
	}*/
    //convert 6D vector to SE3
    Eigen::Vector<Scalar, 6> rt6D_L1_largest;
	double s1, s2, s3, s4;
	
	s1 = fabs(rt6D1(0))+fabs(rt6D1(1))+fabs(rt6D1(2));
	s2 = fabs(rt6D2(0))+fabs(rt6D2(1))+fabs(rt6D2(2));
	s3 = fabs(rt6D3(0))+fabs(rt6D3(1))+fabs(rt6D3(2));
	s4 = fabs(rt6D4(0))+fabs(rt6D4(1))+fabs(rt6D4(2));
	
	/*
	s1 = fabs(rt6D1(3))+fabs(rt6D1(4))+fabs(rt6D1(5));
	s2 = fabs(rt6D2(4))+fabs(rt6D2(4))+fabs(rt6D2(5));
	s3 = fabs(rt6D3(4))+fabs(rt6D3(4))+fabs(rt6D3(5));
	s4 = fabs(rt6D4(4))+fabs(rt6D4(4))+fabs(rt6D4(5));
	*/
	if((s1>=s2)&&(s1>=s3)&&(s1>=s4))
	{
		std::cout<<"\n\nHAHSAHAH\n\n";
		rt6D_L1_largest=rt6D1;
		cnt_task1_large++;
	}
	if((s2>=s1)&&(s2>=s3)&&(s2>=s4))
	{
		rt6D_L1_largest=rt6D2;
		cnt_task2_large++;
	}
	if((s3>=s2)&&(s3>=s1)&&(s3>=s4))
	{
		rt6D_L1_largest=rt6D3;
		cnt_task3_large++;
	}
	if((s4>=s2)&&(s4>=s3)&&(s4>=s1))
	{
		rt6D_L1_largest=rt6D4;
		cnt_task4_large;
	}
	
	Transform iterRtSE3;
	Transform Pre_RtSE3;
	//Eigen::Vector3f Pre_trans(rt6D_L1_largest(0),rt6D_L1_largest(1),rt6D_L1_largest(2));
	Eigen::Vector3f Pre_trans(rt6D_L1_largest(3),rt6D_L1_largest(4),rt6D_L1_largest(5));
    iterRtSE3 = ConstructSE3_GN(rt6D_L1_largest);
	//iterRtSE3 = ConstructSE3(rt6D_L1_largest);
    //chain iterRtSE3 to rtSE3
	std::cout<<iterRtSE3.col(3).head<3>().transpose()<<"\n";
	std::cout<<"\npre: "<<Pre_trans;
	std::cout<<"\nnorm: "<<Pre_trans.norm();
	if(Pre_trans.norm() > 0.3){
		rtSE3_1 = rtSE3_1;
		rtSE3_2 = rtSE3_2;
		rtSE3_3 = rtSE3_3;
		rtSE3_4 = rtSE3_4;
	}
	else{
    	rtSE3_1 = iterRtSE3 * rtSE3_1;
		rtSE3_2 = iterRtSE3 * rtSE3_2;
		rtSE3_3 = iterRtSE3 * rtSE3_3;
		rtSE3_4 = iterRtSE3 * rtSE3_4;
		rtSE3 = rtSE3 * iterRtSE3;
	}
    //chain iterRtSE3 to rtSE3
    

    //chain iterRtSE3 to rtSE3
    

    //chain iterRtSE3 to rtSE3
    
    //check termination
    /*if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }*/
	max_iters = 30;
	if (iterations > max_iters)
	{
		std::cout<<"4 tasks count"<<std::endl;
		std::cout<<cnt_task1_large<<std::endl;
		std::cout<<cnt_task2_large<<std::endl;
		std::cout<<cnt_task3_large<<std::endl;
		std::cout<<cnt_task4_large<<std::endl;		
		break;
	}
  }
  t2 = clock();
  std::cout<<"\nMeta Training time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
  
  
  
  //meta testing
  //icp loop 2 START
  iteration_loop2 = 0;
  t1 = clock();
  while (true) {
  iteration_loop2+=1;
  std::cout<<"\niter2: "<<iteration_loop2;
  //RGB ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  cv::Mat nan_map;
  cv::Mat rej_map;
  //generate point cloud
  keycloud = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  //generate nan map& rej map
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> ((*keycloud).data()+2, (*keycloud).rows());
  
  nan_map = cv::Mat::zeros(height, width, CV_8UC1);
  rej_map = cv::Mat::zeros(height, width, CV_8UC1);
  uchar *nan_map1 = nan_map.data,
           *rej_map1 = rej_map.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map1[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map1[i*width + j] = 255;
	    }
    }
  }
  //canny edge
  //read rgb files
  cv::Mat src;
  src = rgb;
  cv::Mat canny_edge_map;
  cv::Mat out_resized;
  cv::Mat edge_map2;
  cv::Mat edge_map_inner;
  cv::Mat edge_map_outer;
  cv::resize(src, out_resized, cv::Size(320, 240), cv::INTER_CUBIC);

  cv::Mat blurred;
  cv::blur(out_resized, blurred, cv::Size(3, 3));
  //cv::blur(src, blurred, cv::Size(3, 3));
  cv::Canny(blurred, canny_edge_map, 100, 150);
  cv::resize(canny_edge_map, edge_map2, cv::Size(640, 480), cv::INTER_CUBIC);
  //dilation
  cv::Mat dilatemat55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::Mat dilatemat33 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
  cv::dilate(edge_map2, edge_map_inner, dilatemat33);

  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner;
  inv_edge_map_inner = ~edge_map_inner;
  cv::Mat edge_distance_map_inner;
  cv::distanceTransform(inv_edge_map_inner, edge_distance_map_inner, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  std::vector<int> EASInds;
  
  /////////////////////////dilate outer
  cv::Mat dilatemat99 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
  cv::Mat dilatemat77 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::dilate(edge_map2, edge_map_outer, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer, edge_map_inner, edge_map_outer);
  
  cv::Mat inv_edge_map_outer;
  inv_edge_map_outer = ~edge_map_outer;
  cv::Mat edge_distance_map_outer;
  //cv::distanceTransform(inv_edge_map_outer, edge_distance_map_outer, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2;
  std::vector<double> weights2;
  std::vector<int> EASInds2;

  
  //*********************************************
  //Depth ROI detection and transformed to pointcloud
  //*********************************
  //edge detection
  keycloud_depth = std::make_unique<CurrentCloudrgb>(ComputeCurrentCloud(depth));
  cv::Mat edge_map_depth;
  cv::Mat nan_map_depth;
  cv::Mat rej_map_depth;
  cv::Mat edge_map_inner_depth;
  EdgeDetection(*keycloud_depth, edge_map_depth, nan_map_depth, rej_map_depth);
  
  cv::dilate(edge_map_depth, edge_map_inner_depth, dilatemat33);
  
  //calculate edge distance map
  
  cv::Mat inv_edge_map_inner_depth;
  inv_edge_map_inner_depth = ~edge_map_inner_depth;
  cv::Mat edge_distance_map_inner_depth;
  cv::distanceTransform(inv_edge_map_inner_depth, edge_distance_map_inner_depth, cv::DIST_L2, 5);
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds_depth;
  std::vector<double> weights_depth;
  std::vector<int> EASInds_depth;

  
  /////////////////////////dilate outer
  //cv::Mat dilatemat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
  cv::Mat edge_map_outer_depth;
  cv::dilate(edge_map_depth, edge_map_outer_depth, dilatemat77);
  //outer - inner 
  cv::subtract(edge_map_outer_depth, edge_map_inner_depth, edge_map_outer_depth);
  cv::Mat inv_edge_map_outer_depth;
  inv_edge_map_outer_depth = ~edge_map_outer_depth;
  //cv::Mat edge_distance_map_outer_depth;
  //cv::distanceTransform(inv_edge_map_outer_depth, edge_distance_map_outer_depth, cv::DIST_L2, 5);
   
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds2_depth;
  std::vector<double> weights2_depth;
  std::vector<int> EASInds2_depth;
  cv::Mat mapR1;
  cv::Mat mapR2;
  cv::Mat mapR3;
  cv::Mat mapR4;
  mapR1 = edge_map_inner_depth;
  mapR2 = edge_map_outer_depth;
  mapR3 = edge_map_inner;
  mapR4 = edge_map_outer;
  for(int i =0;i<mapR1.rows;i++)
  {
	for(int j =0;j<mapR1.cols;j++)
	{
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR1.at<uchar>(i,j) = 0;
		}		
		if((edge_map_inner.at<uchar>(i,j)>0)||(edge_map_outer.at<uchar>(i,j)>0))
		{
			mapR2.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR3.at<uchar>(i,j) = 0;
		}	
		if((edge_map_inner_depth.at<uchar>(i,j)>0)||(edge_map_outer_depth.at<uchar>(i,j)>0))
		{
			mapR4.at<uchar>(i,j) = 0;
		}
	}		
  }
  
  //Before iteration 20
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_inner, nan_map_depth| rej_map_depth, remindPointInds, weights);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR3, nan_map_depth| rej_map_depth, remindPointInds, weights);
	
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1_before20);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, edge_map_outer, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, mapR4, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2_before20);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_inner_depth, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR1, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3_before20);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	//PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, edge_map_outer_depth, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, mapR2, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4_before20);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	
	
  //After iteration 20
  //4 ROImap pixelwise OR 
  cv::Mat ROI4OR;
  //cv::bitwise_or(edge_map_inner_depth, edge_map_outer_depth, ROI4OR);
  cv::bitwise_or(edge_map_inner, edge_map_outer, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_inner_depth, ROI4OR);
  cv::bitwise_or(ROI4OR, edge_map_outer_depth, ROI4OR);
  if(Useedgeaware==0)
	{	  
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth| rej_map_depth, remindPointInds, weights);
	// 1st 
	EASInds = remindPointInds;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//weighting
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2, weights2);

	// 2nd 
	EASInds2 = remindPointInds2;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
	//calculate normal
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	//weighting
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	EASInds_depth = remindPointInds_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
	//weighting
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, ROI4OR, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	EASInds2_depth = remindPointInds2_depth;
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	//calculate normal
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
  else
  {
	// 1st 
	cv::Mat InvEdgeMap;
	cv::Mat EdgeDistanceMap;
	InvEdgeMap = ~edge_map_depth;
	cv::distanceTransform(InvEdgeMap, EdgeDistanceMap, cv::DIST_L2, 5);
 
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth| rej_map_depth, remindPointInds, weights);
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
	}
    CalculateNormal(*keycloud, EASInds, mSrcCloud1);
	//kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2, weights2);
	
	// 2nd 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2, weights2, EASInds2);
	}
		
	CalculateNormal(*keycloud, EASInds2, mSrcCloud2);
	
	//kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	//3rd
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds_depth, weights_depth);
  
	// 1st sample all points 
	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds_depth, weights_depth, EASInds_depth);
	}
		
	CalculateNormal(*keycloud_depth, EASInds_depth, mSrcCloud3);
		
	//kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	// 4th
	PointRejectionByDepthRangeAndGeometryWeight2(*keycloud_depth, EdgeDistanceMap, nan_map_depth | rej_map_depth, remindPointInds2_depth, weights2_depth);

	
	if(Useweigtedrandomsampling==1)
	{
		WeightedRandomSampling(sampling_size, remindPointInds2_depth, weights2_depth, EASInds2_depth);
	}
	
	CalculateNormal(*keycloud_depth, EASInds2_depth, mSrcCloud4);
	
	
	//weighting
	//kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	}
	kinectNoiseWeightsdepth_before20 = KinectNoiseWighting(mSrcCloud3_before20);
	kinectNoiseWeightsdepth2_before20 = KinectNoiseWighting(mSrcCloud4_before20);
	kinectNoiseWeightsrgb_before20 = KinectNoiseWighting(mSrcCloud1_before20);
	kinectNoiseWeightsrgb2_before20 = KinectNoiseWighting(mSrcCloud2_before20);
	kinectNoiseWeightsdepth = KinectNoiseWighting(mSrcCloud3);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(mSrcCloud4);
	kinectNoiseWeightsrgb = KinectNoiseWighting(mSrcCloud1);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(mSrcCloud2);
	/*kinectNoiseWeightsdepth = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsdepth2 = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb = KinectNoiseWighting(srcCloud);
	kinectNoiseWeightsrgb2 = KinectNoiseWighting(srcCloud);*/
	//
	
	
	
	//sampling end
	
	
    //transform depth source cloud by inital guess
    //SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    //transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    //transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();
    SourceCloud transformedCloudrgb(mSrcCloud1.rows(), 6) ;
	SourceCloud transformedCloudrgb2(mSrcCloud2.rows(), 6) ;
	SourceCloud transformedClouddepth(mSrcCloud3.rows(), 6) ;
	SourceCloud transformedClouddepth2(mSrcCloud4.rows(), 6) ;
	SourceCloud transformedCloudrgb_before20(mSrcCloud1_before20.rows(), 6) ;
	SourceCloud transformedCloudrgb2_before20(mSrcCloud2_before20.rows(), 6) ;
	SourceCloud transformedClouddepth_before20(mSrcCloud3_before20.rows(), 6) ;
	SourceCloud transformedClouddepth2_before20(mSrcCloud4_before20.rows(), 6) ;
	/*SourceCloud transformedCloudrgb(srcCloud.rows(), 6) ;
	SourceCloud transformedCloudrgb2(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth(srcCloud.rows(), 6) ;
	SourceCloud transformedClouddepth2(srcCloud.rows(), 6) ;*/
	if(Useedgeaware==0)
	{
		if(iteration_loop2<=iteration_divide)
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud1_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud1_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud2_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud3_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud3_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud4_before20.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud4_before20.rightCols<3>().transpose()).transpose();
		
		/*//transform rgb source cloud by inital guess
		transformedCloudrgb_before20.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb_before20.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb_before20.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb_before20.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2_before20.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2_before20.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2_before20.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2_before20.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth_before20.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth_before20.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth_before20.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth_before20.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2_before20.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2_before20.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2_before20.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2_before20.rightCols<3>().transpose()).transpose();
		*/}	
		
		else
		{
		//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
		
		/*//transform rgb source cloud by inital guess
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* transformedCloudrgb.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* transformedCloudrgb2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* transformedClouddepth.rightCols<3>().transpose()).transpose();
    
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* transformedClouddepth2.rightCols<3>().transpose()).transpose();
		*/}
	}
	else
	{
		transformedCloudrgb.leftCols(3) = ((rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.leftCols<3>().transpose()).colwise() + rtSE3_3.col(3).head<3>()).transpose();
		transformedCloudrgb.rightCols(3) = (rtSE3_3.topLeftCorner<3,3>()* mSrcCloud1.rightCols<3>().transpose()).transpose();
		//transform rgb cloud cloud by initial guess
		transformedCloudrgb2.leftCols(3) = ((rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.leftCols<3>().transpose()).colwise() + rtSE3_4.col(3).head<3>()).transpose();
		transformedCloudrgb2.rightCols(3) = (rtSE3_4.topLeftCorner<3,3>()* mSrcCloud2.rightCols<3>().transpose()).transpose();
		//transform depth source cloud by inital guess
		transformedClouddepth.leftCols(3) = ((rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.leftCols<3>().transpose()).colwise() + rtSE3_1.col(3).head<3>()).transpose();
		transformedClouddepth.rightCols(3) = (rtSE3_1.topLeftCorner<3,3>()* mSrcCloud3.rightCols<3>().transpose()).transpose();
		
		//transform depth source cloud by inital guess
		transformedClouddepth2.leftCols(3) = ((rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.leftCols<3>().transpose()).colwise() + rtSE3_2.col(3).head<3>()).transpose();
		transformedClouddepth2.rightCols(3) = (rtSE3_2.topLeftCorner<3,3>()* mSrcCloud4.rightCols<3>().transpose()).transpose();
    }

	TargetCloud LastCloud_1 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_2 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_3 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloud_4 = ComputeCurrentCloud(last_depth);
	TargetCloud LastCloudOri = ComputeCurrentCloud(last_depth);

	int cnttrans1, cnttrans2, cnttrans3, cnttrans4;
	cnttrans1 = 0;
	cnttrans2 = 0;
	cnttrans3 = 0;
	cnttrans4 = 0;
    int meanweight1, meanweight2, meanweight3, meanweight4;
    double randomx_thres = 0.1;
	if(Useweigtedrandomsampling==1)
	{
	  randomx_thres = 100;
	}
   
    Eigen::Vector<Scalar, 6> rt6D;

    SourceCloud tmp;
	SourceCloud normal_samp;
	SourceCloud transformedCloudOri(srcCloud.rows(), 6) ;
	normal_samp = JustSampling(*keycloud);
	tmp.resize(srcCloud.rows() + normal_samp.rows(), 6);
	tmp << srcCloud, normal_samp;
	transformedCloudOri.resize(tmp.rows(), 6);
	transformedCloudOri = tmp;
	kinectNoiseWeights = KinectNoiseWighting(transformedCloudOri);

    transformedCloudOri.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* transformedCloudOri.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    transformedCloudOri.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* transformedCloudOri.rightCols<3>().transpose()).transpose();
	if (!MatchingByProject2DAndWalk(transformedCloudOri, tgtCloud)) {
		break; // when correspondence size less than 6
		}
	/*
	if(iteration_loop2 == 1){
		RANSAC_strategy(transformedCloudOri, tgtCloud);
	}
	*/
	std::cout<<"\ncorrs: "<<corrs.rows();
	//rt6D = rt6D1;
	rt6D = MinimizingP2PLErrorMetric(transformedCloudOri(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
	//rt6D = MinimizingP2PLErrorMetricGaussianNewtonRGB(transformedCloudOri(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all),  kinectNoiseWeights(corrs.col(0), Eigen::all), depth ,rgb, last_rgb, rtSE3, LastCloudOri);
    Transform iterRtSE3;
	Eigen::Vector3f Pre_trans(rt6D(0),rt6D(1),rt6D(2));

    iterRtSE3 = ConstructSE3(rt6D);

    //chain iterRtSE3 to rtSE3
    //rtSE3 = iterRtSE3 * rtSE3;
	std::cout<<iterRtSE3.col(3).head<3>().transpose()<<"\n";
	std::cout<<"\npre: "<<Pre_trans;
	std::cout<<"\nnorm: "<<Pre_trans.norm();
	if(Pre_trans.norm() > 0.3){
		rtSE3 = rtSE3;
	}
	else{
		rtSE3 = rtSE3 * iterRtSE3;
	}
    //check termination
    /*if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }*/
	max_iters = 20;
	if (iteration_loop2 > max_iters)
	{
		break;
	}
  }
  t2 = clock();
  std::cout<<"\nMeta Testing time: "<<(t2-t1)/(double)CLOCKS_PER_SEC<<"\n";
  valid = true;
  return rtSE3;
}