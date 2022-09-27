/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICP_VO.cpp

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿年九月一日 (週二) 廿一時九分一秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include "ICP_VO.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
ICP_VO::ICP_VO(const std::string& strSettings):
  pKeyFrame(nullptr),
  pCurrentCloud(nullptr)
{
  backupCloud.second = nullptr;
  cv::FileStorage FsSettings(strSettings.c_str(), cv::FileStorage::READ); 
  pICP = std::make_unique<EAS_ICP> (strSettings);
  //initialize parameters
  //TODO initial from settings
  mDepthMapFactor = FsSettings["DepthMapFactor"];
  //mDepthMapFactor = 5000;
  bUseBackup = (int)(FsSettings["VO.use_backup"]);
  bPredition = (int)(FsSettings["VO.use_prediction"]);
  mThresKeyframeUpdateRot   = FsSettings["VO.keyframe_update_threshold_rot"];
  mThresKeyframeUpdateTrans = FsSettings["VO.keyframe_update_threshold_trans"];
}


ICP_VO::~ICP_VO() {
}
const ICP_VO::Pose& ICP_VO::Track(const cv::Mat& depth, const cv::Mat& rgb, const double& timestamp){
  //calculate 3D points
  pCurrentCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
  //auto pt = pCurrentCloud.get();
  //std::cout<<pt[0]<<"\n";
  return Track(timestamp, rgb, depth);
}
const ICP_VO::Pose& ICP_VO::Track(const cv::Mat& depth, const double& timestamp){
  //calculate 3D points
  pCurrentCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
  return Track(timestamp);
}
const ICP_VO::Pose& ICP_VO::Track(const Cloud& _cloud, const cv::Mat& rgb,  const double& timestamp){
  //calculate 3D points
  pCurrentCloud = std::make_unique<CurrentCloud>(_cloud);
  return Track(timestamp, rgb);
}
const ICP_VO::Pose& ICP_VO::Track(const Cloud& _cloud,  const double& timestamp){
  //calculate 3D points
  pCurrentCloud = std::make_unique<CurrentCloud>(_cloud);
  return Track(timestamp);
}
const ICP_VO::Pose& ICP_VO::Track(const double& timestamp, const cv::Mat& rgb)
{
  cv::Mat rgb1 = rgb;
  cv::Mat rgb2 = rgb;
  cv::Mat rgb3 = rgb;
  //initialize VO if first frame, or track current frame
  if (mPoses.size() == 0) {
    rpK2C = RelativePose::Identity();
    pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud, rgb1), timestamp, Pose::Identity());
    mPoses.push_back(Pose::Identity());
  } else {
    //calculate prediction
    RelativePose pred, predK2C;

    //guess the initial pose
    if (bPredition && mPoses.size() > 1) {
      pred = mPoses.rbegin()[0].inverse()* mPoses.rbegin()[1];
      predK2C = pred*rpK2C;
    } else {
      pred = RelativePose::Identity();
      predK2C = RelativePose::Identity();
    };
    //perform ICP
    rpK2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, predK2C);

    //check ICP valid
    if (bUseBackup && !pICP->isValid() && backupCloud.second != nullptr && mPoses.size() != pKeyFrame->id) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*backupCloud.second, rgb2), backupCloud.first, mPoses.back());
      RelativePose rpB2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, pred);
      if (pICP->isValid()) {
        rpK2C = rpB2C;
      } else {
        rpK2C = pred;
      }
    } else if(bUseBackup && !pICP->isValid() && backupCloud.second != nullptr){
      rpK2C = pred;
    }

    //calculate current pose from keypose
    Pose currentPose = pKeyFrame->pose * (rpK2C.inverse());
    mPoses.push_back(currentPose);

    //check update keyframe
    if (CheckUpdateKeyFrame(rpK2C)) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud, rgb3), timestamp, mPoses.back());
      rpK2C = RelativePose::Identity();
    }

    //backup cloud
    if (bUseBackup) {
      backupCloud.first = timestamp;
      backupCloud.second = std::move(pCurrentCloud);
    } 
  }
  return mPoses.back();
}
const ICP_VO::Pose& ICP_VO::Track(const double& timestamp, const cv::Mat& rgb, const cv::Mat& depth)
{
  std::cout<<"Track!!!!\n";
  
  cv::Mat rgb1 = rgb;
  cv::Mat rgb2 = rgb;
  cv::Mat rgb3 = rgb;
  cv::Mat inten;
  //cv::Vec3b t = rgb.at<cv::Vec3b>(0, 1);
  //cv::Vec3b t2 = lastRGB.at<cv::Vec3b>(0, 1);
  //std::cout<<rgb - lastRGB;

  //std::cout<<"\nt:  "<<t;
  //std::cout<<"\nt2:  "<<t2;

  //std::cout<<"Cloud:\n"<<*pCurrentCloud<<"\n";
  //initialize VO if first frame, or track current frame
  float depth_mean;
  if(true){
    cv::Mat edge_map;
    cv::Mat nan_map;
    cv::Mat rej_map;
    pICP->EdgeDetection(*pCurrentCloud, edge_map, nan_map, rej_map);
    std::cout<<"\nedge MEAN: "<<cv::mean(edge_map).val[0];
    std::string filename("depth_edge.txt");
    std::ofstream file_out;
    file_out.open(filename, std::ios_base::app | std::ios_base::in);
    file_out << cv::mean(edge_map).val[0]<<"\n";
    depth_mean = cv::mean(edge_map).val[0];
    cv::Mat m2, depthCanny;
    cv::normalize(depth, m2, 0, 255, cv::NORM_MINMAX);
    m2.convertTo(m2, CV_8U);
    cv::Canny(m2, depthCanny, 150, 100);

    double depth_quality = cv::mean(depthCanny).val[0] / cv::mean(edge_map).val[0];
    cv::Mat dIdx_quality;
    cv::Mat dIdy_quality;
    pICP->computeDerivativeImages(rgb, dIdx_quality, dIdy_quality);
    cv::Mat final_edges_dI = cv::abs(dIdx_quality | dIdy_quality);
    cv::Mat final_edges;
    cv::Canny(rgb, final_edges, 150, 100);
    /*
    std::cout<<"\n\n"<<cv::mean(final_edges_dI).val[0]<<"\n";
    std::cout<<"\n\n"<< cv::mean(final_edges).val[0]<<"\n";
    */
    double rgb_quality = cv::mean(final_edges).val[0] / cv::mean(final_edges_dI).val[0];
    //std::cout<<"\n\n"<< ans<<"\n";
    //cv::imwrite("rgb_edge_xy.jpg", final_edges);
    //std::cout<<"\nMean: "<<cv::mean(final_edges).val[0];
    std::string filename2("rgb_quality.txt");
    std::ofstream file_out2;
    file_out2.open(filename2, std::ios_base::app | std::ios_base::in);
    //file_out2 << cv::mean(final_edges).val[0]<<"\n";
    file_out2 <<rgb_quality<<"\n";

    std::string filename3("depth_quality.txt");
    std::ofstream file_out3;
    file_out3.open(filename3, std::ios_base::app | std::ios_base::in);
    //file_out2 << cv::mean(final_edges).val[0]<<"\n";
    file_out3 <<depth_quality<<"\n";
    pICP->rgb_quality = rgb_quality;
    pICP->depth_quality = depth_quality;
  }
  
  if (mPoses.size() == 0) {
    rpK2C = RelativePose::Identity();
    pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, Pose::Identity(), rgb);
	mPoses.push_back(Pose::Identity());
	keyframedepthmap = depth * mDepthMapFactor/1000;
  cv::cvtColor(rgb, inten, cv::COLOR_BGR2GRAY);
  keyframeintenmap = inten;
  /*
  pICP->last_rgb = rgb.clone();
  pICP->last_depth = depth.clone();
  */
  //std::cout<<"keyframe:"<<keyframedepthmap<<"\n";
  } 
  else {

    pICP->next_depth = depth.clone();
    //calculate prediction
    RelativePose pred, predK2C;

    //guess the initial pose
    if (bPredition && mPoses.size() > 1) {
      pred = mPoses.rbegin()[0].inverse()* mPoses.rbegin()[1];
      predK2C = pred*rpK2C;
    } else {
      pred = RelativePose::Identity();
      predK2C = RelativePose::Identity();
    }
    //perform ICP

    //std::cout<<cv::sum(keyframergbmap) - cv::sum(inten);
    pICP->pLastCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(pICP->last_depth));
    if(depth_mean >1.0 && pICP->depth_quality > 1.0){
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pICP->pLastCloud), timestamp - 1, mPoses.back(), pICP->last_rgb);
    }
    else
    {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pICP->pLastCloud), timestamp - 1, mPoses.back(), pICP->last_rgb);
      //pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->JustSampling(*pICP->pLastCloud), timestamp - 1, mPoses.back(), pICP->last_rgb);
      //pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->JustSampling(*pICP->pLastCloud), timestamp - 1, mPoses.back(), pICP->last_rgb);
    }
    //pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->JustSampling(*pICP->pLastCloud), timestamp - 1, mPoses.back(), pICP->last_rgb);
    keyframedepthmap = pICP->last_depth * mDepthMapFactor/1000;
    keyframeintenmap = pICP->last_inten;

    cv::cvtColor(rgb, inten, cv::COLOR_BGR2GRAY);
    //rpK2C = pICP->Register(pKeyFrame->keyCloud, keyframedepthmap, rgb, *pCurrentCloud, predK2C, keyframeintenmap);
    //rpK2C = pICP->Register_Ori(pKeyFrame->keyCloud, keyframedepthmap, rgb, *pCurrentCloud, predK2C);
     if(depth_mean >1.0 && pICP->depth_quality > 1.0){
      rpK2C = pICP->Register(pKeyFrame->keyCloud, keyframedepthmap, rgb, *pCurrentCloud, predK2C, keyframeintenmap);
      //rpK2C = pICP->Register_EdgeAdd(pKeyFrame->keyCloud, keyframedepthmap, rgb, *pCurrentCloud, predK2C, keyframeintenmap);
    }
    else
    {
      rpK2C = pICP->Register_EdgeAdd(pKeyFrame->keyCloud, keyframedepthmap, rgb, *pCurrentCloud, predK2C, keyframeintenmap);
      //rpK2C = pICP->RegisterPure(pKeyFrame->keyCloud, keyframedepthmap, rgb, *pCurrentCloud, predK2C, keyframeintenmap);
    }
    //rpK2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, predK2C);
    //check ICP valid
    if (bUseBackup && !pICP->isValid() && backupCloud.second != nullptr && mPoses.size() != pKeyFrame->id) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*backupCloud.second), backupCloud.first, mPoses.back(), rgb);
      RelativePose rpB2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, pred);
      if (pICP->isValid()) {
        rpK2C = rpB2C;
      } else {
        rpK2C = pred;
      }
    } else if(bUseBackup && !pICP->isValid() && backupCloud.second != nullptr){
      rpK2C = pred;
    }

    //calculate current pose from keypose
    Pose currentPose = pKeyFrame->pose * (rpK2C.inverse());
    mPoses.push_back(currentPose);

    //check update keyframe
    if (CheckUpdateKeyFrame(rpK2C)) {
	 // std::cout<<"keyframestart"<<std::endl;
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, mPoses.back(), rgb);
      //std::cout<<"keyframeend"<<std::endl;
	  rpK2C = RelativePose::Identity();
	  keyframedepthmap = depth * mDepthMapFactor/1000;
    keyframeintenmap = inten;
    /*
    pICP->last_rgb = rgb.clone();
    pICP->last_inten = inten.clone();
    pICP->last_depth = depth.clone();
    */
    }
    
    //backup cloud
    if (bUseBackup) {
      backupCloud.first = timestamp;
      backupCloud.second = std::move(pCurrentCloud);
    } 
  }
  pICP->pLastCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
  return mPoses.back();
}

const ICP_VO::Pose& ICP_VO::Track(const double& timestamp)
{
	
  //initialize VO if first frame, or track current frame
  if (mPoses.size() == 0) {
    rpK2C = RelativePose::Identity();
    pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, Pose::Identity());
    mPoses.push_back(Pose::Identity());
  } else {
    //calculate prediction
    RelativePose pred, predK2C;

    //guess the initial pose
    if (bPredition && mPoses.size() > 1) {
      pred = mPoses.rbegin()[0].inverse()* mPoses.rbegin()[1];
      predK2C = pred*rpK2C;
    } else {
      pred = RelativePose::Identity();
      predK2C = RelativePose::Identity();
    }

    //perform ICP
    rpK2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, predK2C);

    //check ICP valid
    if (bUseBackup && !pICP->isValid() && backupCloud.second != nullptr && mPoses.size() != pKeyFrame->id) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*backupCloud.second), backupCloud.first, mPoses.back());
      RelativePose rpB2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, pred);
      if (pICP->isValid()) {
        rpK2C = rpB2C;
      } else {
        rpK2C = pred;
      }
    } else if(bUseBackup && !pICP->isValid() && backupCloud.second != nullptr){
      rpK2C = pred;
    }

    //calculate current pose from keypose
    Pose currentPose = pKeyFrame->pose * (rpK2C.inverse());
    mPoses.push_back(currentPose);

    //check update keyframe
    if (CheckUpdateKeyFrame(rpK2C)) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, mPoses.back());
      rpK2C = RelativePose::Identity();
    }
    
    //backup cloud
    if (bUseBackup) {
      backupCloud.first = timestamp;
      backupCloud.second = std::move(pCurrentCloud);
    } 
  }
  return mPoses.back();
}

const ICP_VO::Pose& ICP_VO::TrackJoint(const cv::Mat& depth, const cv::Mat& rgb, const double& timestamp){
  std::cout<<"Doing Cool things.\n";

  pCurrentCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
    //initialize VO if first frame, or track current frame
  
  ICP_VO::RGB dIdx;
  ICP_VO::RGB dIdy;
  computeDerivativeImages(rgb, dIdx, dIdy);
  //pICP->computeDerivativeImagesHSV(rgb, dIdx, dIdy);
  if (mPoses.size() == 0) {
    std::cout<<"first frame haha\n";
    rpK2C = RelativePose::Identity();
    pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, Pose::Identity());
    mPoses.push_back(Pose::Identity());
    lastRGB = rgb;
    pLastCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
  } else {
    //calculate prediction
    RelativePose pred, predK2C;

    //guess the initial pose
    if (bPredition && mPoses.size() > 1) {
      pred = mPoses.rbegin()[0].inverse()* mPoses.rbegin()[1];
      predK2C = pred*rpK2C;
    } else {
      pred = RelativePose::Identity();
      predK2C = RelativePose::Identity();
    }
    //perform ICP
    Eigen::Matrix<EAS_ICP::Scalar,6, 6,Eigen::RowMajor> A_rgb = Eigen::MatrixXd::Zero(6, 6);
    Eigen::Matrix<EAS_ICP::Scalar, 6, 1> b_rgb = Eigen::MatrixXd::Zero(6, 1);


    pICP->RGBJacobianGet(dIdx, dIdy, depth, rgb, lastRGB, *pCurrentCloud, predK2C, A_rgb, b_rgb, *pLastCloud);

    //Eigen::Vector<EAS_ICP::Scalar, 6> retRGB;
	  //retRGB = A_rgb.ldlt().solve(b_rgb);
    //std::cout<<"\n\nA_rgb: "<<A_rgb<<"\n\n";
    //std::cout<<"\n\nb_rgb: "<<b_rgb<<"\n\n";

    Eigen::Vector<EAS_ICP::Scalar, 6> ret2;
	  ret2 = A_rgb.ldlt().solve(b_rgb);
    rpK2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, predK2C);
    pICP->RGBJacobianGet(dIdx, dIdy, depth, rgb, lastRGB, *pCurrentCloud, (rpK2C.inverse()), A_rgb, b_rgb, *pLastCloud);
    //check ICP valid
    if (bUseBackup && !pICP->isValid() && backupCloud.second != nullptr && mPoses.size() != pKeyFrame->id) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*backupCloud.second), backupCloud.first, mPoses.back());
      RelativePose rpB2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, pred);
      if (pICP->isValid()) {
        rpK2C = rpB2C;
      } else {
        rpK2C = pred;
      }
    } else if(bUseBackup && !pICP->isValid() && backupCloud.second != nullptr){
      rpK2C = pred;
    }

    //calculate current pose from keypose
    Pose currentPose = pKeyFrame->pose * (rpK2C.inverse());
    
    mPoses.push_back(currentPose);

    //check update keyframe
    if (CheckUpdateKeyFrame(rpK2C)) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, mPoses.back());
      rpK2C = RelativePose::Identity();
    }
    
    //backup cloud
    if (bUseBackup) {
      backupCloud.first = timestamp;
      backupCloud.second = std::move(pCurrentCloud);
    }
    lastRGB = rgb;
    pLastCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
  }
  return mPoses.back();
}
bool ICP_VO::CheckUpdateKeyFrame(const RelativePose& rtSE3) {
  Eigen::Vector<Scalar, 6> rt;
  rt = TranslationAndEulerAnglesFromSe3(rtSE3);
  Scalar trans = sqrtf(rt.tail(3).array().pow(2).sum());
  Scalar rot = sqrtf(rt.head(3).array().pow(2).sum());
  if ( (trans > mThresKeyframeUpdateTrans) || (rot > mThresKeyframeUpdateRot)) {
    return true;
  }
  else {
    return false;
  }
}
Eigen::Vector<ICP_VO::Scalar, 6> ICP_VO::TranslationAndEulerAnglesFromSe3(const RelativePose& rtSE3) {
  Eigen::Vector<ICP_VO::Scalar, 6> ret;
  Scalar &x = ret(3), &y = ret(4), &z = ret(5);
  Scalar &roll = ret(0), &pitch = ret(1), &yaw = ret(2);
  x = rtSE3 (0, 3);
  y = rtSE3 (1, 3);
  z = rtSE3 (2, 3);
  roll = std::atan2 (rtSE3 (2, 1), rtSE3 (2, 2));
  pitch = asin (-rtSE3 (2, 0));
  yaw = std::atan2 (rtSE3 (1, 0), rtSE3 (0, 0));
  return ret;
}
ICP_VO::CurrentCloud ICP_VO::ComputeCurrentCloud(const cv::Mat& depth) {
  int width = pICP->width;
  int height = pICP->height;
  CurrentCloud ret(pICP->pixelSize, 3);
 
		
  for (int n = 0; n < height; ++n) {
    for (int m = 0; m < width; ++m) {
      int i = n*width +m;
      Scalar z=0;
      if (depth.type() == CV_32FC1) {
          z = ((float*)depth.data)[i]/ mDepthMapFactor;
         //z = mDepthMapFactor;
         //std::cout<<"DepthFac:"<<mDepthMapFactor<<"\n";
      } else {
         z = ((int16_t*)depth.data)[i]/ mDepthMapFactor;
         //std::cout<<"DepthFac2:"<<mDepthMapFactor<<"\n";
      }
      if (z == 0) {
        for (int j = 0; j < 3; ++j) {
          ret(i, j) = std::numeric_limits<Scalar>::quiet_NaN();
        }
      } else {
        ret(i, 2) = z;
        Scalar x = (Scalar)m;
        Scalar y = (Scalar)n;
        ret(i, 0) = z*(x-pICP->cx) / pICP->fx;
        ret(i, 1) = z*(y-pICP->cy) / pICP->fy;
      }
	  	
		
	}
  }
   if(access("pointcloud.txt",0) < 0)
		{
			std::ofstream fout("pointcloud.txt"); 
			//save the pointcloud as .txt file
			//0, x
			//1, y
			//2, z
			CurrentCloud big   (1, 3);
			CurrentCloud small (1, 3);
			
			for (int n = 0; n < height; ++n) 
			{
				for (int m = 0; m < width; ++m) 
				{
					int i = n*width +m;
					fout << ret(i, 0) << " " 
						<< ret(i, 1) << " " 
						<< ret(i, 2) << std::endl;
					//find the biggest and the smallest x_big, y_big, z_big, x_samll, y_samll, z_small
					if(i==0)
					{
						big  (0, 0) = ret(i, 0);
						big  (0, 1) = ret(i, 1);
						big  (0, 2) = ret(i, 2);
						small(0, 0) = ret(i, 0);
						small(0, 1) = ret(i, 1);
						small(0, 2) = ret(i, 2);
					}
					else
					{
						//big
						if(ret(i, 0) > big(0, 0))
						{
							big(0, 0) = ret(i, 0);
						}
						if(ret(i, 1) > big(0, 1))
						{
							big(0, 1) = ret(i, 1);
						}
						if(ret(i, 2) > big(0, 2))
						{
							big(0, 2) = ret(i, 2);
						}
						//small
						if(ret(i, 0) < small(0, 0))
						{
							small(0, 0) = ret(i, 0);
						}
						if(ret(i, 1) < small(0, 1))
						{
							small(0, 1) = ret(i, 1);
						}
						if(ret(i, 2) < small(0, 2))
						{
							small(0, 2) = ret(i, 2);
						}
						
					}
					//end to find the biggest and the smallest x_big, y_big, z_big, x_samll, y_samll, z_small
				}
			}
			std::cout<<"big_x "  <<big(0, 0)  <<std::endl;
			std::cout<<"big_y "  <<big(0, 1)  <<std::endl;
			std::cout<<"big_z "  <<big(0, 2)  <<std::endl;
			std::cout<<"small_x "<<small(0, 0)<<std::endl;
			std::cout<<"small_y "<<small(0, 1)<<std::endl;
			std::cout<<"small_z "<<small(0, 2)<<std::endl;
			fout.close(); 
			//end of saving pointcloud
		}

  return ret;
}

void ICP_VO::computeDerivativeImages(const cv::Mat& rgb, cv::Mat& dIdx, cv::Mat& dIdy){

  float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
                   0.79451, -0.00000, -0.79451,
                   0.52201,  0.00000, -0.52201};

  float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
                     0.00000, 0.00000, 0.00000,
                    -0.52201, -0.79451, -0.52201};

  cv::Mat intensity;
  cv::cvtColor(rgb, intensity, cv::COLOR_BGR2GRAY);
  
  cv::Mat kernelX(3, 3, CV_32F, gsx3x3);
  cv::Mat kernelY(3, 3, CV_32F, gsy3x3);
  //cv::filter2D( intensity, dIdx, -1 , kernelX, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  cv:: Sobel(intensity, dIdx, CV_32F, 1, 0, 1);
  cv:: Sobel(intensity, dIdy, CV_32F, 0, 1, 1);
  //cv::filter2D( intensity, dIdy, -1 , kernelY, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT);
  //std::cout<<dIdx;

}
void ICP_VO::TrackSource(const cv::Mat& rgb, const cv::Mat& depth){
    cv::Mat inten;
    cv::cvtColor(rgb, inten, cv::COLOR_BGR2GRAY);
    pICP->last_rgb = rgb.clone();
    pICP->last_inten = inten.clone();
    pICP->last_depth = depth.clone();

}