/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICPORBVO_example.cpp

* Purpose :

* Creation Date : 2020-08-29

* Last Modified : 廿廿年八月廿九日 (週六) 十一時卅一分45秒

* Created By : Ji-Ying, Li  

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <ICPORB_VO.h>

void LoadImagePaths(const std::string& tum_association_filepath, std::vector<std::string>& vstr_rgb_filepath, std::vector<std::string>& vstr_depth_filepath, std::vector<std::string>& vstr_timestamp);

int main(int argc, char *argv[])
{
  //parse comment
  if (argc != 6) {
    std::cout << "example: ./build/rgbd_tum path_to_association_file path_to_sequence path_to_setting path_to_output_path path_to_ORB_voc" << std::endl;
    return 1;
  }

  //access association file
  std::vector<std::string> vstr_rgb_filepath;
  std::vector<std::string> vstr_depth_filepath;
  std::vector<std::string> vstr_timestamp;
  LoadImagePaths(argv[1], vstr_rgb_filepath, vstr_depth_filepath, vstr_timestamp);
  
  //open output trajectory file
  std::ofstream fout(argv[4]);
  fout.close();
  fout.open(argv[4], std::fstream::app);

  //instance VO 
  ICPORB_VO vo(argv[5], argv[3]);
  // read meta-icp trajectories
	std::map<uint64_t, Eigen::Isometry3f, std::less<int>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Isometry3f> > > camera_trajectory;
	std::ifstream file;
    std::string line;
    file.open("/home/jjhu/Desktop/jjhu/jy_code/ICPORB_VO/build/360_traj.txt_0.txt");
	uint64_t pose_index = 0;
    while (!file.eof())
    {
        float utime;
        float x, y, z, qx, qy, qz, qw;
        std::getline(file, line);
		std::cout<<line<<std::endl;
        int n = sscanf(line.c_str(), "%f %f %f %f %f %f %f %f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);

        if(file.eof()) 
            break;

        assert(n == 8);

        Eigen::Quaternionf q(qw, qx, qy, qz);
        Eigen::Vector3f t(x, y, z);

        Eigen::Isometry3f T;
        T.setIdentity();
        T.pretranslate(t).rotate(q);
        camera_trajectory[pose_index] = T;
		//std::cout<<utime<<std::endl;
		//std::cout<<camera_trajectory[pose_index].matrix()<<std::endl;
		pose_index++;
    }
  for (uint64_t i = 0; i < vstr_rgb_filepath.size(); ++i) {
    //Read image
    const std::string rgb_path =std::string(argv[2])+'/'+vstr_rgb_filepath[i];
    const std::string depth_path = std::string(argv[2])+'/'+vstr_depth_filepath[i];
    cv::Mat target_color = cv::imread(rgb_path, 1);
    cv::Mat target_depth = cv::imread(depth_path, -1);
	
	
    vo.IncrementalTrack(target_color, target_depth, std::stod(vstr_timestamp[i]), camera_trajectory[i].matrix());
	std::cout<<"camera trajectory"<<std::endl;
	std::cout<<camera_trajectory[i].matrix()<<std::endl;
    Eigen::Isometry3d current_pose_iso;
    current_pose_iso= vo.GetCurrentPose(1);
    Eigen::Quaterniond q(current_pose_iso.rotation());
    fout << vstr_timestamp[i] << " " << current_pose_iso(0,3) 
                            << " " << current_pose_iso(1,3)
                            << " " << current_pose_iso(2,3) 
                            << " " << q.x() << " " << q.y()
                            << " " << q.z() << " " << q.w() << std::endl;


// #ifdef OUTPUT_TUM_POSE
    //show pose on screen
    std::cout << vstr_timestamp[i] << " " << current_pose_iso(0,3) 
                            << " " << current_pose_iso(1,3)
                            << " " << current_pose_iso(2,3) 
                            << " " << q.x() << " " << q.y()
                            << " " << q.z() << " " << q.w() << std::endl;

// #endif
  }
  fout.close();
  return 0;
}

void LoadImagePaths(const std::string& tum_association_filepath, std::vector<std::string>& vstr_rgb_filepath, std::vector<std::string>& vstr_depth_filepath, std::vector<std::string>& vstr_timestamp)
{
  std::ifstream f_association(tum_association_filepath);
  while (!f_association.eof()) {
    std::string s;
    getline(f_association,s);
    if (!s.empty()) {
      std::stringstream ss(s);
      std::string t;
      std::string sRGB, sD;
      ss >> t;
      ss >> sRGB;
      vstr_rgb_filepath.push_back(std::move(sRGB));
      ss >> t;
      vstr_timestamp.push_back(t);
      ss >> sD;
      vstr_depth_filepath.push_back(std::move(sD));
    }
  }
}

