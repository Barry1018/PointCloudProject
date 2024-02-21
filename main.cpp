#include <iostream>
#include "pca_match.h"
#include <omp.h>

int main()
{
    
    
    pca_match pm;
    
   
    pcl::PLYReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>()), p1(new pcl::PointCloud<pcl::PointXYZ>()), p2(new pcl::PointCloud<pcl::PointXYZ>()), p3(new pcl::PointCloud<pcl::PointXYZ>()), p4(new pcl::PointCloud<pcl::PointXYZ>()), p5(new pcl::PointCloud<pcl::PointXYZ>());
        
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>());
    //const std::string filename_ = "D:\\Visual Studio\\pointcloud\\實驗二十八\\15_pile_up4_remove" + std::to_string(2) + ".ply";
    
    //const std::string filename_ = "15_pile_up1_remove" + std::to_string(32) + ".ply";
    const std::string filename_ = "D:\\Visual Studio\\pointcloud\\實驗二十一\\15_pile_up1_remove0.ply";

    //const std::string filename_ = "test.ply";

    //reader.read("oil.ply", *model);//15model_whole//new1//oil.ply
    reader.read("15model_whole.ply", *model);
    pm.setInputModel(model);

    //pm.setfilename(std::to_string(0) + ".png");
    reader.read(filename_, *scene);

    

    pm.setInputCloud(scene);

    

   
    pm.plane_segmentation_hidden();
    pm.model_downsample(1);
    //pm.plane_segmentation3();//segment model
    pm.pass_through();// for scene
    pm.add_noise();//std=0.1
    pm.region_growing();//for scene

     //pm.model_downsample(0.9);//0.9
    pm.find_parts_in_scene_rotate_RMSE2_hull_rank();
    
    //pm.find_parts_in_scene_rotate_RMSE2_hull2();
    //////////////////////////////////////////////////////////////////
    
    //pm.plane_segmentation_hidden_oil();
    // 
    // 
    //pm.pass_through();
    //pm.add_noise();
    //pm.region_growing_oil();
    //pm.model_downsample(1.5);
    //pm.find_parts_in_scene_rotate_oil();

    


    //std::vector<int> v = { 4,2,7,8,1 };
    //std::vector<std::string> m = { "a","b","c","e","f" };
    //std::vector<std::string> m_new = {"","","","",""};


    //std::vector<size_t> indices(v.size());
    //for (size_t i = 0; i < indices.size(); ++i) {
    //    indices[i] = i;
    //}
    //std::sort(indices.begin(), indices.end(), [&v](size_t i, size_t j) {
    //    return v[i] < v[j];
    //    });


    //std::sort(v.begin(), v.end());
    //for (auto& i : v)
    //{
    //    cout << "v" << endl;
    //    cout << i << " ";
    //}

    //for (auto& k : indices)
    //{   
    //    cout << "indice" << endl;
    //    cout << k << " ";
    //}

    //for (size_t j = 0; j < indices.size(); ++j) {
    //    m_new[j] = m[indices[j]];
    //}

    //for (auto& j : m_new)
    //{
    //    cout << "m" << endl;
    //    cout << j << " ";
    //}






	return 0;
}