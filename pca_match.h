#pragma once
#include <iostream>
#include <time.h>
#include <thread>
#include "stdio.h"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/parse.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/console/time.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/passthrough.h>
#include <chrono>
#include <pcl/registration/ppf_registration.h>
#include <pcl/common/random.h>
/*feature*/
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/spin_image.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/moment_of_inertia_estimation.h>
//#include <pcl/kdtree/impl/kdtree_flann.hpp>//這行有問題
/*keypoint*/
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf_descriptor.h>
/*range image*/
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h> 
#include <pcl/filters/frustum_culling.h>
/*plotter*/
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vector>
#include <utility>
/*Harris*/
#include <pcl/keypoints/harris_3d.h>
/*ISS*/
#include <pcl/keypoints/iss_3d.h>
/*plotter*/
#include <pcl/visualization/pcl_plotter.h>
/*match*/
#include <pcl/common/transforms.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/features/board.h>
#include <pcl/registration/sample_consensus_prerejective.h>
//#include <pcl/features/board.h>c
#include <pcl/recognition/cg/geometric_consistency.h>
#include <boost/random.hpp>
#include <pcl/filters/random_sample.h>
#include <pcl/point_traits.h>
#include <pcl/for_each_type.h>
#include <pcl/surface/convex_hull.h>
/**/
#include <vtkRenderingContextOpenGL2Module.h>
#include <vtkContextDevice2D.h>
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2)





class pca_match
{
private:
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr model;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull_;
	Eigen::Vector4f model_center, cloud_center,temp_center;

	pcl::PointCloud<pcl::PointXYZ>::Ptr first_cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr second_cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr third_cloud;

	pcl::PointCloud<pcl::PointXYZ>::Ptr first_model;
	pcl::PointCloud<pcl::PointXYZ>::Ptr second_model;
	pcl::PointCloud<pcl::PointXYZ>::Ptr third_model;

	pcl::PointCloud<pcl::PointXYZ>::Ptr mp1, mp2, mp3, mp4;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cp1, cp2, cp3, cp4;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pass_through;

	pcl::PointCloud<pcl::PointXYZ>::Ptr model_rotated;

	std::vector<Eigen::Vector3f> cloud_centroid, model_centroid;

	std::vector<Eigen::Matrix4f> top3_cloud_pca, top3_model_pca;

	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > model_regions;

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors;  // fpfh特征
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors;

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints;
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_keypoints;

    pcl::PointCloud <pcl::Normal>::Ptr model_normals;
    pcl::PointCloud <pcl::Normal>::Ptr scene_normals;

    std::string filename;
	double model_diameter;
	double cluster_tolerance = 5;

	int MinClusterSize = 50;
	int MaxClusterSize = 2500;

	Eigen::Vector3d camera_location,camera_location1, camera_location2, camera_location3;

public:
	void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_) { cloud = cloud_; }
	void setInputModel(pcl::PointCloud<pcl::PointXYZ>::Ptr& model_) { model = model_; }
    void model_downsample(double leaf_size);
	void setInputSurfaces(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> surfaces);
	void showInputCloud();
	void showInputModel();
	void cluster_cloud();
	void cluster_model(bool rotated);
	void show_cloud_cluster();
	void show_model_cluster();
	void rotate_cloud(double angle, std::string direction, Eigen::Vector3f translation);
	std::vector<Eigen::Matrix4f> compute_cloud_PCA();
	std::vector<Eigen::Matrix4f> compute_model_PCA();
	void registration();
	void registration_by_tri();
	Eigen::Vector3f simplex_method(Eigen::Vector3f init, Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3, Eigen::Vector3f t1, Eigen::Vector3f t2, Eigen::Vector3f t3);
	void plane_segmentation();
	void plane_segmentation2();
	void region_growing();
	void pass_through();
	void triangulation();
	void show_region_growing_part();
	void find_parts_in_scene();
	void region_growing_model();
	void fpfh_model(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors_);
    void fpfh_scene(pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors_);
    void OBB();
    void find_match(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors, pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs);
    void visualize_corrs(pcl::CorrespondencesPtr model_scene_corrs);
    void add_noise();
    void setfilename(std::string file) { filename = file; };
    void find_parts_in_scene_alter();
    void find_parts_in_scene_rotate();

	void find_parts_in_scene_openmp();
    void pca_simplex();
	void find_parts_in_scene_rotate_RMSE();
	void find_parts_in_scene_rotate_RMSE2();
	void find_parts_in_scene_rotate_RMSE2_hull();
	void find_parts_in_scene_rotate_RMSE2_hull2();
	void find_parts_in_scene_rotate_RMSE2_hull_rank();
	void find_parts_in_scene_rotate_oil();


	pcl::PointCloud<pcl::PointXYZ>::Ptr hull_camera_ontop(pcl::PointCloud<pcl::PointXYZ>::Ptr& rotated_model);

	void alpha_shape();
	void plane_segmentation3();
	void plane_segmentation_hidden();
	void plane_segmentation_hidden_oil();
	void region_growing_oil();
	void nearest_hull();
	pcl::PointCloud<pcl::PointXYZ>::Ptr hull(Eigen::Vector3d camera);
	void test();
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> hull_expansion(pcl::PointCloud<pcl::PointXYZ>::Ptr& rotated_model);
};


