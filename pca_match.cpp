#include "pca_match.h"
clock_t start, end, start2, end2;
double cpu_time;
pcl::PointCloud<pcl::PointXYZ>::Ptr randomlySelectPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr pointClouds, int numSelectedPointClouds)
{
    int totalPointClouds = pointClouds->size();

    // Generate random indices
    std::vector<int> randomIndices;
    for (int i = 0; i < numSelectedPointClouds; ++i)
    {
        int randomIndex = rand() % totalPointClouds;
        randomIndices.push_back(randomIndex);
    }

    // Select point clouds based on random indices
    pcl::PointCloud<pcl::PointXYZ>::Ptr selectedPointClouds(new pcl::PointCloud<pcl::PointXYZ>());
    for (int index : randomIndices)
    {
        selectedPointClouds->push_back(pointClouds->points[index]);
    }

    return selectedPointClouds;
}
void pca_match::showInputCloud()
{
    pcl::visualization::PCLVisualizer viewer;
    viewer.getRenderWindow()->GlobalWarningDisplayOff();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 0, 255, 0);
    viewer.addPointCloud(cloud, color_handler, "off_scene_model");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "off_scene_model");

    viewer.initCameraParameters();
    viewer.spin();

    system("pause");
}
void pca_match::showInputModel()
{
    pcl::visualization::PCLVisualizer viewer;
    viewer.getRenderWindow()->GlobalWarningDisplayOff();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(model, 0, 255, 0);
    viewer.addPointCloud(model, color_handler, "off_scene_model");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model");
    viewer.initCameraParameters();
    viewer.spin();

    system("pause");
}

void pca_match::model_downsample(double leaf_size)
{
    //double leaf_size = 1.5;
    pcl::VoxelGrid<pcl::PointXYZ>vox;
    vox.setInputCloud(model);
    vox.setLeafSize(leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    vox.filter(*downsampled_cloud);
    *model = *downsampled_cloud;
    cout << "Model Downsize: " << model->size() << endl;
}
std::vector<Eigen::Matrix4f> pca_match::compute_cloud_PCA()
{
    std::vector<Eigen::Matrix4f> cluster_pca;
    int count = 0;

    while (count < 3)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (count == 0)
        {
            tempCloud = first_cloud;
        }
        else if (count == 1)
        {
            tempCloud = second_cloud;
        }
        else if (count == 2)
        {
            tempCloud = third_cloud;
        }

        pcl::PointXYZ o, pcaZ, pcaY, pcaX, c, pcX, pcY, pcZ;
        Eigen::Vector4f pcaCentroid;
        pcl::compute3DCentroid(*tempCloud, pcaCentroid);
        cloud_centroid.push_back(pcaCentroid.head<3>());
        cout << "CLOUD PCA CENTROID: " << pcaCentroid << endl;
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*tempCloud, pcaCentroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
        Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
        Eigen::Vector3f centroid;
        Eigen::Vector3f p_pi;
        Eigen::Vector3f p;
        centroid[0] = pcaCentroid(0);
        centroid[1] = pcaCentroid(1);
        centroid[2] = pcaCentroid(2);

        Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
        transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
        transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*tempCloud, *transformedCloud, transform);

        //std::cout << "Eigenvalue: \n" << eigenValuesPCA << std::endl;
        //std::cout << "Eigenvector: \n" << eigenVectorsPCA << std::endl;


        o.x = 0.0;
        o.y = 0.0;
        o.z = 0.0;
        Eigen::Affine3f tra_aff(transform);
        Eigen::Vector3f pz = eigenVectorsPCA.col(0);
        Eigen::Vector3f py = eigenVectorsPCA.col(1);
        Eigen::Vector3f px = eigenVectorsPCA.col(2);
        pcl::transformVector(pz, pz, tra_aff);
        pcl::transformVector(py, py, tra_aff);
        pcl::transformVector(px, px, tra_aff);
        int votex, votey{ 0 }, votey_{ 0 }, votez{ 0 }, votez_{ 0 };

        for (size_t i = 0; i < transformedCloud->size(); i++)
        {
            p_pi[0] = 0; p_pi[1] = 0; p_pi[2] = 0;
            p[0] = tempCloud->points[i].x;
            p[1] = tempCloud->points[i].y;
            p[2] = tempCloud->points[i].z;

            p_pi = p - centroid;
            if (p_pi.dot(pz) >= 0)
            {
                votez += 1;
            }
            else if (p_pi.dot(-pz) > 0)
            {
                votez_ += 1;
            }

            if (p_pi.dot(py) >= 0)
            {
                votey += 1;
            }
            else if (p_pi.dot(-py) > 0)
            {
                votey_ += 1;
            }


        }


        if (votez > votez_) {
            eigenVectorsPCA.col(0) = eigenVectorsPCA.col(0);
        }
        else
        {
            eigenVectorsPCA.col(0) = -eigenVectorsPCA.col(0);
        }
        if (votey > votey_) {
            eigenVectorsPCA.col(1) = eigenVectorsPCA.col(1);
        }
        else
        {
            eigenVectorsPCA.col(1) = -eigenVectorsPCA.col(1);
        }

        eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
        transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
        transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());
        cluster_pca.push_back(transform);
        top3_cloud_pca.push_back(transform);
        pcaZ.x = 1000 * pz(0);
        pcaZ.y = 1000 * pz(1);
        pcaZ.z = 1000 * pz(2);

        pcaY.x = 1000 * py(0);
        pcaY.y = 1000 * py(1);
        pcaY.z = 1000 * py(2);

        pcaX.x = 1000 * px(0);
        pcaX.y = 1000 * px(1);
        pcaX.z = 1000 * px(2);

        c.x = pcaCentroid(0);
        c.y = pcaCentroid(1);
        c.z = pcaCentroid(2);


        pcZ.x = 5 * eigenVectorsPCA(0, 0) + c.x;
        pcZ.y = 5 * eigenVectorsPCA(1, 0) + c.y;
        pcZ.z = 5 * eigenVectorsPCA(2, 0) + c.z;

        pcY.x = 5 * eigenVectorsPCA(0, 1) + c.x;
        pcY.y = 5 * eigenVectorsPCA(1, 1) + c.y;
        pcY.z = 5 * eigenVectorsPCA(2, 1) + c.z;

        pcX.x = 5 * eigenVectorsPCA(0, 2) + c.x;
        pcX.y = 5 * eigenVectorsPCA(1, 2) + c.y;
        pcX.z = 5 * eigenVectorsPCA(2, 2) + c.z;


        pcl::visualization::PCLVisualizer viewer;
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(tempCloud, 255, 0, 0);


        viewer.addPointCloud(tempCloud, color_handler, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");


        //viewer.addArrow(pcaZ, o, 0.0, 0.0, 1.0, false, "arrow_Z");
        //viewer.addArrow(pcaY, o, 0.0, 1.0, 0.0, false, "arrow_Y");
        //viewer.addArrow(pcaX, o, 1.0, 0.0, 0.0, false, "arrow_X");

        viewer.addArrow(pcZ, c, 0.0, 0.0, 1.0, false, "arrow_z");
        viewer.addArrow(pcY, c, 0.0, 1.0, 0.0, false, "arrow_y");
        viewer.addArrow(pcX, c, 1.0, 0.0, 0.0, false, "arrow_x");



        viewer.addCoordinateSystem(5);
        viewer.setBackgroundColor(1.0, 1.0, 1.0);
        while (!viewer.wasStopped())
        {
            viewer.spinOnce(100);

        }
        count++;
    }


    return cluster_pca;
}
std::vector<Eigen::Matrix4f> pca_match::compute_model_PCA()
{
    std::vector<Eigen::Matrix4f> cluster_pca;
    int count = 0;

    while (count < 3)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);


        if (count == 0)
        {
            tempCloud = first_model;
        }
        else if (count == 1)
        {
            tempCloud = second_model;
        }
        else if (count == 2)
        {
            tempCloud = third_model;
        }

        pcl::PointXYZ o, pcaZ, pcaY, pcaX, c, pcX, pcY, pcZ;
        Eigen::Vector4f pcaCentroid;

        pcl::compute3DCentroid(*tempCloud, pcaCentroid);
        model_centroid.push_back(pcaCentroid.head<3>());
        cout << "MODEL PCA CENTROID: " << pcaCentroid << endl;
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*tempCloud, pcaCentroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
        Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
        Eigen::Vector3f centroid;
        Eigen::Vector3f p_pi;
        Eigen::Vector3f p;
        centroid[0] = pcaCentroid(0);
        centroid[1] = pcaCentroid(1);
        centroid[2] = pcaCentroid(2);

        Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
        transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
        transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*tempCloud, *transformedCloud, transform);


        //std::cout << "Eigenvalue: \n" << eigenValuesPCA << std::endl;
        //std::cout << "Eigenvector: \n" << eigenVectorsPCA << std::endl;


        o.x = 0.0;
        o.y = 0.0;
        o.z = 0.0;
        Eigen::Affine3f tra_aff(transform);
        Eigen::Vector3f pz = eigenVectorsPCA.col(0);
        Eigen::Vector3f py = eigenVectorsPCA.col(1);
        Eigen::Vector3f px = eigenVectorsPCA.col(2);
        pcl::transformVector(pz, pz, tra_aff);
        pcl::transformVector(py, py, tra_aff);
        pcl::transformVector(px, px, tra_aff);
        int votex, votey{ 0 }, votey_{ 0 }, votez{ 0 }, votez_{ 0 };

        for (size_t i = 0; i < transformedCloud->size(); i++)
        {
            p_pi[0] = 0; p_pi[1] = 0; p_pi[2] = 0;
            p[0] = tempCloud->points[i].x;
            p[1] = tempCloud->points[i].y;
            p[2] = tempCloud->points[i].z;

            p_pi = p - centroid;
            if (p_pi.dot(pz) >= 0)
            {
                votez += 1;
            }
            else if (p_pi.dot(-pz) > 0)
            {
                votez_ += 1;
            }

            if (p_pi.dot(py) >= 0)
            {
                votey += 1;
            }
            else if (p_pi.dot(-py) > 0)
            {
                votey_ += 1;
            }


        }
        if (votez > votez_) {
            eigenVectorsPCA.col(0) = eigenVectorsPCA.col(0);
        }
        else
        {
            eigenVectorsPCA.col(0) = -eigenVectorsPCA.col(0);
        }
        if (votey > votey_) {
            eigenVectorsPCA.col(1) = eigenVectorsPCA.col(1);
        }
        else
        {
            eigenVectorsPCA.col(1) = -eigenVectorsPCA.col(1);
        }

        eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
        transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
        transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());
        cluster_pca.push_back(transform);
        top3_model_pca.push_back(transform);
        pcaZ.x = 1000 * pz(0);
        pcaZ.y = 1000 * pz(1);
        pcaZ.z = 1000 * pz(2);

        pcaY.x = 1000 * py(0);
        pcaY.y = 1000 * py(1);
        pcaY.z = 1000 * py(2);

        pcaX.x = 1000 * px(0);
        pcaX.y = 1000 * px(1);
        pcaX.z = 1000 * px(2);

        c.x = pcaCentroid(0);
        c.y = pcaCentroid(1);
        c.z = pcaCentroid(2);


        pcZ.x = 5 * eigenVectorsPCA(0, 0) + c.x;
        pcZ.y = 5 * eigenVectorsPCA(1, 0) + c.y;
        pcZ.z = 5 * eigenVectorsPCA(2, 0) + c.z;

        pcY.x = 5 * eigenVectorsPCA(0, 1) + c.x;
        pcY.y = 5 * eigenVectorsPCA(1, 1) + c.y;
        pcY.z = 5 * eigenVectorsPCA(2, 1) + c.z;

        pcX.x = 5 * eigenVectorsPCA(0, 2) + c.x;
        pcX.y = 5 * eigenVectorsPCA(1, 2) + c.y;
        pcX.z = 5 * eigenVectorsPCA(2, 2) + c.z;


        pcl::visualization::PCLVisualizer viewer;
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(tempCloud, 255, 0, 0);


        viewer.addPointCloud(tempCloud, color_handler, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");


        //viewer.addArrow(pcaZ, o, 0.0, 0.0, 1.0, false, "arrow_Z");
        //viewer.addArrow(pcaY, o, 0.0, 1.0, 0.0, false, "arrow_Y");
        //viewer.addArrow(pcaX, o, 1.0, 0.0, 0.0, false, "arrow_X");

        viewer.addArrow(pcZ, c, 0.0, 0.0, 1.0, false, "arrow_z");
        viewer.addArrow(pcY, c, 0.0, 1.0, 0.0, false, "arrow_y");
        viewer.addArrow(pcX, c, 1.0, 0.0, 0.0, false, "arrow_x");



        viewer.addCoordinateSystem(5);
        viewer.setBackgroundColor(1.0, 1.0, 1.0);
        while (!viewer.wasStopped())
        {
            viewer.spinOnce(100);

        }
        count++;
    }


    return cluster_pca;
}

void pca_match::cluster_cloud()
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(cloud_pass_through);

    // Euclidean 聚类对象.
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
    // 设置聚类的最小值 2cm (small values may cause objects to be divided
    // in several clusters, whereas big values may join objects in a same cluster).
    clustering.setClusterTolerance(cluster_tolerance);
    // 设置聚类的小点数和最大点云数
    clustering.setMinClusterSize(MinClusterSize);
    clustering.setMaxClusterSize(MaxClusterSize);
    clustering.setSearchMethod(kdtree);
    clustering.setInputCloud(cloud_pass_through);

    std::vector<pcl::PointIndices> cluste;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> top_clusters;

    clustering.extract(cluste);
    // For every cluster...
    int currentClusterNum = 1;
    for (std::vector<pcl::PointIndices>::const_iterator i = cluste.begin(); i != cluste.end(); ++i)
    {
        //添加所有的点云到一个新的点云中
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(cloud->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        // 保存
        if (cluster->points.size() <= 0)
            break;
        top_clusters.push_back(cluster);
        pcl::visualization::PCLVisualizer viewer;
        viewer.getRenderWindow()->GlobalWarningDisplayOff();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(cluster, 255, 0, 0);
        viewer.addPointCloud(cluster, color_handler1, "off_scene_model1");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud_pass_through, 0, 255, 0);
        viewer.addPointCloud(cloud_pass_through, color_handler2, "off_scene_model2");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        viewer.initCameraParameters();
        viewer.spin();

        system("pause");

        currentClusterNum++;
    }
    //first_cloud = top_clusters[0];
    //second_cloud = top_clusters[1];
    //third_cloud = top_clusters[2];




 /*   std::vector<double> v;
    Eigen::Vector4f pcaCentroid0, pcaCentroid1, pcaCentroid2;
    pcl::compute3DCentroid(*top_clusters[0], pcaCentroid0);
    pcl::compute3DCentroid(*top_clusters[1], pcaCentroid1);
    pcl::compute3DCentroid(*top_clusters[2], pcaCentroid2);
    double d1 = sqrt(pow((pcaCentroid0[0] - pcaCentroid1[0]), 2) + pow((pcaCentroid0[1] - pcaCentroid1[1]), 2) +
        pow((pcaCentroid0[2] - pcaCentroid1[2]), 2));
    double d2 = sqrt(pow((pcaCentroid2[0] - pcaCentroid1[0]), 2) + pow((pcaCentroid2[1] - pcaCentroid1[1]), 2) +
        pow((pcaCentroid2[2] - pcaCentroid1[2]), 2));
    double d3 = sqrt(pow((pcaCentroid2[0] - pcaCentroid0[0]), 2) + pow((pcaCentroid2[1] - pcaCentroid0[1]), 2) +
        pow((pcaCentroid2[2] - pcaCentroid0[2]), 2));
    v.push_back(d1);
    v.push_back(d2);
    v.push_back(d3);
    cout << "v: " << endl;
    for (auto& i : v)
    {
        cout << i << endl;
    }*/


}

void pca_match::cluster_model(bool rotated)
{
    if (rotated == true)
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        kdtree->setInputCloud(model_rotated);

        // Euclidean 聚类对象.
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
        // 设置聚类的最小值 2cm (small values may cause objects to be divided
        // in several clusters, whereas big values may join objects in a same cluster).
        clustering.setClusterTolerance(cluster_tolerance);
        // 设置聚类的小点数和最大点云数
        clustering.setMinClusterSize(MinClusterSize);
        clustering.setMaxClusterSize(MaxClusterSize);
        clustering.setSearchMethod(kdtree);
        clustering.setInputCloud(model_rotated);

        std::vector<pcl::PointIndices> cluste;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> top_clusters;

        clustering.extract(cluste);
        // For every cluster...
        int currentClusterNum = 1;
        for (std::vector<pcl::PointIndices>::const_iterator i = cluste.begin(); i != cluste.end(); ++i)
        {
            //添加所有的点云到一个新的点云中
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
                cluster->points.push_back(model_rotated->points[*point]);
            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            // 保存
            if (cluster->points.size() <= 0)
                break;

            top_clusters.push_back(cluster);


            currentClusterNum++;
        }

        first_model = top_clusters[0];
        second_model = top_clusters[1];
        third_model = top_clusters[2];

        std::vector<double> v;
        Eigen::Vector4f pcaCentroid0, pcaCentroid1, pcaCentroid2;
        pcl::compute3DCentroid(*top_clusters[0], pcaCentroid0);
        pcl::compute3DCentroid(*top_clusters[1], pcaCentroid1);
        pcl::compute3DCentroid(*top_clusters[2], pcaCentroid2);
        double d1 = sqrt(pow((pcaCentroid0[0] - pcaCentroid1[0]), 2) + pow((pcaCentroid0[1] - pcaCentroid1[1]), 2) +
            pow((pcaCentroid0[2] - pcaCentroid1[2]), 2));
        double d2 = sqrt(pow((pcaCentroid2[0] - pcaCentroid1[0]), 2) + pow((pcaCentroid2[1] - pcaCentroid1[1]), 2) +
            pow((pcaCentroid2[2] - pcaCentroid1[2]), 2));
        double d3 = sqrt(pow((pcaCentroid2[0] - pcaCentroid0[0]), 2) + pow((pcaCentroid2[1] - pcaCentroid0[1]), 2) +
            pow((pcaCentroid2[2] - pcaCentroid0[2]), 2));
        v.push_back(d1);
        v.push_back(d2);
        v.push_back(d3);
        cout << "v: " << endl;
        for (auto& i : v)
        {
            cout << i << endl;
        }
    }
    else
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        kdtree->setInputCloud(model);

        // Euclidean 聚类对象.
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> clustering;
        // 设置聚类的最小值 2cm (small values may cause objects to be divided
        // in several clusters, whereas big values may join objects in a same cluster).
        clustering.setClusterTolerance(cluster_tolerance);
        // 设置聚类的小点数和最大点云数
        clustering.setMinClusterSize(MinClusterSize);
        clustering.setMaxClusterSize(MaxClusterSize);
        clustering.setSearchMethod(kdtree);
        clustering.setInputCloud(model);

        std::vector<pcl::PointIndices> cluste;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> top_clusters;

        clustering.extract(cluste);
        // For every cluster...
        int currentClusterNum = 1;
        for (std::vector<pcl::PointIndices>::const_iterator i = cluste.begin(); i != cluste.end(); ++i)
        {
            //添加所有的点云到一个新的点云中
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
                cluster->points.push_back(model->points[*point]);
            cluster->width = cluster->points.size();
            cluster->height = 1;
            cluster->is_dense = true;

            // 保存
            if (cluster->points.size() <= 0)
                break;

            top_clusters.push_back(cluster);


            currentClusterNum++;
        }

        first_model = top_clusters[0];
        second_model = top_clusters[1];
        third_model = top_clusters[2];

        std::vector<double> v;
        Eigen::Vector4f pcaCentroid0, pcaCentroid1, pcaCentroid2;
        pcl::compute3DCentroid(*top_clusters[0], pcaCentroid0);
        pcl::compute3DCentroid(*top_clusters[1], pcaCentroid1);
        pcl::compute3DCentroid(*top_clusters[2], pcaCentroid2);
        double d1 = sqrt(pow((pcaCentroid0[0] - pcaCentroid1[0]), 2) + pow((pcaCentroid0[1] - pcaCentroid1[1]), 2) +
            pow((pcaCentroid0[2] - pcaCentroid1[2]), 2));
        double d2 = sqrt(pow((pcaCentroid2[0] - pcaCentroid1[0]), 2) + pow((pcaCentroid2[1] - pcaCentroid1[1]), 2) +
            pow((pcaCentroid2[2] - pcaCentroid1[2]), 2));
        double d3 = sqrt(pow((pcaCentroid2[0] - pcaCentroid0[0]), 2) + pow((pcaCentroid2[1] - pcaCentroid0[1]), 2) +
            pow((pcaCentroid2[2] - pcaCentroid0[2]), 2));
        v.push_back(d1);
        v.push_back(d2);
        v.push_back(d3);
        cout << "v: " << endl;
        for (auto& i : v)
        {
            cout << i << endl;
        }
    }



}


void pca_match::rotate_cloud(double angle, std::string direction, Eigen::Vector3f translation)
{
    //Rotate
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
    float theta = 0;
    if (angle != 0) {
        theta = M_PI / (180 / angle);
    }
    else if (angle == 0)
    {
        theta = 0;
    }
    if (direction == "x" || direction == "X")
    {
        transform_1(1, 1) = std::cos(theta);
        transform_1(1, 2) = -sin(theta);
        transform_1(2, 1) = sin(theta);
        transform_1(2, 2) = std::cos(theta);
    }
    else if (direction == "y" || direction == "Y")
    {
        transform_1(0, 0) = std::cos(theta);
        transform_1(0, 2) = sin(theta);
        transform_1(2, 0) = -sin(theta);
        transform_1(2, 2) = std::cos(theta);
    }
    else if (direction == "z" || direction == "Z")
    {
        transform_1(0, 0) = std::cos(theta);
        transform_1(0, 1) = -sin(theta);
        transform_1(1, 0) = sin(theta);
        transform_1(1, 1) = std::cos(theta);
    }
    transform_1.block<3, 1>(0, 3) = translation;

    pcl::transformPointCloud(*cloud, *cloud, transform_1);
}

void pca_match::show_cloud_cluster()
{
    pcl::visualization::PCLVisualizer viewer;
    viewer.getRenderWindow()->GlobalWarningDisplayOff();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(first_cloud, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(first_cloud, 20, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(first_cloud, 0, 0, 255);

    viewer.addPointCloud(first_cloud, color_handler1, "off_scene_model1");
    viewer.addPointCloud(second_cloud, color_handler2, "off_scene_model2");
    viewer.addPointCloud(third_cloud, color_handler3, "off_scene_model3");

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model3");
    viewer.initCameraParameters();
    viewer.spin();

    system("pause");
}

void pca_match::show_model_cluster()
{
    pcl::visualization::PCLVisualizer viewer;
    viewer.getRenderWindow()->GlobalWarningDisplayOff();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(first_model, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(first_model, 20, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(first_model, 0, 0, 255);

    viewer.addPointCloud(first_model, color_handler1, "off_scene_model1");
    viewer.addPointCloud(second_model, color_handler2, "off_scene_model2");
    viewer.addPointCloud(third_model, color_handler3, "off_scene_model3");

    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model3");
    viewer.initCameraParameters();
    viewer.spin();

    system("pause");
}

void pca_match::registration()
{
    Eigen::Matrix4f cloud_pca, model_pca;
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 3; i++)
    {
        cloud_pca = top3_cloud_pca[i];
        model_pca = top3_model_pca[i];
        cout << "cloud pca: " << cloud_pca << endl;
        cout << "model pca: " << model_pca << endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr rotated(new pcl::PointCloud<pcl::PointXYZ>), rotated_model(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr tempcloud(new pcl::PointCloud<pcl::PointXYZ>), tempmodel(new pcl::PointCloud<pcl::PointXYZ>);
        if (i == 0) { tempcloud = first_cloud; tempmodel = first_model; }
        else if (i == 1) { tempcloud = second_cloud; tempmodel = second_model; }
        else if (i == 2) { tempcloud = third_cloud; tempmodel = third_model; }

        pcl::transformPointCloud(*tempmodel, *rotated, (cloud_pca.inverse()) * model_pca);//(transform2.transpose())*transform1
        pcl::transformPointCloud(*model, *rotated_model, (cloud_pca.inverse()) * model_pca);
        pcl::visualization::PCLVisualizer viewer;


        pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZ>());
        viewer.getRenderWindow()->GlobalWarningDisplayOff();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> off_scene_model_color_handler(rotated, 255, 0, 0);
        viewer.addPointCloud(rotated, off_scene_model_color_handler, "off_scene_model");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model");
        //show keypoints
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_keypoints_color_handler(tempcloud, 0, 0, 255);
        viewer.addPointCloud(tempcloud, scene_keypoints_color_handler, "scene_keypoints");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");


        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color_handler(rotated_model, 255, 0, 0);
        viewer.addPointCloud(rotated_model, model_color_handler, "off_scene");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene");
        //show keypoints
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 0, 0, 255);
        viewer.addPointCloud(cloud, color_handler, "scene");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene");



        viewer.initCameraParameters();
        viewer.spin();

        system("pause");
    }

}

void pca_match::registration_by_tri()
{

    Eigen::Vector3f A_center, B_center, t;
    Eigen::Matrix3f A, B, H, R;
    Eigen::Matrix4f transform(Eigen::Matrix4f::Identity()), translation_first(Eigen::Matrix4f::Identity());
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_tranlated(new pcl::PointCloud<pcl::PointXYZ>);
    A_center = (model_centroid[0] + model_centroid[1] + model_centroid[2]) / 3;
    B_center = (cloud_centroid[0] + cloud_centroid[1] + cloud_centroid[2]) / 3;
    translation_first.block<3, 1>(0, 3) = B_center - A_center;
    pcl::transformPointCloud(*model, *model_tranlated, translation_first);
    A.block<3, 1>(0, 0) = model_centroid[0] - A_center;
    A.block<3, 1>(0, 1) = model_centroid[1] - A_center;
    A.block<3, 1>(0, 2) = model_centroid[2] - A_center;

    B.block<3, 1>(0, 0) = cloud_centroid[0] - B_center;
    B.block<3, 1>(0, 1) = cloud_centroid[1] - B_center;
    B.block<3, 1>(0, 2) = cloud_centroid[2] - B_center;

    H = A * B.transpose();
    //cout << "A: \n" << A << endl;
    //cout << "B: \n" << B_center << endl;
    //cout << "H: \n" << H << endl;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f V = svd.matrixV(), U = svd.matrixU();
    R = V * U.transpose();
    t = B_center - R * A_center;
    transform.block<3, 3>(0, 0) = R;
    transform.block<3, 1>(0, 3) = t;
    cout << "transform: \n" << transform << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated(new pcl::PointCloud<pcl::PointXYZ>), rotated_model(new pcl::PointCloud<pcl::PointXYZ>), rotated_model2(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);


    pcl::transformPointCloud(*model, *rotated_model, transform);
    model_rotated = rotated_model;
    pca_match::cluster_model(true);
    pca_match::compute_model_PCA();
    pcl::visualization::PCLVisualizer viewer;


    pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZ>());
    viewer.getRenderWindow()->GlobalWarningDisplayOff();
    //show keypoints
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color_handler(rotated_model, 255, 0, 0);
    //viewer.addPointCloud(rotated_model, model_color_handler, "off_scene");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene");
    //show keypoints
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 0, 0, 255);
    viewer.addPointCloud(cloud, color_handler, "scene");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene");

    //ICP

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    icp.setInputSource(rotated_model);
    icp.setInputTarget(cloud);
    //icp.setMaxCorrespondenceDistance(0.05);
    icp.setMaximumIterations(500);
    icp.setTransformationEpsilon(1e-10);
    icp.align(*cloud_icp);
    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    cout << "Final transformation: \n" << transformation << endl;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_icp_color_h(cloud_icp, 180, 0, 0);
    viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_icp_v2");



    //PCA Alignment
    //std::vector<Eigen::Matrix4f> model_rotated_pcas;

    //pcl::transformPointCloud(*rotated_model, *rotated_model2, top3_cloud_pca[0] * top3_model_pca[0].inverse());
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color_handler2(rotated_model2, 0, 255, 0);
    //viewer.addPointCloud(rotated_model2, model_color_handler2, "pca1");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pca1");
    viewer.initCameraParameters();
    viewer.spin();
    system("pause");
}
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    stable_sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

    return idx;
}
struct costfuncval {
    double u, v, w;
    Eigen::Vector3f p1, p2, p3, t1, t2, t3;
    double calculate() {
        Eigen::Vector3f Rp1, Rp2, Rp3;
        Eigen::Matrix3f yaw = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f pitch = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f roll= Eigen::Matrix3f::Identity();
        yaw(0, 0) = cosf(u); yaw(0, 1) = -sinf(u); yaw(1, 0) = sinf(u); yaw(1, 1) = cosf(u);
        pitch(0, 0) = cosf(v); pitch(0, 2) = sinf(v); pitch(2, 0) = -sinf(v); pitch(2, 2) = cosf(v);
        roll(1, 1) = cosf(w); roll(1, 2) = -sinf(w); roll(2, 1) = sinf(w); roll(2, 2) = cosf(w);
        Eigen::Matrix3f R;
        R = yaw * pitch * roll;
        Rp1 = R * p1; Rp2 = R * p2; Rp3 = R * p3;
        double result = sqrt(2 - 2 * (Rp1.dot(t1))) + sqrt(2 - 2 * (Rp2.dot(t2))) + sqrt(2 - 2 * (Rp3.dot(t3)));
        
        return result;
    }
};

Eigen::Vector3f pca_match::simplex_method(Eigen::Vector3f init, Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3,Eigen::Vector3f t1, Eigen::Vector3f t2, Eigen::Vector3f t3)
{
    //Initialize parameters

    int n = init.size();
    int maxiter = 200 * n;
    int maxfun = 200 * n;
    int rho = 1;
    int chi = 2;
    double psi = 0.5;
    double sigma = 0.5;
    double tolx = 1e-2;
    double tolf = 1e-2;
    int np1 = n + 1;
    int itercount = 0;

    //Continue setting up the initial simplex.
    //Set up a simplex near the initial guess.

    
    Eigen::MatrixXf v(n, np1), v_(n, np1), v_temp(n, np1);
    v_.block<3, 1>(0, 0) = init;
    std::vector<double> fv(np1), fv_(np1), fv_temp(np1);
    double usual_delta = 0.05;
    double zero_term_delta = 0.00025;
    Eigen::Vector3f y = init;
    costfuncval cost;
    cost.u = y[0]; cost.v = y[1]; cost.w = y[2];
    cost.t1 = t1; cost.t2 = t2; cost.t3 = t3;
    cost.p1 = p1; cost.p2 = p2; cost.p3 = p3;
    
    fv_[0] = cost.calculate();
    
    std::vector<double> cost_list{};
    for (int j = 0; j < n; j++)
    {
        y = init;
        if (y[j] != 0)
        {
            y[j] = (1 + usual_delta) * y[j];
        }
        else
        {
            y[j] = zero_term_delta;
        }
        v_.block<3, 1>(0, j + 1) = y;
        cost.u = y[0]; cost.v = y[1]; cost.w = y[2];
        double cost_val = cost.calculate();
        fv_[j + 1] = cost_val;

    }

    std::vector<size_t> idx = sort_indexes(fv_);

    for (int i = 0; i < idx.size(); i++)
    {
        v.block<3, 1>(0, i) = v_.block<3, 1>(0, idx[i]);
        fv[i] = fv_[idx[i]];

    }


    itercount = itercount + 1;
    int func_evals = np1;


    //cout << std::all_of(fv.begin(), fv.end(), [fv, tolf](int i) {for (i = 1; i < 3; i++)return std::abs(fv[0] - fv[i]) <= std::max(tolf, 10 * std::nextafter(0.0, fv[0])); });
    //Main algorithm :

    while (func_evals < maxfun && itercount < maxiter)
    {

        std::vector<double>vv;
        for (int i = 1; i < np1; i++)
        {
            vv.push_back((v.block<3, 1>(0, i) - v.block<3, 1>(0, 0)).lpNorm<Eigen::Infinity>());
        }
        if (std::all_of(fv.begin(), fv.end(), [fv, tolf](int i) {for (i = 1; i < 4; i++)return std::abs(fv[0] - fv[i]) <= tolf; }) &&
            std::all_of(vv.begin(), vv.end(), [tolx](int i) {return i > tolx; }))
        {

            break;
        }
        //Compute the reflection point
        //cout << "\n v:\n " << v << endl;
        Eigen::Vector3f xbar = v.block<3, 3>(0, 0).rowwise().mean();
        Eigen::Vector3f xr = (1 + rho) * xbar - rho * v.block<3, 1>(0, 3);
        cost.u = xr[0]; cost.v = xr[1]; cost.w = xr[2];
        double fxr = cost.calculate();
        func_evals += 1;

        if (fxr < fv[0])
        {
            //Calculate the expansion point
            Eigen::Vector3f xe = (1 + rho * chi) * xbar - rho * chi * v.block<3, 1>(0, 3);
            cost.u = xe[0]; cost.v = xe[1]; cost.w = xe[2];
            double fxe = cost.calculate();
            func_evals += 1;
            if (fxe < fxr)
            {
                v.block<3, 1>(0, 3) = xe;
                fv[np1 - 1] = fxe;
            }
            else
            {
                v.block<3, 1>(0, 3) = xr;
                fv[np1 - 1] = fxr;
            }
        }
        else
        {
            if (fxr < fv[n - 1])//reflect
            {
                v.block<3, 1>(0, 3) = xr;
                fv[np1 - 1] = fxr;
            }
            else//Perform contraction
            {
                if (fxr < fv[np1 - 1])//Perform an outside contraction
                {
                    Eigen::Vector3f xc = (1 + psi * rho) * xbar - psi * rho * v.block<3, 1>(0, 3);
                    cost.u = xc[0]; cost.v = xc[1]; cost.w = xc[2];
                    double fxc = cost.calculate();
                    func_evals += 1;

                    if (fxc <= fxr)//contract uotside
                    {
                        v.block<3, 1>(0, 3) = xc;
                        fv[np1 - 1] = fxc;
                    }
                    else
                    {
                        //shrink
                        for (int j = 1; j < np1; j++)
                        {
                            v.block<3, 1>(0, j) = v.block<3, 1>(0, 0) + sigma * (v.block<3, 1>(0, j) - v.block<3, 1>(0, 0));
                            cost.u = v(0, j); cost.v = v(1, j); cost.w = v(2, j);
                            fv[j] = cost.calculate();
                        }
                        func_evals += 1;
                    }
                }
                else
                {
                    //Perform an inside contraction
                    Eigen::Vector3f xcc = (1 - psi) * xbar + psi * v.block<3, 1>(0, 3);
                    cost.u = xcc[0]; cost.v = xcc[1]; cost.w = xcc[2];
                    double fxcc = cost.calculate();
                    func_evals += 1;

                    if (fxcc < fv[np1 - 1])//contract inside
                    {
                        v.block<3, 1>(0, 3) = xcc;
                        fv[np1 - 1] = fxcc;
                    }
                    else
                    {
                        //shrink
                        for (int j = 1; j < np1 - 1; j++)
                        {
                            v.block<3, 1>(0, j) = v.block<3, 1>(0, 0) + sigma * (v.block<3, 1>(0, j) - v.block<3, 1>(0, 0));
                            cost.u = v(0, j); cost.v = v(1, j); cost.w = v(2, j);
                            fv[j] = cost.calculate();
                        }
                        func_evals += 1;
                    }
                }

            }
        }

        std::vector<size_t> idx_ = sort_indexes(fv);
        v_temp = v;
        fv_temp = fv;
        for (int i = 0; i < idx_.size(); i++)
        {
            v(0, i) = v_temp(0, idx_[i]);
            v(1, i) = v_temp(1, idx_[i]);
            v(2, i) = v_temp(2, idx_[i]);
            fv[i] = fv_temp[idx_[i]];

        }
        itercount += 1;


    }
    Eigen::Vector3f opt_v = v.block<3, 1>(0, 0);
    //cout << "Iterations: " << itercount << endl;
    return opt_v;

}

void pca_match::plane_segmentation()
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory

    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_PROSAC);
    seg.setNumberOfThreads(16);//8
    seg.setDistanceThreshold(0.12);//0.12
    seg.setAxis(Eigen::Vector3f::UnitZ());

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    *model_ = *model;
    const double model_size = model->size();
    while (model_->size() > 0.3 * model_size)
    {

        seg.setInputCloud(model_);
        seg.segment(*inliers, *coefficients);

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(model_);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);
        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(*model_);
        if (plane->size() < 500)//500
        {
            continue;
        }
        model_regions.push_back(plane);
        ////cout << plane->size() << endl;
        //pcl::visualization::PCLVisualizer viewer;
        //viewer.getRenderWindow()->GlobalWarningDisplayOff();
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
        //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
        //viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        //viewer.initCameraParameters();
        //viewer.spin();

        //system("pause");
        inliers->indices.clear();  
        coefficients->values.clear();
    }



}

void pca_match::plane_segmentation2()
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory

    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_PROSAC);
    seg.setNumberOfThreads(16);//8
    seg.setDistanceThreshold(0.01);//0.05
    //seg.setEpsAngle(M_PI / 18);
    seg.setAxis(Eigen::Vector3f::UnitZ());

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    *model_ = *model;
    const double model_size = model->size();
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
    int i = 0;
    while (model_->size() > 0.3 * model_size)//0.3
    { 
        
        seg.setInputCloud(model_);
        seg.segment(*inliers, *coefficients);

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(model_);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(*model_);

        if (plane->size() < 500)//500
        {
            i++;
            continue;
        }
        //if (i == 2 || i == 17 || i == 9)//2 17 9
        //{
        //    model_regions.push_back(plane);

        //    pcl::visualization::PCLVisualizer viewer;
        //    viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //    viewer.getRenderWindow()->GlobalWarningDisplayOff();
        //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
        //    viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
        //    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
        //    viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
        //    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        //    viewer.initCameraParameters();
        //    viewer.spin();

        //    system("pause");
        //}
        
        //pcl::visualization::PCLVisualizer viewer;
        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //viewer.getRenderWindow()->GlobalWarningDisplayOff();
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
        //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
        //viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        //viewer.initCameraParameters();
        //viewer.spin();

        //system("pause");
        //cout << plane->size() << endl;
        //cout << "i: " << i << endl;

        inliers->indices.clear();
        coefficients->values.clear();
        i++;
    }



}

void pca_match::plane_segmentation3()
{
    //find cylinder first
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg2;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(model);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    // Create the segmentation object for cylinder segmentation and set all the parameters
    seg2.setOptimizeCoefficients(true);
    seg2.setModelType(pcl::SACMODEL_CYLINDER);
    seg2.setMethodType(pcl::SAC_RANSAC);
    seg2.setNormalDistanceWeight(0.1);
    seg2.setDistanceThreshold(0.03);
    seg2.setRadiusLimits(0, 10);
    seg2.setInputCloud(model);
    seg2.setInputNormals(cloud_normals);
   

    pcl::ConcaveHull<pcl::PointXYZ> hull;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    hull.setInputCloud(model);
    hull.setAlpha(10);
    hull.reconstruct(*cloud_hull);
    cloud_hull_ = cloud_hull;


    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory

    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_PROSAC);
    seg.setNumberOfThreads(16);//8
    seg.setDistanceThreshold(0.03);//0.05
    //seg.setEpsAngle(M_PI / 18);
    seg.setAxis(Eigen::Vector3f::UnitZ());



    pcl::PointCloud<pcl::PointXYZ>::Ptr model_(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    *model_ = *cloud_hull;
    const double model_size = cloud_hull->size();
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
    int i = 0;
    while (model_->size() > 0.3 * model_size)//0.3
    {
        if (i == 0)
        {
            seg2.setInputCloud(model);
            seg2.segment(*inliers, *coefficients);
            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZ> extract;

            pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setInputCloud(model);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*plane);
            model_regions.push_back(plane);

            //pcl::visualization::PCLVisualizer viewer;
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.getRenderWindow()->GlobalWarningDisplayOff();
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
            //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
            //viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
            //viewer.initCameraParameters();
            //viewer.spin();

            //system("pause");
        }
        seg.setInputCloud(model_);
        seg.segment(*inliers, *coefficients);

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(model_);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(*model_);

        if (plane->size() < 500)//500
        {
            i++;
            continue;
        }
        model_regions.push_back(plane);


        //pcl::visualization::PCLVisualizer viewer;
        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //viewer.getRenderWindow()->GlobalWarningDisplayOff();
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
        //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
        //viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        //viewer.initCameraParameters();
        //viewer.spin();

        //system("pause");
        //cout << plane->size() << endl;
        //cout << "i: " << i << endl;

        inliers->indices.clear();
        coefficients->values.clear();
        i++;
    }
}



Eigen::Vector3d ComputeMinBound(
    const std::vector<Eigen::Vector3d>& points)  
{
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
        points.begin(), points.end(), points[0],
        [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
            return a.array().min(b.array()).matrix();
        });
}

Eigen::Vector3d ComputeMaxBound(
    const std::vector<Eigen::Vector3d>& points) 
{
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
        points.begin(), points.end(), points[0],
        [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
            return a.array().max(b.array()).matrix();
        });
}

void pca_match::plane_segmentation_hidden()
{
    pcl::visualization::PCLVisualizer viewer;
    std::vector<Eigen::Vector3d> model_points{model->size()};
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points)- ComputeMinBound(model_points);
    double diameter = max_min.norm();
    double radius = diameter*250 ;//*250
    //double radius = 100;

    std::vector<pcl::visualization::Camera> cam;

    //find cylinder first
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg2;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);


    for (int k = 0; k < 3; k++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>), cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
        if (k==0)
        {
            viewer.setCameraPosition(-diameter, 0, 0, 5, 0, 0);
            camera_location1[0] = diameter;
            camera_location1[1] = 0;
            camera_location1[2] = 15;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }

            
            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);

        }
        else if (k==1)
        {
            viewer.setCameraPosition(diameter, 0, 10, 0, 0, 0);
            camera_location2[0] = 0;
            camera_location2[1] = diameter;
            camera_location2[2] = 15;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }

            
            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }
        else if (k==2)
        {
            viewer.setCameraPosition(0, 0, -diameter, 0, 0, 0);
            camera_location3[0] = 0;
            camera_location3[1] = 0;
            camera_location3[2] = -diameter;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }

            
            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }

        //ransac plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory

        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_PROSAC);
        seg.setNumberOfThreads(16);//8
        seg.setDistanceThreshold(1);//0.05
        //seg.setEpsAngle(M_PI / 18);
        seg.setAxis(Eigen::Vector3f::UnitZ());

        // Create the segmentation object for cylinder segmentation and set all the parameters
        seg2.setOptimizeCoefficients(true);
        seg2.setModelType(pcl::SACMODEL_CYLINDER);
        seg2.setMethodType(pcl::SAC_RANSAC);
        seg2.setNormalDistanceWeight(0.1);
        seg2.setDistanceThreshold(0.03);
        seg2.setRadiusLimits(0, 10);
        seg2.setInputCloud(cloud_p);
        seg2.setInputNormals(cloud_normals);

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        *model_ = *cloud_p;
        const double model_size = cloud_p->size();
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
       

        if (k == 1)
        {

            // Estimate point normals
            ne.setSearchMethod(tree);
            ne.setInputCloud(cloud_p);
            ne.setKSearch(50);
            ne.compute(*cloud_normals);

            seg2.setInputCloud(cloud_p);
            seg2.segment(*inliers, *coefficients);
            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZ> extract;

            pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setInputCloud(cloud_p);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*plane);
            model_regions.push_back(plane);


            //pcl::visualization::PCLVisualizer viewer;
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.getRenderWindow()->GlobalWarningDisplayOff();
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
            //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
            //viewer.addPointCloud(cloud_p, color_handler2, "off_scene_model2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
            //viewer.initCameraParameters();
            //viewer.spin();

            //system("pause");
            //cout << plane->size() << endl;
            //cout << "i: " << i << endl;

            inliers->indices.clear();
            coefficients->values.clear();
            
        }
        else
        {
            seg.setInputCloud(model_);
            seg.segment(*inliers, *coefficients);

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZ> extract;

            pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setInputCloud(model_);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*plane);
            model_regions.push_back(plane);
            // Remove the planar inliers, extract the rest
            extract.setNegative(true);
            extract.filter(*model_);

            //pcl::visualization::PCLVisualizer viewer;
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.getRenderWindow()->GlobalWarningDisplayOff();
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
            //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
            //viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
            //viewer.initCameraParameters();
            //viewer.spin();

            //system("pause");
            //cout << plane->size() << endl;
            //cout << "i: " << i << endl;

            inliers->indices.clear();
            coefficients->values.clear();
            
        }
            





    }
    


   



    //cout << "radius: " << radius << std::endl;
    //cout << "Original Cloud size " << model->points.size() << std::endl;
    //cout << "New Cloud's size " << newCloud->points.size() << std::endl;
    //

    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.getRenderWindow()->GlobalWarningDisplayOff();
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(newCloud, 255, 0, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model, 0, 0, 255);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(cloud_hull, 0, 255, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(cloud_p, 0, 255, 0);
    ////viewer.addPointCloud(newCloud, color_handler1, "off_scene_model1");
    ////viewer.addPointCloud(model, color_handler2, "off_scene_model2");
    ////viewer.addPointCloud(cloud_hull, color_handler3, "off_scene_model3");
    //viewer.addPointCloud(cloud_p, color_handler4, "off_scene_model4");
    //
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model1");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model2");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model3");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "off_scene_model4");

    //
    ////viewer.initCameraParameters();
    //viewer.resetCamera();
    //viewer.spin();
    //system("pause");
}
pcl::PointCloud<pcl::PointXYZ>::Ptr pca_match::hull(Eigen::Vector3d camera)
{
    //pcl::visualization::PCLVisualizer viewer;
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;
    double radius = diameter * 250;//*250
    //double radius = 100;



    //std::vector<pcl::visualization::Camera> cam;
    //viewer.setCameraPosition(camera[0], camera[1], camera[2], 0, 0, 0);
    //viewer.getCameras(cam);
    //pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
    pcl::PointXYZ cameraPoint = pcl::PointXYZ(camera[0], camera[1], camera[2]);

    //cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
    Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
    std::vector<Eigen::Vector3d> spherical_proojection;
    pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

    //step1:spherical projection
    for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
    {
        pcl::PointXYZ currentPoint = model->points[pidx];
        Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

        Eigen::Vector3d projected_point = currentVector - camera_location;
        double norm = projected_point.norm();
        //if (norm == 1)
        //{
        //    norm = 0.0001;
        //}
        spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
    }
    size_t origin_pidx = spherical_proojection.size();
    spherical_proojection.push_back(Eigen::Vector3d(camera[0], camera[1], camera[2]));
    //spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
    for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
    {
        Eigen::Vector3d currentVector = spherical_proojection.at(i);
        pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
        newCloud->push_back(currentPoint);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConvexHull<pcl::PointXYZ> chull;
    pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
    chull.setInputCloud(newCloud);
    chull.reconstruct(*cloud_hull);
    chull.getHullPointIndices(*hull_indices);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(model);
    extract.setIndices(hull_indices);
    extract.filter(*cloud_p);

    //pcl::visualization::PCLVisualizer viewer;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_p, 255, 0, 0);


    //viewer.addPointCloud(cloud_p, color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.spin();
    //system("pause");

    return cloud_p;
}



void pca_match::alpha_shape()
{
    pcl::ConcaveHull<pcl::PointXYZ> hull;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    hull.setInputCloud(model);
    hull.setAlpha(2.5);//new1:3
    hull.reconstruct(*cloud_hull);
    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    viewer.getRenderWindow()->GlobalWarningDisplayOff();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(cloud_hull, 255, 0, 0);
    viewer.addPointCloud(cloud_hull, color_handler1, "off_scene_model1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
 /*   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model, 0, 255, 0);
    viewer.addPointCloud(model, color_handler2, "off_scene_model2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");*/
    viewer.initCameraParameters();
    viewer.spin();

    system("pause");
}

void pca_match::region_growing()
{
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud_pass_through);
    normal_estimator.setKSearch(9);
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);//70
    reg.setMaxClusterSize(3000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(20);
    reg.setInputCloud(cloud_pass_through);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(4/ 180.0 * M_PI);// for stack.ply set to 3/180//for 15_pile_up set to 8/180 with noise 5/180
    reg.setCurvatureThreshold(0.8);//1.0
    std::vector <pcl::PointIndices> clusters;

    reg.extract(clusters);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_pass_through);

    //cout << "OKKKK" << endl;

    // For every cluster...
    int currentClusterNum = 1;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        //添加所有的点云到一个新的点云中
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(cloud_pass_through->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        // 保存
        if (cluster->points.size() <= 0)
            break;
        //std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
        //std::string fileName = "cluster" + boost::to_string(currentClusterNum) + ".pcd";
        //pcl::io::savePCDFileASCII(fileName, *cluster);
        regions.push_back(cluster);// region in class private

        currentClusterNum++;
    }

    //pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    //pcl::visualization::CloudViewer viewer("Cluster viewer");
    //viewer.showCloud(colored_cloud);

    //while (!viewer.wasStopped())
    //{
    //}
}

void pca_match::pass_through()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass; // 声明直通滤波
    pass.setInputCloud(cloud); // 传入点云数据
    pass.setFilterFieldName("z"); // 设置操作的坐标轴
    pass.setFilterLimits(0.6, 200);// 设置坐标范围
    // pass.setFilterLimitsNegative(true); // 保留数据函数
    pass.filter(*cloud_filtered);  // 进行滤波输出

    //pcl::visualization::PCLVisualizer viewer;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_filtered, 0, 255, 0);


    //viewer.addPointCloud(cloud_filtered, color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //while (!viewer.wasStopped())
    //{
    //    viewer.spinOnce();
    //}
    cloud_pass_through = cloud_filtered;
}

void pca_match::triangulation()
{
    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    n.setInputCloud(cloud);
    n.setSearchMethod(tree);
    n.setKSearch(10);
    n.compute(*normals);

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud(cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points(maximum edge length)
    gp3.setSearchRadius(0.025);

    // Set typical values for the parameters
    gp3.setMu(1.0);
    gp3.setMaximumNearestNeighbors(20);
    gp3.setMaximumSurfaceAngle(M_PI / 6); // 45 degrees
    gp3.setMinimumAngle(M_PI / 30); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree2);
    gp3.reconstruct(triangles);

    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPolygonMesh(triangles, "polygon", 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);

    }

}
bool compare(pcl::PointCloud<pcl::PointXYZ>::Ptr a, pcl::PointCloud<pcl::PointXYZ>::Ptr b)
{
    double mean_a{ 0 }, mean_b{ 0 };
    for (int i = 0; i < a->size(); i++)
    {
        mean_a += a->points[i].z;
    }
    for (int j = 0; j < b->size(); j++)
    {
        mean_b += b->points[j].z;
    }
    mean_a = mean_a / a->size();
    mean_b = mean_b / b->size();

    return (a->size() > b->size()&& mean_a > mean_b); //a->size() > b->size();
}
void pca_match::show_region_growing_part()
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    std::sort(regions_.begin(), regions_.end(), compare);
    cout << "Totally " << regions.size() << " Parts." << endl;
    for (int j = 0; j < regions.size(); j++)
    {
        pcl::visualization::PCLVisualizer viewer;
        viewer.getRenderWindow()->GlobalWarningDisplayOff();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(regions_[j], 255, 0, 0);
        viewer.addPointCloud(regions_[j], color_handler1, "off_scene_model1");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud_pass_through, 0, 255, 0);
        viewer.addPointCloud(cloud_pass_through, color_handler2, "off_scene_model2");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "off_scene_model2");
        //viewer.initCameraParameters();
        viewer.spin();
        system("pause");

    }

}

Eigen::Vector3f compute_region_pca(pcl::PointCloud<pcl::PointXYZ>::Ptr region)//return normal vectors
{
    pcl::PointXYZ o, pcaZ, pcaY, pcaX, c, pcX, pcY, pcZ;
    Eigen::Vector4f pcaCentroid;

    pcl::compute3DCentroid(*region, pcaCentroid);

    //cout << "MODEL PCA CENTROID: " << pcaCentroid << endl;
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*region, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    Eigen::Vector3f centroid;
    Eigen::Vector3f p_pi;
    Eigen::Vector3f p;
    centroid[0] = pcaCentroid(0);
    centroid[1] = pcaCentroid(1);
    centroid[2] = pcaCentroid(2);

    Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
    transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*region, *transformedCloud, transform);


    //std::cout << "Eigenvalue: \n" << eigenValuesPCA << std::endl;
    //std::cout << "Eigenvector: \n" << eigenVectorsPCA << std::endl;


    o.x = 0.0;
    o.y = 0.0;
    o.z = 0.0;
    Eigen::Affine3f tra_aff(transform);
    Eigen::Vector3f pz = eigenVectorsPCA.col(0);
    Eigen::Vector3f py = eigenVectorsPCA.col(1);
    Eigen::Vector3f px = eigenVectorsPCA.col(2);
    pcl::transformVector(pz, pz, tra_aff);
    pcl::transformVector(py, py, tra_aff);
    pcl::transformVector(px, px, tra_aff);
    int votex, votey{ 0 }, votey_{ 0 }, votez{ 0 }, votez_{ 0 };

    for (size_t i = 0; i < transformedCloud->size(); i++)
    {
        p_pi[0] = 0; p_pi[1] = 0; p_pi[2] = 0;
        p[0] = region->points[i].x;
        p[1] = region->points[i].y;
        p[2] = region->points[i].z;

        p_pi = p - centroid;
        if (p_pi.dot(pz) >= 0)
        {
            votez += 1;
        }
        else if (p_pi.dot(-pz) > 0)
        {
            votez_ += 1;
        }

        if (p_pi.dot(py) >= 0)
        {
            votey += 1;
        }
        else if (p_pi.dot(-py) > 0)
        {
            votey_ += 1;
        }


    }
    if (votez > votez_) {
        eigenVectorsPCA.col(0) = eigenVectorsPCA.col(0);
    }
    else
    {
        eigenVectorsPCA.col(0) = -eigenVectorsPCA.col(0);
    }
    if (votey > votey_) {
        eigenVectorsPCA.col(1) = eigenVectorsPCA.col(1);
    }
    else
    {
        eigenVectorsPCA.col(1) = -eigenVectorsPCA.col(1);
    }

    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

    pcaZ.x = 1000 * pz(0);
    pcaZ.y = 1000 * pz(1);
    pcaZ.z = 1000 * pz(2);

    pcaY.x = 1000 * py(0);
    pcaY.y = 1000 * py(1);
    pcaY.z = 1000 * py(2);

    pcaX.x = 1000 * px(0);
    pcaX.y = 1000 * px(1);
    pcaX.z = 1000 * px(2);

    c.x = pcaCentroid(0);
    c.y = pcaCentroid(1);
    c.z = pcaCentroid(2);


    pcZ.x = 10 * eigenVectorsPCA(0, 0) + c.x;//normal vector
    pcZ.y = 10 * eigenVectorsPCA(1, 0) + c.y;
    pcZ.z = 10 * eigenVectorsPCA(2, 0) + c.z;

    pcY.x = 5 * eigenVectorsPCA(0, 1) + c.x;
    pcY.y = 5 * eigenVectorsPCA(1, 1) + c.y;
    pcY.z = 5 * eigenVectorsPCA(2, 1) + c.z;

    pcX.x = 5 * eigenVectorsPCA(0, 2) + c.x;
    pcX.y = 5 * eigenVectorsPCA(1, 2) + c.y;
    pcX.z = 5 * eigenVectorsPCA(2, 2) + c.z;
    //pcl::visualization::PCLVisualizer viewer;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(region, 255, 0, 0);


    //viewer.addPointCloud(region, color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");


    ////viewer.addArrow(pcaZ, o, 0.0, 0.0, 1.0, false, "arrow_Z");
    ////viewer.addArrow(pcaY, o, 0.0, 1.0, 0.0, false, "arrow_Y");
    ////viewer.addArrow(pcaX, o, 1.0, 0.0, 0.0, false, "arrow_X");

    //viewer.addArrow(pcZ, c, 0.0, 0.0, 1.0, false, "arrow_z");
    //viewer.addArrow(pcY, c, 0.0, 1.0, 0.0, false, "arrow_y");
    //viewer.addArrow(pcX, c, 1.0, 0.0, 0.0, false, "arrow_x");



    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.spin();


    return eigenVectorsPCA.col(0);
}

Eigen::Matrix4f Rotation_matrix(Eigen::Vector3f norm_a, Eigen::Vector3f norm_b, Eigen::Vector4f center_a, Eigen::Vector4f center_b)
{


    Eigen::Vector3f v = norm_a.cross(norm_b);
    Eigen::Matrix3f vx;
    vx << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    double c = norm_a.dot(norm_b);
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity() + vx + vx * vx * (1 / (1 + c));
    Eigen::Matrix4f temp = Eigen::Matrix4f::Identity();
    temp.block<3, 3>(0, 0) = R;
    temp.block<3, 1>(0, 3) = center_b.block<3, 1>(0, 0) - R * center_a.block<3, 1>(0, 0);
    return temp;
}

void rot2ang(Eigen::Matrix4f matrix)
{
    double x1{ 0 }, x2{ 0 }, y1{ 0 }, y2{ 0 }, z1{ 0 }, z2{ 0 };
    y1 = -std::asinf(matrix(2, 0));
    y2 = M_PI - x1;
    x1 = std::atan2f(matrix(2, 1) / cosf(y1), matrix(2, 2) / cosf(y1));
    x2 = std::atan2f(matrix(2, 1) / cosf(y2), matrix(2, 2) / cosf(y2));
    z1 = std::atan2f(matrix(1, 0) / cosf(y1), matrix(0, 0) / cosf(y1));
    z2 = std::atan2f(matrix(1, 0) / cosf(y2), matrix(0, 0) / cosf(y2));

    cout << "1. XYZ: " << x1 * 180 / M_PI << " , " << y1 * 180 / M_PI << " , " << z1 * 180 / M_PI << endl;
    cout << "2. XYZ: " << x2 * 180 / M_PI << " , " << y2 * 180 / M_PI << " , " << z2 * 180 / M_PI << endl;


}

void pca_match::find_parts_in_scene()
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;
    //double min_dist = 100;
    //double min_dist2 = 200;
    //double min_dist3 = 300;

    //for (int i = 0; i < regions_.size(); i++)
    //{
    //    Eigen::Vector4f centroid;
    //    pcl::compute3DCentroid(*regions_[i], centroid);
    //    center_list.push_back(centroid);
    //}

    /*for (int k =  1; k < regions_.size(); k++)
    {
        Eigen::Vector4f c1 = center_list[0];
        Eigen::Vector4f c2 = center_list[k];
        double dist = (c1 - c2).norm();
        if (dist < min_dist)
        {
            min_dist3 = min_dist2;
            adjacent3 = adjacent2;

            min_dist2 = min_dist;
            adjacent2 = adjacent;

            min_dist = dist;
            adjacent = regions_[k];
        }
        else if (dist < min_dist2)
        {
            min_dist3 = min_dist2;
            adjacent3 = adjacent2;

            min_dist2 = dist;
            adjacent2 = regions_[k];
        }
        else if (dist < min_dist3)
        {
            min_dist3 = dist;
            adjacent3 = regions_[k];
        }
    }*/
    /*pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(adjacent, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(adjacent2, 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(adjacent2, 255, 0, 255);


    viewer.addPointCloud(adjacent, color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud3");
    viewer.addPointCloud(adjacent2, color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(adjacent3, color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    viewer.spin();*/
    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point2(new pcl::PointCloud<pcl::PointXYZ>());
    
    

    for (int i = 0; i < 2; i++)//regions_.size()
    {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        pcl::compute3DCentroid(*regions_[i], cloud_center);
        
        for (int j = 0; j < model_regions.size(); j++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);

            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(rotatedCloud, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(transformed_final, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_point, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[0], 0, 0, 255);
            //viewer.addPointCloud(rotatedCloud, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(transformed_final, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(nearest_point, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(regions_[0], color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //while (!viewer.wasStopped()) {
            //    viewer.spinOnce();
            //}
            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;

            for (int j = 0; j < transformed_final->size(); j++)
            {

                searchPoint.x = transformed_final->points[j].x;
                searchPoint.y = transformed_final->points[j].y;
                searchPoint.z = transformed_final->points[j].z;

                searchPoint_r.x = rotatedCloud->points[j].x;
                searchPoint_r.y = rotatedCloud->points[j].y;
                searchPoint_r.z = rotatedCloud->points[j].z;


                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {   
                    total_dist += pointKNNSquaredDistance[0];
                }
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    total_dist_r += pointKNNSquaredDistance[0];
                }

            }

            
            total_dist /= model_temp->size();
            total_dist_r /= model_temp->size();

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);

            transformation_cloud_list.push_back(transformed_final);
            transformation_cloud_list.push_back(rotatedCloud);

            transformation_matrix_list.push_back(transformation);

            //cout << "Total dist: " << total_dist << endl;
            //cout << "Total dist rotation: " << total_dist_r << endl;


  
        }

        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];

        //rot2ang(transformation_matrix_list[index]);
        int K = 1;
        double total_dist = 0;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);
        pcl::PointXYZ searchPoint2;
        pcl::PointXYZ searchPoint2_;

        
        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < transformed_show->size(); j++)
        {
            searchPoint2.x = transformed_show->points[j].x;
            searchPoint2.y = transformed_show->points[j].y;
            searchPoint2.z = transformed_show->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.3)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }
        
        
        near_list.push_back(nearest_point->size());
        dist_list.clear();
        nearest_point->clear();
    }
    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);


    cout << "Nearest points size: " << near_list[index_near] << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_point, 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_point, color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();
    
    
}

void pca_match::region_growing_model()
{
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(model);
    normal_estimator.setKSearch(5);
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(2000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(5);
    reg.setInputCloud(model);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(1.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(2.0);
    std::vector <pcl::PointIndices> clusters;

    reg.extract(clusters);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(model);



    // For every cluster...
    int currentClusterNum = 1;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        //添加所有的点云到一个新的点云中
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(model->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        // 保存
        if (cluster->points.size() <= 0)
            break;
        //std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
        //std::string fileName = "cluster" + boost::to_string(currentClusterNum) + ".pcd";
        //pcl::io::savePCDFileASCII(fileName, *cluster);
        regions.push_back(cluster);// region in class private

        currentClusterNum++;
    }

    for (int j = 0; j < regions.size(); j++)
    {
        pcl::visualization::PCLVisualizer viewer;
        viewer.getRenderWindow()->GlobalWarningDisplayOff();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(regions[j], 255, 0, 0);
        viewer.addPointCloud(regions[j], color_handler1, "off_scene_model1");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model, 0, 255, 0);
        viewer.addPointCloud(model, color_handler2, "off_scene_model2");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model2");
        viewer.initCameraParameters();
        viewer.spin();
        cout << regions[j]->size() << endl;
        system("pause");

    }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud < pcl::PointNormal > ::Ptr subsampleAndCalculateNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    const Eigen::Vector4f subsampling_leaf_size(0.02f, 0.02f, 0.02f, 0.0f);
    constexpr float normal_estimation_search_radius = 0.05f;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_subsampled(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> subsampling_filter;
    subsampling_filter.setInputCloud(cloud);
    subsampling_filter.setLeafSize(subsampling_leaf_size);
    subsampling_filter.filter(*cloud_subsampled);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_subsampled_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation_filter;
    normal_estimation_filter.setInputCloud(cloud_subsampled);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr search_tree(new pcl::search::KdTree<pcl::PointXYZ>);
    normal_estimation_filter.setSearchMethod(search_tree);
    normal_estimation_filter.setRadiusSearch(normal_estimation_search_radius);
    normal_estimation_filter.compute(*cloud_subsampled_normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_subsampled_with_normals(
        new pcl::PointCloud<pcl::PointNormal>());
    concatenateFields(
        *cloud_subsampled, *cloud_subsampled_normals, *cloud_subsampled_with_normals);

    PCL_INFO("Cloud dimensions before / after subsampling: %zu / %zu\n",
        static_cast<std::size_t>(cloud->size()),
        static_cast<std::size_t>(cloud_subsampled->size()));
    return cloud_subsampled_with_normals;
}


void pca_match::setInputSurfaces(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> surfaces)
{
    model_regions = surfaces;
}

void pca_match::OBB()
{
    pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(model);
    feature_extractor.compute();

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    feature_extractor.getMomentOfInertia(moment_of_inertia);
    feature_extractor.getEccentricity(eccentricity);
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues(major_value, middle_value, minor_value);
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter(mass_center);
    //for (auto& i : moment_of_inertia)cout << "moment_of_inertia: " << i;
    cout << "min_point_OBB: " << min_point_OBB << endl;
    cout << "max_point_OBB: " << max_point_OBB << endl;
    cout << "major_vector" << major_vector << endl;
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->addPointCloud<pcl::PointXYZ>(model, "sample cloud");
    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Quaternionf quat(rotational_matrix_OBB);
    viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB");
    pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
    pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
    pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
    pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
    viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
    viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
    viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();

    }
}

void pca_match::fpfh_model(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors_)
{
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    //pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors_(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(model);
    normal_estimator.setKSearch(10);
    normal_estimator.setNumberOfThreads(32);
    normal_estimator.compute(*normals);
    model_normals = normals;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints;
    pcl::PointXYZ keypoints;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_list(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector4d center;

    int numSelectedPointClouds = 50;
    pcl::PointCloud<pcl::PointXYZ>::Ptr selectedPointClouds = randomlySelectPointClouds(model, numSelectedPointClouds);
    



    //pcl::compute3DCentroid(*model_regions[0], center);
    //keypoints.y = center[0];
    //keypoints.y = center[1];
    //keypoints.z = center[2];



    //keypoints_list->push_back(keypoints);
    //
    fpfh.setInputCloud(selectedPointClouds);  // 计算keypoints处的特征//selectedPointClouds
    fpfh.setInputNormals(normals);   // cloud的法线
    fpfh.setSearchSurface(model); 
    fpfh.setSearchMethod(kdtree);
    fpfh.setNumberOfThreads(32);
    fpfh.setRadiusSearch(5.0);
    
    fpfh.compute(*model_descriptors_);

    pcl::visualization::PCLVisualizer viewer("registration Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color(model, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_keypoint_color(selectedPointClouds, 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_region_color(model_regions[0], 255, 0, 255);

    
    viewer.setBackgroundColor(255, 255, 255);
    viewer.addPointCloud(model, model_color, "model");
    viewer.addPointCloud(model_regions[0], model_region_color, "region");
    viewer.addPointCloud(selectedPointClouds, model_keypoint_color, "model_keypoints");
    
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "region");

    //viewer.addSphere((*keypoints_list)[0], 20, 1, 0.0, 0.0, "sphere");
    
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        
    }
    //model_keypoints = model_regions[0];
    model_keypoints = selectedPointClouds;
    //pcl::visualization::PCLPlotter plotter;
    //plotter.addFeatureHistogram(*descriptors, 33); //设置的很坐标长度，该值越大，则显示的越细致
    //
    //plotter.plot();
   

}

void pca_match::fpfh_scene(pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors_)
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    std::sort(regions_.begin(), regions_.end(), compare);
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    //pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors_(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    int numSelectedPointClouds = 20;
    pcl::PointCloud<pcl::PointXYZ>::Ptr selectedPointClouds = randomlySelectPointClouds(regions_[0], numSelectedPointClouds);
    scene_keypoints = regions_[0];
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(10);
    normal_estimator.setNumberOfThreads(32);
    normal_estimator.setSearchSurface(cloud);
    normal_estimator.compute(*normals);
    scene_normals = normals;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints;
    pcl::PointXYZ keypoints;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_list(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector4d center;

    




    
    fpfh.setInputCloud(regions_[0]);  // 计算keypoints处的特征
    fpfh.setInputNormals(normals);   // cloud的法线
    fpfh.setSearchSurface(cloud); 
    fpfh.setSearchMethod(kdtree);
    fpfh.setNumberOfThreads(32);
    fpfh.setRadiusSearch(5.0);
    
    fpfh.compute(*scene_descriptors_);

    pcl::visualization::PCLVisualizer viewer("registration Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> region_color(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> selected_point_color(selectedPointClouds, 255, 0, 255);

    
    viewer.setBackgroundColor(255, 255, 255);
    viewer.addPointCloud(cloud, scene_color, "scene");
    viewer.addPointCloud(regions_[0], region_color, "region");
    viewer.addPointCloud(selectedPointClouds, selected_point_color, "scene_keypoints");
    
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "region");

    //viewer.addSphere((*keypoints_list)[0], 20, 1, 0.0, 0.0, "sphere");
    
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        
    }


    //pcl::visualization::PCLPlotter plotter;
    //plotter.addFeatureHistogram(*descriptors, 33); //设置的很坐标长度，该值越大，则显示的越细致
    //
    //plotter.plot();
    
}

void pca_match::find_match(pcl::PointCloud<pcl::FPFHSignature33>::Ptr model_descriptors, pcl::PointCloud<pcl::FPFHSignature33>::Ptr scene_descriptors, pcl::CorrespondencesPtr model_scene_corrs)
{
    
    pcl::KdTreeFLANN<pcl::FPFHSignature33> matching;
    matching.setInputCloud(model_descriptors);
    for (size_t i = 0; i < scene_descriptors->size(); ++i)
    {
        std::vector<int> neighbors(1);
        std::vector<float> squaredDistances(1);
        // Ignore NaNs.
        if (std::isfinite(scene_descriptors->at(i).histogram[0]))
        {
            // Find the nearest neighbor (in descriptor space)...
            int neighborCount = matching.nearestKSearch(scene_descriptors->at(i), 1, neighbors, squaredDistances);
            // ...and add a new correspondence if the distance is less than a threshold
            // (SHOT distances are between 0 and 1, other descriptors use different metrics).
            cout << squaredDistances[0] << endl;
            cout << neighbors[0] << endl;
            if (neighborCount == 1 && squaredDistances[0] < 300.0)
            {
                pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i), squaredDistances[0]);
                model_scene_corrs->push_back(correspondence);
                cout << "( " << correspondence.index_query << "," << correspondence.index_match << " )" << endl;
            }
        }
    }

    std::cout << "Found " << model_scene_corrs->size() << " correspondences." << std::endl;

    //Actual Clustering
    float model_ss_(0.01f);
    float scene_ss_(0.03f);
    float rf_rad_(2.0f);
    
    float cg_size_(2.0f);
    float cg_thresh_(2.0f);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
    rf_est.setFindHoles(true);
    
    rf_est.setRadiusSearch(rf_rad_);
    rf_est.setInputCloud(model_keypoints);
    rf_est.setInputNormals(model_normals);
    rf_est.setSearchSurface(model);
    rf_est.compute(*model_rf);
    
    rf_est.setInputCloud(scene_keypoints);
    rf_est.setInputNormals(scene_normals);
    rf_est.setSearchSurface(cloud);
    rf_est.compute(*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
    clusterer.setHoughBinSize(cg_size_);
    clusterer.setHoughThreshold(cg_thresh_);
    clusterer.setUseInterpolation(true);
    clusterer.setUseDistanceWeight(false);

    clusterer.setInputCloud(model_keypoints);
    clusterer.setInputRf(model_rf);
    clusterer.setSceneCloud(scene_keypoints);
    clusterer.setSceneRf(scene_rf);
    clusterer.setModelSceneCorrespondences(model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize(rototranslations, clustered_corrs);
    pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
    viewer.addPointCloud(cloud, "scene_cloud");
    pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZ>());


    //  We are translating the model so that it doesn't end in the middle of the scene representation
    pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
    pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
    viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");
    


    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_keypoints_color_handler(scene_keypoints, 0, 0, 255);
    viewer.addPointCloud(scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, 0, 0, 255);
    viewer.addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");


    for (std::size_t i = 0; i < rototranslations.size(); ++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_model(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rotated_model_color_handler(rotated_model, 255, 0, 0);
        viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());


        for (std::size_t j = 0; j < clustered_corrs[i].size(); ++j)
        {
            std::stringstream ss_line;
            ss_line << "correspondence_line" << i << "_" << j;
            pcl::PointXYZ& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
            pcl::PointXYZ& scene_point = scene_keypoints->at(clustered_corrs[i][j].index_match);

            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(model_point, scene_point, 0, 255, 0, ss_line.str());
        }
        
    }

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

}

void pca_match::visualize_corrs(pcl::CorrespondencesPtr model_scene_corrs)
{
    // 添加关键点
    pcl::visualization::PCLVisualizer viewer("corrs Viewer");
    viewer.setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_color(model, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_color(cloud, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_keypoint_color(model_keypoints, 0, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_keypoint_color(scene_keypoints, 0, 0, 255);
    viewer.addPointCloud(model, model_color, "model");
    viewer.addPointCloud(cloud, scene_color, "scene");
    viewer.addPointCloud(model_keypoints, model_keypoint_color, "model_keypoints");
    viewer.addPointCloud(scene_keypoints, scene_keypoint_color, "scene_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "model_keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "scene_keypoints");

    // 可视化对应关系
    cout << model_keypoints->size() << endl;
    cout << scene_keypoints->size() << endl;

    viewer.addCorrespondences<pcl::PointXYZ>(model_keypoints, scene_keypoints, *model_scene_corrs);
        //添加对应关系
        /*int i=1;*/
        //for(auto iter=model_scene_corrs->begin();iter!=model_scene_corrs->end();++iter)
        //{
        //    std::stringstream ss_line;
        //    ss_line << "correspondence_line" << i ;
        //    i++;
        //    PointType& model_point = model_keypoints->at (iter->index_query);  // 从corrs中获取对应点
        //    PointType& scene_point = scene_keypoints->at (iter->index_match);
        //    viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 0, 0, ss_line.str ());
        //    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, ss_line.str ());   // 设置线宽
    
    
    
        //}

        // 显示
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        
    }

}

void pca_match::add_noise()
{
    const double noise_mean = 0.0;  // Mean of the noise distribution
    const double noise_stddev = 0.1;  // Standard deviation of the noise distribution

    pcl::common::UniformGenerator<float> rand_gen;  // Random number generator
    
    rand_gen.setSeed(time(0));

    for (pcl::PointXYZ& point : cloud_pass_through->points) {
        // Generate random noise
        const double noise_x = rand_gen.run() * noise_stddev + noise_mean;
        const double noise_y = rand_gen.run() * noise_stddev + noise_mean;
        const double noise_z = rand_gen.run() * noise_stddev + noise_mean;

        // Add noise to the point coordinates
        point.x += noise_x;
        point.y += noise_y;
        point.z += noise_z;
    }
}

Eigen::Matrix3f compute_region_pca_all(pcl::PointCloud<pcl::PointXYZ>::Ptr region)//return normal vectors
{
    pcl::PointXYZ o, pcaZ, pcaY, pcaX, c, pcX, pcY, pcZ;
    Eigen::Vector4f pcaCentroid;

    pcl::compute3DCentroid(*region, pcaCentroid);

    //cout << "MODEL PCA CENTROID: " << pcaCentroid << endl;
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*region, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    Eigen::Vector3f centroid;
    Eigen::Vector3f p_pi;
    Eigen::Vector3f p;
    centroid[0] = pcaCentroid(0);
    centroid[1] = pcaCentroid(1);
    centroid[2] = pcaCentroid(2);

    Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
    transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*region, *transformedCloud, transform);


    //std::cout << "Eigenvalue: \n" << eigenValuesPCA << std::endl;
    //std::cout << "Eigenvector: \n" << eigenVectorsPCA << std::endl;


    o.x = 0.0;
    o.y = 0.0;
    o.z = 0.0;
    Eigen::Affine3f tra_aff(transform);
    Eigen::Vector3f pz = eigenVectorsPCA.col(0);
    Eigen::Vector3f py = eigenVectorsPCA.col(1);
    Eigen::Vector3f px = eigenVectorsPCA.col(2);
    pcl::transformVector(pz, pz, tra_aff);
    pcl::transformVector(py, py, tra_aff);
    pcl::transformVector(px, px, tra_aff);
    int votex, votey{ 0 }, votey_{ 0 }, votez{ 0 }, votez_{ 0 };

    for (size_t i = 0; i < transformedCloud->size(); i++)
    {
        p_pi[0] = 0; p_pi[1] = 0; p_pi[2] = 0;
        p[0] = region->points[i].x;
        p[1] = region->points[i].y;
        p[2] = region->points[i].z;

        p_pi = p - centroid;
        if (p_pi.dot(pz) >= 0)
        {
            votez += 1;
        }
        else if (p_pi.dot(-pz) > 0)
        {
            votez_ += 1;
        }

        if (p_pi.dot(py) >= 0)
        {
            votey += 1;
        }
        else if (p_pi.dot(-py) > 0)
        {
            votey_ += 1;
        }


    }
    if (votez > votez_) {
        eigenVectorsPCA.col(0) = eigenVectorsPCA.col(0);
    }
    else
    {
        eigenVectorsPCA.col(0) = -eigenVectorsPCA.col(0);
    }
    if (votey > votey_) {
        eigenVectorsPCA.col(1) = eigenVectorsPCA.col(1);
    }
    else
    {
        eigenVectorsPCA.col(1) = -eigenVectorsPCA.col(1);
    }
    if (eigenVectorsPCA.col(0).z() < 0)
    {
        eigenVectorsPCA.col(0) = -eigenVectorsPCA.col(0);
    }
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

    pcaZ.x = 1000 * pz(0);
    pcaZ.y = 1000 * pz(1);
    pcaZ.z = 1000 * pz(2);

    pcaY.x = 1000 * py(0);
    pcaY.y = 1000 * py(1);
    pcaY.z = 1000 * py(2);

    pcaX.x = 1000 * px(0);
    pcaX.y = 1000 * px(1);
    pcaX.z = 1000 * px(2);

    c.x = pcaCentroid(0);
    c.y = pcaCentroid(1);
    c.z = pcaCentroid(2);


    pcZ.x = 10 * eigenVectorsPCA(0, 0) + c.x;//normal vector
    pcZ.y = 10 * eigenVectorsPCA(1, 0) + c.y;
    pcZ.z = 10 * eigenVectorsPCA(2, 0) + c.z;

    pcY.x = 5 * eigenVectorsPCA(0, 1) + c.x;
    pcY.y = 5 * eigenVectorsPCA(1, 1) + c.y;
    pcY.z = 5 * eigenVectorsPCA(2, 1) + c.z;

    pcX.x = 5 * eigenVectorsPCA(0, 2) + c.x;
    pcX.y = 5 * eigenVectorsPCA(1, 2) + c.y;
    pcX.z = 5 * eigenVectorsPCA(2, 2) + c.z;
    //pcl::visualization::PCLVisualizer viewer;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(region, 255, 0, 0);


    //viewer.addPointCloud(region, color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");


    ////viewer.addArrow(pcaZ, o, 0.0, 0.0, 1.0, false, "arrow_Z");
    ////viewer.addArrow(pcaY, o, 0.0, 1.0, 0.0, false, "arrow_Y");
    ////viewer.addArrow(pcaX, o, 1.0, 0.0, 0.0, false, "arrow_X");

    //viewer.addArrow(pcZ, c, 0.0, 0.0, 1.0, false, "arrow_z");
    //viewer.addArrow(pcY, c, 0.0, 1.0, 0.0, false, "arrow_y");
    //viewer.addArrow(pcX, c, 1.0, 0.0, 0.0, false, "arrow_x");



    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.spin();


    return eigenVectorsPCA;
}
void pca_match::find_parts_in_scene_alter()
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_r(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<double>nearest_point_list;


    for (int i = 0; i < 1; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());

        //transformation_cloud_list.clear();
        Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        pcl::compute3DCentroid(*regions_[i], cloud_center);
        double total_dist = 0;
        double total_dist_r = 0;
        for (int j = 0; j < model_regions.size(); j++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>);
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            
            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);


            // K nearest neighbor search
            //Evaluation
            int K = 1;


            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint2;
            pcl::PointXYZ searchPoint2_;

            transformation_cloud_list.push_back(transformed_final);
            transformation_cloud_list.push_back(rotatedCloud);
            for (int j = 0; j < model_temp->size(); j++)
            {

                searchPoint.x = transformed_final->points[j].x;
                searchPoint.y = transformed_final->points[j].y;
                searchPoint.z = transformed_final->points[j].z;

                searchPoint_r.x = rotatedCloud->points[j].x;
                searchPoint_r.y = rotatedCloud->points[j].y;
                searchPoint_r.z = rotatedCloud->points[j].z;


                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {
                    total_dist += pointKNNSquaredDistance[0];
                    if (pointKNNSquaredDistance[0] < 0.3)//0.3
                    {
                        nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    }
                }
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    total_dist_r += pointKNNSquaredDistance[0];
                    if (pointKNNSquaredDistance[0] < 0.3)//0.3
                    {
                        nearest_point_r->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    }
                }


            }
            total_dist /= model_temp->size();
            total_dist_r /= model_temp->size();
            cout << "total dist, r dist: " << total_dist << " , " << total_dist_r <<" , "<< nearest_point->size() <<" , "<< nearest_point_r->size() << endl;
            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);

            nearest_point_list.push_back(nearest_point->size());
            nearest_point_list.push_back(nearest_point_r->size());


            total_dist = 0;
            total_dist_r = 0;
            nearest_point->clear();
            nearest_point_r->clear();

            pcl::visualization::PCLVisualizer viewer;
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(rotatedCloud, 255, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(transformed_final, 0, 0, 255);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_point, 0, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[0], 0, 0, 255);
            viewer.addPointCloud(rotatedCloud, color_handler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            viewer.addPointCloud(cloud, color_handler2, "cloud2");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            viewer.addPointCloud(transformed_final, color_handler3, "cloud3");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            viewer.addPointCloud(nearest_point, color_handler4, "cloud4");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            viewer.addPointCloud(regions_[0], color_handler5, "cloud5");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            viewer.addCoordinateSystem(5);
            viewer.setBackgroundColor(1.0, 1.0, 1.0);
            viewer.spin();
            system("pause");

        }
        for (auto& i : dist_list)
        {
            cout << i << endl;
        }
        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        auto maxElement = std::max_element(dist_list.begin(), dist_list.end());

        int index = std::distance(dist_list.begin(), minElement);
        int indexM = std::distance(dist_list.begin(), maxElement);

        transformed_show = transformation_cloud_list[index];
        cout << "index: " << index << endl;
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];
        cout << "Dist, nearpoint: " << dist_list[index]/ dist_list[indexM] <<" , "<< 1-nearest_point_list[index]/16750 << endl;
        //near_list.push_back(nearest_point->size());
        //dist_list.clear();
        //nearest_point->clear();
        auto max = std::max_element(nearest_point_list.begin(), nearest_point_list.end());
        int index_near = std::distance(nearest_point_list.begin(), max);
        transformed_show2 = transformation_cloud_list[index_near];
        cout << "Dist2, nearpoint2: " << dist_list[index_near] / dist_list[indexM] << " , " << 1-nearest_point_list[index_near]/16750 << endl;
        
        pcl::visualization::PCLVisualizer viewer;
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(transformed_show, 0, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(transformed_show2, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[0], 0, 0, 255);

       /* viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");*/
        viewer.addPointCloud(cloud, color_handler2, "cloud2");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");
        viewer.addPointCloud(transformed_show, color_handler3, "cloud3");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud3");
        viewer.addPointCloud(transformed_show2, color_handler4, "cloud4");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
        viewer.addPointCloud(regions_[0], color_handler5, "cloud5");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
        viewer.addCoordinateSystem(5);
        viewer.setBackgroundColor(1.0, 1.0, 1.0);
        while (!viewer.wasStopped()) {
            viewer.spinOnce();
            //viewer.saveScreenshot(filename);
        }


        regions_.clear();
    }
    

}

void pca_match::find_parts_in_scene_rotate()
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;
  
    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());

    

    start2 = clock();


    for (int i = 0; i < 2; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);


            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(rotatedCloud, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(transformed_final, 0, 125, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_point, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[0], 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(rotatedCloud2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(rotatedCloud3, 255, 0, 255);

            //viewer.addPointCloud(rotatedCloud, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(transformed_final, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(nearest_point, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(regions_[0], color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addPointCloud(rotatedCloud2, color_handler6, "cloud6");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud6");
            //viewer.addPointCloud(rotatedCloud3, color_handler7, "cloud7");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud7");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //while (!viewer.wasStopped()) {
            //    viewer.spinOnce();
            //}
            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;


            for (int j = 0; j < transformed_final->size(); j++)
            {

                searchPoint.x = transformed_final->points[j].x;
                searchPoint.y = transformed_final->points[j].y;
                searchPoint.z = transformed_final->points[j].z;

                searchPoint_r.x = rotatedCloud->points[j].x;
                searchPoint_r.y = rotatedCloud->points[j].y;
                searchPoint_r.z = rotatedCloud->points[j].z;

                searchPoint_r2.x = rotatedCloud2->points[j].x;
                searchPoint_r2.y = rotatedCloud2->points[j].y;
                searchPoint_r2.z = rotatedCloud2->points[j].z;

                searchPoint_r3.x = rotatedCloud3->points[j].x;
                searchPoint_r3.y = rotatedCloud3->points[j].y;
                searchPoint_r3.z = rotatedCloud3->points[j].z;

                double constraint = 2000;
                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {
                    if (pointKNNSquaredDistance[0] < constraint)
                    {
                        total_dist += pointKNNSquaredDistance[0];
                        nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    }


                }
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (pointKNNSquaredDistance[0] < constraint)
                    {
                        total_dist_r += pointKNNSquaredDistance[0];
                        nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    }
                }
                if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (pointKNNSquaredDistance[0] < constraint)
                    {
                        total_dist_r2 += pointKNNSquaredDistance[0];
                        nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    }

                }
                if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (pointKNNSquaredDistance[0] < constraint)
                    {
                        total_dist_r3 += pointKNNSquaredDistance[0];
                        nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    }
                }

            }




            total_dist /= model_temp->size();
            total_dist_r /= model_temp->size();
            total_dist_r2 /= model_temp->size();
            total_dist_r3 /= model_temp->size();

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            transformation_cloud_list.push_back(transformed_final);
            transformation_cloud_list.push_back(rotatedCloud);
            transformation_cloud_list.push_back(rotatedCloud2);
            transformation_cloud_list.push_back(rotatedCloud3);

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            //cout << "Total dist: " << total_dist << endl;
            //cout << "Total dist rotation: " << total_dist_r << endl;
            //cout << "Total dist rotation2: " << total_dist_r2 << endl;
            //cout << "Total dist rotation3: " << total_dist_r3 << endl;


            //for (auto& i : dist_list)
            //{
            //    cout << " dist_list: " << i << endl;
            //}


        }

       

        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        //cout << "index: " << index << endl;
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];

        //rot2ang(transformation_matrix_list[index]);
        int K = 1;
        double total_dist = 0;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);
        pcl::PointXYZ searchPoint2;
        pcl::PointXYZ searchPoint2_;


        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < transformed_show->size(); j++)
        {
            searchPoint2.x = transformed_show->points[j].x;
            searchPoint2.y = transformed_show->points[j].y;
            searchPoint2.z = transformed_show->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.3)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }


        near_list.push_back(nearest_point->size());
        dist_list.clear();
        nearest_point->clear();
        
    }
    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);
    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_near] << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped()) 
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();

}



std::vector<Eigen::Vector3f> region_pca(pcl::PointCloud<pcl::PointXYZ>::Ptr region)//return normal vectors
{
    pcl::PointXYZ o, pcaZ, pcaY, pcaX, c, pcX, pcY, pcZ;
    Eigen::Vector4f pcaCentroid;

    pcl::compute3DCentroid(*region, pcaCentroid);

    //cout << "MODEL PCA CENTROID: " << pcaCentroid << endl;
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*region, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
    Eigen::Vector3f centroid;
    Eigen::Vector3f p_pi;
    Eigen::Vector3f p;
    centroid[0] = pcaCentroid(0);
    centroid[1] = pcaCentroid(1);
    centroid[2] = pcaCentroid(2);

    Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
    transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*region, *transformedCloud, transform);


    //std::cout << "Eigenvalue: \n" << eigenValuesPCA << std::endl;
    //std::cout << "Eigenvector: \n" << eigenVectorsPCA << std::endl;


    o.x = 0.0;
    o.y = 0.0;
    o.z = 0.0;
    Eigen::Affine3f tra_aff(transform);
    Eigen::Vector3f pz = eigenVectorsPCA.col(0);
    Eigen::Vector3f py = eigenVectorsPCA.col(1);
    Eigen::Vector3f px = eigenVectorsPCA.col(2);
    pcl::transformVector(pz, pz, tra_aff);
    pcl::transformVector(py, py, tra_aff);
    pcl::transformVector(px, px, tra_aff);
    int votex, votey{ 0 }, votey_{ 0 }, votez{ 0 }, votez_{ 0 };

    for (size_t i = 0; i < transformedCloud->size(); i++)
    {
        p_pi[0] = 0; p_pi[1] = 0; p_pi[2] = 0;
        p[0] = region->points[i].x;
        p[1] = region->points[i].y;
        p[2] = region->points[i].z;

        p_pi = p - centroid;
        if (p_pi.dot(pz) >= 0)
        {
            votez += 1;
        }
        else if (p_pi.dot(-pz) > 0)
        {
            votez_ += 1;
        }

        if (p_pi.dot(py) >= 0)
        {
            votey += 1;
        }
        else if (p_pi.dot(-py) > 0)
        {
            votey_ += 1;
        }


    }
    if (votez > votez_) {
        eigenVectorsPCA.col(0) = eigenVectorsPCA.col(0);
    }
    else
    {
        eigenVectorsPCA.col(0) = -eigenVectorsPCA.col(0);
    }
    if (votey > votey_) {
        eigenVectorsPCA.col(1) = eigenVectorsPCA.col(1);
    }
    else
    {
        eigenVectorsPCA.col(1) = -eigenVectorsPCA.col(1);
    }

    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    transform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    transform.block<3, 1>(0, 3) = -1.0f * (transform.block<3, 3>(0, 0)) * (pcaCentroid.head<3>());
    if (eigenVectorsPCA(0,2) < 0)
    {
        eigenVectorsPCA(0, 2) = -eigenVectorsPCA(0, 2);
    }
    pcaZ.x = 1000 * pz(0);
    pcaZ.y = 1000 * pz(1);
    pcaZ.z = 1000 * pz(2);

    pcaY.x = 1000 * py(0);
    pcaY.y = 1000 * py(1);
    pcaY.z = 1000 * py(2);

    pcaX.x = 1000 * px(0);
    pcaX.y = 1000 * px(1);
    pcaX.z = 1000 * px(2);

    c.x = pcaCentroid(0);
    c.y = pcaCentroid(1);
    c.z = pcaCentroid(2);


    pcZ.x = 10 * eigenVectorsPCA(0, 0) + c.x;//normal vector
    pcZ.y = 10 * eigenVectorsPCA(1, 0) + c.y;
    pcZ.z = 10 * eigenVectorsPCA(2, 0) + c.z;

    pcY.x = 5 * eigenVectorsPCA(0, 1) + c.x;
    pcY.y = 5 * eigenVectorsPCA(1, 1) + c.y;
    pcY.z = 5 * eigenVectorsPCA(2, 1) + c.z;

    pcX.x = 5 * eigenVectorsPCA(0, 2) + c.x;
    pcX.y = 5 * eigenVectorsPCA(1, 2) + c.y;
    pcX.z = 5 * eigenVectorsPCA(2, 2) + c.z;

    std::vector<Eigen::Vector3f> pca;
    pca.push_back(eigenVectorsPCA.col(0));
    pca.push_back(eigenVectorsPCA.col(1));
    pca.push_back(eigenVectorsPCA.col(2));

    //pcl::visualization::PCLVisualizer viewer;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(region, 255, 0, 0);


    //viewer.addPointCloud(region, color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");


    ////viewer.addArrow(pcaZ, o, 0.0, 0.0, 1.0, false, "arrow_Z");
    ////viewer.addArrow(pcaY, o, 0.0, 1.0, 0.0, false, "arrow_Y");
    ////viewer.addArrow(pcaX, o, 1.0, 0.0, 0.0, false, "arrow_X");

    //viewer.addArrow(pcZ, c, 0.0, 0.0, 1.0, false, "arrow_z");
    //viewer.addArrow(pcY, c, 0.0, 1.0, 0.0, false, "arrow_y");
    //viewer.addArrow(pcX, c, 1.0, 0.0, 0.0, false, "arrow_x");



    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.spin();


    return pca;
}
void pca_match::pca_simplex() 
{
    Eigen::Vector3f init;
    init << 5, 5, 5;
    
    
    



    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;
    //double min_dist = 100;
    //double min_dist2 = 200;
    //double min_dist3 = 300;

    //for (int i = 0; i < regions_.size(); i++)
    //{
    //    Eigen::Vector4f centroid;
    //    pcl::compute3DCentroid(*regions_[i], centroid);
    //    center_list.push_back(centroid);
    //}

    /*for (int k =  1; k < regions_.size(); k++)
    {
        Eigen::Vector4f c1 = center_list[0];
        Eigen::Vector4f c2 = center_list[k];
        double dist = (c1 - c2).norm();
        if (dist < min_dist)
        {
            min_dist3 = min_dist2;
            adjacent3 = adjacent2;

            min_dist2 = min_dist;
            adjacent2 = adjacent;

            min_dist = dist;
            adjacent = regions_[k];
        }
        else if (dist < min_dist2)
        {
            min_dist3 = min_dist2;
            adjacent3 = adjacent2;

            min_dist2 = dist;
            adjacent2 = regions_[k];
        }
        else if (dist < min_dist3)
        {
            min_dist3 = dist;
            adjacent3 = regions_[k];
        }
    }*/
    /*pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(adjacent, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(adjacent2, 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(adjacent2, 255, 0, 255);


    viewer.addPointCloud(adjacent, color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud3");
    viewer.addPointCloud(adjacent2, color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(adjacent3, color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    viewer.spin();*/
    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point2(new pcl::PointCloud<pcl::PointXYZ>());



    for (int i = 0; i < 3; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        std::vector<Eigen::Vector3f>scene_pca = region_pca(regions_[i]);
        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);
            //cout << "transform cloud size:　" << transformed_cloud->size() << endl;
            std::vector<Eigen::Vector3f>model_pca = region_pca(transformed_cloud);
            Eigen::Vector3f uvw = pca_match::simplex_method(init, model_pca[0], model_pca[1], model_pca[2], scene_pca[0], scene_pca[1], scene_pca[2]);
            Eigen::Matrix3f yaw = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f pitch = Eigen::Matrix3f::Identity();
            Eigen::Matrix3f roll = Eigen::Matrix3f::Identity();
            double u = uvw(0); double v = uvw(1); double w = uvw(2);
            yaw(0, 0) = cosf(u); yaw(0, 1) = -sinf(u); yaw(1, 0) = sinf(u); yaw(1, 1) = cosf(u);
            pitch(0, 0) = cosf(v); pitch(0, 2) = sinf(v); pitch(2, 0) = -sinf(v); pitch(2, 2) = cosf(v);
            roll(1, 1) = cosf(w); roll(1, 2) = -sinf(w); roll(2, 1) = sinf(w); roll(2, 2) = cosf(w);
            Eigen::Matrix3f R;
            R = yaw * pitch * roll;
            pcl::compute3DCentroid(*transformed_cloud, temp_center);
            Eigen::Matrix4f transformation=Eigen::Matrix4f::Identity();
            transformation.block<3, 3>(0, 0) = R;
            transformation.block<3, 1>(0, 3) = cloud_center.block<3, 1>(0, 0) - R * temp_center.block<3, 1>(0, 0);
            
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);

            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);

            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(model, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(transformed_final, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_point, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);
            //viewer.addPointCloud(model, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(transformed_final, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            ////viewer.addPointCloud(nearest_point, color_handler4, "cloud4");
            ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            ////viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
            ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //while (!viewer.wasStopped()) {
            //    viewer.spinOnce();
            //}
            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;

            for (int j = 0; j < model_temp->size(); j++)
            {

                searchPoint.x = transformed_final->points[j].x;
                searchPoint.y = transformed_final->points[j].y;
                searchPoint.z = transformed_final->points[j].z;

                searchPoint_r.x = rotatedCloud->points[j].x;
                searchPoint_r.y = rotatedCloud->points[j].y;
                searchPoint_r.z = rotatedCloud->points[j].z;


                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {
                    total_dist += pointKNNSquaredDistance[0];
                }
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    total_dist_r += pointKNNSquaredDistance[0];
                }

            }


            total_dist /= model_temp->size();
            total_dist_r /= model_temp->size();

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);

            transformation_cloud_list.push_back(transformed_final);
            transformation_cloud_list.push_back(rotatedCloud);

            transformation_matrix_list.push_back(transformation);

            //cout << "Total dist: " << total_dist << endl;


        }

        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];

        //rot2ang(transformation_matrix_list[index]);
        int K = 1;
        double total_dist = 0;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);
        pcl::PointXYZ searchPoint2;
        pcl::PointXYZ searchPoint2_;


        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < transformed_show->size(); j++)
        {
            searchPoint2.x = transformed_show->points[j].x;
            searchPoint2.y = transformed_show->points[j].y;
            searchPoint2.z = transformed_show->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.3)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }


        near_list.push_back(nearest_point->size());
        dist_list.clear();
        nearest_point->clear();
    }
    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);


    cout << "Nearest points size: " << near_list[index_near] << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_point, 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_point, color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();

}


struct dist_and_point {

    double dist;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point;
}dp;

double evaluation(pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_final_, pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list,pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pass_through_)
{
    int K = 1;

    double dist= 0;
    pcl::PointXYZ searchPoint;
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> pointIdxKNNSearch(K);
    std::vector<float> pointKNNSquaredDistance(K);
    kdtree.setInputCloud(cloud_pass_through_);
    double constraint = 2000;

    for (int j = 0; j < transformed_final_->size(); j++)
    {

        searchPoint.x = transformed_final_->points[j].x;
        searchPoint.y = transformed_final_->points[j].y;
        searchPoint.z = transformed_final_->points[j].z;



        if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
        {
            
            if (pointKNNSquaredDistance[0] < constraint)
            {
                dist += pointKNNSquaredDistance[0];
                nearest_point_list->push_back((*cloud_pass_through_)[pointIdxKNNSearch[0]]);
            }
        }

    }
    return dist;
}



void pca_match::find_parts_in_scene_openmp()
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());


    start = clock();


    
    for (int i = 0; i < 2; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);

            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            //int K = 1;
            //double total_dist = 0;
            //double total_dist_r = 0;
            //double total_dist_r2 = 0; double total_dist_r3 = 0;

            //pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            //std::vector<int> pointIdxKNNSearch(K);
            //std::vector<float> pointKNNSquaredDistance(K);

 /*           kdtree.setInputCloud(cloud_pass_through);*/
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;
            double constraint = 2000;
            int id;
            //omp_set_num_threads(4);
            //omp_set_nested(1);

#pragma omp parallel sections 
                {
#pragma omp section
                    {

                        double total_dist = evaluation(transformed_final, nearest_point_list, cloud_pass_through);

                        total_dist /= model_temp->size();
                        //cout << "Total dist: " << total_dist << endl;
                        //printf("Section1 T:%d\n", omp_get_thread_num());
                        dist_list.push_back(total_dist);
                        transformation_cloud_list.push_back(transformed_final);
                        //id = omp_get_thread_num();
                        //printf("<T:%d> ", id);
                    }

#pragma omp section
                    {
                        Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
                        rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
                        rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
                        rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));

                        pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
                        double total_dist_r = evaluation(rotatedCloud, nearest_point_list_r1, cloud_pass_through);

                        total_dist_r /= model_temp->size();
                        dist_list.push_back(total_dist_r);
                        transformation_cloud_list.push_back(rotatedCloud);
                        //printf("Section2 T:%d\n", omp_get_thread_num());
                        //id = omp_get_thread_num();
                        //printf("<T:%d> ", id);
                    }

#pragma omp section
                    {
                        Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
                        rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
                        rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
                        rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));

                        pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
                        double total_dist_r2 = evaluation(rotatedCloud2, nearest_point_list_r2, cloud_pass_through);

                        total_dist_r2 /= model_temp->size();
                        dist_list.push_back(total_dist_r2);
                        transformation_cloud_list.push_back(rotatedCloud2);
                        //printf("Section3 T:%d\n", omp_get_thread_num());
                        //id = omp_get_thread_num();
                        //printf("<T:%d> ", id);
                    }
#pragma omp section
                    {
                        Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
                        rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
                        rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
                        rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));

                        pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);
                        double total_dist_r3 = evaluation(rotatedCloud3, nearest_point_list_r3, cloud_pass_through);

                        total_dist_r3 /= model_temp->size();
                        dist_list.push_back(total_dist_r3);
                        transformation_cloud_list.push_back(rotatedCloud3);
                        //printf("Section4 T:%d\n", omp_get_thread_num());
                        //id = omp_get_thread_num();
                        //printf("<T:%d> ", id);
                    }

                }






            // K nearest neighbor search
            //Evaluation


            //for (auto& i : dist_list)
            //{
            //    cout << " dist_list: " << i << endl;
            //}










            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            //cout << "Total dist: " << total_dist << endl;
            //cout << "Total dist rotation: " << total_dist_r << endl;
            //cout << "Total dist rotation2: " << total_dist_r2 << endl;
            //cout << "Total dist rotation3: " << total_dist_r3 << endl;

        }





        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        //cout << "index: " << index << endl;
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];

        //rot2ang(transformation_matrix_list[index]);
        int K = 1;
        double total_dist = 0;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);
        pcl::PointXYZ searchPoint2;
        pcl::PointXYZ searchPoint2_;


        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < transformed_show->size(); j++)
        {
            searchPoint2.x = transformed_show->points[j].x;
            searchPoint2.y = transformed_show->points[j].y;
            searchPoint2.z = transformed_show->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.3)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }


        near_list.push_back(nearest_point->size());
        dist_list.clear();
        nearest_point->clear();

    }
    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);

    end = clock();
    cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);
    cout << "Nearest points size: " << near_list[index_near] << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();



}

void rmse(pcl::PointXYZ searchPoint, pcl::PointXYZ targets)
{
    double x_err = 0; double y_err = 0; double z_err = 0;
    x_err += (searchPoint.x - targets.x) * (searchPoint.x - targets.x);
    y_err += (searchPoint.y - targets.y) * (searchPoint.y - targets.y);
    z_err += (searchPoint.z - targets.z) * (searchPoint.z - targets.z);

}



void pca_match::find_parts_in_scene_rotate_RMSE()
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());



    start2 = clock();


    for (int i = 0; i < 2; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);

            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;

            double x_err = 0;
            double y_err = 0;
            double z_err = 0;

            double x_err_r1 = 0;
            double y_err_r1 = 0;
            double z_err_r1 = 0;

            double x_err_r2 = 0;
            double y_err_r2 = 0;
            double z_err_r2 = 0;

            double x_err_r3 = 0;
            double y_err_r3 = 0;
            double z_err_r3 = 0;


            for (int j = 0; j < transformed_final->size(); j++)
            {

                searchPoint.x = transformed_final->points[j].x;
                searchPoint.y = transformed_final->points[j].y;
                searchPoint.z = transformed_final->points[j].z;

                searchPoint_r.x = rotatedCloud->points[j].x;
                searchPoint_r.y = rotatedCloud->points[j].y;
                searchPoint_r.z = rotatedCloud->points[j].z;

                searchPoint_r2.x = rotatedCloud2->points[j].x;
                searchPoint_r2.y = rotatedCloud2->points[j].y;
                searchPoint_r2.z = rotatedCloud2->points[j].z;

                searchPoint_r3.x = rotatedCloud3->points[j].x;
                searchPoint_r3.y = rotatedCloud3->points[j].y;
                searchPoint_r3.z = rotatedCloud3->points[j].z;

                double constraint = 1000;
                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {
                    
                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist += pointKNNSquaredDistance[0];
                        nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE 1
                        //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                        x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }



                }
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r += pointKNNSquaredDistance[0];
                        nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r1
                        
                        x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }

                }
                if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r2 += pointKNNSquaredDistance[0];
                        nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r2
                        x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }

                }
                if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r3 += pointKNNSquaredDistance[0];
                        nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r3
                        x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }

                }

            }


            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();
            //x_err = sqrt(x_err); x_err_r1 = sqrt(x_err_r1); x_err_r2 = sqrt(x_err_r2); x_err_r3 = sqrt(x_err_r3);
            //y_err = sqrt(y_err); y_err_r1 = sqrt(y_err_r1); y_err_r2 = sqrt(y_err_r2); y_err_r3 = sqrt(y_err_r3);
            //z_err = sqrt(z_err); z_err_r1 = sqrt(z_err_r1); z_err_r2 = sqrt(z_err_r2); z_err_r3 = sqrt(z_err_r3);

            cout << "total_err: " << x_err+ y_err+ z_err << endl;
            cout << "total_err r1: " << x_err_r1 + y_err_r1 + z_err_r1 << endl; 
            cout << "total_err r2: " << x_err_r2 + y_err_r2 + z_err_r2 << endl;
            cout << "total_err r3: " << x_err_r3 + y_err_r3 + z_err_r3 << endl;

            total_dist = sqrt(x_err* x_err + y_err* y_err + z_err* z_err);//+ y_err + z_err
            total_dist_r = sqrt(x_err_r1* x_err_r1 + y_err_r1* y_err_r1 + z_err_r1* z_err_r1);//+ y_err_r1 + z_err_r1
            total_dist_r2 = sqrt(x_err_r2* x_err_r2 + y_err_r2* y_err_r2 + z_err_r2* z_err_r2);// + y_err_r2 + z_err_r2
            total_dist_r3 = sqrt(x_err_r3* x_err_r3 + y_err_r3* y_err_r3 + z_err_r3* z_err_r3);// + y_err_r3 + z_err_r3

            //total_dist = sqrt(total_dist / model_temp->size());
            //total_dist_r = sqrt(total_dist_r / model_temp->size());
            //total_dist_r2 = sqrt(total_dist_r2 / model_temp->size());
            //total_dist_r3 = sqrt(total_dist_r3 / model_temp->size());

            //total_dist /= model_temp->size();
            //total_dist_r /= model_temp->size();
            //total_dist_r2 /= model_temp->size();
            //total_dist_r3 /= model_temp->size();

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            transformation_cloud_list.push_back(transformed_final);
            transformation_cloud_list.push_back(rotatedCloud);
            transformation_cloud_list.push_back(rotatedCloud2);
            transformation_cloud_list.push_back(rotatedCloud3);

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            //cout << "Total dist: " << total_dist << endl;
            //cout << "Total dist rotation: " << total_dist_r << endl;
            //cout << "Total dist rotation2: " << total_dist_r2 << endl;
            //cout << "Total dist rotation3: " << total_dist_r3 << endl;


            //for (auto& i : dist_list)
            //{
            //    cout << " dist_list: " << i << endl;
            //}


        }



        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        cout << "index: " << dist_list[index] << endl;
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];

        //rot2ang(transformation_matrix_list[index]);
        int K = 1;
        double total_dist = 0;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);
        pcl::PointXYZ searchPoint2;
        pcl::PointXYZ searchPoint2_;


        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < transformed_show->size(); j++)
        {
            searchPoint2.x = transformed_show->points[j].x;
            searchPoint2.y = transformed_show->points[j].y;
            searchPoint2.z = transformed_show->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.3)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }


        near_list.push_back(nearest_point->size());
        dist_list.clear();
        nearest_point->clear();

    }
    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);
    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_near] << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();
}

void pca_match::find_parts_in_scene_rotate_RMSE2()
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());



    start2 = clock();


    for (int i = 0; i < 2; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);

            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;

            double x_err = 0;
            double y_err = 0;
            double z_err = 0;

            double x_err_r1 = 0;
            double y_err_r1 = 0;
            double z_err_r1 = 0;

            double x_err_r2 = 0;
            double y_err_r2 = 0;
            double z_err_r2 = 0;

            double x_err_r3 = 0;
            double y_err_r3 = 0;
            double z_err_r3 = 0;


            for (int j = 0; j < transformed_final->size(); j++)
            {

                searchPoint.x = transformed_final->points[j].x;
                searchPoint.y = transformed_final->points[j].y;
                searchPoint.z = transformed_final->points[j].z;

                searchPoint_r.x = rotatedCloud->points[j].x;
                searchPoint_r.y = rotatedCloud->points[j].y;
                searchPoint_r.z = rotatedCloud->points[j].z;

                searchPoint_r2.x = rotatedCloud2->points[j].x;
                searchPoint_r2.y = rotatedCloud2->points[j].y;
                searchPoint_r2.z = rotatedCloud2->points[j].z;

                searchPoint_r3.x = rotatedCloud3->points[j].x;
                searchPoint_r3.y = rotatedCloud3->points[j].y;
                searchPoint_r3.z = rotatedCloud3->points[j].z;

                double constraint = 1000;
                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist += pointKNNSquaredDistance[0];
                        nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE 1
                        //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                        x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }



                }
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r += pointKNNSquaredDistance[0];
                        nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r1

                        x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }

                }
                if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r2 += pointKNNSquaredDistance[0];
                        nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r2
                        x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }

                }
                if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r3 += pointKNNSquaredDistance[0];
                        nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r3
                        x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                    }

                }

            }


            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();
            //x_err = sqrt(x_err); x_err_r1 = sqrt(x_err_r1); x_err_r2 = sqrt(x_err_r2); x_err_r3 = sqrt(x_err_r3);
            //y_err = sqrt(y_err); y_err_r1 = sqrt(y_err_r1); y_err_r2 = sqrt(y_err_r2); y_err_r3 = sqrt(y_err_r3);
            //z_err = sqrt(z_err); z_err_r1 = sqrt(z_err_r1); z_err_r2 = sqrt(z_err_r2); z_err_r3 = sqrt(z_err_r3);

            cout << "total_err: " << x_err + y_err + z_err << endl;
            cout << "total_err r1: " << x_err_r1 + y_err_r1 + z_err_r1 << endl;
            cout << "total_err r2: " << x_err_r2 + y_err_r2 + z_err_r2 << endl;
            cout << "total_err r3: " << x_err_r3 + y_err_r3 + z_err_r3 << endl;

            total_dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);//+ y_err + z_err
            total_dist_r = sqrt(x_err_r1 * x_err_r1 + y_err_r1 * y_err_r1 + z_err_r1 * z_err_r1);//+ y_err_r1 + z_err_r1
            total_dist_r2 = sqrt(x_err_r2 * x_err_r2 + y_err_r2 * y_err_r2 + z_err_r2 * z_err_r2);// + y_err_r2 + z_err_r2
            total_dist_r3 = sqrt(x_err_r3 * x_err_r3 + y_err_r3 * y_err_r3 + z_err_r3 * z_err_r3);// + y_err_r3 + z_err_r3

            //total_dist = sqrt(total_dist / model_temp->size());
            //total_dist_r = sqrt(total_dist_r / model_temp->size());
            //total_dist_r2 = sqrt(total_dist_r2 / model_temp->size());
            //total_dist_r3 = sqrt(total_dist_r3 / model_temp->size());

            //total_dist /= model_temp->size();
            //total_dist_r /= model_temp->size();
            //total_dist_r2 /= model_temp->size();
            //total_dist_r3 /= model_temp->size();

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            transformation_cloud_list.push_back(transformed_final);
            transformation_cloud_list.push_back(rotatedCloud);
            transformation_cloud_list.push_back(rotatedCloud2);
            transformation_cloud_list.push_back(rotatedCloud3);

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            //cout << "Total dist: " << total_dist << endl;
            //cout << "Total dist rotation: " << total_dist_r << endl;
            //cout << "Total dist rotation2: " << total_dist_r2 << endl;
            //cout << "Total dist rotation3: " << total_dist_r3 << endl;


            //for (auto& i : dist_list)
            //{
            //    cout << " dist_list: " << i << endl;
            //}


        }



        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        cout << "index: " << dist_list[index] << endl;
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);
        //transformed_show2 = transformation_cloud_list[index];

        //rot2ang(transformation_matrix_list[index]);
        int K = 1;
        double total_dist = 0;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);
        pcl::PointXYZ searchPoint2;
        pcl::PointXYZ searchPoint2_;


        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < transformed_show->size(); j++)
        {
            searchPoint2.x = transformed_show->points[j].x;
            searchPoint2.y = transformed_show->points[j].y;
            searchPoint2.z = transformed_show->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.3)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }


        near_list.push_back(nearest_point->size());
        dist_list.clear();
        nearest_point->clear();

    }
    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);
    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_near] << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();
}
void pca_match::find_parts_in_scene_rotate_RMSE2_hull()
{
    pcl::visualization::PCLVisualizer viewer;
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_hull(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point(new pcl::PointCloud<pcl::PointXYZ>()), hull(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point2(new pcl::PointCloud<pcl::PointXYZ>()), hull2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point3(new pcl::PointCloud<pcl::PointXYZ>()), hull3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point4(new pcl::PointCloud<pcl::PointXYZ>()), hull4(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector3d transformed_cam, transformed_cam2, transformed_cam3, transformed_cam4;
    pcl::PointXYZ cam;
    start2 = clock();


    for (int i = 0; i < 2; i++)//regions_.size()
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);
        
        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {
            /*Original Camera Location*/
            if (j == 0)
            {
                camera_location1[0] = model_diameter;
                camera_location1[1] = 0;
                camera_location1[2] = 0;
                cam.x = camera_location1[0]; cam.y = camera_location1[1]; cam.z = camera_location1[2];
                camera_point->push_back(cam);
                //original_hull = pca_match::hull(camera_location1);
                //cout << "cam: " << cam << endl;
            }
            else if (j == 1)
            {
                camera_location1[0] = 0;
                camera_location1[1] = 0;
                camera_location1[2] = -model_diameter;
                cam.x = camera_location1[0]; cam.y = camera_location1[1]; cam.z = camera_location1[2];
                camera_point->push_back(cam);
            }
            else if (j == 2)
            {
                camera_location1[0] = -model_diameter;
                camera_location1[1] = 0;
                camera_location1[2] = 0;
                cam.x = camera_location1[0]; cam.y = camera_location1[1]; cam.z = camera_location1[2];
                camera_point->push_back(cam);
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*cloud_hull_, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            Eigen::Matrix4f transformation_r1, transformation_r2, transformation_r3;
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedhull(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull2(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull3(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull4(new pcl::PointCloud<pcl::PointXYZ>);

            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);

            /*Rotate Camera*/
            //1.normal aligned transformation
            pcl::transformPointCloud(*camera_point, *camera_point, transform_matrix);
            //2.ICP transformation 
            pcl::transformPointCloud(*camera_point, *camera_point, transformation);
            pcl::transformPointCloud(*camera_point, *camera_point2, transformation);
            pcl::transformPointCloud(*camera_point, *camera_point3, transformation);
            pcl::transformPointCloud(*camera_point, *camera_point4, transformation);
            //3.rotation
            pcl::transformPointCloud(*camera_point2, *camera_point2, rotation);
            pcl::transformPointCloud(*camera_point3, *camera_point3, rotation2);
            pcl::transformPointCloud(*camera_point4, *camera_point4, rotation3);

            transformed_cam.x() = camera_point->points[0].x;
            transformed_cam.y() = camera_point->points[0].y;
            transformed_cam.z() = camera_point->points[0].z;
            hull = pca_match::hull(transformed_cam);

            transformed_cam2.x() = camera_point2->points[0].x;
            transformed_cam2.y() = camera_point2->points[0].y;
            transformed_cam2.z() = camera_point2->points[0].z;
            hull2 = pca_match::hull(transformed_cam2);

            transformed_cam3.x() = camera_point3->points[0].x;
            transformed_cam3.y() = camera_point3->points[0].y;
            transformed_cam3.z() = camera_point3->points[0].z;
            hull3 = pca_match::hull(transformed_cam3);

            transformed_cam4.x() = camera_point4->points[0].x;
            transformed_cam4.y() = camera_point4->points[0].y;
            transformed_cam4.z() = camera_point4->points[0].z;
            hull4 = pca_match::hull(transformed_cam4);

            /*Rotate Hull*/
            //1.normal aligned transformation
            pcl::transformPointCloud(*hull, *rotatedhull, transform_matrix);
            pcl::transformPointCloud(*hull2, *rotatedhull2, transform_matrix);
            pcl::transformPointCloud(*hull3, *rotatedhull3, transform_matrix);
            pcl::transformPointCloud(*hull4, *rotatedhull4, transform_matrix);
            //2.ICP transformation
            pcl::transformPointCloud(*rotatedhull, *rotatedhull, transformation);
            pcl::transformPointCloud(*rotatedhull2, *rotatedhull2, transformation);
            pcl::transformPointCloud(*rotatedhull3, *rotatedhull3, transformation);
            pcl::transformPointCloud(*rotatedhull4, *rotatedhull4, transformation);
            //3.Rotation
            pcl::transformPointCloud(*rotatedhull2, *rotatedhull2, rotation);
            pcl::transformPointCloud(*rotatedhull3, *rotatedhull3, rotation2);
            pcl::transformPointCloud(*rotatedhull4, *rotatedhull4, rotation3);



            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(rotatedhull, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(rotatedhull2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(rotatedhull3, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(rotatedhull4, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(camera_point, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(camera_point2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler8(camera_point3, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler9(camera_point4, 0, 0, 255);




            //viewer.addPointCloud(rotatedhull, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(rotatedhull2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(rotatedhull3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(rotatedhull4, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addPointCloud(camera_point, color_handler6, "cloud6");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud6");
            //viewer.addPointCloud(camera_point2, color_handler7, "cloud7");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud7");
            //viewer.addPointCloud(camera_point3, color_handler8, "cloud8");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud8");
            //viewer.addPointCloud(camera_point4, color_handler9, "cloud9");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud9");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");





            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;

            double x_err = 0;
            double y_err = 0;
            double z_err = 0;

            double x_err_r1 = 0;
            double y_err_r1 = 0;
            double z_err_r1 = 0;

            double x_err_r2 = 0;
            double y_err_r2 = 0;
            double z_err_r2 = 0;

            double x_err_r3 = 0;
            double y_err_r3 = 0;
            double z_err_r3 = 0;


            double constraint = 10;
            int x = 0;
            for (int j = 0; j < rotatedhull->size(); j++)
            {
                
                //use hull to find nearest point
                searchPoint.x = rotatedhull->points[j].x;
                searchPoint.y = rotatedhull->points[j].y;
                searchPoint.z = rotatedhull->points[j].z;
                if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist += pointKNNSquaredDistance[0];
                        nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE 1
                        //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                        x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                        x += 1;
                    }



                }
            }
            cout << "nearest point count: " << x << endl;
            int y = 0;
            for (int i = 0; i < rotatedhull2->size(); i++)
            {
                searchPoint_r.x = rotatedhull2->points[i].x;
                searchPoint_r.y = rotatedhull2->points[i].y;
                searchPoint_r.z = rotatedhull2->points[i].z;
                if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r += pointKNNSquaredDistance[0];
                        nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r1

                        x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                        y += 1;
                    }

                }
            }
            cout << "nearest point count 2 : " << y << endl;
            int z = 0;
            for (int k = 0; k < rotatedhull3->size(); k++)
            {
                searchPoint_r2.x = rotatedhull3->points[k].x;
                searchPoint_r2.y = rotatedhull3->points[k].y;
                searchPoint_r2.z = rotatedhull3->points[k].z;
                if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r2 += pointKNNSquaredDistance[0];
                        nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r2
                        x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                        z += 1;
                    }

                }
            }
            cout << "nearest point count 3 : " << z << endl;
            int a = 0;
            for (int h = 0; h < rotatedhull4->size(); h++)
            {
                searchPoint_r3.x = rotatedhull4->points[h].x;
                searchPoint_r3.y = rotatedhull4->points[h].y;
                searchPoint_r3.z = rotatedhull4->points[h].z;

                if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                {

                    if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                    {
                        //total_dist_r3 += pointKNNSquaredDistance[0];
                        nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                        //RMSE r3
                        x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                        y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                        z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                        a += 1;
                    }

                }
            }
            cout << "nearest point count 4 : " << a << endl;
                       



               




            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();
            //x_err = sqrt(x_err); x_err_r1 = sqrt(x_err_r1); x_err_r2 = sqrt(x_err_r2); x_err_r3 = sqrt(x_err_r3);
            //y_err = sqrt(y_err); y_err_r1 = sqrt(y_err_r1); y_err_r2 = sqrt(y_err_r2); y_err_r3 = sqrt(y_err_r3);
            //z_err = sqrt(z_err); z_err_r1 = sqrt(z_err_r1); z_err_r2 = sqrt(z_err_r2); z_err_r3 = sqrt(z_err_r3);

            //cout << "total_err: " << x_err + y_err + z_err << endl;
            //cout << "total_err r1: " << x_err_r1 + y_err_r1 + z_err_r1 << endl;
            //cout << "total_err r2: " << x_err_r2 + y_err_r2 + z_err_r2 << endl;
            //cout << "total_err r3: " << x_err_r3 + y_err_r3 + z_err_r3 << endl;

            total_dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);//+ y_err + z_err
            total_dist_r = sqrt(x_err_r1 * x_err_r1 + y_err_r1 * y_err_r1 + z_err_r1 * z_err_r1);//+ y_err_r1 + z_err_r1
            total_dist_r2 = sqrt(x_err_r2 * x_err_r2 + y_err_r2 * y_err_r2 + z_err_r2 * z_err_r2);// + y_err_r2 + z_err_r2
            total_dist_r3 = sqrt(x_err_r3 * x_err_r3 + y_err_r3 * y_err_r3 + z_err_r3 * z_err_r3);// + y_err_r3 + z_err_r3

            cout << "total_err: " << total_dist << endl;
            cout << "total_err r1: " << total_dist_r << endl;
            cout << "total_err r2: " << total_dist_r2 << endl;
            cout << "total_err r3: " << total_dist_r3 << endl;

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            transformation_cloud_list.push_back(rotatedhull);
            transformation_cloud_list.push_back(rotatedhull2);
            transformation_cloud_list.push_back(rotatedhull3);
            transformation_cloud_list.push_back(rotatedhull4);

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            pcl::visualization::PCLVisualizer viewer;
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_final, 0, 0, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(rotatedCloud, 0, 255, 0);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(rotatedCloud2, 0, 0, 255);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(rotatedCloud3, 0, 255, 255);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(cloud, 255, 0, 255);

            viewer.addPointCloud(transformed_final, color_handler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(rotatedCloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(rotatedCloud2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(rotatedCloud3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            viewer.addPointCloud(cloud, color_handler5, "cloud5");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            viewer.addCoordinateSystem(5);
            viewer.setBackgroundColor(1.0, 1.0, 1.0);
            viewer.spin();
            system("pause");


        }



        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        cout << "smallest dist: " << dist_list[index]  << endl;
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);

        //transformed_show2 = transformation_cloud_list[index];


        //pcl::visualization::PCLVisualizer viewer;
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_show, 255, 0, 0);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
        //
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

        //viewer.addPointCloud(transformed_show, color_handler, "cloud");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
        //viewer.addPointCloud(cloud, color_handler2, "cloud2");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
        //viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
        //
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
        ////viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
        ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
        //viewer.addCoordinateSystem(5);
        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //while (!viewer.wasStopped())
        //{
        //    viewer.spinOnce();
        //    //viewer.saveScreenshot(filename);
        //}

        cout << "Compare amount: " << compare_cloud_list.size() << endl;
        //rot2ang(transformation_matrix_list[index]);
        
        


        
        dist_list.clear();
        nearest_point->clear();

    }
    int K = 1;
    double total_dist = 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> pointIdxKNNSearch(K);
    std::vector<float> pointKNNSquaredDistance(K);
    pcl::PointXYZ searchPoint2;
    pcl::PointXYZ searchPoint2_;

    for (int i = 0; i < compare_cloud_list.size(); i++)
    {
        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < compare_cloud_list[i]->size(); j++)
        {
            searchPoint2.x = compare_cloud_list[i]->points[j].x;
            searchPoint2.y = compare_cloud_list[i]->points[j].y;
            searchPoint2.z = compare_cloud_list[i]->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (pointKNNSquaredDistance[0] < 0.05)//0.3
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }

        }
        near_list.push_back(nearest_point->size());
    }

    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);
    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_near] << endl;
    cout << "index: " << index_near << endl;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    //viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    //viewer.addPointCloud(cloud, color_handler2, "cloud2");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    //viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    //viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    //viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //while (!viewer.wasStopped())
    //{
    //    viewer.spinOnce();
    //    //viewer.saveScreenshot(filename);
    //}


    regions_.clear();
}
void pca_match::test()
{
    //transform
    Eigen::Vector3d camera(80, 0, 0);
    Eigen::Vector3d new_camera;

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_model_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cam(new pcl::PointCloud<pcl::PointXYZ>);



    double theta = M_PI / 4.0;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitY()));
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cam(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointXYZ cam_;
    pcl::PointXYZ new_cam;
    cam_.x = camera.x(); cam_.y = camera.y(); cam_.z = camera.z();
    cam->push_back(cam_);



    pcl::transformPointCloud(*cam, *transformed_cam, transform);
    
    new_camera.x() = transformed_cam->points[0].x;
    new_camera.y() = transformed_cam->points[0].y;
    new_camera.z() = transformed_cam->points[0].z;
    new_cam.x = new_camera.x();
    new_cam.y = new_camera.y();
    new_cam.z = new_camera.z();

    model_hull = pca_match::hull(new_camera);
    original_model_hull = pca_match::hull(camera);
    
    
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(model_hull, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(transformed_cam, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(original_model_hull, 0, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(cam, 0, 0, 255);


    viewer.addPointCloud(model_hull, color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    viewer.addPointCloud(transformed_cam, color_handler2, "cloud1");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud1");
    viewer.addPointCloud(original_model_hull, color_handler3, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");
    viewer.addPointCloud(cam, color_handler4, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");

    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }
}

void pca_match::find_parts_in_scene_rotate_RMSE2_hull2()
{

    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_hull(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point(new pcl::PointCloud<pcl::PointXYZ>()), hull(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point2(new pcl::PointCloud<pcl::PointXYZ>()), hull2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point3(new pcl::PointCloud<pcl::PointXYZ>()), hull3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point4(new pcl::PointCloud<pcl::PointXYZ>()), hull4(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector3d transformed_cam, transformed_cam2, transformed_cam3, transformed_cam4;
    pcl::PointXYZ cam;
    start2 = clock();


    for (int i = 0; i < regions_.size(); i++)//regions_.size()
    {
        if (i > 1) { break; }
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {


            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            Eigen::Matrix4f transformation_r1, transformation_r2, transformation_r3;
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedhull(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull2(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull3(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull4(new pcl::PointCloud<pcl::PointXYZ>);

            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);


            hull = pca_match::hull_camera_ontop(transformed_final);
            hull2 = pca_match::hull_camera_ontop(rotatedCloud);
            hull3 = pca_match::hull_camera_ontop(rotatedCloud2);
            hull4 = pca_match::hull_camera_ontop(rotatedCloud3);



            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(hull, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(hull2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(hull3, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(hull4, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(camera_point, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(camera_point2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler8(camera_point3, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler9(camera_point4, 0, 0, 255);




            //viewer.addPointCloud(hull, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(hull2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(hull3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(hull4, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addPointCloud(camera_point, color_handler6, "cloud6");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud6");
            //viewer.addPointCloud(camera_point2, color_handler7, "cloud7");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud7");
            //viewer.addPointCloud(camera_point3, color_handler8, "cloud8");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud8");
            //viewer.addPointCloud(camera_point4, color_handler9, "cloud9");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud9");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");



            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(model_regions[j], 255, 0, 0);
            //viewer.addPointCloud(model_regions[j], color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");

            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;

            double x_err = 0;
            double y_err = 0;
            double z_err = 0;

            double x_err_r1 = 0;
            double y_err_r1 = 0;
            double z_err_r1 = 0;

            double x_err_r2 = 0;
            double y_err_r2 = 0;
            double z_err_r2 = 0;

            double x_err_r3 = 0;
            double y_err_r3 = 0;
            double z_err_r3 = 0;


            double constraint = 100;//15
            double mean_constraint = 0;
            int x = 0;
            int method = 1;//0 for whole model//1 for hull
            if (method == 0)
            {
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }
                        



                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < transformed_final->size(); i++)//hull2
                {
                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }
                        

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < transformed_final->size(); k++)//hull3
                {
                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }
                       

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < transformed_final->size(); h++)//hull4
                {
                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }
                       
                    }
                }
                cout << "nearest point count 4 : " << a << endl;
            }
            else if (method == 1)
            {
                for (int j = 0; j < hull->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = hull->points[j].x; //searchPoint.x = transformed_final->points[j].x;
                    searchPoint.y = hull->points[j].y; //searchPoint.y = transformed_final->points[j].y;
                    searchPoint.z = hull->points[j].z; //searchPoint.z = transformed_final->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }



                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < hull2->size(); i++)//hull2
                {
                    searchPoint_r.x = hull2->points[i].x; //searchPoint_r.x = rotatedCloud->points[i].x;
                    searchPoint_r.y = hull2->points[i].y; //searchPoint_r.y = rotatedCloud->points[i].y;
                    searchPoint_r.z = hull2->points[i].z; //searchPoint_r.z = rotatedCloud->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < hull3->size(); k++)//hull3
                {
                    searchPoint_r2.x = hull3->points[k].x; //searchPoint_r2.x = rotatedCloud2->points[k].x;
                    searchPoint_r2.y = hull3->points[k].y; //searchPoint_r2.y = rotatedCloud2->points[k].y;
                    searchPoint_r2.z = hull3->points[k].z; //searchPoint_r2.z = rotatedCloud2->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < hull4->size(); h++)//hull4
                {
                    searchPoint_r3.x = hull4->points[h].x; //searchPoint_r3.x = rotatedCloud3->points[h].x;
                    searchPoint_r3.y = hull4->points[h].y; //searchPoint_r3.y = rotatedCloud3->points[h].y;
                    searchPoint_r3.z = hull4->points[h].z; //searchPoint_r3.z = rotatedCloud3->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }

                    }
                }
                cout << "nearest point count 4 : " << a << endl;
            }
            else if (method == 2)
            {
                int iter = 0;
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;

                    searchPoint_r.x = rotatedCloud->points[j].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[j].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[j].z; //searchPoint_r.z = hull2->points[i].z;

                    searchPoint_r2.x = rotatedCloud2->points[j].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[j].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[j].z; //searchPoint_r2.z = hull3->points[k].z;

                    searchPoint_r3.x = rotatedCloud3->points[j].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[j].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[j].z; //searchPoint_r3.z = hull4->points[h].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }

                }
                mean_constraint = mean_constraint / iter;
                cout << "mean_constraint: " << mean_constraint << endl;
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }



                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < transformed_final->size(); i++)//hull2
                {
                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < transformed_final->size(); k++)//hull3
                {
                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < transformed_final->size(); h++)//hull4
                {
                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }

                    }
                }

                cout << "nearest point count 4 : " << a << endl;
            }








            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();


            total_dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);//+ y_err + z_err
            total_dist_r = sqrt(x_err_r1 * x_err_r1 + y_err_r1 * y_err_r1 + z_err_r1 * z_err_r1);//+ y_err_r1 + z_err_r1
            total_dist_r2 = sqrt(x_err_r2 * x_err_r2 + y_err_r2 * y_err_r2 + z_err_r2 * z_err_r2);// + y_err_r2 + z_err_r2
            total_dist_r3 = sqrt(x_err_r3 * x_err_r3 + y_err_r3 * y_err_r3 + z_err_r3 * z_err_r3);// + y_err_r3 + z_err_r3

            cout << "total_err: " << total_dist << endl;
            cout << "total_err r1: " << total_dist_r << endl;
            cout << "total_err r2: " << total_dist_r2 << endl;
            cout << "total_err r3: " << total_dist_r3 << endl;

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            transformation_cloud_list.push_back(hull);
            transformation_cloud_list.push_back(hull2);
            transformation_cloud_list.push_back(hull3);
            transformation_cloud_list.push_back(hull4);

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_final, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(rotatedCloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(rotatedCloud2, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(rotatedCloud3, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(cloud, 255, 0, 255);

            //viewer.addPointCloud(transformed_final, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(rotatedCloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(rotatedCloud2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(rotatedCloud3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(cloud, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");


        }



        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        cout << "smallest dist: " << dist_list[index] << endl;
        transformed_show = transformation_cloud_list[index];
        compare_cloud_list.push_back(transformed_show);

        //transformed_show2 = transformation_cloud_list[index];


        //pcl::visualization::PCLVisualizer viewer;
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_show, 255, 0, 0);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
        //
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

        //viewer.addPointCloud(transformed_show, color_handler, "cloud");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
        //viewer.addPointCloud(cloud, color_handler2, "cloud2");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
        //viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
        //
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
        ////viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
        ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
        //viewer.addCoordinateSystem(5);
        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //viewer.spin();
        //system("pause");

        






        dist_list.clear();
        nearest_point->clear();

    }
    int K = 1;
    double total_dist = 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> pointIdxKNNSearch(K);
    std::vector<float> pointKNNSquaredDistance(K);
    pcl::PointXYZ searchPoint2;
    pcl::PointXYZ searchPoint2_;
    cout << "Compare amount: " << compare_cloud_list.size() << endl;
    for (int i = 0; i < compare_cloud_list.size(); i++)
    {
        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < compare_cloud_list[i]->size(); j++)
        {
            searchPoint2.x = compare_cloud_list[i]->points[j].x;
            searchPoint2.y = compare_cloud_list[i]->points[j].y;
            searchPoint2.z = compare_cloud_list[i]->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                if (sqrt(pointKNNSquaredDistance[0]) < 1)//10
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }
        }
        near_list.push_back(nearest_point->size());
        cout << "Near point coounts: " << nearest_point->size() << endl;
    }

    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);
    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_near] << endl;
    cout << "index: " << index_near << endl;

    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();
    
}

//void pca_match::find_parts_in_scene_rotate_RMSE2_hull_rank()
//{
//    std::vector<Eigen::Vector3d> model_points{ model->size() };
//    for (int i = 0; i < model->size(); i++)
//    {
//        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
//    }
//
//    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
//    double diameter = max_min.norm();
//    model_diameter = diameter;
//
//    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
//    cout << "Total " << regions.size() << " planes." << endl;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
//    std::sort(regions_.begin(), regions_.end(), compare);
//    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
//    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/
//
//    std::vector<int> near_list;
//    std::vector<double> dist_compare;
//    //Extract regions from 'regions'vector in private numbers
//    std::vector<Eigen::Vector4f> center_list;
//
//    std::vector<Eigen::Matrix4f> transformation_matrix_list;
//    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list, transformation_cloud_list_sorted;
//    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
//    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
//    std::vector<double> dist_list;
//    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
//    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr original_hull(new pcl::PointCloud<pcl::PointXYZ>());
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point(new pcl::PointCloud<pcl::PointXYZ>()), hull(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point2(new pcl::PointCloud<pcl::PointXYZ>()), hull2(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point3(new pcl::PointCloud<pcl::PointXYZ>()), hull3(new pcl::PointCloud<pcl::PointXYZ>());
//    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point4(new pcl::PointCloud<pcl::PointXYZ>()), hull4(new pcl::PointCloud<pcl::PointXYZ>());
//    Eigen::Vector3d transformed_cam, transformed_cam2, transformed_cam3, transformed_cam4;
//    pcl::PointXYZ cam;
//    start2 = clock();
//
//
//    for (int i = 0; i < regions_.size(); i++)//regions_.size()
//    {
//        //if (regions_.size() == 1) { i = 0; continue; }
//        if (i > 0) { break; }
//        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
//        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>()), transformed_show3(new pcl::PointCloud<pcl::PointXYZ>());
//        transformation_cloud_list.clear();
//        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
//        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
//        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
//        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);
//
//        pcl::compute3DCentroid(*regions_[i], cloud_center);
//
//        for (int j = 0; j < model_regions.size(); j++)
//        {
//
//
//            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
//            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
//            pcl::compute3DCentroid(*model_regions[j], model_center);
//            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);
//
//            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
//            pcl::transformPointCloud(*model, *model_temp, transform_matrix);
//
//
//
//            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
//            icp.setInputSource(transformed_cloud);
//            icp.setInputTarget(regions_[i]);
//            //icp.setMaxCorrespondenceDistance(0.05);
//            icp.setMaximumIterations(20);
//            icp.setTransformationEpsilon(1e-5);
//            icp.align(*cloud_icp);
//            Eigen::Matrix4f transformation = icp.getFinalTransformation();
//            Eigen::Matrix4f transformation_r1, transformation_r2, transformation_r3;
//            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
//            //Rotate around normal vector
//            double angle = M_PI;
//            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
//            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
//            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
//            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
//            //Rotate around another 2 axis
//            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
//            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
//            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
//            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
//            //
//            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
//            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
//            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
//            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
//
//
//            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
//            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedhull(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull2(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull3(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull4(new pcl::PointCloud<pcl::PointXYZ>);
//
//            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
//            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
//            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);
//
//
//            hull = pca_match::hull_camera_ontop(transformed_final);
//            hull2 = pca_match::hull_camera_ontop(rotatedCloud);
//            hull3 = pca_match::hull_camera_ontop(rotatedCloud2);
//            hull4 = pca_match::hull_camera_ontop(rotatedCloud3);
//
//
//
//            //pcl::visualization::PCLVisualizer viewer;
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(hull, 255, 0, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(hull2, 0, 255, 255);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(hull3, 0, 0, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(hull4, 0, 0, 255);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(camera_point, 255, 0, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(camera_point2, 0, 255, 255);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler8(camera_point3, 0, 0, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler9(camera_point4, 0, 0, 255);
//
//
//
//
//            //viewer.addPointCloud(hull, color_handler, "cloud");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
//            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
//            //viewer.addPointCloud(hull2, color_handler3, "cloud3");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
//            //viewer.addPointCloud(hull3, color_handler4, "cloud4");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
//            //viewer.addPointCloud(hull4, color_handler5, "cloud5");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
//            //viewer.addPointCloud(camera_point, color_handler6, "cloud6");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud6");
//            //viewer.addPointCloud(camera_point2, color_handler7, "cloud7");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud7");
//            //viewer.addPointCloud(camera_point3, color_handler8, "cloud8");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud8");
//            //viewer.addPointCloud(camera_point4, color_handler9, "cloud9");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud9");
//            //viewer.addCoordinateSystem(5);
//            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
//            //viewer.spin();
//            //system("pause");
//
//
//
//            //pcl::visualization::PCLVisualizer viewer;
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(model_regions[j], 255, 0, 0);
//            //viewer.addPointCloud(model_regions[j], color_handler, "cloud");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
//            //viewer.addCoordinateSystem(5);
//            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
//            //viewer.spin();
//            //system("pause");
//
//            // K nearest neighbor search
//            //Evaluation
//            int K = 1;
//            double total_dist = 0;
//            double total_dist_r = 0;
//            double total_dist_r2 = 0; double total_dist_r3 = 0;
//
//            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//            std::vector<int> pointIdxKNNSearch(K);
//            std::vector<float> pointKNNSquaredDistance(K);
//            std::vector<int> pointIdxRadiusSearch;
//            std::vector<float> pointRadiusSquaredDistance;
//            kdtree.setInputCloud(cloud_pass_through);
//            pcl::PointXYZ searchPoint;
//            pcl::PointXYZ searchPoint_r;
//            pcl::PointXYZ searchPoint_r2;
//            pcl::PointXYZ searchPoint_r3;
//
//            double x_err = 0;
//            double y_err = 0;
//            double z_err = 0;
//
//            double x_err_r1 = 0;
//            double y_err_r1 = 0;
//            double z_err_r1 = 0;
//
//            double x_err_r2 = 0;
//            double y_err_r2 = 0;
//            double z_err_r2 = 0;
//
//            double x_err_r3 = 0;
//            double y_err_r3 = 0;
//            double z_err_r3 = 0;
//
//
//            double constraint = 50;//15
//            double mean_constraint = 0;
//            int x = 0;
//            int method = 0;//0 for whole model//1 for hull
//            if (method == 0)
//            {
//                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
//                {
//
//                    //use hull to find nearest point
//                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
//                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
//                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
//                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//                        //cout << "dist: " << sqrt(pointKNNSquaredDistance[0]);
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist += pointKNNSquaredDistance[0];
//                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE 1
//                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
//                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            x += 1;
//                        }
//                        else
//                        {
//                            //x_err += 100;
//                            //y_err += 100;
//                            //z_err += 100;
//                            //x += 1;
//                        }
//
//
//
//
//                    }
//                }
//                cout << "nearest point count: " << x << endl;
//                int y = 0;
//                for (int i = 0; i < transformed_final->size(); i++)//hull2
//                {
//                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
//                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
//                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
//                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist_r += pointKNNSquaredDistance[0];
//                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r1
//
//                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            y += 1;
//                        }
//                        else
//                        {
//                            //x_err_r1 += 100;
//                            //y_err_r1 += 100;
//                            //z_err_r1 += 100;
//                            //y += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 2 : " << y << endl;
//                int z = 0;
//                for (int k = 0; k < transformed_final->size(); k++)//hull3
//                {
//                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
//                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
//                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
//                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist_r2 += pointKNNSquaredDistance[0];
//                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r2
//                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            z += 1;
//                        }
//                        else
//                        {
//                            //x_err_r2 += 100;
//                            //y_err_r2 += 100;
//                            //z_err_r2 += 100;
//                            z += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 3 : " << z << endl;
//                int a = 0;
//                for (int h = 0; h < transformed_final->size(); h++)//hull4
//                {
//                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
//                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
//                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;
//
//                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist_r3 += pointKNNSquaredDistance[0];
//                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r3
//                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            a += 1;
//                        }
//                        else
//                        {
//                            //x_err_r3 += 100;
//                            //y_err_r3 += 100;
//                            //z_err_r3 += 100;
//                            a += 1;
//                        }
//                    }
//                }
//                cout << "nearest point count 4 : " << a << endl;
//            }
//            else if (method == 1)
//            {
//                for (int j = 0; j < hull->size(); j++)//(int j = 0; j < hull->size(); j++)
//                {
//
//                    //use hull to find nearest point
//                    searchPoint.x = hull->points[j].x; //searchPoint.x = transformed_final->points[j].x;
//                    searchPoint.y = hull->points[j].y; //searchPoint.y = transformed_final->points[j].y;
//                    searchPoint.z = hull->points[j].z; //searchPoint.z = transformed_final->points[j].z;
//                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist += pointKNNSquaredDistance[0];
//                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE 1
//                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
//                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            x += 1;
//                        }
//
//
//
//                    }
//                }
//                cout << "nearest point count: " << x << endl;
//                int y = 0;
//                for (int i = 0; i < hull2->size(); i++)//hull2
//                {
//                    searchPoint_r.x = hull2->points[i].x; //searchPoint_r.x = rotatedCloud->points[i].x;
//                    searchPoint_r.y = hull2->points[i].y; //searchPoint_r.y = rotatedCloud->points[i].y;
//                    searchPoint_r.z = hull2->points[i].z; //searchPoint_r.z = rotatedCloud->points[i].z;
//                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist_r += pointKNNSquaredDistance[0];
//                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r1
//
//                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            y += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 2 : " << y << endl;
//                int z = 0;
//                for (int k = 0; k < hull3->size(); k++)//hull3
//                {
//                    searchPoint_r2.x = hull3->points[k].x; //searchPoint_r2.x = rotatedCloud2->points[k].x;
//                    searchPoint_r2.y = hull3->points[k].y; //searchPoint_r2.y = rotatedCloud2->points[k].y;
//                    searchPoint_r2.z = hull3->points[k].z; //searchPoint_r2.z = rotatedCloud2->points[k].z;
//                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist_r2 += pointKNNSquaredDistance[0];
//                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r2
//                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            z += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 3 : " << z << endl;
//                int a = 0;
//                for (int h = 0; h < hull4->size(); h++)//hull4
//                {
//                    searchPoint_r3.x = hull4->points[h].x; //searchPoint_r3.x = rotatedCloud3->points[h].x;
//                    searchPoint_r3.y = hull4->points[h].y; //searchPoint_r3.y = rotatedCloud3->points[h].y;
//                    searchPoint_r3.z = hull4->points[h].z; //searchPoint_r3.z = rotatedCloud3->points[h].z;
//
//                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
//                        {
//                            //total_dist_r3 += pointKNNSquaredDistance[0];
//                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r3
//                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            a += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 4 : " << a << endl;
//            }
//            else if (method == 2)
//            {
//                int iter = 0;
//                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
//                {
//
//                    //use hull to find nearest point
//                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
//                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
//                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
//
//                    searchPoint_r.x = rotatedCloud->points[j].x; //searchPoint_r.x = hull2->points[i].x;
//                    searchPoint_r.y = rotatedCloud->points[j].y; //searchPoint_r.y = hull2->points[i].y;
//                    searchPoint_r.z = rotatedCloud->points[j].z; //searchPoint_r.z = hull2->points[i].z;
//
//                    searchPoint_r2.x = rotatedCloud2->points[j].x; //searchPoint_r2.x = hull3->points[k].x;
//                    searchPoint_r2.y = rotatedCloud2->points[j].y; //searchPoint_r2.y = hull3->points[k].y;
//                    searchPoint_r2.z = rotatedCloud2->points[j].z; //searchPoint_r2.z = hull3->points[k].z;
//
//                    searchPoint_r3.x = rotatedCloud3->points[j].x; //searchPoint_r3.x = hull4->points[h].x;
//                    searchPoint_r3.y = rotatedCloud3->points[j].y; //searchPoint_r3.y = hull4->points[h].y;
//                    searchPoint_r3.z = rotatedCloud3->points[j].z; //searchPoint_r3.z = hull4->points[h].z;
//                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
//                        iter += 1;
//                    }
//                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
//                        iter += 1;
//                    }
//                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
//                        iter += 1;
//                    }
//                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
//                        iter += 1;
//                    }
//
//                }
//                mean_constraint = mean_constraint / iter;
//                cout << "mean_constraint: " << mean_constraint << endl;
//                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
//                {
//
//                    //use hull to find nearest point
//                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
//                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
//                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
//                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
//                        {
//                            //total_dist += pointKNNSquaredDistance[0];
//                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE 1
//                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
//                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            x += 1;
//                        }
//
//
//
//                    }
//                }
//                cout << "nearest point count: " << x << endl;
//                int y = 0;
//                for (int i = 0; i < transformed_final->size(); i++)//hull2
//                {
//                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
//                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
//                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
//                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
//                        {
//                            //total_dist_r += pointKNNSquaredDistance[0];
//                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r1
//
//                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            y += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 2 : " << y << endl;
//                int z = 0;
//                for (int k = 0; k < transformed_final->size(); k++)//hull3
//                {
//                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
//                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
//                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
//                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
//                        {
//                            //total_dist_r2 += pointKNNSquaredDistance[0];
//                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r2
//                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            z += 1;
//                        }
//
//                    }
//                }
//                cout << "nearest point count 3 : " << z << endl;
//                int a = 0;
//                for (int h = 0; h < transformed_final->size(); h++)//hull4
//                {
//                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
//                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
//                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;
//
//                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//                    {
//
//                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
//                        {
//                            //total_dist_r3 += pointKNNSquaredDistance[0];
//                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                            //RMSE r3
//                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
//                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
//                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
//                            a += 1;
//                        }
//
//                    }
//                }
//
//                cout << "nearest point count 4 : " << a << endl;
//            }
//
//
//
//
//
//
//
//
//            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
//            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
//            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();
//
//
//            total_dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);//+ y_err + z_err
//            total_dist_r = sqrt(x_err_r1 * x_err_r1 + y_err_r1 * y_err_r1 + z_err_r1 * z_err_r1);//+ y_err_r1 + z_err_r1
//            total_dist_r2 = sqrt(x_err_r2 * x_err_r2 + y_err_r2 * y_err_r2 + z_err_r2 * z_err_r2);// + y_err_r2 + z_err_r2
//            total_dist_r3 = sqrt(x_err_r3 * x_err_r3 + y_err_r3 * y_err_r3 + z_err_r3 * z_err_r3);// + y_err_r3 + z_err_r3
//
//            cout << "total_err: " << total_dist << endl;
//            cout << "total_err r1: " << total_dist_r << endl;
//            cout << "total_err r2: " << total_dist_r2 << endl;
//            cout << "total_err r3: " << total_dist_r3 << endl;
//
//            dist_list.push_back(total_dist);
//            dist_list.push_back(total_dist_r);
//            dist_list.push_back(total_dist_r2);
//            dist_list.push_back(total_dist_r3);
//
//            transformation_cloud_list.push_back(hull);
//            transformation_cloud_list.push_back(hull2);
//            transformation_cloud_list.push_back(hull3);
//            transformation_cloud_list.push_back(hull4);
//
//            nearest_cloud_list.push_back(nearest_point_list);
//            nearest_cloud_list.push_back(nearest_point_list_r1);
//            nearest_cloud_list.push_back(nearest_point_list_r2);
//            nearest_cloud_list.push_back(nearest_point_list_r3);
//            transformation_matrix_list.push_back(transformation);
//
//            //pcl::visualization::PCLVisualizer viewer;
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_final, 0, 0, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(rotatedCloud, 0, 255, 0);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(rotatedCloud2, 0, 0, 255);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(rotatedCloud3, 0, 255, 255);
//            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(cloud, 255, 0, 255);
//
//            //viewer.addPointCloud(transformed_final, color_handler, "cloud");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
//            //viewer.addPointCloud(rotatedCloud, color_handler2, "cloud2");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
//            //viewer.addPointCloud(rotatedCloud2, color_handler3, "cloud3");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
//            //viewer.addPointCloud(rotatedCloud3, color_handler4, "cloud4");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
//            //viewer.addPointCloud(cloud, color_handler5, "cloud5");
//            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
//            //viewer.addCoordinateSystem(5);
//            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
//            //viewer.spin();
//            //system("pause");
//
//
//        }
//        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list_sorted = transformation_cloud_list;
//        std::vector<size_t> indices(dist_list.size());
//        for (size_t i = 0; i < indices.size(); ++i) {
//            indices[i] = i;
//        }
//        std::sort(indices.begin(), indices.end(), [&dist_list](size_t i, size_t j) {
//            return dist_list[i] < dist_list[j];
//            });
//
//        std::sort(dist_list.begin(), dist_list.end());
//
//
//        for (size_t j = 0; j < indices.size(); ++j) {
//            transformation_cloud_list_sorted[j] = transformation_cloud_list[indices[j]];
//        }
//
//
//
//
//        //pcl::visualization::PCLVisualizer viewer;
//        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformation_cloud_list_sorted[0], 255, 0, 0);
//        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(transformation_cloud_list_sorted[1], 255, 255, 0);
//        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(transformation_cloud_list_sorted[2], 0, 255, 255);
//
//
//        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
//        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
//        //
//        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);
//
//        //viewer.addPointCloud(transformation_cloud_list_sorted[0], color_handler, "cloud");
//        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
//
//        //viewer.addPointCloud(transformation_cloud_list_sorted[1], color_handler6, "cloud5");
//        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud5");
//        //viewer.addPointCloud(transformation_cloud_list_sorted[2], color_handler7, "cloud8");
//        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud8");
//
//        //viewer.addPointCloud(cloud, color_handler2, "cloud2");
//        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
//        //viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
//        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
//        //
//        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
//        ////viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
//        ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
//        //viewer.addCoordinateSystem(5);
//        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
//        //viewer.spin();
//        //system("pause");
//
//
//
//        //Filter the distance part
//        
//        double sum = std::accumulate(std::begin(dist_list), std::end(dist_list), 0.0);
//        double mean = sum / dist_list.size();
//        cout << "mean: " << mean << endl;
//        cout << "total compare: " << dist_list.size() << endl;
//
//
//
//        for (int i = 0; i < transformation_cloud_list_sorted.size(); i++)
//        {
//            if (dist_list[i] < mean*100)
//            {
//                compare_cloud_list.push_back(transformation_cloud_list_sorted[i]);
//                dist_compare.push_back(dist_list[i]);
//            }
//            
//        }
//
//
//        dist_list.clear();
//
//
//    }
//
//
//    int K = 1;
//    
//    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
//    std::vector<int> pointIdxKNNSearch(K);
//    std::vector<float> pointKNNSquaredDistance(K);
//    std::vector<double> dist_mean;
//    pcl::PointXYZ searchPoint2;
//    pcl::PointXYZ searchPoint2_;
//    cout << "Compare amount: " << compare_cloud_list.size() << endl;
//    for (int i = 0; i < compare_cloud_list.size(); i++)
//    {
//        double total_dist = 0;
//        kdtree.setInputCloud(cloud_pass_through);
//        for (int j = 0; j < compare_cloud_list[i]->size(); j++)
//        {
//            searchPoint2.x = compare_cloud_list[i]->points[j].x;
//            searchPoint2.y = compare_cloud_list[i]->points[j].y;
//            searchPoint2.z = compare_cloud_list[i]->points[j].z;
//
//            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
//            {
//                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
//                
//                //cout << "Nearest dist: " << sqrt(pointKNNSquaredDistance[0]) << " ";
//                if (sqrt(pointKNNSquaredDistance[0]) < 0.4)//10
//                {
//                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//                    total_dist += sqrt(pointKNNSquaredDistance[0]);
//                }
//                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
//
//            }
//        }
//        dist_mean.push_back(nearest_point->size() / total_dist);
//        near_list.push_back(nearest_point->size());
//        cout << "Near point coounts: " << nearest_point->size() << endl;
//        //cout << "total_dist: " << nearest_point->size()/total_dist << endl;
//
//        nearest_point->clear();
//    }
//
//
//
//    ////Filter the nearest point part
//    //double sum_point = std::accumulate(std::begin(near_list), std::end(near_list), 0.0);
//    //double mean_point = sum_point / near_list.size();
//    //
//
//    //for (int i = 0; i < compare_cloud_list.size(); i++)
//    //{
//    //    if (near_list[i] > mean_point)
//    //    {
//    //        compare_cloud_list.push_back(transformation_cloud_list_sorted[i]);
//    //    }
//    //}
//
//    
//    for (auto& i : dist_mean)
//    {
//        cout << "dist mean: " << i << endl;
//    }
//    cout << endl;
//    for (auto& j : dist_compare)
//    {
//        cout << "dist_compare: " << j << endl;
//    }
//    auto max = std::max_element(near_list.begin(), near_list.end());//near_list.end()
//    auto min = std::min_element(dist_compare.begin(), dist_compare.end());//near_list.end()
//
//    //int index_near = std::distance(dist_compare.begin(), min);
//    int index_near = std::distance(near_list.begin(), max);
//
//
//
//    end2 = clock();
//    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
//    printf("Time = %f\n", cpu_time);
//
//    cout << "Nearest points size: " << near_list[index_near] << endl;
//    cout << "index: " << index_near << endl;
//
//    pcl::visualization::PCLVisualizer viewer;
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(compare_cloud_list[0], 0, 0, 255);
//
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);
//
//    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
//    viewer.addPointCloud(cloud, color_handler2, "cloud2");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
//    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
//    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
//    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
//    viewer.addPointCloud(compare_cloud_list[0], color_handler6, "cloud6");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud6");
//    viewer.addCoordinateSystem(5);
//    viewer.setBackgroundColor(1.0, 1.0, 1.0);
//    while (!viewer.wasStopped())
//    {
//        viewer.spinOnce();
//        //viewer.saveScreenshot(filename);
//    }
//
//
//    regions_.clear();
//}

float cosineSimilarity(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
    float dotProduct = v1.dot(v2);
    float magnitude1 = v1.norm();
    float magnitude2 = v2.norm();

    return dotProduct / (magnitude1 * magnitude2);
}

struct Vector3D {
    double x, y, z;

    // Constructor
    Vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

    // Function to calculate the magnitude of the vector
    double magnitude() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // Function to normalize the vector
    void normalize() {
        double mag = magnitude();

        // Check if the magnitude is not zero to avoid division by zero
        if (mag != 0.0) {
            x /= mag;
            y /= mag;
            z /= mag;
        }
    }
};
void pca_match::find_parts_in_scene_rotate_RMSE2_hull_rank()/////////test
{
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<double> near_list;
    std::vector<double> dist_compare;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list, transformation_cloud_list_sorted,best_hull;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> final_nearest_cloud_list;

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list,best_list,compare_dist;

    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>()), nearest_point_temp(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_hull(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point(new pcl::PointCloud<pcl::PointXYZ>()), hull(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point2(new pcl::PointCloud<pcl::PointXYZ>()), hull2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point3(new pcl::PointCloud<pcl::PointXYZ>()), hull3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point4(new pcl::PointCloud<pcl::PointXYZ>()), hull4(new pcl::PointCloud<pcl::PointXYZ>()), hull_test(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector3d transformed_cam, transformed_cam2, transformed_cam3, transformed_cam4;
    pcl::PointXYZ cam;
    start2 = clock();


    for (int i = 0; i < regions_.size(); i++)//regions_.size()
    {
        //if (regions_.size() == 1) { i = 0; continue; }
        if (i > 0) { break; }
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>()), transformed_show3(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {


            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            Eigen::Matrix4f transformation_r1, transformation_r2, transformation_r3;
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedhull(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull2(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull3(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull4(new pcl::PointCloud<pcl::PointXYZ>);

            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);

            std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr> hull_expand;

            hull = pca_match::hull_camera_ontop(transformed_final);
            hull2 = pca_match::hull_camera_ontop(rotatedCloud);
            hull3 = pca_match::hull_camera_ontop(rotatedCloud2);
            hull4 = pca_match::hull_camera_ontop(rotatedCloud3);

            //hull_expand = pca_match::hull_expansion(rotatedCloud);


            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;

            double x_err = 0;
            double y_err = 0;
            double z_err = 0;

            double x_err_r1 = 0;
            double y_err_r1 = 0;
            double z_err_r1 = 0;

            double x_err_r2 = 0;
            double y_err_r2 = 0;
            double z_err_r2 = 0;

            double x_err_r3 = 0;
            double y_err_r3 = 0;
            double z_err_r3 = 0;


            double constraint = 200;//70
            double mean_constraint = 0;
            int x = 0;
            int method = 1;//0 for whole model//1 for hull
            if (method == 0)
            {
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        //cout << "dist: " << sqrt(pointKNNSquaredDistance[0]);
                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }
                        else
                        {
                            //x_err += 100;
                            //y_err += 100;
                            //z_err += 100;
                            //x += 1;
                        }




                    }
                }
                //cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < transformed_final->size(); i++)//hull2
                {
                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }
                        else
                        {
                            //x_err_r1 += 100;
                            //y_err_r1 += 100;
                            //z_err_r1 += 100;
                            //y += 1;
                        }

                    }
                }
                //cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < transformed_final->size(); k++)//hull3
                {
                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }
                        else
                        {
                            //x_err_r2 += 100;
                            //y_err_r2 += 100;
                            //z_err_r2 += 100;
                            z += 1;
                        }

                    }
                }
                //cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < transformed_final->size(); h++)//hull4
                {
                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }
                        else
                        {
                            //x_err_r3 += 100;
                            //y_err_r3 += 100;
                            //z_err_r3 += 100;
                            a += 1;
                        }
                    }
                }
                //cout << "nearest point count 4 : " << a << endl;
            }
            else if (method == 1)//constaint=10
            {
                for (int j = 0; j < hull->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = hull->points[j].x; //searchPoint.x = transformed_final->points[j].x;
                    searchPoint.y = hull->points[j].y; //searchPoint.y = transformed_final->points[j].y;
                    searchPoint.z = hull->points[j].z; //searchPoint.z = transformed_final->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }



                    }
                }
                //cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < hull2->size(); i++)//hull2
                {
                    searchPoint_r.x = hull2->points[i].x; //searchPoint_r.x = rotatedCloud->points[i].x;
                    searchPoint_r.y = hull2->points[i].y; //searchPoint_r.y = rotatedCloud->points[i].y;
                    searchPoint_r.z = hull2->points[i].z; //searchPoint_r.z = rotatedCloud->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }

                    }
                }
                //cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < hull3->size(); k++)//hull3
                {
                    searchPoint_r2.x = hull3->points[k].x; //searchPoint_r2.x = rotatedCloud2->points[k].x;
                    searchPoint_r2.y = hull3->points[k].y; //searchPoint_r2.y = rotatedCloud2->points[k].y;
                    searchPoint_r2.z = hull3->points[k].z; //searchPoint_r2.z = rotatedCloud2->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }

                    }
                }
                //cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < hull4->size(); h++)//hull4
                {
                    searchPoint_r3.x = hull4->points[h].x; //searchPoint_r3.x = rotatedCloud3->points[h].x;
                    searchPoint_r3.y = hull4->points[h].y; //searchPoint_r3.y = rotatedCloud3->points[h].y;
                    searchPoint_r3.z = hull4->points[h].z; //searchPoint_r3.z = rotatedCloud3->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }

                    }
                }
                //cout << "nearest point count 4 : " << a << endl;
            }
            else if (method == 2)
            {
                int iter = 0;
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;

                    searchPoint_r.x = rotatedCloud->points[j].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[j].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[j].z; //searchPoint_r.z = hull2->points[i].z;

                    searchPoint_r2.x = rotatedCloud2->points[j].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[j].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[j].z; //searchPoint_r2.z = hull3->points[k].z;

                    searchPoint_r3.x = rotatedCloud3->points[j].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[j].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[j].z; //searchPoint_r3.z = hull4->points[h].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }

                }
                mean_constraint = mean_constraint / iter;
                cout << "mean_constraint: " << mean_constraint << endl;
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }



                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < transformed_final->size(); i++)//hull2
                {
                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < transformed_final->size(); k++)//hull3
                {
                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < transformed_final->size(); h++)//hull4
                {
                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }

                    }
                }

                cout << "nearest point count 4 : " << a << endl;
            }


            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();


            total_dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);//+ y_err + z_err
            total_dist_r = sqrt(x_err_r1 * x_err_r1 + y_err_r1 * y_err_r1 + z_err_r1 * z_err_r1);//+ y_err_r1 + z_err_r1
            total_dist_r2 = sqrt(x_err_r2 * x_err_r2 + y_err_r2 * y_err_r2 + z_err_r2 * z_err_r2);// + y_err_r2 + z_err_r2
            total_dist_r3 = sqrt(x_err_r3 * x_err_r3 + y_err_r3 * y_err_r3 + z_err_r3 * z_err_r3);// + y_err_r3 + z_err_r3

            //cout << "total_err: " << total_dist << endl;
            //cout << "total_err r1: " << total_dist_r << endl;
            //cout << "total_err r2: " << total_dist_r2 << endl;
            //cout << "total_err r3: " << total_dist_r3 << endl;

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            auto min_d = std::min_element(dist_list.begin(), dist_list.end());
            int best_d = std::distance(dist_list.begin(), min_d);

            transformation_cloud_list.push_back(hull);
            transformation_cloud_list.push_back(hull2);
            transformation_cloud_list.push_back(hull3);
            transformation_cloud_list.push_back(hull4);

            compare_cloud_list.push_back(transformation_cloud_list[best_d]);
            compare_dist.push_back(dist_list[best_d]);
            cout << "Best dist pushback: " << dist_list[best_d] << endl;
            cout << endl;
            

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            
            transformation_matrix_list.push_back(transformation);

            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_final, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(rotatedCloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(rotatedCloud2, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(rotatedCloud3, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(cloud, 255, 0, 255);

            //viewer.addPointCloud(transformed_final, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(rotatedCloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(rotatedCloud2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(rotatedCloud3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(cloud, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");

            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformation_cloud_list[best_d], 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(hull2, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(hull3, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(hull4, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(cloud, 255, 0, 255);

            //viewer.addPointCloud(transformation_cloud_list[best_d], color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud");
            ////viewer.addPointCloud(hull2, color_handler2, "cloud2");
            ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            ////viewer.addPointCloud(hull3, color_handler3, "cloud3");
            ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            ////viewer.addPointCloud(hull4, color_handler4, "cloud4");
            ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(cloud, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");

            dist_list.clear();
            transformation_cloud_list.clear();
        }



    }


    int K = 1;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> pointIdxKNNSearch(K);
    std::vector<float> pointKNNSquaredDistance(K);
    std::vector<double> dist_mean;
    pcl::PointXYZ searchPoint2;
    pcl::PointXYZ searchPoint2_;
    std::vector<double> distance_list[3];



    for (int i = 0; i < compare_cloud_list.size(); i++)
    {
        double total_dist = 0;
        kdtree.setInputCloud(cloud_pass_through);
        Eigen::Vector3f hull_point_pca = compute_region_pca_all(compare_cloud_list[i]).col(0);
        
        for (int j = 0; j < compare_cloud_list[i]->size(); j++)
        {
            searchPoint2.x = compare_cloud_list[i]->points[j].x;
            searchPoint2.y = compare_cloud_list[i]->points[j].y;
            searchPoint2.z = compare_cloud_list[i]->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;

                //cout << "Nearest dist: " << sqrt(pointKNNSquaredDistance[0]) << " ";
                if (sqrt(pointKNNSquaredDistance[0]) < 0.8)//1
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                    total_dist += sqrt(pointKNNSquaredDistance[0]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }
        }
        *nearest_point_temp = *nearest_point;
        final_nearest_cloud_list.push_back(nearest_point_temp);

        Eigen::Vector3f nearest_point_pca = compute_region_pca_all(nearest_point).col(0);
        
        //cout <<"nearest_point_pca:\n " << nearest_point_pca << endl;
        //cout << "hull_pca_list:\n " << hull_point_pca << endl;
        float distance = cosineSimilarity(nearest_point_pca, hull_point_pca);
        
        distance_list->push_back(distance);
        dist_mean.push_back(nearest_point->size() / total_dist);
        near_list.push_back(nearest_point->size());
        cout << "pca difference: " << distance << endl;
        cout << "Near point coounts: " << nearest_point->size() << endl;
        cout << "total_dist: " << nearest_point->size()/total_dist << endl;
        cout << endl;
        //cout << "final_nearest_cloud_list: " << final_nearest_cloud_list[0]->size() << endl;
        nearest_point->clear();
        
    }

    Vector3D near_list_norm(near_list[0], near_list[1], near_list[2]);
    Vector3D distance_list_norm(distance_list->at(0), distance_list->at(1), distance_list->at(2));
    Vector3D compare_dist_norm(compare_dist[0], compare_dist[1], compare_dist[2]);
    near_list_norm.normalize();
    distance_list_norm.normalize();
    compare_dist_norm.normalize();
    near_list[0] = near_list_norm.x; near_list[1] = near_list_norm.y; near_list[2] = near_list_norm.z;
    distance_list->at(0) = distance_list_norm.x; distance_list->at(1) = distance_list_norm.y; distance_list->at(2) = distance_list_norm.z;
    compare_dist[0] = 1-compare_dist_norm.x; compare_dist[1] = 1-compare_dist_norm.y; compare_dist[2] = 1-compare_dist_norm.z;




    auto max = std::max_element(near_list.begin(), near_list.end());//near_list.end()
    auto min = std::max_element(distance_list->begin(), distance_list->end());//near_list.end()
    auto min_d = std::max_element(compare_dist.begin(), compare_dist.end());//near_list.end()


    //int index_near = std::distance(dist_compare.begin(), min);
    int index_pca = std::distance(distance_list->begin(), min);
    int index_count= std::distance(near_list.begin(), max);
    int index_min_d= std::distance(compare_dist.begin(), min_d);

    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_count] << endl;
    cout << "Nearest distance: " << compare_dist[index_min_d] << endl;
    cout << "PCA distance: " << distance_list->at(index_pca) << endl;
    cout << endl << endl;

    cout << "index_count:" << index_count << endl;
    cout << "index_min_d:" << index_min_d << endl;
    cout << "index_pca:" << index_pca << endl;

    //Final Evaluation
    //int final_result_index;;
    std::vector<int> chart{ 0,0,0 };
    float dist_weight = 0.45;
    float near_point_weight = 0.35;
    float pca_weight = 0.20;
    std::vector<double> final_result;
    double first_result = compare_dist[0] * dist_weight + near_list[0] * near_point_weight + distance_list->at(0) * pca_weight;
    double second_result = compare_dist[1] * dist_weight + near_list[1] * near_point_weight + distance_list->at(1) * pca_weight;
    double third_result = compare_dist[2] * dist_weight + near_list[2] * near_point_weight + distance_list->at(2) * pca_weight;


    final_result.push_back(first_result); 
    final_result.push_back(second_result); 
    final_result.push_back(third_result);
    for (auto& i : final_result)
    {
        cout << "Final score: " << i << endl;
    }
    auto max_result = std::max_element(final_result.begin(), final_result.end());
    int final_result_index= std::distance(final_result.begin(), max_result);



    //chart[index_min_d] += 1;//dsit
    //chart[index_count] += 1;//near points
    //chart[index_pca] += 1;//pca

    //if (chart[index_min_d] == 3)
    //{
    //    final_result_index = index_min_d;
    //    cout << "Perfect match!" << endl;
    //}
    //else if (chart[index_min_d] != 3)
    //{
    //    if (chart[index_min_d] == 2)
    //    {
    //        //if((compare_dist[index_min_d] / compare_dist[index_count]) *dist_weight)
    //        final_result_index = index_min_d;
    //    }
    //    if (chart[index_count] == 2)
    //    {
    //        //if((compare_dist[index_min_d] / compare_dist[index_count]) *dist_weight)
    //        final_result_index = index_count;
    //    }
    //    else if ((near_list[index_count] / near_list[index_min_d]) * near_point_weight > (compare_dist[index_count] / compare_dist[index_min_d]) * dist_weight)
    //    {
    //        final_result_index = index_count;
    //        cout << "Nearest Point is more." << endl;

    //    }
    //    else if ((near_list[index_count] / near_list[index_min_d]) * near_point_weight < (compare_dist[index_count] / compare_dist[index_min_d]) * dist_weight)
    //    {
    //        final_result_index = index_min_d;
    //        cout << "Distance is Better." << endl;
    //    }
    //    

    //}
    cout << "Final Index: " << final_result_index << endl;
    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[final_result_index], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(compare_cloud_list[index_count], 255, 255, 125);//Yellow
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(compare_cloud_list[index_min_d], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(final_nearest_cloud_list[index_pca], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);
    

    viewer.addPointCloud(compare_cloud_list[final_result_index], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
    viewer.addPointCloud(final_nearest_cloud_list[index_pca], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud5");
    //viewer.addPointCloud(compare_cloud_list[index_count], color_handler6, "cloud6");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud6");
    //viewer.addPointCloud(compare_cloud_list[index_min_d], color_handler7, "cloud7");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud7");

    viewer.addCoordinateSystem(20);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();
}


std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>pca_match::hull_expansion(pcl::PointCloud<pcl::PointXYZ>::Ptr& rotated_model)
{
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> output;
    //pcl::visualization::PCLVisualizer viewer;
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;
    double radius = diameter * 250;//*250
    //double radius = 100;
    // Define the rotation axis
    Eigen::Vector3f axis(1.0, 0.0, 0.0);  // Rotate around the z-axis
    // Define the rotation angle in radians
    float angle = M_PI / 2;  // 90 degrees
    // Create a transformation matrix
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(angle, axis));

    
    


    //std::vector<pcl::visualization::Camera> cam;
    //viewer.setCameraPosition(camera[0], camera[1], camera[2], 0, 0, 0);
    //viewer.getCameras(cam);
    //pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
    pcl::PointXYZ cameraPoint = pcl::PointXYZ(0, 0, diameter);//(0,0,2d)
    pcl::PointXYZ cameraPoint2;
    // Apply the transformation to the point
    cout << "Original camera point: " << cameraPoint << endl;


    Eigen::Vector3f transformed_point = transform * Eigen::Vector3f(10, 0, 2 * diameter);
    cameraPoint2.x = transformed_point[0];
    cameraPoint2.y = transformed_point[1];
    cameraPoint2.z = transformed_point[2];
    cout << "Transformed camera point: " << cameraPoint2 << endl;

    //cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
    Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
    std::vector<Eigen::Vector3d> spherical_proojection;
    
    for (int k = 0; k < 2; k++)
    {
        //step1:spherical projection
        for (size_t pidx = 0; pidx < rotated_model->points.size(); ++pidx)
        {
            pcl::PointXYZ currentPoint = rotated_model->points[pidx];
            Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

            Eigen::Vector3d projected_point = currentVector - camera_location;
            double norm = projected_point.norm();
            //if (norm == 1)
            //{
            //    norm = 0.0001;
            //}
            spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
        }

        size_t origin_pidx = spherical_proojection.size();
        if (k == 0) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);
            cout << "rotated_cloud size" << rotated_model->size() << endl;
            spherical_proojection.push_back(Eigen::Vector3d(cameraPoint.x, cameraPoint.y, cameraPoint.z));
            //spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size(); i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }
            
            cout << "newCloud" << newCloud->size() << endl;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            


            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(rotated_model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);
            output.push_back(cloud_p);
            cout << "Hull size: " << cloud_p->size() << endl;

            pcl::visualization::PCLVisualizer viewer;
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_p, 255, 0, 0);
            viewer.addPointCloud(cloud_p, color_handler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

            viewer.addCoordinateSystem(5);
            viewer.setBackgroundColor(1.0, 1.0, 1.0);
            viewer.spin();
            system("pause");
            spherical_proojection.clear();
        }
        else if (k == 1)
        {
            


            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud2(new pcl::PointCloud<pcl::PointXYZ>);
            
            spherical_proojection.push_back(Eigen::Vector3d(cameraPoint2.x, cameraPoint2.y, cameraPoint2.z));
            //spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size(); i++)
            {
                Eigen::Vector3d currentVector2 = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint2(currentVector2.x(), currentVector2.y(), currentVector2.z());
                newCloud2->push_back(currentPoint2);
            }
            cout << "newCloud" << newCloud2->size() << endl;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull2(new pcl::PointCloud<pcl::PointXYZ>), cloud_p2(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::ConvexHull<pcl::PointXYZ> chull2;
            pcl::PointIndices::Ptr hull_indices2(new  pcl::PointIndices());
            chull2.setInputCloud(newCloud2);
            chull2.reconstruct(*cloud_hull2);
            chull2.getHullPointIndices(*hull_indices2);
            pcl::ExtractIndices<pcl::PointXYZ> extract2;
            extract2.setInputCloud(rotated_model);
            extract2.setIndices(hull_indices2);
            extract2.filter(*cloud_p2);
            output.push_back(cloud_p2);
            cout << "Hull size: " << cloud_p2->size() << endl;

            pcl::visualization::PCLVisualizer viewer;
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_p2, 255, 0, 0);
            viewer.addPointCloud(cloud_p2, color_handler, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

            viewer.addCoordinateSystem(5);
            viewer.setBackgroundColor(1.0, 1.0, 1.0);
            viewer.spin();
            system("pause");
        }

        

        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
        //pcl::ConvexHull<pcl::PointXYZ> chull;
        //pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
        //chull.setInputCloud(newCloud);
        //chull.reconstruct(*cloud_hull);
        //chull.getHullPointIndices(*hull_indices);
        //pcl::ExtractIndices<pcl::PointXYZ> extract;
        //extract.setInputCloud(rotated_model);
        //extract.setIndices(hull_indices);
        //extract.filter(*cloud_p);
        //output.push_back(cloud_p);
        //cout << "Hull size: " << cloud_p->size() << endl;

        //pcl::visualization::PCLVisualizer viewer;
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_p, 255, 0, 0);
        //viewer.addPointCloud(cloud_p, color_handler, "cloud");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

        //viewer.addCoordinateSystem(5);
        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //viewer.spin();
        //system("pause");

        //cloud_p->clear();
        //spherical_proojection.clear();
        //newCloud->clear();
        //hull_indices->indices.clear();
        //cloud_hull->clear();
    }


    
    return output;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr pca_match::hull_camera_ontop(pcl::PointCloud<pcl::PointXYZ>::Ptr& rotated_model)
{
    //pcl::visualization::PCLVisualizer viewer;
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;
    double radius = diameter * 250;//*250
    //double radius = 100;



    //std::vector<pcl::visualization::Camera> cam;
    //viewer.setCameraPosition(camera[0], camera[1], camera[2], 0, 0, 0);
    //viewer.getCameras(cam);
    //pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
    //pcl::PointXYZ cameraPoint = pcl::PointXYZ(-120, -120, 2* diameter);//(0,0,2d)
    pcl::PointXYZ cameraPoint = pcl::PointXYZ(10, 10, 2 * diameter);
    //cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
    Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
    std::vector<Eigen::Vector3d> spherical_proojection;
    pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

    //step1:spherical projection
    for (size_t pidx = 0; pidx < rotated_model->points.size(); ++pidx)
    {
        pcl::PointXYZ currentPoint = rotated_model->points[pidx];
        Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

        Eigen::Vector3d projected_point = currentVector - camera_location;
        double norm = projected_point.norm();
        //if (norm == 1)
        //{
        //    norm = 0.0001;
        //}
        spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
    }
    size_t origin_pidx = spherical_proojection.size();
    spherical_proojection.push_back(Eigen::Vector3d(cameraPoint.x, cameraPoint.y, cameraPoint.z));
    //spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
    for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
    {
        Eigen::Vector3d currentVector = spherical_proojection.at(i);
        pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
        newCloud->push_back(currentPoint);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConvexHull<pcl::PointXYZ> chull;
    pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
    chull.setInputCloud(newCloud);
    chull.reconstruct(*cloud_hull);
    chull.getHullPointIndices(*hull_indices);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(rotated_model);
    extract.setIndices(hull_indices);
    extract.filter(*cloud_p);

    //pcl::visualization::PCLVisualizer viewer;
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud_p, 255, 0, 0);


    //viewer.addPointCloud(cloud_p, color_handler, "cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    //viewer.addCoordinateSystem(5);
    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.spin();
    //system("pause");

    return cloud_p;
}



void pca_match::plane_segmentation_hidden_oil()
{
    pcl::visualization::PCLVisualizer viewer;
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    double radius = diameter * 250;//*250
    //double radius = 100;

    std::vector<pcl::visualization::Camera> cam;

    //find cylinder first
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg2;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);


    for (int k = 3; k < 6; k++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>), cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
        if (k == 0)
        {
            viewer.setCameraPosition(-diameter, 0, 0, 0, 0, 0);
            camera_location1[0] = diameter;
            camera_location1[1] = 0;
            camera_location1[2] = 15;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }


            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);

        }
        else if (k == 1)
        {
            viewer.setCameraPosition(diameter, 0, 0, 0, 0, 0);
            camera_location2[0] = 0;
            camera_location2[1] = diameter;
            camera_location2[2] = 15;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }


            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }
        else if (k == 2)
        {
            viewer.setCameraPosition(20, 0, -diameter, 0, 0, 0);
            camera_location3[0] = 0;
            camera_location3[1] = 0;
            camera_location3[2] = -diameter;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }


            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }

        else if (k == 3)
        {
            viewer.setCameraPosition(5, 12 , 1.5*diameter, 0, 0, 0);
            camera_location3[0] = 0;
            camera_location3[1] = 0;
            camera_location3[2] = -diameter;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }


            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }
        else if (k == 4)
        {
            viewer.setCameraPosition(0,diameter , 0, 0, 0, 0);
            camera_location3[0] = 0;
            camera_location3[1] = 0;
            camera_location3[2] = -diameter;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }


            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }
        else if (k == 5)
        {
            viewer.setCameraPosition(0, -diameter, 0, 0, 0, 0);
            camera_location3[0] = 0;
            camera_location3[1] = 0;
            camera_location3[2] = -diameter;

            viewer.getCameras(cam);
            pcl::PointXYZ cameraPoint = pcl::PointXYZ(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]);
            cout << "Camera Point: " << cam[0].pos[0] << " , " << cam[0].pos[1] << " , " << cam[0].pos[2] << endl;
            Eigen::Vector3d camera_location(cameraPoint.x, cameraPoint.y, cameraPoint.z);
            std::vector<Eigen::Vector3d> spherical_proojection;
            pcl::PointCloud<pcl::PointXYZ>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZ>);

            //step1:spherical projection
            for (size_t pidx = 0; pidx < model->points.size(); ++pidx)
            {
                pcl::PointXYZ currentPoint = model->points[pidx];
                Eigen::Vector3d currentVector(currentPoint.x, currentPoint.y, currentPoint.z);

                Eigen::Vector3d projected_point = currentVector - camera_location;
                double norm = projected_point.norm();
                //if (norm == 1)
                //{
                //    norm = 0.0001;
                //}
                spherical_proojection.push_back(projected_point + 2 * (radius - norm) * projected_point / norm);
            }
            size_t origin_pidx = spherical_proojection.size();
            //spherical_proojection.push_back(Eigen::Vector3d(cam[0].pos[0], cam[0].pos[1], cam[0].pos[2]));
            spherical_proojection.push_back(Eigen::Vector3d(0, 0, 0));
            for (std::size_t i = 0; i < spherical_proojection.size() - 1; i++)
            {
                Eigen::Vector3d currentVector = spherical_proojection.at(i);
                pcl::PointXYZ currentPoint(currentVector.x(), currentVector.y(), currentVector.z());
                newCloud->push_back(currentPoint);
            }


            pcl::ConvexHull<pcl::PointXYZ> chull;
            pcl::PointIndices::Ptr hull_indices(new  pcl::PointIndices());
            chull.setInputCloud(newCloud);
            chull.reconstruct(*cloud_hull);
            chull.getHullPointIndices(*hull_indices);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(model);
            extract.setIndices(hull_indices);
            extract.filter(*cloud_p);



        }

        //ransac plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory

        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_PROSAC);
        seg.setNumberOfThreads(16);//8
        seg.setDistanceThreshold(0.05);//0.05
        //seg.setEpsAngle(M_PI / 18);
        if (k == 3)
        {
            seg.setAxis(Eigen::Vector3f::UnitZ());
        }
        else
        {
            seg.setAxis(Eigen::Vector3f::UnitZ());
        }
        

        // Create the segmentation object for cylinder segmentation and set all the parameters
        seg2.setOptimizeCoefficients(true);
        seg2.setModelType(pcl::SACMODEL_CYLINDER);
        seg2.setMethodType(pcl::SAC_RANSAC);
        seg2.setNormalDistanceWeight(0.1);
        seg2.setDistanceThreshold(0.03);
        seg2.setRadiusLimits(0, 10);
        seg2.setInputCloud(cloud_p);
        seg2.setInputNormals(cloud_normals);

        pcl::PointCloud<pcl::PointXYZ>::Ptr model_(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>), cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        *model_ = *cloud_p;
        const double model_size = cloud_p->size();
        //pcl::PointCloud<pcl::PointXYZ>::Ptr plane_temp(new pcl::PointCloud<pcl::PointXYZ>);


        //if (k == 1)
        //{

        //    // Estimate point normals
        //    ne.setSearchMethod(tree);
        //    ne.setInputCloud(cloud_p);
        //    ne.setKSearch(50);
        //    ne.compute(*cloud_normals);

        //    seg2.setInputCloud(cloud_p);
        //    seg2.segment(*inliers, *coefficients);
        //    // Extract the planar inliers from the input cloud
        //    pcl::ExtractIndices<pcl::PointXYZ> extract;

        //    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
        //    extract.setInputCloud(cloud_p);
        //    extract.setIndices(inliers);
        //    extract.setNegative(false);
        //    extract.filter(*plane);
        //    model_regions.push_back(plane);


        //    //pcl::visualization::PCLVisualizer viewer;
        //    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //    //viewer.getRenderWindow()->GlobalWarningDisplayOff();
        //    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
        //    //viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
        //    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        //    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
        //    //viewer.addPointCloud(cloud_p, color_handler2, "off_scene_model2");
        //    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        //    //viewer.initCameraParameters();
        //    //viewer.spin();

        //    //system("pause");
        //    //cout << plane->size() << endl;
        //    //cout << "i: " << i << endl;

        //    inliers->indices.clear();
        //    coefficients->values.clear();

        //}

        seg.setInputCloud(model_);
        seg.segment(*inliers, *coefficients);

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>), plane_temp(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(model_);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);
        model_regions.push_back(plane);
        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(*model_);

        pcl::visualization::PCLVisualizer viewer;
        viewer.setBackgroundColor(1.0, 1.0, 1.0);
        viewer.getRenderWindow()->GlobalWarningDisplayOff();
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(plane, 255, 0, 0);
        viewer.addPointCloud(plane, color_handler1, "off_scene_model1");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model1");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model_, 0, 255, 0);
        viewer.addPointCloud(model_, color_handler2, "off_scene_model2");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "off_scene_model2");
        viewer.initCameraParameters();
        viewer.spin();

        system("pause");
        //cout << plane->size() << endl;
        //cout << "i: " << i << endl;

        inliers->indices.clear();
        coefficients->values.clear();








    }







    //cout << "radius: " << radius << std::endl;
    //cout << "Original Cloud size " << model->points.size() << std::endl;
    //cout << "New Cloud's size " << newCloud->points.size() << std::endl;
    //

    //viewer.setBackgroundColor(1.0, 1.0, 1.0);
    //viewer.getRenderWindow()->GlobalWarningDisplayOff();
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(newCloud, 255, 0, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(model, 0, 0, 255);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(cloud_hull, 0, 255, 0);
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(cloud_p, 0, 255, 0);
    ////viewer.addPointCloud(newCloud, color_handler1, "off_scene_model1");
    ////viewer.addPointCloud(model, color_handler2, "off_scene_model2");
    ////viewer.addPointCloud(cloud_hull, color_handler3, "off_scene_model3");
    //viewer.addPointCloud(cloud_p, color_handler4, "off_scene_model4");
    //
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model1");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model2");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "off_scene_model3");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "off_scene_model4");

    //
    ////viewer.initCameraParameters();
    //viewer.resetCamera();
    //viewer.spin();
    //system("pause");
}
void pca_match::find_parts_in_scene_rotate_oil()
{
    std::vector<Eigen::Vector3d> model_points{ model->size() };
    for (int i = 0; i < model->size(); i++)
    {
        model_points[i] = Eigen::Vector3d(model->points[i].x, model->points[i].y, model->points[i].z);
    }

    Eigen::Vector3d max_min = ComputeMaxBound(model_points) - ComputeMinBound(model_points);
    double diameter = max_min.norm();
    model_diameter = diameter;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > regions_ = regions;
    cout << "Total " << regions.size() << " planes." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjacent, adjacent2, adjacent3;
    std::sort(regions_.begin(), regions_.end(), compare);
    /*cout << "Biggest Plane: " << regions_[0]->size() << endl;
    cout << "2nd Biggest Plane: " << regions_[1]->size() << endl;*/

    std::vector<int> near_list;
    //Extract regions from 'regions'vector in private numbers
    std::vector<Eigen::Vector4f> center_list;

    std::vector<Eigen::Matrix4f> transformation_matrix_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> transformation_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nearest_cloud_list;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> compare_cloud_list;
    std::vector<double> dist_list;
    /*pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearest_point_list_r3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_hull(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point(new pcl::PointCloud<pcl::PointXYZ>()), hull(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point2(new pcl::PointCloud<pcl::PointXYZ>()), hull2(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point3(new pcl::PointCloud<pcl::PointXYZ>()), hull3(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_point4(new pcl::PointCloud<pcl::PointXYZ>()), hull4(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector3d transformed_cam, transformed_cam2, transformed_cam3, transformed_cam4;
    pcl::PointXYZ cam;
    start2 = clock();


    for (int i = 0; i < regions_.size(); i++)//regions_.size()
    {
        //if (regions_.size() == 1) { i = 0; continue; }
        if (i > 1) { break; }
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_show2(new pcl::PointCloud<pcl::PointXYZ>());
        transformation_cloud_list.clear();
        //Eigen::Vector3f normal_scene = compute_region_pca(regions_[i]);
        Eigen::Vector3f normal_scene = compute_region_pca_all(regions_[i]).col(0);
        Eigen::Vector3f normal_scene2 = compute_region_pca_all(regions_[i]).col(1);
        Eigen::Vector3f normal_scene3 = compute_region_pca_all(regions_[i]).col(2);

        pcl::compute3DCentroid(*regions_[i], cloud_center);

        for (int j = 0; j < model_regions.size(); j++)
        {


            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>()), transformed_final(new pcl::PointCloud<pcl::PointXYZ>()), model_temp(new pcl::PointCloud<pcl::PointXYZ>());
            Eigen::Vector3f normal_model = compute_region_pca(model_regions[j]);
            pcl::compute3DCentroid(*model_regions[j], model_center);
            Eigen::Matrix4f transform_matrix = Rotation_matrix(normal_model, normal_scene, model_center, cloud_center);

            pcl::transformPointCloud(*model_regions[j], *transformed_cloud, transform_matrix);
            pcl::transformPointCloud(*model, *model_temp, transform_matrix);



            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(regions_[i]);
            //icp.setMaxCorrespondenceDistance(0.05);
            icp.setMaximumIterations(20);
            icp.setTransformationEpsilon(1e-5);
            icp.align(*cloud_icp);
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            Eigen::Matrix4f transformation_r1, transformation_r2, transformation_r3;
            pcl::transformPointCloud(*model_temp, *transformed_final, transformation);
            //Rotate around normal vector
            double angle = M_PI;
            Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
            rotation.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation.rotate(Eigen::AngleAxisf(angle, normal_scene));
            rotation.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //Rotate around another 2 axis
            Eigen::Affine3f rotation2 = Eigen::Affine3f::Identity();
            rotation2.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation2.rotate(Eigen::AngleAxisf(angle, normal_scene2));
            rotation2.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));
            //
            Eigen::Affine3f rotation3 = Eigen::Affine3f::Identity();
            rotation3.translate(Eigen::Vector3f(cloud_center[0], cloud_center[1], cloud_center[2]));
            rotation3.rotate(Eigen::AngleAxisf(angle, normal_scene3));
            rotation3.translate(Eigen::Vector3f(-cloud_center[0], -cloud_center[1], -cloud_center[2]));


            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedCloud(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud2(new pcl::PointCloud<pcl::PointXYZ>), rotatedCloud3(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr rotatedhull(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull2(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull3(new pcl::PointCloud<pcl::PointXYZ>), rotatedhull4(new pcl::PointCloud<pcl::PointXYZ>);

            pcl::transformPointCloud(*transformed_final, *rotatedCloud, rotation);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud2, rotation2);
            pcl::transformPointCloud(*transformed_final, *rotatedCloud3, rotation3);


            hull = pca_match::hull_camera_ontop(transformed_final);
            hull2 = pca_match::hull_camera_ontop(rotatedCloud);
            hull3 = pca_match::hull_camera_ontop(rotatedCloud2);
            hull4 = pca_match::hull_camera_ontop(rotatedCloud3);



            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(hull, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(hull2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(hull3, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(hull4, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler6(camera_point, 255, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler7(camera_point2, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler8(camera_point3, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler9(camera_point4, 0, 0, 255);




            //viewer.addPointCloud(hull, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(cloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(hull2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(hull3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(hull4, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addPointCloud(camera_point, color_handler6, "cloud6");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud6");
            //viewer.addPointCloud(camera_point2, color_handler7, "cloud7");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud7");
            //viewer.addPointCloud(camera_point3, color_handler8, "cloud8");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud8");
            //viewer.addPointCloud(camera_point4, color_handler9, "cloud9");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "cloud9");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");



            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(model_regions[j], 255, 0, 0);
            //viewer.addPointCloud(model_regions[j], color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");

            // K nearest neighbor search
            //Evaluation
            int K = 1;
            double total_dist = 0;
            double total_dist_r = 0;
            double total_dist_r2 = 0; double total_dist_r3 = 0;

            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::vector<int> pointIdxKNNSearch(K);
            std::vector<float> pointKNNSquaredDistance(K);
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            kdtree.setInputCloud(cloud_pass_through);
            pcl::PointXYZ searchPoint;
            pcl::PointXYZ searchPoint_r;
            pcl::PointXYZ searchPoint_r2;
            pcl::PointXYZ searchPoint_r3;

            double x_err = 0;
            double y_err = 0;
            double z_err = 0;

            double x_err_r1 = 0;
            double y_err_r1 = 0;
            double z_err_r1 = 0;

            double x_err_r2 = 0;
            double y_err_r2 = 0;
            double z_err_r2 = 0;

            double x_err_r3 = 0;
            double y_err_r3 = 0;
            double z_err_r3 = 0;


            double constraint = 5;//15
            double mean_constraint = 0;
            int x = 0;
            int method = 0;//0 for whole model//1 for hull
            if (method == 0)
            {
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        //cout << "dist: " << sqrt(pointKNNSquaredDistance[0]);
                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }
                        else
                        {
                            x_err += 100;
                            y_err += 100;
                            z_err += 100;
                            //x += 1;
                        }




                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < transformed_final->size(); i++)//hull2
                {
                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }
                        else
                        {
                            x_err_r1 += 100;
                            y_err_r1 += 100;
                            z_err_r1 += 100;
                            y += 1;
                        }

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < transformed_final->size(); k++)//hull3
                {
                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }
                        else
                        {
                            x_err_r2 += 100;
                            y_err_r2 += 100;
                            z_err_r2 += 100;
                            z += 1;
                        }

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < transformed_final->size(); h++)//hull4
                {
                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }
                        else
                        {
                            x_err_r3 += 100;
                            y_err_r3 += 100;
                            z_err_r3 += 100;
                            a += 1;
                        }
                    }
                }
                cout << "nearest point count 4 : " << a << endl;
            }
            else if (method == 1)
            {
                for (int j = 0; j < hull->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = hull->points[j].x; //searchPoint.x = transformed_final->points[j].x;
                    searchPoint.y = hull->points[j].y; //searchPoint.y = transformed_final->points[j].y;
                    searchPoint.z = hull->points[j].z; //searchPoint.z = transformed_final->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }



                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < hull2->size(); i++)//hull2
                {
                    searchPoint_r.x = hull2->points[i].x; //searchPoint_r.x = rotatedCloud->points[i].x;
                    searchPoint_r.y = hull2->points[i].y; //searchPoint_r.y = rotatedCloud->points[i].y;
                    searchPoint_r.z = hull2->points[i].z; //searchPoint_r.z = rotatedCloud->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < hull3->size(); k++)//hull3
                {
                    searchPoint_r2.x = hull3->points[k].x; //searchPoint_r2.x = rotatedCloud2->points[k].x;
                    searchPoint_r2.y = hull3->points[k].y; //searchPoint_r2.y = rotatedCloud2->points[k].y;
                    searchPoint_r2.z = hull3->points[k].z; //searchPoint_r2.z = rotatedCloud2->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < hull4->size(); h++)//hull4
                {
                    searchPoint_r3.x = hull4->points[h].x; //searchPoint_r3.x = rotatedCloud3->points[h].x;
                    searchPoint_r3.y = hull4->points[h].y; //searchPoint_r3.y = rotatedCloud3->points[h].y;
                    searchPoint_r3.z = hull4->points[h].z; //searchPoint_r3.z = rotatedCloud3->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }

                    }
                }
                cout << "nearest point count 4 : " << a << endl;
            }
            else if (method == 2)
            {
                int iter = 0;
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;

                    searchPoint_r.x = rotatedCloud->points[j].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[j].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[j].z; //searchPoint_r.z = hull2->points[i].z;

                    searchPoint_r2.x = rotatedCloud2->points[j].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[j].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[j].z; //searchPoint_r2.z = hull3->points[k].z;

                    searchPoint_r3.x = rotatedCloud3->points[j].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[j].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[j].z; //searchPoint_r3.z = hull4->points[h].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {
                        mean_constraint += sqrt(pointKNNSquaredDistance[0]);
                        iter += 1;
                    }

                }
                mean_constraint = mean_constraint / iter;
                cout << "mean_constraint: " << mean_constraint << endl;
                for (int j = 0; j < transformed_final->size(); j++)//(int j = 0; j < hull->size(); j++)
                {

                    //use hull to find nearest point
                    searchPoint.x = transformed_final->points[j].x; //searchPoint.x = hull->points[j].x;
                    searchPoint.y = transformed_final->points[j].y; //searchPoint.y = hull->points[j].y;
                    searchPoint.z = transformed_final->points[j].z; //searchPoint.z = hull->points[j].z;
                    if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist += pointKNNSquaredDistance[0];
                            nearest_point_list->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE 1
                            //rmse(searchPoint, (*cloud_pass_through)[pointIdxKNNSearch[0]]);  
                            x_err += (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err += (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err += (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            x += 1;
                        }



                    }
                }
                cout << "nearest point count: " << x << endl;
                int y = 0;
                for (int i = 0; i < transformed_final->size(); i++)//hull2
                {
                    searchPoint_r.x = rotatedCloud->points[i].x; //searchPoint_r.x = hull2->points[i].x;
                    searchPoint_r.y = rotatedCloud->points[i].y; //searchPoint_r.y = hull2->points[i].y;
                    searchPoint_r.z = rotatedCloud->points[i].z; //searchPoint_r.z = hull2->points[i].z;
                    if (kdtree.nearestKSearch(searchPoint_r, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r += pointKNNSquaredDistance[0];
                            nearest_point_list_r1->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r1

                            x_err_r1 += (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r1 += (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r1 += (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            y += 1;
                        }

                    }
                }
                cout << "nearest point count 2 : " << y << endl;
                int z = 0;
                for (int k = 0; k < transformed_final->size(); k++)//hull3
                {
                    searchPoint_r2.x = rotatedCloud2->points[k].x; //searchPoint_r2.x = hull3->points[k].x;
                    searchPoint_r2.y = rotatedCloud2->points[k].y; //searchPoint_r2.y = hull3->points[k].y;
                    searchPoint_r2.z = rotatedCloud2->points[k].z; //searchPoint_r2.z = hull3->points[k].z;
                    if (kdtree.nearestKSearch(searchPoint_r2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r2 += pointKNNSquaredDistance[0];
                            nearest_point_list_r2->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r2
                            x_err_r2 += (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r2.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r2 += (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r2.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r2 += (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r2.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            z += 1;
                        }

                    }
                }
                cout << "nearest point count 3 : " << z << endl;
                int a = 0;
                for (int h = 0; h < transformed_final->size(); h++)//hull4
                {
                    searchPoint_r3.x = rotatedCloud3->points[h].x; //searchPoint_r3.x = hull4->points[h].x;
                    searchPoint_r3.y = rotatedCloud3->points[h].y; //searchPoint_r3.y = hull4->points[h].y;
                    searchPoint_r3.z = rotatedCloud3->points[h].z; //searchPoint_r3.z = hull4->points[h].z;

                    if (kdtree.nearestKSearch(searchPoint_r3, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
                    {

                        if (sqrt(pointKNNSquaredDistance[0]) < mean_constraint)
                        {
                            //total_dist_r3 += pointKNNSquaredDistance[0];
                            nearest_point_list_r3->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                            //RMSE r3
                            x_err_r3 += (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x) * (searchPoint_r3.x - (*cloud_pass_through)[pointIdxKNNSearch[0]].x);
                            y_err_r3 += (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y) * (searchPoint_r3.y - (*cloud_pass_through)[pointIdxKNNSearch[0]].y);
                            z_err_r3 += (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z) * (searchPoint_r3.z - (*cloud_pass_through)[pointIdxKNNSearch[0]].z);
                            a += 1;
                        }

                    }
                }

                cout << "nearest point count 4 : " << a << endl;
            }








            x_err /= model_temp->size(); x_err_r1 /= model_temp->size(); x_err_r2 /= model_temp->size(); x_err_r3 /= model_temp->size();
            y_err /= model_temp->size(); y_err_r1 /= model_temp->size(); y_err_r2 /= model_temp->size(); y_err_r3 /= model_temp->size();
            z_err /= model_temp->size(); z_err_r1 /= model_temp->size(); z_err_r2 /= model_temp->size(); z_err_r3 /= model_temp->size();


            total_dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);//+ y_err + z_err
            total_dist_r = sqrt(x_err_r1 * x_err_r1 + y_err_r1 * y_err_r1 + z_err_r1 * z_err_r1);//+ y_err_r1 + z_err_r1
            total_dist_r2 = sqrt(x_err_r2 * x_err_r2 + y_err_r2 * y_err_r2 + z_err_r2 * z_err_r2);// + y_err_r2 + z_err_r2
            total_dist_r3 = sqrt(x_err_r3 * x_err_r3 + y_err_r3 * y_err_r3 + z_err_r3 * z_err_r3);// + y_err_r3 + z_err_r3

            cout << "total_err: " << total_dist << endl;
            cout << "total_err r1: " << total_dist_r << endl;
            cout << "total_err r2: " << total_dist_r2 << endl;
            cout << "total_err r3: " << total_dist_r3 << endl;

            dist_list.push_back(total_dist);
            dist_list.push_back(total_dist_r);
            dist_list.push_back(total_dist_r2);
            dist_list.push_back(total_dist_r3);

            transformation_cloud_list.push_back(hull);
            transformation_cloud_list.push_back(hull2);
            transformation_cloud_list.push_back(hull3);
            transformation_cloud_list.push_back(hull4);

            nearest_cloud_list.push_back(nearest_point_list);
            nearest_cloud_list.push_back(nearest_point_list_r1);
            nearest_cloud_list.push_back(nearest_point_list_r2);
            nearest_cloud_list.push_back(nearest_point_list_r3);
            transformation_matrix_list.push_back(transformation);

            //pcl::visualization::PCLVisualizer viewer;
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_final, 0, 0, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(rotatedCloud, 0, 255, 0);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(rotatedCloud2, 0, 0, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(rotatedCloud3, 0, 255, 255);
            //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(cloud, 255, 0, 255);

            //viewer.addPointCloud(transformed_final, color_handler, "cloud");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            //viewer.addPointCloud(rotatedCloud, color_handler2, "cloud2");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
            //viewer.addPointCloud(rotatedCloud2, color_handler3, "cloud3");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
            //viewer.addPointCloud(rotatedCloud3, color_handler4, "cloud4");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
            //viewer.addPointCloud(cloud, color_handler5, "cloud5");
            //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
            //viewer.addCoordinateSystem(5);
            //viewer.setBackgroundColor(1.0, 1.0, 1.0);
            //viewer.spin();
            //system("pause");


        }



        auto minElement = std::min_element(dist_list.begin(), dist_list.end());
        int index = std::distance(dist_list.begin(), minElement);
        cout << "///////smallest dist: " << dist_list[index] << endl;
        transformed_show = transformation_cloud_list[index];
        dist_list.erase(dist_list.begin() + index);
        transformation_cloud_list.erase(transformation_cloud_list.begin() + index);

        auto minElement2 = std::min_element(dist_list.begin(), dist_list.end());
        int index2 = std::distance(dist_list.begin(), minElement2);
        transformed_show2 = transformation_cloud_list[index2];
        cout << "//////second smallest dist: " << dist_list[index2] << endl;




        compare_cloud_list.push_back(transformed_show);
        //compare_cloud_list.push_back(transformed_show2);


        //transformed_show2 = transformation_cloud_list[index];


        //pcl::visualization::PCLVisualizer viewer;
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(transformed_show, 255, 0, 0);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
        //
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

        //viewer.addPointCloud(transformed_show, color_handler, "cloud");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
        //viewer.addPointCloud(cloud, color_handler2, "cloud2");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
        //viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");
        //
        //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
        ////viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
        ////viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
        //viewer.addCoordinateSystem(5);
        //viewer.setBackgroundColor(1.0, 1.0, 1.0);
        //viewer.spin();
        //system("pause");








        dist_list.clear();
        nearest_point->clear();

    }
    int K = 1;
    double total_dist = 0;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> pointIdxKNNSearch(K);
    std::vector<float> pointKNNSquaredDistance(K);
    pcl::PointXYZ searchPoint2;
    pcl::PointXYZ searchPoint2_;
    cout << "Compare amount: " << compare_cloud_list.size() << endl;
    for (int i = 0; i < compare_cloud_list.size(); i++)
    {
        kdtree.setInputCloud(cloud_pass_through);
        for (int j = 0; j < compare_cloud_list[i]->size(); j++)
        {
            searchPoint2.x = compare_cloud_list[i]->points[j].x;
            searchPoint2.y = compare_cloud_list[i]->points[j].y;
            searchPoint2.z = compare_cloud_list[i]->points[j].z;

            if (kdtree.nearestKSearch(searchPoint2, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
            {
                //cout << "pointKNNSquaredDistance[1]: " << pointKNNSquaredDistance[0] << endl;
                total_dist += pointKNNSquaredDistance[0];
                //cout << "Nearest dist: " << sqrt(pointKNNSquaredDistance[0]) << " ";
                if (sqrt(pointKNNSquaredDistance[0]) < 0.4)//10
                {
                    nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);
                }
                //nearest_point->push_back((*cloud_pass_through)[pointIdxKNNSearch[0]]);

            }
        }
        near_list.push_back(nearest_point->size());
        cout << "Near point coounts: " << nearest_point->size() << endl;
    }

    auto max = std::max_element(near_list.begin(), near_list.end());
    int index_near = std::distance(near_list.begin(), max);
    end2 = clock();
    cpu_time = ((double)(end2 - start2)) / CLOCKS_PER_SEC;
    printf("Time = %f\n", cpu_time);

    cout << "Nearest points size: " << near_list[index_near] << endl;
    cout << "index: " << index_near << endl;

    pcl::visualization::PCLVisualizer viewer;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(compare_cloud_list[index_near], 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(regions_[0], 0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler4(nearest_cloud_list[index_near], 0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler5(regions_[1], 0, 0, 255);

    viewer.addPointCloud(compare_cloud_list[index_near], color_handler, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer.addPointCloud(cloud, color_handler2, "cloud2");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
    viewer.addPointCloud(regions_[0], color_handler3, "cloud3");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud3");
    viewer.addPointCloud(nearest_cloud_list[index_near], color_handler4, "cloud4");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "cloud4");
    viewer.addPointCloud(regions_[1], color_handler5, "cloud5");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud5");
    viewer.addCoordinateSystem(5);
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
        //viewer.saveScreenshot(filename);
    }


    regions_.clear();
}
void pca_match::region_growing_oil()
{
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud_pass_through);
    normal_estimator.setKSearch(9);
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);//70
    reg.setMaxClusterSize(3000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(20);
    reg.setInputCloud(cloud_pass_through);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(8 / 180.0 * M_PI);// for stack.ply set to 3/180//for 15_pile_up set to 8/180 with noise 5/180
    reg.setCurvatureThreshold(1.2);//1.0
    std::vector <pcl::PointIndices> clusters;

    reg.extract(clusters);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_pass_through);

    //cout << "OKKKK" << endl;

    // For every cluster...
    int currentClusterNum = 1;
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
    {
        //添加所有的点云到一个新的点云中
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
            cluster->points.push_back(cloud_pass_through->points[*point]);
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        // 保存
        if (cluster->points.size() <= 0)
            break;
        //std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
        //std::string fileName = "cluster" + boost::to_string(currentClusterNum) + ".pcd";
        //pcl::io::savePCDFileASCII(fileName, *cluster);
        regions.push_back(cluster);// region in class private

        currentClusterNum++;
    }

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    pcl::visualization::CloudViewer viewer("Cluster viewer");
    viewer.showCloud(colored_cloud);

    while (!viewer.wasStopped())
    {
    }
}