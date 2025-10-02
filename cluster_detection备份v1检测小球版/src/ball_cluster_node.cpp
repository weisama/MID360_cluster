#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>               // compute3DCentroid
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <nlohmann/json.hpp>   // 单头文件版
// ROS ⇄ PCL 转换
#include <pcl_conversions/pcl_conversions.h>

using namespace std::chrono_literals;

class BallClusterNode : public rclcpp::Node
{
public:
    BallClusterNode() : Node("ball_cluster_node")
    {
        // 读取JSON配置文件
        std::string home_dir = std::getenv("HOME");
        std::string config_file = home_dir + "/lidar.json";
        
        try {
            std::ifstream f(config_file);
            config_ = nlohmann::json::parse(f);
            RCLCPP_INFO(this->get_logger(), "成功加载配置文件: %s", config_file.c_str());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "无法加载配置文件: %s", e.what());
            return;
        }

        // 获取参数
        intensity_threshold_ = config_["intensity_threshold"];
        point_cloud_topic_ = config_["node_settings"]["point_cloud_filter_topic"];
        frame_id_ = config_["node_settings"]["frame_id"];
        
        eps_ = config_["dbscan_parameters"]["eps"];
        min_points_ = config_["dbscan_parameters"]["min_points"];
        max_points_ = config_["dbscan_parameters"]["max_points"];
        
        target_diameter_ = config_["ball_parameters"]["target_diameter"];
        diameter_tolerance_ = config_["ball_parameters"]["diameter_tolerance"];
        min_cluster_size_ = config_["ball_parameters"]["min_cluster_size"];
        max_cluster_size_ = config_["ball_parameters"]["max_cluster_size"];

        // 打印参数
        print_parameters();

        // 创建订阅者和发布者
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            point_cloud_topic_, 10,
            std::bind(&BallClusterNode::point_cloud_callback, this, std::placeholders::_1));

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ball_markers", 10);

        cluster_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/clustered_balls", 10);

        RCLCPP_INFO(this->get_logger(), "Ball Cluster节点已启动");
    }

private:
    void print_parameters()
    {
        RCLCPP_INFO(this->get_logger(), "=== 聚类参数 ===");
        RCLCPP_INFO(this->get_logger(), "强度阈值: %d", intensity_threshold_);
        RCLCPP_INFO(this->get_logger(), "点云话题: %s", point_cloud_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "坐标系: %s", frame_id_.c_str());
        RCLCPP_INFO(this->get_logger(), "DBSCAN eps: %.3f", eps_);
        RCLCPP_INFO(this->get_logger(), "DBSCAN 最小点数: %d", min_points_);
        RCLCPP_INFO(this->get_logger(), "目标直径: %.3f m", target_diameter_);
        RCLCPP_INFO(this->get_logger(), "直径容差: %.3f m", diameter_tolerance_);
    }

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 转换点云格式
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);

        // 过滤强度
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto& point : cloud->points) {
            if (point.intensity > intensity_threshold_) {
                filtered_cloud->points.push_back(point);
            }
        }

        if (filtered_cloud->points.empty()) {
            return;
        }

        // 创建KD树用于聚类
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(filtered_cloud);

        // 执行欧几里得聚类（类似DBSCAN）
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(eps_);
        ec.setMinClusterSize(min_points_);
        ec.setMaxClusterSize(max_points_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(filtered_cloud);
        ec.extract(cluster_indices);

        // 处理聚类结果
        process_clusters(filtered_cloud, cluster_indices, msg->header);
    }

    void process_clusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                         const std::vector<pcl::PointIndices>& cluster_indices,
                         const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        pcl::PointCloud<pcl::PointXYZI> cluster_cloud;
        int ball_id = 0;

        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            // 提取单个聚类
            pcl::PointCloud<pcl::PointXYZI> single_cluster;
            for (const auto& idx : cluster_indices[i].indices) {
                single_cluster.points.push_back(cloud->points[idx]);
            }

            // 计算聚类尺寸
            pcl::PointXYZI min_pt, max_pt;
            pcl::getMinMax3D(single_cluster, min_pt, max_pt);
            
            double dx = max_pt.x - min_pt.x;
            double dy = max_pt.y - min_pt.y;
            double dz = max_pt.z - min_pt.z;
            double diameter = std::max({dx, dy, dz});

            // 检查是否为小球（直径15mm±容差）
            if (diameter >= (target_diameter_ - diameter_tolerance_) && 
                diameter <= (target_diameter_ + diameter_tolerance_) &&
                diameter >= min_cluster_size_ && diameter <= max_cluster_size_) {
                
                // 计算聚类中心
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(single_cluster, centroid);

                // 创建小球标记
                auto marker = create_ball_marker(centroid, diameter, ball_id, header);
                marker_array.markers.push_back(marker);

                // 将聚类点云添加到总点云中
                for (const auto& point : single_cluster.points) {
                    pcl::PointXYZI new_point;
                    new_point.x = point.x;
                    new_point.y = point.y;
                    new_point.z = point.z;
                    new_point.intensity = ball_id + 1;  // 用强度值表示球ID
                    cluster_cloud.points.push_back(new_point);
                }

                RCLCPP_INFO(this->get_logger(), "检测到小球 %d - 位置: (%.3f, %.3f, %.3f), 直径: %.3f m", 
                           ball_id, centroid[0], centroid[1], centroid[2], diameter);
                
                ball_id++;
            }
        }

        // 发布标记
        if (!marker_array.markers.empty()) {
            marker_pub_->publish(marker_array);
        }

        // 发布聚类点云
        if (!cluster_cloud.points.empty()) {
            sensor_msgs::msg::PointCloud2 cluster_msg;
            pcl::toROSMsg(cluster_cloud, cluster_msg);
            cluster_msg.header = header;
            cluster_cloud_pub_->publish(cluster_msg);
        }
    }

    visualization_msgs::msg::Marker create_ball_marker(const Eigen::Vector4f& centroid, 
                                                      double diameter, int id,
                                                      const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "balls";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = centroid[0];
        marker.pose.position.y = centroid[1];
        marker.pose.position.z = centroid[2];
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = diameter*3;
        marker.scale.y = diameter*3;
        marker.scale.z = diameter*3;
        
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        
        marker.lifetime = rclcpp::Duration(1s);
        
        return marker;
    }

    // 参数
    nlohmann::json config_;
    int intensity_threshold_;
    std::string point_cloud_topic_;
    std::string frame_id_;
    double eps_;
    int min_points_;
    int max_points_;
    double target_diameter_;
    double diameter_tolerance_;
    double min_cluster_size_;
    double max_cluster_size_;

    // ROS2组件
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_cloud_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallClusterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
