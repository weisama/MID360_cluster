#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <nlohmann/json.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <fstream>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cmath>

using namespace std::chrono_literals;

// 圆柱跟踪数据结构
struct TrackedCylinder {
    int id;
    Eigen::Vector3f position;
    int consecutive_detections;  // 连续检测到的帧数
    int consecutive_misses;      // 连续未检测到的帧数
    rclcpp::Time last_seen;
    
    // 默认构造函数
    TrackedCylinder() 
        : id(-1), position(Eigen::Vector3f::Zero()), consecutive_detections(0), 
          consecutive_misses(0), last_seen(rclcpp::Clock().now()) {}
    
    // 带参数的构造函数
    TrackedCylinder(int cylinder_id, const Eigen::Vector3f& pos) 
        : id(cylinder_id), position(pos), consecutive_detections(1), 
          consecutive_misses(0), last_seen(rclcpp::Clock().now()) {}
};

class RingClusterNode : public rclcpp::Node
{
public:
    RingClusterNode() : Node("ring_cluster_node")
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
        
        target_diameter_ = config_["ring_parameters"]["target_diameter"];
        ring_radius_ = target_diameter_ / 2.0; // 60mm
        cylinder_height_ = 0.03; // 30mm高度
        
        // 跟踪参数
        tracking_threshold_ = 0.2; // 跟踪匹配阈值（米）
        max_consecutive_misses_ = 3; // 最大连续丢失帧数

        // 打印参数
        print_parameters();

        // 创建订阅者和发布者
        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            point_cloud_topic_, 10,
            std::bind(&RingClusterNode::point_cloud_callback, this, std::placeholders::_1));

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/cylinder_markers", 10);

        center_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/cylinder_centers", 10);

        // 初始化ID计数器
        next_cylinder_id_ = 0;

        RCLCPP_INFO(this->get_logger(), "圆柱跟踪节点已启动，点云频率: 10Hz");
    }

private:
    void print_parameters()
    {
        RCLCPP_INFO(this->get_logger(), "=== 圆柱检测参数 ===");
        RCLCPP_INFO(this->get_logger(), "强度阈值: %d", intensity_threshold_);
        RCLCPP_INFO(this->get_logger(), "点云话题: %s", point_cloud_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "坐标系: %s", frame_id_.c_str());
        RCLCPP_INFO(this->get_logger(), "DBSCAN eps: %.3f", eps_);
        RCLCPP_INFO(this->get_logger(), "DBSCAN 最小点数: %d", min_points_);
        RCLCPP_INFO(this->get_logger(), "目标直径: %.3f m", target_diameter_);
        RCLCPP_INFO(this->get_logger(), "圆柱半径: %.3f m", ring_radius_);
        RCLCPP_INFO(this->get_logger(), "圆柱高度: %.3f m", cylinder_height_);
        RCLCPP_INFO(this->get_logger(), "跟踪阈值: %.3f m", tracking_threshold_);
        RCLCPP_INFO(this->get_logger(), "最大连续丢失帧数: %d", max_consecutive_misses_);
    }

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        auto start_time = this->now();
        
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
            RCLCPP_DEBUG(this->get_logger(), "过滤后的点云为空");
            // 即使没有检测到圆柱，也要更新跟踪状态
            update_tracking(std::vector<Eigen::Vector3f>(), msg->header);
            return;
        }

        RCLCPP_DEBUG(this->get_logger(), "过滤后点云点数: %zu", filtered_cloud->points.size());

        // 创建KD树用于聚类
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(filtered_cloud);

        // 执行欧几里得聚类
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(eps_);
        ec.setMinClusterSize(min_points_);
        ec.setMaxClusterSize(max_points_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(filtered_cloud);
        ec.extract(cluster_indices);

        RCLCPP_DEBUG(this->get_logger(), "检测到 %zu 个聚类", cluster_indices.size());

        // 处理聚类结果
        std::vector<Eigen::Vector3f> detected_cylinders = process_clusters(filtered_cloud, cluster_indices);
        
        // 更新跟踪状态
        update_tracking(detected_cylinders, msg->header);
        
        auto end_time = this->now();
        auto processing_time = (end_time - start_time).seconds() * 1000.0; // 转换为毫秒
        RCLCPP_DEBUG(this->get_logger(), "点云处理时间: %.2f ms", processing_time);
    }

    std::vector<Eigen::Vector3f> process_clusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                         const std::vector<pcl::PointIndices>& cluster_indices)
    {
        std::vector<Eigen::Vector3f> detected_cylinders;

        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            // 提取单个聚类
            pcl::PointCloud<pcl::PointXYZI> single_cluster;
            for (const auto& idx : cluster_indices[i].indices) {
                single_cluster.points.push_back(cloud->points[idx]);
            }

            // 计算聚类质点
            Eigen::Vector4f cluster_centroid;
            if (pcl::compute3DCentroid(single_cluster, cluster_centroid) == 0) {
                RCLCPP_WARN(this->get_logger(), "无法计算聚类 %zu 的质心", i);
                continue;
            }

            // 计算圆柱中心（从雷达原点指向质点，在XY平面上延长60mm）
            Eigen::Vector3f lidar_origin(0, 0, 0); // 雷达坐标系原点
            Eigen::Vector3f centroid_vec(cluster_centroid[0], cluster_centroid[1], cluster_centroid[2]);
            
            // 计算从雷达指向质点的方向向量（只在XY平面）
            Eigen::Vector3f direction = centroid_vec - lidar_origin;
            direction.z() = 0; // 只在XY平面
            double distance = direction.norm();
            if (distance > 0) {
                direction.normalize();
            } else {
                // 如果质点在原点，使用默认方向
                direction = Eigen::Vector3f(1, 0, 0);
            }
            
            // 圆柱中心 = 质点 + 方向向量 * 圆柱半径（只在XY平面）
            Eigen::Vector3f cylinder_center = centroid_vec + direction * ring_radius_;
            cylinder_center.z() = cluster_centroid[2]; // 保持原始Z坐标

            detected_cylinders.push_back(cylinder_center);
            
            RCLCPP_DEBUG(this->get_logger(), 
                "检测到圆柱 - 中心: (%.3f, %.3f, %.3f), 距离: %.3f m", 
                cylinder_center[0], cylinder_center[1], cylinder_center[2], distance);
        }

        return detected_cylinders;
    }

    void update_tracking(const std::vector<Eigen::Vector3f>& detected_cylinders, 
                        const std_msgs::msg::Header& header)
    {
        auto current_time = this->now();
        
        // 第一步：更新已跟踪圆柱的状态
        for (auto& cylinder_pair : tracked_cylinders_) {
            auto& cylinder = cylinder_pair.second;
            cylinder.consecutive_misses++; // 先假设这一帧没检测到
        }
        
        // 第二步：匹配检测到的圆柱与已跟踪的圆柱
        std::vector<bool> matched_detections(detected_cylinders.size(), false);
        
        for (auto& cylinder_pair : tracked_cylinders_) {
            auto& cylinder = cylinder_pair.second;
            float min_distance = std::numeric_limits<float>::max();
            int best_match_idx = -1;
            
            // 寻找最近的检测圆柱
            for (size_t i = 0; i < detected_cylinders.size(); ++i) {
                if (matched_detections[i]) continue; // 已经匹配过的跳过
                
                float distance = (cylinder.position - detected_cylinders[i]).norm();
                if (distance < tracking_threshold_ && distance < min_distance) {
                    min_distance = distance;
                    best_match_idx = i;
                }
            }
            
            // 如果找到匹配
            if (best_match_idx != -1) {
                // 更新圆柱位置
                cylinder.position = detected_cylinders[best_match_idx];
                cylinder.consecutive_detections++;
                cylinder.consecutive_misses = 0; // 重置连续丢失计数
                cylinder.last_seen = current_time;
                
                matched_detections[best_match_idx] = true;
                
                RCLCPP_DEBUG(this->get_logger(), 
                    "更新圆柱 %d - 位置: (%.3f, %.3f, %.3f)", 
                    cylinder.id, cylinder.position[0], cylinder.position[1], cylinder.position[2]);
            }
        }
        
        // 第三步：为未匹配的检测创建新的跟踪圆柱
        for (size_t i = 0; i < detected_cylinders.size(); ++i) {
            if (!matched_detections[i]) {
                int new_id = next_cylinder_id_++;
                tracked_cylinders_.emplace(new_id, TrackedCylinder(new_id, detected_cylinders[i]));
                
                RCLCPP_INFO(this->get_logger(), 
                    "创建新圆柱 %d - 位置: (%.3f, %.3f, %.3f)", 
                    new_id, detected_cylinders[i][0], detected_cylinders[i][1], detected_cylinders[i][2]);
            }
        }
        
        // 第四步：移除连续丢失超过阈值的圆柱
        std::vector<int> cylinders_to_remove;
        for (const auto& cylinder_pair : tracked_cylinders_) {
            if (cylinder_pair.second.consecutive_misses > max_consecutive_misses_) {
                cylinders_to_remove.push_back(cylinder_pair.first);
                RCLCPP_INFO(this->get_logger(), "移除圆柱 %d - 连续丢失 %d 帧", 
                           cylinder_pair.first, cylinder_pair.second.consecutive_misses);
            }
        }
        
        for (int id : cylinders_to_remove) {
            tracked_cylinders_.erase(id);
        }
        
        // 第五步：发布所有活跃的圆柱标记
        publish_cylinders(header);
        
        RCLCPP_DEBUG(this->get_logger(), "活跃圆柱数量: %zu", tracked_cylinders_.size());
    }

    void publish_cylinders(const std_msgs::msg::Header& header)
    {
        auto marker_array = std::make_shared<visualization_msgs::msg::MarkerArray>();
        
        // 发布所有活跃的圆柱
        for (const auto& cylinder_pair : tracked_cylinders_) {
            const auto& cylinder = cylinder_pair.second;
            auto cylinder_marker = create_cylinder_marker(cylinder.position, cylinder.id, header);
            marker_array->markers.push_back(cylinder_marker);
            
            // 发布圆柱中心坐标
            publish_center_point(cylinder.position, header);
        }
        
        // 发布标记
        if (!marker_array->markers.empty()) {
            marker_pub_->publish(*marker_array);
        }
    }

    visualization_msgs::msg::Marker create_cylinder_marker(const Eigen::Vector3f& center, 
                                                          int id,
                                                          const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.header.stamp = this->now();
        marker.ns = "tracked_cylinders";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = center[0];
        marker.pose.position.y = center[1];
        marker.pose.position.z = center[2];
        marker.pose.orientation.w = 1.0;
        
        // 设置圆柱体尺寸：直径120mm，高度30mm
        marker.scale.x = target_diameter_;
        marker.scale.y = target_diameter_;
        marker.scale.z = cylinder_height_;
        
        // 根据跟踪状态设置颜色
        // 新检测的为红色，稳定跟踪的为蓝色
        auto it = tracked_cylinders_.find(id);
        if (it != tracked_cylinders_.end() && it->second.consecutive_detections > 3) {
            // 稳定跟踪 - 蓝色
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        } else {
            // 新检测 - 红色
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        marker.color.a = 0.7;
        
        // 设置较长的生存时间，确保不会过早消失
        marker.lifetime = rclcpp::Duration(500ms);
        
        return marker;
    }

    void publish_center_point(const Eigen::Vector3f& center, 
                             const std_msgs::msg::Header& header)
    {
        auto point_msg = std::make_shared<geometry_msgs::msg::PointStamped>();
        point_msg->header = header;
        point_msg->header.stamp = this->now();
        point_msg->point.x = center[0];
        point_msg->point.y = center[1];
        point_msg->point.z = center[2];
        
        center_point_pub_->publish(*point_msg);
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
    double ring_radius_;
    double cylinder_height_;
    
    // 跟踪参数
    float tracking_threshold_;
    int max_consecutive_misses_;
    int next_cylinder_id_;
    
    // 跟踪状态
    std::unordered_map<int, TrackedCylinder> tracked_cylinders_;

    // ROS2组件
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr center_point_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RingClusterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
