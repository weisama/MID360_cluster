#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace std::chrono_literals;

class PointCloudFilter : public rclcpp::Node
{
public:
    PointCloudFilter() : Node("point_cloud_filter")
    {
        // 读取JSON配置文件
        this->load_configuration();
        
        // 创建订阅者和发布者
        subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, 10,
            std::bind(&PointCloudFilter::pointcloud_callback, this, std::placeholders::_1));
            
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_, 10);
            
        RCLCPP_INFO(this->get_logger(), "点云滤波器已启动");
        RCLCPP_INFO(this->get_logger(), "强度阈值: %f", intensity_threshold_);
        RCLCPP_INFO(this->get_logger(), "输入话题: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "输出话题: %s", output_topic_.c_str());
    }

private:
    void load_configuration()
    {
        std::string json_path = std::string(getenv("HOME")) + "/lidar.json";
        std::ifstream json_file(json_path);
        
        if (!json_file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "无法打开配置文件: %s", json_path.c_str());
            // 使用默认值
            intensity_threshold_ = 0.3;
            input_topic_ = "/livox/lidar";
            output_topic_ = "/filtered_pointcloud";
            return;
        }
        
        try {
            nlohmann::json config = nlohmann::json::parse(json_file);
            
            intensity_threshold_ = config.value("intensity_threshold", 0.3);
            input_topic_ = config["node_settings"].value("livox_ros_driver2_topic", "/livox/lidar");
            output_topic_ = config["node_settings"].value("point_cloud_filter_topic", "/filtered_pointcloud");
            frame_id_ = config["node_settings"].value("frame_id", "livox_frame");
            
            RCLCPP_INFO(this->get_logger(), "配置文件加载成功: %s", json_path.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "配置文件解析错误: %s", e.what());
            // 使用默认值
            intensity_threshold_ = 0.3;
            input_topic_ = "/livox/lidar";
            output_topic_ = "/filtered_pointcloud";
            frame_id_ = "livox_frame";
        }
    }
    
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // 将ROS2点云消息转换为PCL点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // 创建过滤后的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        
        // 过滤强度低于阈值的点:cite[1]:cite[6]
        for (const auto& point : cloud->points) {
            if (point.intensity >= intensity_threshold_) {
                filtered_cloud->points.push_back(point);
            }
        }
        
        // 保持点云的宽度、高度和头信息
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->header = cloud->header;
        
        // 将PCL点云转换回ROS2消息
        sensor_msgs::msg::PointCloud2 filtered_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_msg);
        
        // 保持原始坐标系:cite[1]
        filtered_msg.header.frame_id = frame_id_;
        filtered_msg.header.stamp = this->now();
        
        // 发布过滤后的点云
        publisher_->publish(filtered_msg);
        
        RCLCPP_DEBUG(this->get_logger(), "点云过滤完成: 原始点数=%lu, 过滤后点数=%lu", 
                     cloud->points.size(), filtered_cloud->points.size());
    }
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    
    double intensity_threshold_;
    std::string input_topic_;
    std::string output_topic_;
    std::string frame_id_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudFilter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
