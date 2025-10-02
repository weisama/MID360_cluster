// ball_cluster_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <Eigen/Dense>

using namespace std::chrono_literals;

// ----- 卡尔曼滤波：常加速度 (CA) 模型 (9维状态) -----
class KalmanFilterCA {
private:
    Eigen::VectorXd x;
    Eigen::MatrixXd P, F, H, Q, R;
    bool initialized_ = false;

public:
    KalmanFilterCA(double dt = 0.1, double process_noise = 0.1, double measurement_noise = 0.1) {
        x = Eigen::VectorXd::Zero(9);
        P = Eigen::MatrixXd::Identity(9,9)*0.1;
        F = Eigen::MatrixXd::Identity(9,9);
        setDt(dt);
        H = Eigen::MatrixXd::Zero(3,9);
        H(0,0)=1; H(1,1)=1; H(2,2)=1;
        Q = Eigen::MatrixXd::Identity(9,9)*process_noise;
        R = Eigen::MatrixXd::Identity(3,3)*measurement_noise;
    }

    void setDt(double dt) {
        F.setIdentity();
        F(0,3)=dt; F(0,6)=0.5*dt*dt;
        F(1,4)=dt; F(1,7)=0.5*dt*dt;
        F(2,5)=dt; F(2,8)=0.5*dt*dt;
        F(3,6)=dt; F(4,7)=dt; F(5,8)=dt;
    }

    void initialize(const Eigen::Vector3f& pos) {
        x.setZero();
        x(0)=pos[0]; x(1)=pos[1]; x(2)=pos[2];
        P = Eigen::MatrixXd::Identity(9,9)*0.1;
        initialized_ = true;
    }

    void predict(double dt) {
        if(!initialized_) return;
        setDt(dt);
        x = F*x;
        P = F*P*F.transpose() + Q;
    }

    void update(const Eigen::Vector3f& measurement) {
        if(!initialized_) { initialize(measurement); return; }
        Eigen::Vector3d z; z << measurement[0], measurement[1], measurement[2];
        Eigen::MatrixXd S = H*P*H.transpose() + R;
        Eigen::MatrixXd K = P*H.transpose()*S.inverse();
        x = x + K*(z - H*x);
        P = (Eigen::MatrixXd::Identity(9,9)-K*H)*P;
    }

    Eigen::Vector3f getPosition() const { return Eigen::Vector3f(x(0),x(1),x(2)); }
    Eigen::Vector3f getVelocity() const { return Eigen::Vector3f(x(3),x(4),x(5)); }
    Eigen::Vector3f getAcceleration() const { return Eigen::Vector3f(x(6),x(7),x(8)); }
    bool isInitialized() const { return initialized_; }
};

// ----- 跟踪圆柱结构 -----
struct TrackedCylinder {
    int id;
    Eigen::Vector3f position;
    Eigen::Vector3f filtered_position;
    Eigen::Vector3f velocity;
    int consecutive_detections;
    int consecutive_misses;
    rclcpp::Time last_seen;
    rclcpp::Time last_update;
    KalmanFilterCA kf;
    std::deque<std::pair<Eigen::Vector3f, rclcpp::Time>> history; // 轨迹 + 时间戳

    double history_duration_s = 3.0; // 保留3秒轨迹

    TrackedCylinder(int cylinder_id, const Eigen::Vector3f& pos, const rclcpp::Time& now,
                    double kf_dt, double process_noise, double meas_noise)
        : id(cylinder_id), position(pos), filtered_position(pos), velocity(Eigen::Vector3f::Zero()),
          consecutive_detections(1), consecutive_misses(0), last_seen(now), last_update(now),
          kf(kf_dt, process_noise, meas_noise)
    {
        kf.initialize(pos);
        history.push_back({pos, now});
    }

    void updateKF(const Eigen::Vector3f& measurement, const rclcpp::Time& now) {
        double dt = std::max(1e-6, (now - last_update).seconds());
        kf.predict(dt);
        kf.update(measurement);
        filtered_position = kf.getPosition();
        velocity = kf.getVelocity();
        position = measurement;
        last_update = now;
        last_seen = now;

        history.push_back({filtered_position, now});
        cleanHistory(now);
    }

    void predictKF(const rclcpp::Time& now) {
        double dt = std::max(1e-6, (now - last_update).seconds());
        kf.predict(dt);
        filtered_position = kf.getPosition();
        velocity = kf.getVelocity();
        last_update = now;
        cleanHistory(now);
    }

    void cleanHistory(const rclcpp::Time& now) {
        while(!history.empty() && (now - history.front().second).seconds() > history_duration_s) {
            history.pop_front();
        }
    }
};

// ----- 主节点 -----
class RingClusterNode : public rclcpp::Node {
public:
    RingClusterNode() : Node("ring_cluster_node") {
        std::string home_dir = std::getenv("HOME") ? std::getenv("HOME") : ".";
        std::string config_file = home_dir + "/lidar.json";
        try {
            std::ifstream f(config_file);
            config_ = nlohmann::json::parse(f);
            RCLCPP_INFO(this->get_logger(), "加载配置: %s", config_file.c_str());
        } catch(const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "无法加载配置 %s: %s, 使用默认参数", config_file.c_str(), e.what());
        }

        intensity_threshold_ = config_.value("intensity_threshold",100.0);
        point_cloud_topic_ = config_.value("node_settings",nlohmann::json::object()).value("point_cloud_filter_topic",std::string("/points"));
        frame_id_ = config_.value("node_settings",nlohmann::json::object()).value("frame_id",std::string("map"));
        eps_ = config_.value("dbscan_parameters",nlohmann::json::object()).value("eps",0.05);
        min_points_ = config_.value("dbscan_parameters",nlohmann::json::object()).value("min_points",5);
        max_points_ = config_.value("dbscan_parameters",nlohmann::json::object()).value("max_points",10000);
        target_diameter_ = config_.value("ring_parameters",nlohmann::json::object()).value("target_diameter",0.12);
        ring_radius_ = target_diameter_/2.0;
        cylinder_height_ = config_.value("ring_parameters",nlohmann::json::object()).value("cylinder_height",0.03);

        kf_process_noise_ = config_.value("kalman_filter",nlohmann::json::object()).value("process_noise",0.1);
        kf_measurement_noise_ = config_.value("kalman_filter",nlohmann::json::object()).value("measurement_noise",0.1);
        use_kalman_filter_ = config_.value("kalman_filter",nlohmann::json::object()).value("enabled",true);
        prediction_enabled_ = config_.value("kalman_filter",nlohmann::json::object()).value("prediction_enabled",true);
        kf_dt_default_ = config_.value("kalman_filter",nlohmann::json::object()).value("default_dt",0.1);

        tracking_threshold_ = config_.value("tracking_parameters",nlohmann::json::object()).value("tracking_threshold",0.2f);
        max_consecutive_misses_ = config_.value("tracking_parameters",nlohmann::json::object()).value("max_consecutive_misses",3);
        min_detections_for_stable_ = config_.value("tracking_parameters",nlohmann::json::object()).value("min_detections_for_stable",3);

        point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            point_cloud_topic_, rclcpp::QoS(10),
            std::bind(&RingClusterNode::point_cloud_callback,this,std::placeholders::_1));

        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/cylinder_markers",10);
        center_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/cylinder_centers",10);
        filtered_center_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/filtered_cylinder_centers",10);

        next_cylinder_id_ = 0;
        RCLCPP_INFO(this->get_logger(),"节点启动: %s",this->get_name());
    }

private:
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto start_time = this->now();
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg,*cloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        filtered_cloud->points.reserve(cloud->points.size());
        std::copy_if(cloud->points.begin(),cloud->points.end(),std::back_inserter(filtered_cloud->points),
                     [&](const pcl::PointXYZI& p){ return p.intensity>intensity_threshold_; });
        filtered_cloud->width = static_cast<uint32_t>(filtered_cloud->points.size());
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = cloud->is_dense;

        if(filtered_cloud->points.empty()) {
            update_tracking(std::vector<Eigen::Vector3f>(),msg->header);
            return;
        }

        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(filtered_cloud);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(eps_);
        ec.setMinClusterSize(min_points_);
        ec.setMaxClusterSize(max_points_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(filtered_cloud);
        ec.extract(cluster_indices);

        std::vector<Eigen::Vector3f> detected_cylinders;
        detected_cylinders.reserve(cluster_indices.size());
        for(const auto& ci : cluster_indices){
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*filtered_cloud,ci.indices,centroid);
            Eigen::Vector3f centroid_vec(centroid[0],centroid[1],centroid[2]);
            Eigen::Vector3f dir = centroid_vec; dir.z()=0;
            if(dir.norm()>1e-6) dir.normalize(); else dir=Eigen::Vector3f(1,0,0);
            Eigen::Vector3f cylinder_center = centroid_vec + dir*static_cast<float>(ring_radius_);
            cylinder_center.z() = centroid[2];
            detected_cylinders.push_back(cylinder_center);
        }

        update_tracking(detected_cylinders,msg->header);
    }

    void update_tracking(const std::vector<Eigen::Vector3f>& detected_cylinders,const std_msgs::msg::Header& header){
        auto current_time = this->now();

        for(auto& kv: tracked_cylinders_) {
            auto &c = kv.second;
            if(use_kalman_filter_ && prediction_enabled_) c.predictKF(current_time);
            c.consecutive_misses++;
        }

        std::vector<bool> matched(detected_cylinders.size(),false);
        for(auto& kv: tracked_cylinders_) {
            auto &c = kv.second;
            float best_dist = std::numeric_limits<float>::max();
            int best_idx=-1;
            Eigen::Vector3f ref = use_kalman_filter_ ? c.filtered_position : c.position;
            for(size_t i=0;i<detected_cylinders.size();++i){
                if(matched[i]) continue;
                float d = (ref - detected_cylinders[i]).norm();
                if(d<best_dist && d<tracking_threshold_){ best_dist=d; best_idx=i; }
            }
            if(best_idx!=-1){
                if(use_kalman_filter_) c.updateKF(detected_cylinders[best_idx],current_time);
                else {
                    c.position = detected_cylinders[best_idx];
                    c.filtered_position = c.position;
                    c.last_update = current_time;
                    c.last_seen = current_time;
                    c.history.push_back({c.filtered_position,current_time});
                    c.cleanHistory(current_time);
                }
                c.consecutive_detections++;
                c.consecutive_misses=0;
                matched[best_idx]=true;
            }
        }

        for(size_t i=0;i<detected_cylinders.size();++i){
            if(!matched[i]){
                int id = next_cylinder_id_++;
                tracked_cylinders_.emplace(id,TrackedCylinder(id,detected_cylinders[i],current_time,kf_dt_default_,kf_process_noise_,kf_measurement_noise_));
            }
        }

        std::vector<int> to_remove;
        for(const auto& kv: tracked_cylinders_){
            if(kv.second.consecutive_misses>max_consecutive_misses_) to_remove.push_back(kv.first);
        }
        for(int id: to_remove) tracked_cylinders_.erase(id);

        publish_cylinders(header);
    }

    void set_marker_color(visualization_msgs::msg::Marker& marker,bool stable,bool use_kf){
        if(stable){ marker.color.r=0.0f; marker.color.g=use_kf?0.3f:0.0f; marker.color.b=1.0f; }
        else { marker.color.r=1.0f; marker.color.g=use_kf?0.3f:0.0f; marker.color.b=0.0f; }
        marker.color.a=0.8f;
    }

    void publish_cylinders(const std_msgs::msg::Header& header){
        visualization_msgs::msg::MarkerArray markers;
        for(const auto& kv: tracked_cylinders_){
            const auto &c = kv.second;
            Eigen::Vector3f disp = use_kalman_filter_ ? c.filtered_position : c.position;

            visualization_msgs::msg::Marker cyl_marker;
            cyl_marker.header = header;
            cyl_marker.header.stamp = this->now();
            cyl_marker.ns = "tracked_cylinders";
            cyl_marker.id = c.id;
            cyl_marker.type = visualization_msgs::msg::Marker::CYLINDER;
            cyl_marker.action = visualization_msgs::msg::Marker::ADD;
            cyl_marker.pose.position.x = disp.x();
            cyl_marker.pose.position.y = disp.y();
            cyl_marker.pose.position.z = disp.z();
            cyl_marker.pose.orientation.w = 1.0;
            cyl_marker.scale.x = target_diameter_;
            cyl_marker.scale.y = target_diameter_;
            cyl_marker.scale.z = cylinder_height_;
            bool stable = c.consecutive_detections >= min_detections_for_stable_;
            set_marker_color(cyl_marker,stable,use_kalman_filter_);
            cyl_marker.lifetime = rclcpp::Duration(500ms);
            markers.markers.push_back(cyl_marker);

            if(!c.history.empty()){
                visualization_msgs::msg::Marker traj;
                traj.header = header;
                traj.header.stamp = this->now();
                traj.ns = "tracked_cylinders_traj";
                traj.id = c.id + 100000;
                traj.type = visualization_msgs::msg::Marker::LINE_STRIP;
                traj.action = visualization_msgs::msg::Marker::ADD;
                traj.scale.x = 0.01;
                traj.pose.orientation.w=1.0;
                traj.color.r=0.0f; traj.color.g=1.0f; traj.color.b=0.0f; traj.color.a=0.6f;
                traj.lifetime = rclcpp::Duration(500ms);

                for(const auto &pt_pair: c.history){
                    geometry_msgs::msg::Point gp;
                    gp.x=pt_pair.first.x(); gp.y=pt_pair.first.y(); gp.z=pt_pair.first.z();
                    traj.points.push_back(gp);
                }
                markers.markers.push_back(traj);
            }

            publish_center_point(c.position,header,center_point_pub_);
            if(use_kalman_filter_) publish_center_point(c.filtered_position,header,filtered_center_point_pub_);
        }

        if(!markers.markers.empty()) marker_pub_->publish(markers);
    }

    void publish_center_point(const Eigen::Vector3f& center,const std_msgs::msg::Header& header,
                              const rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr& pub){
        geometry_msgs::msg::PointStamped p;
        p.header=header;
        p.header.stamp=this->now();
        p.point.x=center.x(); p.point.y=center.y(); p.point.z=center.z();
        pub->publish(p);
    }

    nlohmann::json config_;
    double intensity_threshold_=100.0;
    std::string point_cloud_topic_="/points";
    std::string frame_id_="map";
    double eps_=0.05;
    int min_points_=5;
    int max_points_=10000;
    double target_diameter_=0.12;
    double ring_radius_=0.06;
    double cylinder_height_=0.03;

    double kf_process_noise_=0.1;
    double kf_measurement_noise_=0.1;
    bool use_kalman_filter_=true;
    bool prediction_enabled_=true;
    double kf_dt_default_=0.1;

    float tracking_threshold_=0.2f;
    int max_consecutive_misses_=3;
    int min_detections_for_stable_=3;
    int next_cylinder_id_=0;

    std::unordered_map<int,TrackedCylinder> tracked_cylinders_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr center_point_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr filtered_center_point_pub_;
};

int main(int argc, char** argv){
    rclcpp::init(argc,argv);
    auto node = std::make_shared<RingClusterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

