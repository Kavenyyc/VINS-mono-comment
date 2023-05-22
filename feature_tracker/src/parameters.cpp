#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");//ans-->config_file
    std::cout<<"config_file"<<config_file<<endl;//自己加入的，用于查看config_file
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;//image_topic："/cam0/image_raw"
    fsSettings["imu_topic"] >> IMU_TOPIC;//imu_topic: "/imu0"
    MAX_CNT = fsSettings["max_cnt"];//tracking最大特征点数量为150
    MIN_DIST = fsSettings["min_dist"];//两个特征点之间的最小距离30
    ROW = fsSettings["image_height"];//图像高度为480
    COL = fsSettings["image_width"];//图像宽度为752
    FREQ = fsSettings["freq"];//publish tracking result 的频率，默认值为10Hz
    F_THRESHOLD = fsSettings["F_threshold"];//RANSAC threshold (pixel)数值为1
    SHOW_TRACK = fsSettings["show_track"];//发布image topic 数值为1
    EQUALIZE = fsSettings["equalize"];//如果图像太暗或者太亮，打开equalize来找到足够的特征，默认为1
    FISHEYE = fsSettings["fisheye"];//默认不使用fisheye， 默认值为0
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);//string类型的config_file push到CAM_NAMES vector中

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;//如果是双目相机，将这个参数改为true
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
