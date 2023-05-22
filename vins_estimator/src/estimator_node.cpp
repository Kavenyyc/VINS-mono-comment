#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf; // 订阅pose graph node发布的回环帧数据，存到relo_buf队列中，供重定位使用
int sum_of_wait = 0;

std::mutex m_buf; // 用于处理多个线程使用imu_buf和feature_buf的冲突
std::mutex m_state; // 用于处理多个线程使用当前里程计信息（即tmp_P、tmp_Q、tmp_V）的冲突
std::mutex i_buf;
std::mutex m_estimator; // 用于处理多个线程使用VINS系统对象（即Estimator类的实例estimator）的冲突

double latest_time; // 最近一次里程计信息对应的IMU时间戳
// 当前里程计信息
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;

// 当前里程计信息对应的IMU bias
Eigen::Vector3d tmp_Ba;//加速度计 bias
Eigen::Vector3d tmp_Bg;//陀螺仪 bias

// 当前里程计信息对应的IMU测量值
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

bool init_feature = 0;
bool init_imu = 1; // true：第一次接收IMU数据
double last_imu_t = 0; // 最近一帧IMU数据的时间戳


// 通过IMU测量值积分更新里程计信息
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    //1、判断是否第一帧imu_msg
    if (init_imu) // 第一次接收IMU数据
    {
        latest_time = t;
        init_imu = 0;
        return;//return使程序跳出if所在函数，返回到母函数中继续执行
    }
    //2、计算当前imu_msg距离上一个imu_msg的时间间隔
    double dt = t - latest_time;
    latest_time = t;

    //3、取x,y,z三个方向上的线加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    //4、获取x,y,z三个方向上的角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //5、tmp_P,tmp_V,tmp_Q更新，这一部分有公式推导，利用中值积分实现
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;//a理想值=R(a测量值-Ba)-g w理想值=w测量值-BW 此式为k时刻的加速度的模型

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;//中值法：取k时刻和k+1时刻的角速度再除2
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);//旋转的更新

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;//k+1时刻的线加速度

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    //6、迭代赋值
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 当处理完measurements中的所有数据后，如果VINS系统正常完成滑动窗口优化，那么需要用优化后的结果更新里程计数据
void update()
{
    TicToc t_predict;
    latest_time = current_time;

    // 首先获取滑动窗口中最新帧的P、V、Q
    tmp_P = estimator.Ps[WINDOW_SIZE];//vector3d类型
    tmp_Q = estimator.Rs[WINDOW_SIZE];//vector3d类型
    tmp_V = estimator.Vs[WINDOW_SIZE];//vector3d类型
    tmp_Ba = estimator.Bas[WINDOW_SIZE];//vector3d类型
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];//vector3d类型
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    // 滑动窗口中最新帧并不是当前帧，中间隔着缓存队列的数据，所以还需要使用缓存队列中的IMU数据进行积分得到当前帧的里程计信息
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())//pop是出队，队列先进先出
        predict(tmp_imu_buf.front());//front指队首指针

}


//sensor_msgs::PointCloudConstPtr 表示某一帧图像的feature_points
//std::vector<sensor_msgs::ImuConstPtr> 表示当前帧和上一帧时间间隔中的所有IMU数据
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()//看看这返回类型，这么长！
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    // 直到把imu_buf或者feature_buf中的数据全部取出，才会退出while循环
    while (true)
    {
        //[1]进入大循环，imu信息接收到后会存入imu_buf，feature消息收到后存入feature_buf中。判断队列是否为空，为空返回measurements
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //[2]对imu数据和image数据在时间戳上的判断（时间要对齐才能进行组合）        
        //imu_buf中最后一个imu消息的时间戳比feature_buf中第一个feature消息的时间戳还要小，说明imu数据发出来太早了
        // imu_buf队尾元素的时间戳，早于或等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要等待接收IMU数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            //imu数据比图像数据要早，所以返回，等待时间上和图像对齐的imu数据到来  为什么不直接把这些imu的数据从imu_buf里丢掉了？这里等，这些数据还在imu_buf里啊
            sum_of_wait++;
            return measurements;
        }

        // imu_buf队首元素的时间戳，晚于或等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要剔除feature_buf队首多余的特征点数据
        //imu_buf中第一个数据的时间戳比feature_buf的第一个数据的时间戳还要大
if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            //imu数据比图像数据滞后，所以图像数据要出队列
            feature_buf.pop();
            continue;
        }   

        //[3]获取最早发布的图像数据
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front(); // 返回队首元素的值，读取feature_buf队首的数据
        feature_buf.pop(); // 剔除feature_buf队首的数据，疑似和上面重复
        
        std::vector<sensor_msgs::ImuConstPtr> IMUs;//两帧图像之间有很多IMU数据，因此使用vector
        //【6】imu_buf的最前边的数据时间戳小于图像数据的时间戳的话，就将imu_buf中最前边的数据存入IMUs当中

        // 一帧图像特征点数据，对应多帧imu数据,把它们进行对应，然后塞入measurements
        // 一帧图像特征点数据，与它和上一帧图像特征点数据之间的时间间隔内所有的IMU数据，以及时间戳晚于当前帧图像的第一帧IMU数据对应
        // 如下图所示：
        //   *             *             *             *             *            （IMU数据）
        //                                                      |                 （图像特征点数据）
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            //将imu_buf中最上边的元素插入到IMUs的尾部
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front()); // 时间戳晚于当前帧图像的第一帧IMU数据，为何不pop？

        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        //将和该img_msg时间上“对齐”的imu数据连同该img_msg放入measurements的尾部
        //所以这里的measurements中存放的就是在时间上“对齐”的IMU数据和图像数据
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //1、当前IMU采样时间小于上次IMU采样时间，说明有bug，跳出此条件
    if (imu_msg->header.stamp.toSec() <= last_imu_t)//last_imu_t初值为0
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec(); 

    // 2、在修改多个线程共享的变量时进行上锁。防止多个线程同时访问该变量
    m_buf.lock();
    imu_buf.push(imu_msg);//将IMU数据存入IMU数据缓存队列imu_buf
    m_buf.unlock();
    con.notify_one(); // 唤醒作用于process线程中的获取观测值数据的函数getMeasurements()读取缓存imu_buf和feature_buf中的观测数据

    // 通过IMU测量值积分更新并发布里程计信息
    last_imu_t = imu_msg->header.stamp.toSec(); // 这一行代码似乎重复了，上面有着一模一样的代码

    {
        //3、预测tmp_P,tmp_V,tmp_Q
        std::lock_guard<std::mutex> lg(m_state);//模板类std::lock_guard，在构造时就能提供已锁的互斥量，并在析构的时候进行解锁,在std::lock_guard对象构造时，传入的mutex对象(即它所管理的mutex对象)会被当前线程锁住
        predict(imu_msg);
        //4、如果solver_flag为非线性优化，发布最新的里程计消息
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // VINS初始化已完成，正处于滑动窗口非线性优化状态，如果VINS还在初始化，则不发布里程计信息
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // 发布里程计信息，发布频率很高（与IMU数据同频），每次获取IMU数据都会及时进行更新，而且发布的是当前的里程计信息。
            // 还有一个pubOdometry()函数，似乎也是发布里程计信息，但是它是在estimator每次处理完一帧图像特征点数据后才发布的，有延迟，而且频率也不高（至多与图像同频）
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    //1、判断是否是第一个特征，如果是第一个特征直接跳过
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    // 2、将图像特征点数据存入图像特征点数据缓存队列feature_buf
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one(); // 唤醒process线程中的getMeasurements()读取缓存imu_buf和feature_buf中的观测数据
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        //1、当feature_buf和imu_buf非空时，循环删除队首元素
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        //2、清除估计器的状态和参数，时间重置
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}
//主线程
// thread: visual-inertial odometry
// process()是measurement_process线程的线程函数，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
void process()
{
    while (true)
    {
        //1、进入循环构建measurement复合数据类型。将IMU和图像数据组合起来
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

        // unique_lock对象lk以独占所有权的方式管理mutex对象m_buf的上锁和解锁操作，所谓独占所有权，就是没有其他的 unique_lock对象同时拥有m_buf的所有权，
        // 新创建的unique_lock对象lk管理Mutex对象m_buf，并尝试调用m_buf.lock()对Mutex对象m_buf进行上锁，如果此时另外某个unique_lock对象已经管理了该Mutex对象m_buf,
        // 则当前线程将会被阻塞；如果此时m_buf本身就处于上锁状态，当前线程也会被阻塞（我猜的）。
        // 在unique_lock对象lk的声明周期内，它所管理的锁对象m_buf会一直保持上锁状态
        std::unique_lock<std::mutex> lk(m_buf);//锁定m_buf
        //当 getMeasurements()).size() = 0时wait() 阻塞当前线程，并且在收到imucallback,imagecallback的通知后getMeasurements()).size() != 0即有数据时时才会被解除阻塞

        // std::condition_variable::wait(std::unique_lock<std::mutex>& lock, Predicate pred)的功能：
        // while (!pred()) 
        // {
        //     wait(lock);
        // }
        // 当pred为false的时候，才会调用wait(lock)，阻塞当前线程，当同一条件变量在其它线程中调用了notify_*函数时，当前线程被唤醒。
        // 直到pred为ture的时候，退出while循环。

        // [&]{return (measurements = getMeasurements()).size() != 0;}是lamda表达式（匿名函数）
        // 先调用匿名函数，从缓存队列中读取IMU数据和图像特征点数据，如果measurements为空，则匿名函数返回false，调用wait(lock)，
        // 释放m_buf（为了使图像和IMU回调函数可以访问缓存队列），阻塞当前线程，等待被con.notify_one()唤醒
        // 直到measurements不为空时（成功从缓存队列获取数据），匿名函数返回true，则可以退出while循环。
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock(); // 从缓存队列中读取数据完成，解锁
        
        //2、遍历measurements，其实就是遍历获取每一个img_msg和其对应的imu_msg对数据进行处理
        m_estimator.lock();//上锁
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

        //3、遍历和当前img_msg时间上“对齐”的IMU数据,处理imu数据processIMU()
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec(); // 最新IMU数据的时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // 图像特征点数据的时间戳，补偿了通过优化得到的一个时间偏移
                //由于一帧image图像消息对应多个IMU消息，所以针对每一个IMU消息的时间戳，
                // 需要判断IMU的时间戳和image消息的时间戳之间的大小关系，也就是需要判断IMU消息早于image消息先收到，
                // 还是IMU消息晚于image消息收到。            
                if (t <= img_t) // 补偿时间偏移后，图像特征点数据的时间戳晚于或等于IMU数据的时间戳，IMU的数据比图像数据早到
                {
                    //第一次时current_time为-1 
                    if (current_time < 0) // 第一次接收IMU数据时会出现这种情况
                        current_time = t;//IMU采样时间
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t; // 更新最近一次接收的IMU数据的时间戳
                    //dx,dy,dz分别是IMU在三个轴方向上的线加速度

                    // IMU数据测量值
                    // 3轴加速度测量值
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;

                    // 3轴角速度测量值
                    //rx,ry,rz分别是IMU在三个轴方向上的角速度
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //对每一个IMU值进行预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else // IMU时间戳晚于图像特征点数据（时间偏移补偿后）的第一帧IMU数据（也是一组measurement中的唯一一帧），对IMU数据进行插值，得到图像帧时间戳对应的IMU数据
                {
                    // 时间戳的对应关系如下图所示：
                    //                                            current_time         t
                    // *               *               *               *               *     （IMU数据）
                    //                                                          |            （图像特征点数据）
                    //                                                        img_t
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    //对每一个IMU值进行预积分
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 4、设置重定位用的回环帧
            // setReloFrame()
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;//去看传感器数据的definition
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();//返回队首元素值
                relo_buf.pop();//删除队列首元素
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec(); // 回环帧的时间戳
                //遍历relo_msg中的points特征点
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                //[重定位帧的平移向量T的x,y,z，旋转四元数w,x,y,z和索引值]
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r); // 设置回环帧
            }

            //5、处理图像processImage()
            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            TicToc t_s;
            // 将图像特征点数据存到一个map容器中，key是特征点id
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image; // 为什么键值是一个vector啊，一个id的特征点对应一个vector，难道是因为可能有多个相机
            //遍历img_msg中的特征点
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5; // ？？？这是什么操作
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                //获取img_msg中第i个点的x,y,z坐标，这个是归一化后的坐标值

                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                //获取像素的坐标值
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                //获取像素点在x,y方向上的速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);//原地构造，不需要调用构造函数和拷贝构造函数
            }
            }
            estimator.processImage(image, img_msg->header);

            //6、发布message
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            // 每处理完一帧图像特征点数据，都要发布这些话题
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();//解锁
        
        //7、当solver_flag为NON_LINEAR时,update()
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            // VINS系统完成滑动窗口优化后，用优化后的结果，更新里程计数据
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    //[1]初始化，设置句柄，设置logger级别    
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    //【2】从config_file中读取参数，imu话题，Ba,Bg,Na,Ng，最大迭代次数等
    readParameters(n);
    //【3】优化器设置参数，包括相机IMU外参tic和ric
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    //RViz相关话题
    //[4]注册visualization.cpp中创建的发布器registerPub(n)
    registerPub(n); // 注册visualization.cpp中创建的发布器 

    //[5]订阅imu话题，订阅/feature_tracker/feature，/feature_tracker/restart，/pose_graph/match_points等话题,执行回调函数    
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay()); // IMU数据
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback); // 图像特征点数据
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback); // ？？？接收一个bool值，判断是否重启estimator
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback); // ？？？根据回环检测信息进行重定位

    // estimator_node中的线程由旧版本中的3个改为现在的1个
    // 回环检测和全局位姿图优化在新增的一个ROS节点中运行
    // measurement_process线程的线程函数是process()，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
    //[6]创建主线程函数process()，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
    std::thread measurement_process{process}; //创建线程measurement_process对象，一创建线程对象就启动线程 运行process
    ros::spin();

    return 0;
}