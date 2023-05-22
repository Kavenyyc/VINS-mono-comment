#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * 把当前帧图像（frame_count）的特征点添加到feature容器中
 * 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧）
 * 也就是说当前帧图像特征点存入feature中后，并不会立即判断是否将当前帧添加为新的关键帧，而是去判断当前帧的前一帧（第2最新帧）。
 * 当前帧图像要在下一次接收到图像时进行判断（那个时候，当前帧已经变成了第2最新帧）
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    //image数据类型解释：feature_id,camera_id，p.x,p.y,p.z,u,v,u_velocity,v_velocity
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());//计算滑动窗口内被track过的特征点的数量
    double parallax_sum = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的总视差
    int parallax_num = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的数量
    last_track_num = 0; // 当前帧（第1最新帧）图像跟踪到的特征点的数量

    //[1]遍历图像image中所有的特征点，和已经记录了特征点的容器feature中进行比较
    // 把当前帧图像特征点数据image添加到feature容器中
    // feature容器按照特征点id组织特征点数据，对于每个id的特征点，记录它被滑动窗口中哪些图像帧观测到了
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);//7*1 x,y,z,u,v,u的速度，v的速度

        int feature_id = id_pts.first;//获取特征点id
        //在feature中查找该feature_id的feature是否存在

        /**
         * STL find_if的用法：
         * find_if (begin, end, func)
         * 就是从begin开始 ，到end为止，返回第一个让 func这个函数返回true的iterator
         */
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 返回尾部迭代器，说明该特征点第一次出现（在当前帧中新检测的特征点），需要在feature中新建一个FeaturePerId对象
        if (it == feature.end())//这里的feature的数据类型: list<FeaturePerId> feature;
        {
            //未找到该feature的id，则把特征点放入到feature的list容器中
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);//list链表，.back()表示返回最后一个元素
        }
        else if (it->feature_id == feature_id)

        {
/**
 * 如果找到了相同ID特征点，就在其FeaturePerFrame内增加此特征点在此帧的位置以及其他信息，
 * it的feature_per_frame容器中存放的是该feature能够被哪些帧看到，存放的是在这些帧中该特征点的信息
 * 所以，feature_per_frame.size的大小就表示有多少个帧可以看到该特征点
 * */
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++; // 当前帧（第1最新帧）图像跟踪到的特征点的数量，特征点被跟踪的次数+1
        }
    }

    // 1. 当前帧的帧号小于2，即为0或1，为0，则没有第2最新帧，为1，则第2最新帧是滑动窗口中的第1帧
    // 2. 当前帧（第1最新帧）跟踪到的特征点数量小于20（？？？为什么当前帧的跟踪质量不好，就把第2最新帧当作关键帧？？？）
    // 出现以上2种情况的任意一种，则认为第2最新帧是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true; // 第2最新帧是关键帧 marg最老帧

//[2]遍历每一个feature，计算能被当前帧和其前两帧共同看到的特征点视差
    // 计算第2最新帧和第3最新帧之间跟踪到的特征点的平均视差
    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            //计算特征点it_per_id在倒数第二帧和倒数第三帧之间的视差，并求所有视差的累加和
            // 对于给定id的特征点，计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
            //（需要使用IMU数据补偿由于旋转造成的视差）
            parallax_sum += compensatedParallax2(it_per_id, frame_count);//开始补偿视差
            parallax_num++;//满足条件的特征点数量+1
        }
    }

    if (parallax_num == 0)
    {
        // 如果第2最新帧和第3最新帧之间跟踪到的特征点的数量为0，则把第2最新帧添加为关键帧
        // ？？怎么会出现这种情况？？？？
        // 如果出现这种情况，那么第2最新帧和第3最新帧之间的视觉约束关系不就没有了？？？
        return true;
    }
    else
    {
        // 计算平均视差 视差总和/参与计算视差的特征点个数=每个特征点的平均视差数值
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX; 
        // 如果平均视差大于设定的阈值，则把第2最新帧当作关键帧
        //当平均视差大于等于最小视差，返回true，marg最老帧，当平均视差小于最小视差，marg掉次新帧
        //当平均视差小于最小视差的意思是表示新来的图像帧时间和前两帧挨得很近，图像的信息很相似，因此可以去掉相邻的次新帧
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature) // 对于每个id的特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 每个id的特征点被多少帧图像观测到了
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            // 如果该特征点被两帧及两帧以上的图像观测到，
            // 且观测到该特征点的第一帧图像应该早于或等于滑动窗口第4最新关键帧
            // 也就是说，至少是第4最新关键帧和第3最新关键帧观测到了该特征点（第2最新帧似乎是紧耦合优化的最新帧）

            continue; // 跳过

        if (it_per_id.estimated_depth > 0) // 该id的特征点深度值大于0，该值在初始化时为-1，如果大于0，说明该点被三角化过
            continue; // 跳过

        // imu_i：观测到该特征点的第一帧图像在滑动窗口中的帧号
        // imu_j：观测到该特征点的最后一帧图像在滑动窗口中的帧号
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0; // 似乎是[R | T]的形式，是一个位姿
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity(); // 单位旋转矩阵
        P0.rightCols<1>() = Eigen::Vector3d::Zero(); // 0平移向量

        for (auto &it_per_frame : it_per_id.feature_per_frame) // 对于观测到该id特征点的每一图像帧
        {
            imu_j++; // 观测到该特征点的最后一帧图像在滑动窗口中的帧号

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j) // 在第一次进入for循环的时候，这个条件成立，这时候循环体都执行完了，continue发挥不了什么作用啊？？？
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method; // 似乎是得到了该特征点的深度
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1) // 如果估计出来的深度小于0.1（单位是啥？？？），则把它替换为一个设定的值
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * 对于给定id的特征点
 * 计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
 * （需要使用IMU数据补偿由于旋转造成的视差）
 * it_per_id 从特征点list上取下来的一个feature
 * frame_count 当前滑动窗口中的frame个数
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //传入第一个参数为滑窗内的每一个feature，第二参数表示当前的帧数
    //check the second last frame is keyframe or not检查次新帧是否为关键帧
    //parallax betwwen seconde last frame and third last 
    //说白了就是在算一个特征点最后两帧数据中点的坐标在归一化平面中的距离
    //FeaturePerId &it_per_id 对应的是一个特征点变量，该变量里存放了该点在每一历史帧中的信息
    //feature_per_frame[]表示包含这个特征的关键帧的管理器
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // 第3最新帧
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // 第2最新帧

    double ans = 0;// 初始化视差

    Vector3d p_j = frame_j.point;//某特征点在倒数第一帧里的坐标
    //由于特征点都是经过归一化之后的点，因此深度均为1，所以这里不用再进行归一化去除深度，下面进行深度去除,效果相同
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;//某特征点在倒数第二帧里的坐标
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    //p_i中存放的是该point的x,y,z的坐标值，p_i(2)就是z，也就是深度值，这里是归一化坐标所以z为1
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;//该点在前后两帧图像里归一化点的坐标差

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // ？？？？ 开算术平方根还能开出负数吗？？？？ 不用比也是后者大啊？？？？
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;//返回最大的坐标差？
}