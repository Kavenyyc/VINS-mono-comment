#include "feature_tracker.h"

int FeatureTracker::n_id = 0;
bool inBorder(const cv::Point2f &pt)//判断点是否在图像内
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);//CVRound（）：返回跟参数最接近的整数值，即四舍五入；
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}
//ps:cvFloor()：返回不大于参数的最大整数值，即向下取整；
// cvCeil()：返回不小于参数的最小整数值，即向上取整;

//去除无法跟踪的特征点，status为0的点直接跳过，否则v[j++]=v[j]，留下来，最后v.resize(j)根据最新的j安排内存
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)//根据statues进行重组，将statue中为1的对应点保存下来，为0的对应点去除掉
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)//根据statues进行重组，将statue中为1的对应点保存下来，为0的对应点去除掉
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}


//对跟踪点进行排序，并去除密集的点
//对跟踪到的特征点，按照被追踪到的次数排序并依次选点，使用mask来进行类似非极大值抑制的做法，半径为30，去掉密集点
//是特征点分布更加均匀，对跟踪到的特征点从大到小进行排序，并去除密集的点
void FeatureTracker::setMask()
{
    //判断是否为鱼眼相机，采取相应的掩膜方式
    if(FISHEYE)
        mask = fisheye_mask.clone();//如果是鱼眼镜头，直接clone即可，否则创建空白板
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time 倾向于保留追踪时间较长的特征点
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;//定义复合数据类型,构造（cnt，pts，id）序列，三个数据类型分别表示特征点被track次数、特征点、特征点的id
    //将当前帧中的特征点frow_pts及特征点的id和被track的次数打包push进cont_pts_id中
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    //对光流跟踪到的特征点frow_pts，按照被跟踪的次数cnt从大到小排序（lambda表达式）
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });
    //清空track_cnt，frow_pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();
    //遍历cnt_pts_id，操作目的为是图像提取的特征点更均匀
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)// 这个特征点对应的mask值为255，表明点是黑的，还没占有
        {
            //则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //图片，点，半径，颜色为0表示在角点检测在该点不起作用,粗细（-1）表示填充
            //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            /*cv::circle 用于绘制圆圈
            img=mask图像
            center=it.second.first圆心坐标
            radius=MIN_DIST圆形半径
            color=0线条颜色
            thickness=-1圆形轮廓的粗细（若为正值），负值表示要绘制实心圆
            line_type 线条的类型，默认是8
            shift 圆心坐标点和半径值的小数点位数
            
            */ 
        }
    }
}

void FeatureTracker::addPoints()//把新提取的特征点放到frow_pts中，id初始化为-1，track_cnt初始化为1
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);//新提取的特征点id初始化为-1
        track_cnt.push_back(1);//新提取的特征点被跟踪的次数初始化为1
    }
}


/*
图像数据变量：
prev_img： 上一次发布数据时对应的图像帧
cur_img： 光流跟踪的前一帧图像，而不是“当前帧”
forw_img： 光流跟踪的后一帧图像，真正意义上的“当前帧”

特征点数据变量：
prev_pts： 上一次发布的，且能够被当前帧（forw）跟踪到的特征点
cur_pts： 在光流跟踪的前一帧图像中，能够被当前帧（forw）跟踪到的特征点
forw_pts： 光流跟踪的后一帧图像，即当前帧中的特征点（除了跟踪到的特征点，可能还包含新检测的特征点）
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    //1、判断图像是否需要均衡化
    if (EQUALIZE)//均衡化增强图像对比度，默认数值为1，表示图像太亮或者太暗
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));//opencv中的函数，用于生成自适应均衡化图像
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    //2、如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据，将读入的图像数据赋给当前帧forw_img，同时，还将读入的图像赋给prev_img、cur_mig
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;//如果当前帧不为空，则存入新图像
    }

    forw_pts.clear();//此时forw_pts中保存的是上一帧图像中的特征点，因此将其清空

    //判断光流跟踪的当前帧中的特征点规模是否大于0，大于0表示有图像数据点，对其进行图像跟踪
    if (cur_pts.size() > 0)//如果上一帧有特征点
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //调用cv::calcOpticalFlowPyrLK()对上一帧的特征点cur_pts到当前帧的特征点forw_pts进行LK金字塔光流跟踪，得到forw_pts
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0

        for (int i = 0; i < int(forw_pts.size()); i++)//光流跟踪结束后，判断跟踪成功的角点是否都在图像中，不在图像内的焦点status置为0
            if (status[i] && !inBorder(forw_pts[i]))//将当前帧跟踪的位于图像边界外的点标记为0
                status[i] = 0;
        //根据statues，把跟踪失败的点剔除
        //不仅要从forw_pts中剔除，而且还要从cur_unpts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点的id和ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    //4、光流追踪成功，特征点被成功跟踪的次数track_cnt就加1 
    for (auto &n : track_cnt)//track_cnt为每个角点的跟踪次数，数值越大，说明被追踪的就越久
        n++;

    //5、发布这一数据，PUB_THIS_FRAME=1 发布特征点
    if (PUB_THIS_FRAME)
    {
        //6、rejectWithF()通过基本矩阵F剔除outliers
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();//设置mask，保证相邻特征点之间相隔30个像素
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        //寻找新的特征点 goodFeaturesToTrack()
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
                  /**
    * cv::goodFeaturesToTrack()
    * @brief   在mask中不为0的区域检测新的特征点
    * @optional    ref:https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    * @param[in]    InputArray _image=forw_img 输入图像
    * @param[out]   OutputArray _corners=n_pts 存放检测到的角点的vector
    * @param[in]    int maxCorners=MAX_CNT - forw_pts.size() 返回的角点的数量的最大值
    * @param[in]    double qualityLevel=0.01 角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
    * @param[in]    double minDistance=MIN_DIST 返回角点之间欧式距离的最小值
    * @param[in]    InputArray _mask=mask=noArray() 和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
    * @param[in]    int blockSize=3：计算协方差矩阵时的窗口大小
    * @param[in]     bool useHarrisDetector=false：指示是否使用Harris角点检测，如不指定，则计算shi-tomasi角点
    * @param[in]     double harrisK=0.04：Harris角点检测需要的k值
    * @return      void
    */

            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();//n_pts是用来存储提取的新特征点的，这个条件下表示不需要提取新的特征点，将上一帧中提取的点clear
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //addPoints()向forw_pts添加新的追踪点
        ROS_DEBUG("add feature begins");
        TicToc t_a;

        //添加新检测到的特征点n_pts添加到forw_pts中，id初始化为-1，track_cnt初始化为1
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    //更新帧、特征点
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    prev_img = cur_img;//在第一帧处理中，还是等于当前帧frow_img
    prev_pts = cur_pts;//第一帧中不做处理
    prev_un_pts = cur_un_pts;//第一帧中不做处理
    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;
    //从第二张图像输入后，每进行一次循环，根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度
    undistortedPoints();
    prev_time = cur_time;
}

//通过基本矩阵F剔除外点outliers
//首先将特征点转化到归一化相机坐标系，然后计算F矩阵，再根据status清除为0的特征点
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)//当前帧（追踪到的）特征点数量足够多
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        //构建cv::Point2f类型的点为cv::findDundamentalMat()做准备
        //1、遍历所有点，转化为归一化相机坐标系
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)//遍历上一帧所有特征点
        {
            Eigen::Vector3d tmp_p;
            //根据不同的相机类型，将二维坐标转换至三维坐标
            //对于PINHOLE（针孔相机）可将像素坐标转换到归一化平面并去畸变
            //对于CATA（卡特鱼眼相机）将像素坐标投影到单位圆内
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            //un_cur_pts和un_frow_pts转化为归一化像素坐标，上一帧和当前帧
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
    
        // 2. 调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵，RANSAC去除outliers，需要归一化相机系，z=1. 
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        //3. 根据status删除一些特征点，进行重组
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}
/*特征点id相当于特征点的身份证号，对数据关联（data association）至关重要。需要注意的是，
更新特征点id的步骤被特意放到了回调函数img_callback()中，而不是FeatureTracker::readImage()函数内部
基于线程安全的考虑，因为n_id为一个static成员变量
FeatureTracker类的多个实例对象会共享一个n_id，在readImage()函数内部更新特征点id的话，
如果多个相机并行调用readImage()，它们都要去访问n_id并改变它的值，可能会产生问题。
我有一个疑问：为什么会出现多个相机并行调用readImage()的情况，因为从源代码来说，可以保证多个相机的调用存在时序上的先后关系
*/
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())//每当检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

//显示去畸变矫正后的特征点 name为图像帧名称
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
//对角点图像进行去畸变矫正，转换到归一化坐标系上，并计算每个角点的速度
void FeatureTracker::undistortedPoints()//进行畸变矫正
{   
    //清除cur_un_pts、cur_un_pts_map的状态
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    //归一化相机坐标系
    //给cur_un_pts、cur_un_pts_map装入新的值
    for (unsigned int i = 0; i < cur_pts.size(); i++)//遍历所有特征点
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        //根据不同的相机模型将二维坐标转换到归一化相机三维坐标系
        m_camera->liftProjective(a, b);//转像素坐标
        //再延伸到深度归一化平面上
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // 【3】判断上一帧中特征点点是否为0，不为0则计算点跟踪的速度
    // caculate points velocity计算每个特征点的速度pts_velocity
    if (!prev_un_pts_map.empty())// 2.1 地图不是空的判断是否新的点
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();// pts_velocity表示当前帧相对前一帧特征点沿x,y方向的像素移动速度
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {   // 2.2 通过id判断不是最新的点
            if (ids[i] != -1)//如果点不是第一次被track
            {
                std::map<int, cv::Point2f>::iterator it;// 地图的迭代器
                it = prev_un_pts_map.find(ids[i]);// 找到对应的id
                if (it != prev_un_pts_map.end())// 2.3 在地图中寻找是否出现过id判断是否最新点
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;//计算x方向上的速度
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;//计算y方向上的速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));// 之前出现过，push_back即可
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));// 之前没出现过，先放进去但是速度为0
            }
            else//点第一次被track
            {
                pts_velocity.push_back(cv::Point2f(0, 0));// 是最新的点，速度为0
            }
        }
    }
    else//为0则push一个值
    {
    // 如果地图是空的，速度是0
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    //更新地图，当前帧中的点和点的id传递到上一帧
    prev_un_pts_map = cur_un_pts_map;
    }
}
