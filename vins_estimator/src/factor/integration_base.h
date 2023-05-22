#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>
using namespace Eigen;

class IntegrationBase
{
  public:
    IntegrationBase() = delete;//表示删除默认构造函数
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}
//p,v,q,ba,bg每个都是3维，所以是15×1维，这里影响雅克比和协方差的维度
    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();//向主对角线上添加元素
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }
////把传递的参数值push到相应的vector中
    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();//雅可比矩阵初值设置为单位矩阵
        covariance.setZero();//协方差矩阵初值为0矩阵
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    //离散形式下PVQ增量误差，这一部分基本上是根据公式来求的，求误差的雅克比和协方差的迭代更新公式
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");a_q为相对预积分参考系的旋转四元数
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);//k时刻的加速度
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;//中值积分的角速度
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);//旋转的更新
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);//k+1时刻的加速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);//中值积分的加速度
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;//位置P更新
        result_delta_v = delta_v + un_acc * _dt;//速度v更新
        result_linearized_ba = linearized_ba;//ba的迭代，极短时间内，我们认为数值不变，因此直接赋值即可
        result_linearized_bg = linearized_bg;//bg同上
        //能否更新与积分测量关于IMU_Bais的雅克比矩阵！！！离散形式下的
        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;//角速度
            //计算_acc_0这个观测线加速度对应的实际加速度
            Vector3d a_0_x = _acc_0 - linearized_ba;//线加速度
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            /**
                        *         | 0      -W_z     W_y |
                        * [W]_x = | W_z     0      -W_x |
                        *         | -W_y   W_x       0  |
                       */
            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            /**
             * matrix.block(i,j, p, q) : 表示返回从矩阵(i, j)开始，每行取p个元素，每列取q个元素所组成的临时新矩阵对象，原矩阵的元素不变；
             * matrix.block<p,q>(i, j) :<p, q>可理解为一个p行q列的子矩阵，该定义表示从原矩阵中第(i, j)开始，获取一个p行q列的子矩阵，
             * 返回该子矩阵组成的临时矩阵对象，原矩阵的元素不变；
            */

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;//雅克比矩阵的更新，迭代公式为：J_(k+1)=F*J_k,J_0=I
            /*
            *covariance为协方差，协方差的迭代公式：P_(k+1) ​= F*P_k*​F^T + V*Q*V^T，P_0​ = 0
            * P_k就是协方差，Q为noise，其初值为18*18的单位矩阵
            */
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();//协方差的更新
        }

    }

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
     
    }

    // 计算IMU观测残差
    //求残差的表达式
    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;//上面的都是为了写residuals
    }

    // 中值积分需要使用前后两个时刻的IMU数据
    double dt; // 前后两个时刻的时间间隔
    Eigen::Vector3d acc_0, gyr_0; // 前一帧IMU数据中的加速度计测量值和陀螺仪测量值
    Eigen::Vector3d acc_1, gyr_1; // 后一帧IMU数据中的加速度计测量值和陀螺仪测量值

    const Eigen::Vector3d linearized_acc, linearized_gyr; // 这一段预积分初始时刻的IMU测量值，作为常量一直保存，在IntegrationBase对象创建时指定
    Eigen::Vector3d linearized_ba, linearized_bg; // 这一段预积分对应的加速度计偏置和陀螺仪偏置

    // jacobian: 当前误差状态量关于预积分初始时刻误差状态量的雅可比矩阵
    // covariance: 误差状态的协方差矩阵
    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian; //似乎是用来调试程序的临时变量
    Eigen::Matrix<double, 15, 18> step_V; //似乎是用来调试程序的临时变量
    Eigen::Matrix<double, 18, 18> noise; // 误差状态传播方程中的噪声的协方差矩阵

    double sum_dt; // 这一段预积分的总时间间隔

    // delta_p、delta_q和delta_v是标称状态的预积分
    // delta_p表示该段预积分初始时刻本体坐标系下，当前时刻本体坐标系的位置
    // delta_q表示该段预积分初始时刻本体坐标系下，当前时刻本体坐标系的旋转
    // delta_v表示该段预积分初始时刻本体坐标系下，当前时刻本体坐标系的速度
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    // 该段预积分所使用的IMU数据的缓存vector
    // 这3个缓存的作用是：当bias变换过大时，需要使用这些数据重新进行预积分
    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

};
/*

    void eulerIntegration(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        result_delta_p = delta_p + delta_v * _dt + 0.5 * (delta_q * (_acc_1 - linearized_ba)) * _dt * _dt;
        result_delta_v = delta_v + delta_q * (_acc_1 - linearized_ba) * _dt;
        Vector3d omg = _gyr_1 - linearized_bg;
        omg = omg * _dt / 2;
        Quaterniond dR(1, omg(0), omg(1), omg(2));
        result_delta_q = (delta_q * dR);   
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Vector3d w_x = _gyr_1 - linearized_bg;
            Vector3d a_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_x<<0, -a_x(2), a_x(1),
                a_x(2), 0, -a_x(0),
                -a_x(1), a_x(0), 0;

            MatrixXd A = MatrixXd::Zero(15, 15);
            // one step euler 0.5
            A.block<3, 3>(0, 3) = 0.5 * (-1 * delta_q.toRotationMatrix()) * R_a_x * _dt;
            A.block<3, 3>(0, 6) = MatrixXd::Identity(3,3);
            A.block<3, 3>(0, 9) = 0.5 * (-1 * delta_q.toRotationMatrix()) * _dt;
            A.block<3, 3>(3, 3) = -R_w_x;
            A.block<3, 3>(3, 12) = -1 * MatrixXd::Identity(3,3);
            A.block<3, 3>(6, 3) = (-1 * delta_q.toRotationMatrix()) * R_a_x;
            A.block<3, 3>(6, 9) = (-1 * delta_q.toRotationMatrix());
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd U = MatrixXd::Zero(15,12);
            U.block<3, 3>(0, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            U.block<3, 3>(3, 3) =  MatrixXd::Identity(3,3);
            U.block<3, 3>(6, 0) =  delta_q.toRotationMatrix();
            U.block<3, 3>(9, 6) = MatrixXd::Identity(3,3);
            U.block<3, 3>(12, 9) = MatrixXd::Identity(3,3);

            // put outside
            Eigen::Matrix<double, 12, 12> noise = Eigen::Matrix<double, 12, 12>::Zero();
            noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(6, 6) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
            noise.block<3, 3>(9, 9) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

            //write F directly
            MatrixXd F, V;
            F = (MatrixXd::Identity(15,15) + _dt * A);
            V = _dt * U;
            step_jacobian = F;
            step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }     


    void checkJacobian(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0, 
                                   const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
    {
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 0);

        Vector3d turb_delta_p;
        Quaterniond turb_delta_q;
        Vector3d turb_delta_v;
        Vector3d turb_linearized_ba;
        Vector3d turb_linearized_bg;

        Vector3d turb(0.0001, -0.003, 0.003);

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p + turb, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb p       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff " << (step_jacobian.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q * Quaterniond(1, turb(0) / 2, turb(1) / 2, turb(2) / 2), delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb q       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v + turb,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb v       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba + turb, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb ba       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 9) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg + turb,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb bg       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_jacobian.block<3, 3>(0, 12) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_jacobian.block<3, 3>(3, 12) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_jacobian.block<3, 3>(6, 12) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_jacobian.block<3, 3>(9, 12) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_jacobian.block<3, 3>(12, 12) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0 + turb, _gyr_0, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 0) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 0) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 0) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 0) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 0) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0 + turb, _acc_1 , _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_0       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 3) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 3) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 3) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 3) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 3) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 + turb, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb acc_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 6) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 6) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 6) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 6) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 6) * turb).transpose() << endl;

        midPointIntegration(_dt, _acc_0, _gyr_0, _acc_1 , _gyr_1 + turb, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            turb_delta_p, turb_delta_q, turb_delta_v,
                            turb_linearized_ba, turb_linearized_bg, 0);
        cout << "turb _gyr_1       " << endl;
        cout << "p diff       " << (turb_delta_p - result_delta_p).transpose() << endl;
        cout << "p jacob diff " << (step_V.block<3, 3>(0, 9) * turb).transpose() << endl;
        cout << "q diff       " << ((result_delta_q.inverse() * turb_delta_q).vec() * 2).transpose() << endl;
        cout << "q jacob diff " << (step_V.block<3, 3>(3, 9) * turb).transpose() << endl;
        cout << "v diff       " << (turb_delta_v - result_delta_v).transpose() << endl;
        cout << "v jacob diff " << (step_V.block<3, 3>(6, 9) * turb).transpose() << endl;
        cout << "ba diff      " << (turb_linearized_ba - result_linearized_ba).transpose() << endl;
        cout << "ba jacob diff" << (step_V.block<3, 3>(9, 9) * turb).transpose() << endl;
        cout << "bg diff      " << (turb_linearized_bg - result_linearized_bg).transpose() << endl;
        cout << "bg jacob diff" << (step_V.block<3, 3>(12, 9) * turb).transpose() << endl;
    }
    */