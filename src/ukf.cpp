#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */

    /**
     * TODO: Complete the initialization. See ukf.h for other member properties.
     * Hint: one or more values initialized above might be wildly off...
     */

    is_initialized_ = false;
    n_x_ = 5;

    n_aug_ = n_x_ + 2;
    lambda_ = 3 - n_aug_;
    int num_sigma_points = 2 * n_aug_ + 1;

    Xsig_pred_ = MatrixXd(n_x_, num_sigma_points);
    CalculateWeights();
}

UKF::~UKF() {}
void UKF::SquashAngle(double& angle)
{
  while (angle > M_PI)
  {
    angle -= 2. * M_PI;
  }
  while (angle < -M_PI)
  {
    angle += 2. * M_PI;
  }
}
void UKF::InitializeLidarMatrices(MeasurementPackage measurement)
{
    double x = measurement.raw_measurements_(0);
    double y = measurement.raw_measurements_(1);

    x_ << x, y, 0, 0, 0;
    P_ << Eigen::MatrixXd::Identity(n_x_, n_x_) * 1;
    P_(0, 0) = std_laspx_ * std_laspx_;
    P_(1, 1) = std_laspx_ * std_laspx_;

    time_us_ = measurement.timestamp_;
    is_initialized_ = true;
}
void UKF::InitializeRadarMatrices(MeasurementPackage measurement)
{
    double rho = measurement.raw_measurements_(0);
    double phi = measurement.raw_measurements_(1);
    double rho_dot = measurement.raw_measurements_(2);

    double x = rho * std::sin(phi);
    double y = rho * std::cos(phi);

    x_ << x, y, 0, 0, 0;

    // P_ << Eigen::MatrixXd::Identity(n_x_, n_x_) * 1;
    // P_(0, 0) = std_radr_ * std_radr_;
    // P_(1, 1) = std_radr_ * std_radr_;
    // P_(2, 2) = std_radrd_ * std_radrd_;
    // P_(3, 3) = std_radphi_ * std_radphi_;

    time_us_ = measurement.timestamp_;
    is_initialized_ = true;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

    if (!is_initialized_)
    {
        if (meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            InitializeLidarMatrices(meas_package);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            InitializeRadarMatrices(meas_package);
        }
    }

    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        UpdateRadar(meas_package);
    }

}

Eigen::MatrixXd UKF::CreateAugmentedMatrix()
{
    VectorXd x_aug = VectorXd::Zero(n_aug_);
    // augumented state covariance
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    // augumented sigma point matrix
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    MatrixXd L = P_aug.llt().matrixL();
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i)
    {
        Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    return Xsig_aug;
}

void UKF::PredictMean()
{
    // Predict Mean
    x_.fill(0.0);
    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        x_ = x_ + (weights_(col_no) * Xsig_pred_.col(col_no));
    }
}

void UKF::CalculateWeights()
{
    weights_ = VectorXd(2 * n_aug_ + 1);
    double w_major = lambda_ / (lambda_ + n_aug_);
    double w_minor = 0.5 / (lambda_ + n_aug_);

    weights_.fill(w_minor);
    weights_(0) = w_major;
}
void UKF::PredictCovariance()
{
    P_.fill(0.0);
    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        VectorXd residual = Xsig_pred_.col(col_no) - x_;
        SquashAngle(residual(3));
        P_ = P_ + (weights_(col_no) * (residual * residual.transpose()));
    }
}

void UKF::PropagateSigmaPoints(Eigen::MatrixXd Xsig_aug, double delta_t)
{
    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        // extract state
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // predict state
        double px_p, py_p, v_p, yaw_p, yawd_p;
        if (fabs(yawd) > 0.001)
        {
            // curve path
            px_p = p_x + v/yawd*(sin(yaw+yawd*delta_t) - sin(yaw)) + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
            py_p = p_y + v/yawd*(-cos(yaw+yawd*delta_t) + cos(yaw)) + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
            v_p = v + delta_t*nu_a;
            yaw_p = yaw + yawd*delta_t + 0.5*delta_t*delta_t*nu_yawdd;
            yawd_p = yawd + delta_t*nu_yawdd;
        }
        else
        {
            // straight path
            px_p = p_x + v*cos(yaw)*delta_t + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
            py_p = p_y + v*sin(yaw)*delta_t + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
            v_p = v + delta_t*nu_a;
            yaw_p = yaw + yawd*delta_t + 0.5*delta_t*delta_t*nu_yawdd;
            yawd_p = yawd + delta_t*nu_yawdd;
        }

        // update Xsig_pred
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
}

void UKF::Prediction(double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */
    Eigen::MatrixXd Xsig_aug = CreateAugmentedMatrix();
    PropagateSigmaPoints(Xsig_aug, delta_t);
    // Predict Mean and Covariance
    // predict state mean and covariance-----------------
    VectorXd final_x = VectorXd::Zero(5);
    MatrixXd final_P = MatrixXd::Zero(5, 5);

    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        final_x += weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; ++i)
    {
        VectorXd x_diff = Xsig_pred_.col(i) - final_x;

        while (x_diff(3) > M_PI) x_diff(3) -= 2.0*M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.0*M_PI;

        final_P += weights_(i) * x_diff * x_diff.transpose();
    }


    // write result
    x_ = final_x;
    P_ = final_P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
    int n_z = 2;
    VectorXd z = meas_package.raw_measurements_;
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    VectorXd z_pred = VectorXd(n_z);

    MatrixXd S = MatrixXd(n_z, n_z);

    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        double px = Xsig_pred_(0, col_no);
        double py = Xsig_pred_(1, col_no);

        Zsig(0, col_no) = px;
        Zsig(1, col_no) = py;
    }

    z_pred.fill(0.0);
    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        z_pred = z_pred + weights_(col_no) * Zsig.col(col_no);
    }

    S.fill(0.0);
    MatrixXd R(n_z, n_z);
    R.fill(0.0);
    R(0, 0) = std_laspx_ * std_laspx_;
    R(1, 1) = std_laspy_ * std_laspy_;
    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        VectorXd residual = Zsig.col(col_no) - z_pred;
        S = S + weights_(col_no) * (residual * residual.transpose());
    }
    S = S + R;

    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);
    for (int col_no = 0; col_no < Zsig.cols(); ++col_no)
    {
        VectorXd state_residual = Xsig_pred_.col(col_no) - x_;
        VectorXd measurement_residual = Zsig.col(col_no) - z_pred;
        SquashAngle(state_residual(3));
        Tc = Tc + weights_(col_no) * (state_residual * measurement_residual.transpose());
    }

    MatrixXd K = Tc * S.inverse();

    VectorXd z_residual = z - z_pred;

    x_ = x_ + K * (z_residual);
    P_ = P_ - K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */

    int n_z = 3;
    VectorXd z = meas_package.raw_measurements_;
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd R(n_z, n_z);

    Zsig = CalculateMeasurementSigmaPoints();
    z_pred = CalculatePredictedMeasurement(Zsig);

    S.fill(0.0);
    R.fill(0.0);
    R(0, 0) = std_radr_ * std_radr_;
    R(1, 1) = std_radphi_ * std_radphi_;
    R(2, 2) = std_radrd_ * std_radrd_;

    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        VectorXd residual = Zsig.col(col_no) - z_pred;
        SquashAngle(residual(1));
        S = S + weights_(col_no) * (residual * residual.transpose());
    }
    S = S + R;

    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);
    for (int col_no = 0; col_no < Zsig.cols(); ++col_no)
    {
        VectorXd state_residual = Xsig_pred_.col(col_no) - x_;
        VectorXd measurement_residual = Zsig.col(col_no) - z_pred;
        SquashAngle(state_residual(3));
        SquashAngle(measurement_residual(1));
        Tc = Tc + weights_(col_no) * (state_residual * measurement_residual.transpose());
    }

    MatrixXd K = Tc * S.inverse();

    VectorXd z_residual = z - z_pred;

    SquashAngle(z_residual(1));

    x_ = x_ + K * (z_residual);
    P_ = P_ - K * S * K.transpose();
}

Eigen::MatrixXd UKF::CalculateMeasurementSigmaPoints()
{
    int n_z = 3;
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);
    for (int col_no = 0; col_no < Xsig_pred_.cols(); ++col_no)
    {
        double px = Xsig_pred_(0, col_no);
        double py = Xsig_pred_(1, col_no);
        double v = Xsig_pred_(2, col_no);
        double phi = Xsig_pred_(3, col_no);

        double vx = std::cos(phi) * v;
        double vy = std::sin(phi) * v;

        Zsig(0, col_no) = std::sqrt(px * px + py * py);
        Zsig(1, col_no) = std::atan2(py, px);
        Zsig(2, col_no) = (px * vx + py * vy) / (std::sqrt(px * px + py * py));
    }
    return Zsig;
}

Eigen::VectorXd UKF::CalculatePredictedMeasurement(Eigen::MatrixXd Zsig)
{
    int n_z = 3;
    VectorXd z_pred = VectorXd(n_z);

    z_pred.fill(0.0);
    for (int col_no = 0; col_no < Zsig.cols(); ++col_no)
    {
        z_pred = z_pred + weights_(col_no) * Zsig.col(col_no);
    }
    return z_pred;
}
