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

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

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

    x_ << x, y, rho_dot, phi, 0;

    P_ << Eigen::MatrixXd::Identity(n_x_, n_x_) * 1;
    P_(0, 0) = std_radr_ * std_radr_;
    P_(1, 1) = std_radr_ * std_radr_;
    P_(2, 2) = std_radrd_ * std_radrd_;
    P_(3, 3) = std_radphi_ * std_radphi_;

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
  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();
  MatrixXd sig_matrix = A * std::sqrt((lambda_ + n_x_));
  Eigen::MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);

  // create sigma point matrix
  MatrixXd sigma_matrix = A * std::sqrt(lambda_ + n_aug_);
  Xsig_aug.col(0) = x_aug;

  for (int col_no = 0; col_no < n_aug_; ++col_no)
  {
    Xsig_aug.col(col_no + 1) = x_aug + sigma_matrix.col(col_no);
    Xsig_aug.col(col_no + n_aug_ + 1) = x_aug - sigma_matrix.col(col_no);
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
    for (int col_no = 0; col_no < Xsig_aug.cols(); ++col_no)
    {
        double p_x = Xsig_aug(0, col_no);
        double p_y = Xsig_aug(1, col_no);
        double v = Xsig_aug(2, col_no);
        double psi = Xsig_aug(3, col_no);
        double psi_dot = Xsig_aug(4, col_no);
        double nu = Xsig_aug(5, col_no);
        double nu_dot_dot = Xsig_aug(6, col_no);

        double px_pred{ 0.0 };
        double py_pred{ 0.0 };

        if (fabs(psi_dot) < 0.001)
        {
            px_pred = p_x + v * std::cos(psi) * delta_t;
            py_pred = p_y + v * std::sin(psi) * delta_t;
        }
        else
        {
            double factor = v / psi_dot;
            double predicted_psi = psi + psi_dot * delta_t;
            px_pred = p_x + factor * (std::sin(predicted_psi) - std::sin(psi));
            py_pred = p_y + factor * (std::cos(psi) - cos(predicted_psi));
        }
        double v_pred = v;
        double psi_pred = psi + psi * delta_t;
        double psi_dot_pred = psi_dot;

        px_pred += 0.5 * delta_t * delta_t * std::cos(psi) * nu;
        py_pred += 0.5 * delta_t * delta_t * std::sin(psi) * nu;
        v_pred += delta_t * nu;
        psi_pred += 0.5 * delta_t * delta_t * nu_dot_dot;
        psi_dot_pred += delta_t * nu_dot_dot;

        Xsig_pred_(0, col_no) = px_pred;
        Xsig_pred_(1, col_no) = py_pred;
        Xsig_pred_(2, col_no) = v_pred;
        Xsig_pred_(3, col_no) = psi_pred;
        Xsig_pred_(4, col_no) = psi_dot_pred;
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
    PredictMean();
    PredictCovariance();

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
}
