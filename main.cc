#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include "gpio/i2c.h"

using Eigen::Vector3f;
using Eigen::VectorXd;
using Eigen::Matrix3f;
using Eigen::MatrixXd;

namespace {

int i2cfd_;
Vector3f magadj_;

bool InitMPU9150() {
  i2c_write(i2cfd_, 0x68, 107, 0x80);  // reset
  usleep(10000);
  i2c_write(i2cfd_, 0x68, 107, 0);  // wake up
  i2c_write(i2cfd_, 0x68, 107, 1);  // use gyro clock
  i2c_write(i2cfd_, 0x68, 108, 0);  // enable accel + gyro
  i2c_write(i2cfd_, 0x68, 55, 0x32);  // enable bypass, int pin latch
  i2c_write(i2cfd_, 0x68, 0x1a, 0x03);  // set filters
  i2c_write(i2cfd_, 0x68, 0x19, 4);  // samplerate divisor
  i2c_write(i2cfd_, 0x68, 0x1a, 0);  // accel filter
  i2c_write(i2cfd_, 0x68, 0x1b, 0);  // gyro filter
  i2c_write(i2cfd_, 0x68, 0x38, 1);  // DRDY int enable

  uint8_t id;
  i2c_read(i2cfd_, 0x68, 117, 1, &id);  // whoami
  fprintf(stderr, "\r\nMPU-9150 id: %02x\r\n", id);

  i2c_read(i2cfd_, 0x0c, 0x00, 1, &id);  // mag device id
  fprintf(stderr, "AK8975C id: %02x\r\n", id);

  uint8_t magadj8[3];
  i2c_write(i2cfd_, 0x0c, 0x0a, 0x0f);  // mag fuse rom access
  i2c_read(i2cfd_, 0x0c, 0x10, 3, magadj8);  // mag device id
  i2c_write(i2cfd_, 0x0c, 0x0a, 0x01);  // mag enable
  magadj_ = Vector3f(
      1 + (magadj8[0] - 128.0f) / 256.0f,
      1 + (magadj8[1] - 128.0f) / 256.0f,
      1 + (magadj8[2] - 128.0f) / 256.0f);

  fprintf(stderr, "AK8975C mag adjust: %f %f %f\r\n",
         magadj_[0], magadj_[1], magadj_[2]);

  return true;  // FIXME
}

bool ReadMag(Vector3f *mag) {
  uint8_t readbuf[14];
  if (i2c_read(i2cfd_, 0x0c, 2, 1, readbuf)) {  // akc8075c magnetometer
    if (readbuf[0] & 0x01) {
      i2c_read(i2cfd_, 0x0c, 0x03, 6, readbuf);
      int16_t x = (readbuf[1] << 8) | readbuf[0],
              y = (readbuf[3] << 8) | readbuf[2],
              z = (readbuf[5] << 8) | readbuf[4];
      *mag = Vector3f(x, y, z).cwiseProduct(magadj_);
      i2c_write(i2cfd_, 0x0c, 0x0a, 0x01);  // mag enable
      return true;
    }
  }
  return false;
}

MatrixXd YTY_(10, 10);

Matrix3f proj_;
Vector3f center_;

// Fits an ellipsoid to the input data using least-squares.
//
// Solution takes the same amount of time no matter how many datapoints were
// added in to YTY_, but they need to be good datapoints as least squares is
// not robust to outliers.
//
// "OLS" solution in Markovsky, Kukush, Van Huffel,
// "Consistent Fitting of Ellipsoids"
// http://eprints.soton.ac.uk/263295/1/ellest_comp_published.pdf
bool SolveMagCalibration() {
  static Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver_(10);
  // Solution is the eigenvector corresponding to the smallest eigenvalue,
  // which is always the first eigenvector returned by eigen_solver
  eigen_solver_.compute(YTY_);
  VectorXd B(10);
  // save in B, then unpack into A, b, d
  B = eigen_solver_.eigenvectors().col(0);
  Matrix3f A_;
  Vector3f b;
  A_ << B[0], B[1], B[2],
     0, B[3], B[4],
     0,    0, B[5];
  Matrix3f A = A_.selfadjointView<Eigen::Upper>();
  b << B[6], B[7], B[8];
  float d = B[9];
  Eigen::LDLT<Matrix3f> ALDLT = A.ldlt();
  Vector3f c = -0.5 * ALDLT.solve(b);
  float scale = 1.0f / (c.transpose() * A * c - d);
  // if any element in scale * vectorD is <0, then we are not calibrated
  Vector3f D = scale * ALDLT.vectorD();
  if (D[0] < 0 || D[1] < 0 || D[2] < 0) {
    return false;
  }

  Vector3f sqrtD = D.cwiseSqrt();
  Matrix3f sqrtDD = sqrtD.asDiagonal();
  proj_ = sqrtDD * ALDLT.matrixU();
  center_ = c;

  return true;
}

// returns (approximately unit) vector pointing towards magnetic north, after
// compensating for ellipsoid shape
bool CalibrateMag(const Vector3f &mag, bool is_calibrated, Vector3f *north) {
  if (!is_calibrated) {
    // Add our datapoint to the YTY_ matrix.
    //
    // roll up H^-T Y^T Y H^-1 by doing a rank update of (y H^-1)
    // H^-1 = 1/sqrt(2) for elements w/ coefficient 2, 1 elsewhere
    // so 2/sqrt(2) = sqrt(2) = r2
    const double r2 = sqrt(2.0);  // root 2
    VectorXd y(10);
    y << mag[0] * mag[0], r2 * mag[0] * mag[1], r2 * mag[0] * mag[2],
      mag[1] * mag[1], r2 * mag[1] * mag[2],
      mag[2] * mag[2],
      mag[0], mag[1], mag[2], 1.0f;
    YTY_.selfadjointView<Eigen::Lower>().rankUpdate(y, 1);

    if (!SolveMagCalibration())
      return false;
  }

  *north = proj_ * (mag - center_);
  return true;
}

bool LoadMagCalibration() {
  // what we store in magcal.bin is actually the raw cumulative sufficient
  // statistics, so we need to recompute the projection also
  FILE *fp = fopen("magcal.bin", "rb");
  if (fp) {
    fread(YTY_.data(), 10*10, sizeof(double), fp);
    fclose(fp);
    if (SolveMagCalibration()) {
      std::cout << "mag calibration center " << center_.transpose()
          << " projection:" << std::endl << proj_ << std::endl;
      return true;
    }
  }
  return false;
}

bool SaveMagCalibration() {
  FILE *fp = fopen("magcal.bin", "wb");
  fwrite(YTY_.data(), 10*10, sizeof(double), fp);
  fclose(fp);
  return true;
}

bool ReadIMU(Vector3f *accel, Vector3f *gyro) {
  uint8_t readbuf[14];
  // mpu-9150 accel & gyro
  if (i2c_read(i2cfd_, 0x68, 0x3b, 14, readbuf)) {
    int16_t ax = (readbuf[0] << 8) | readbuf[1],
            ay = (readbuf[2] << 8) | readbuf[3],
            az = (readbuf[4] << 8) | readbuf[5];
    // int16_t t = (readbuf[6] << 8) | readbuf[7];
    int16_t gx = (readbuf[ 8] << 8) | readbuf[ 9],
            gy = (readbuf[10] << 8) | readbuf[11],
            gz = (readbuf[12] << 8) | readbuf[13];
    *accel = Vector3f(ax, ay, az);
    // TODO: temp calibration
    *gyro = Vector3f(gx, gy, gz);
    return true;
  }
  return false;
}

}  // empty namespace

int main() {
  i2cfd_ = open("/dev/i2c-1", O_RDWR);
  if (i2cfd_ == -1) {
    perror("/dev/i2c-1");
    return 1;
  }

  InitMPU9150();
  bool mag_calibrated = LoadMagCalibration();

  time_t t0 = time(NULL);
  for (;;) {
    Vector3f mag;
    if (ReadMag(&mag)) {
      if (CalibrateMag(mag, mag_calibrated, &mag)) {
        printf("[%f,%f,%f]\n", mag[0], mag[1], mag[2]);
        fflush(stdout);
        time_t t1 = time(NULL);
        // if we've spent 10 seconds in a row calibrated, save it
        if (t1 >= (t0 + 10)) {
          t0 = t1;
          SaveMagCalibration();
          mag_calibrated = true;
          fprintf(stderr, "[saved calibration]\e[K\n");
        }
      } else {
        t0 = time(NULL);
      }
    }
  }

  return 0;
}
