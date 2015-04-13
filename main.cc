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
  printf("\r\nMPU-9150 id: %02x\r\n", id);

  i2c_read(i2cfd_, 0x0c, 0x00, 1, &id);  // mag device id
  printf("AK8975C id: %02x\r\n", id);

  uint8_t magadj8[3];
  i2c_write(i2cfd_, 0x0c, 0x0a, 0x0f);  // mag fuse rom access
  i2c_read(i2cfd_, 0x0c, 0x10, 3, magadj8);  // mag device id
  i2c_write(i2cfd_, 0x0c, 0x0a, 0x01);  // mag enable
  magadj_ = Vector3f(
      1 + (magadj8[0] - 128.0f) / 256.0f,
      1 + (magadj8[1] - 128.0f) / 256.0f,
      1 + (magadj8[2] - 128.0f) / 256.0f);

  printf("AK8975C mag adjust: %f %f %f\r\n",
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

  MatrixXd YTY(10, 10);  // TODO: use Triangular
  const double r2 = sqrt(2.0);  // root 2
  int nmag = 0;
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(10);
  for (;;) {
    Vector3f mag;
    if (ReadMag(&mag)) {
      VectorXd y(10);
      // roll up H^-T Y^T Y H^-1 by doing a rank update of (y H^-1)
      // H^-1 = 1/sqrt(2) for elements w/ coefficient 2, 1 elsewhere
      // so 2/sqrt(2) = sqrt(2) = r2
      y << mag[0] * mag[0], r2 * mag[0] * mag[1], r2 * mag[0] * mag[2],
        mag[1] * mag[1], r2 * mag[1] * mag[2],
        mag[2] * mag[2],
        mag[0], mag[1], mag[2], 1.0f;
      YTY.selfadjointView<Eigen::Lower>().rankUpdate(y, 1);
      nmag++;
      // printf("%f %f %f\r", mag[0], mag[1], mag[2]);
      // fflush(stdout);
      // std::cout << YTY << std::endl;
#if 1
      if (nmag > 5) {
        // Solution is the eigenvector corresponding to the smallest eigenvalue,
        // which is always the first eigenvector returned by eigen_solver
        eigen_solver.compute(YTY);
        VectorXd B(10);
        // save in B, then unpack into A, b, d
        B = eigen_solver.eigenvectors().col(0);
        // std::cout << B.transpose() << std::endl;
        Matrix3f A_;  // TODO: use triangular/selfadjointView
        Vector3f b;
        A_ << B[0], B[1], B[2],
                 0, B[3], B[4],
                 0,    0, B[5];
        Matrix3f A = A_.selfadjointView<Eigen::Upper>();
        b << B[6], B[7], B[8];
        float d = B[9];
        Eigen::LDLT<Matrix3f> ALDLT = A.ldlt();
        Vector3f c = -0.5 * ALDLT.solve(b);
        float scale1 = c.transpose() * A * c;
        float scale = 1.0f / (scale1 - d);
        Matrix3f Ae = scale * A;  // we don't actually need Ae, just verifying
        std::cout << "center: " << c.transpose();
        // std::cout << "Ae: (scale1=" << scale1 << " scale=" << scale << ")\n"
        //     << Ae << std::endl;
        float scale2 = (mag - c).transpose() * Ae * (mag - c);
        std::cout << " D: " << (scale * ALDLT.vectorD()).transpose()
            << std::endl;
        // if any element in scale * vectorD is <0, then we are not calibrated
        Vector3f sqrtD = (scale * ALDLT.vectorD()).cwiseSqrt();
        Matrix3f sqrtDD = sqrtD.asDiagonal();
        Matrix3f proj = sqrtDD * ALDLT.matrixU();
        std::cout << mag.transpose() << " -> (" << scale2 << ") "
            << (proj * (mag - c)).transpose() << std::endl;
      }
#endif
    }
  }
  /*
  timeval t1;
  gettimeofday(&t1, NULL);
  printf("%d.%06d\n", t1.tv_sec, t1.tv_usec);
  t1.tv_sec -= t0.tv_sec;
  t1.tv_usec -= t0.tv_usec;
  if (t1.tv_usec < 0) {
    t1.tv_usec += 1000000;
    t1.tv_sec += 1;
  }
  float dt = (t1.tv_sec + t1.tv_usec / 1e6f);
  float rate = naccel / dt;
  printf("%d.%06d %d %d %f/sec\n",
         t1.tv_sec, t1.tv_usec, naccel, nmag, rate);
         */

  return 0;
}
