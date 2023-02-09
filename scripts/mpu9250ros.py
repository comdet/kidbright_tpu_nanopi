#!/usr/bin/env python
# license removed for brevity
import rospy
import FaBo9Axis_MPU9250
from sensor_msgs.msg import Imu
from kidbright_tpu.msg import lino_imu

#ACCEL_FACTOR = 0.000598550415   # (ADC_Value / Scale) * 9.80665            => Range : +- 2[g]  Scale : +- 16384
#GYRO_FACTOR  = 0.0010642        # (ADC_Value/Scale) * (pi/180)             => Range : +- 2000[deg/s]Scale : +- 16.4[deg/s]
#MAG_FACTOR   = 15e-8


ACCEL_FACTOR = 1   # (ADC_Value / Scale) * 9.80665            => Range : +- 2[g]  Scale : +- 16384
GYRO_FACTOR  = 1       # (ADC_Value/Scale) * (pi/180)             => Range : +- 2000[deg/s]Scale : +- 16.4[deg/s]
MAG_FACTOR   = 1

def talker():
    mpu9250 = FaBo9Axis_MPU9250.MPU9250()
    mpu9250.configMPU9250(FaBo9Axis_MPU9250.GFS_2000, FaBo9Axis_MPU9250.AFS_2G)
    imu_msg = Imu()
    imu_msg.orientation_covariance[0] = -1
    imu_msg.angular_velocity_covariance[0] = -1
    imu_msg.linear_acceleration_covariance[0] = -1
    pub_imu = rospy.Publisher('raw_imu', lino_imu, queue_size=10)
    rospy.init_node('mpu9250', anonymous=True)
    rate = rospy.Rate(20) # 10hz


    while not rospy.is_shutdown():
        accel = mpu9250.readAccel()
        gyro = mpu9250.readGyro()
        mag = mpu9250.readMagnet()
        imu_msg = lino_imu()
        imu_msg.linear_acceleration.x = accel['x']*ACCEL_FACTOR
        imu_msg.linear_acceleration.y = accel['y']*ACCEL_FACTOR
        imu_msg.linear_acceleration.z = accel['z']*ACCEL_FACTOR 

        imu_msg.angular_velocity.x = gyro['x']*GYRO_FACTOR*3.1416/180.0
        imu_msg.angular_velocity.y = gyro['y']*GYRO_FACTOR*3.1416/180.0
        imu_msg.angular_velocity.z = gyro['z']*GYRO_FACTOR*3.1416/180.0

        imu_msg.magnetic_field.x = mag['x']*MAG_FACTOR
        imu_msg.magnetic_field.y = mag['y']*MAG_FACTOR
        imu_msg.magnetic_field.z = mag['z']*MAG_FACTOR






        pub_imu.publish(imu_msg)
        rate.sleep()

if __name__ == '__main__':
    
    try:
        talker()
    except rospy.ROSInterruptException:
        pass