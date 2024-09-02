# Interactive-DJI-Drone-Plots-for-Path-and-Velocity

**DJI Flight Path Reconstruction**

This Python script reconstructs the flight path of a DJI drone using velocity and orientation data when GPS signal is lost or unreliable. It utilizes a Runge-Kutta 4th order (RK4) integration scheme to calculate the drone's position over time.

**Problem Statement**

DJI drones may occasionally lose GPS signal, resulting in inaccurate or non-existent path data when using services like Airdata. This script provides an alternative method to reconstruct the flight path using the drone's velocity and orientation data.
Features

* Loads and processes DJI flight record CSV filesReconstructs the drone's path using velocity data and RK4 integration
* Visualizes the reconstructed flight path
* Displays real-time velocity and orientation information
* Includes an interactive time slider for exploring the flight data

**Example**

![FlightPathPlot](https://github.com/user-attachments/assets/30ff745e-44ef-4982-b958-929cfa540b8b)

**Video**



https://github.com/user-attachments/assets/db6309a9-1360-4579-bc66-20194cab2be9




**Requirements**

* pandas
* matplotlib
* numpy
* scipy

