## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.


Installation
---

Clone the Github repository and change the directory path in the main_pipeline.py file in line 20. Namely, it should be the project repository path in your local: 
![fig.0](https://raw.githubusercontent.com/chatmoon/SDC_PRJ04_Lane_Finding/master/_1_wip/output_images/_001_readme_installation_directory%20path_.PNG)


Usage
---

Edit the main_pipeline.py file and select (uncomment) which video to run the test. By default, the video called `project_video.mp4` is the video used by the pipeline. The video ouput is saved into 'output_images' folder.

![fig.1](https://raw.githubusercontent.com/chatmoon/SDC_PRJ04_Lane_Finding/master/_1_wip/output_images/_002_readme_usage_video%20input.PNG)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



