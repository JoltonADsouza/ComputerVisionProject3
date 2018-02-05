// XykenMiniProj.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/core/core.hpp"
//#include "opencv2/legacy/legacy.hpp"
#include "opencv2/opencv.hpp"
//#include "gvfc.h"
//#include "common.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
RNG rng(12345);


Mat src, src1, src_gray;
Mat dst, detected_edges;


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event != EVENT_LBUTTONDOWN)
		return;
	cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
}


int main(int argc, char* argv[])
{
	std::vector<Mat> vec_images;
	VideoCapture capture(argv[1]);

	while (1)
	{
		Mat frame;
		capture.read(frame);
		if (frame.empty())
		{
			break;
		}
		else
		{
			vec_images.push_back(frame);
			//namedWindow("video", WINDOW_NORMAL);
			flip(frame, frame, -1);
			/*imshow("video", frame);
			waitKey(30);*/
		}
	}
	cout << vec_images.size() << " number of images captured in the video" << endl;
	//imshow("image", vec_images[150]);



	src = vec_images[100];
	Mat clone = src.clone();

	//src1 = vec_images[50];
	imshow("soruce_img", src);

	//Hough Transform for circles
	Mat src_gray;
	if (!src.data)
	{
		return -1;
	}

	//Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	//Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

	namedWindow("gray", CV_WINDOW_AUTOSIZE);
	imshow("gray", src_gray);


	vector<Vec3f> circles;
	vector<Point> largest_contour;
	int step = 2;


	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		cout << circles[i][0] << "\n" << circles[i][1] << "\n" << circles[i][2] << endl;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		cout << "The centre of the detected circle is " << center << endl;
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius*1.5, Scalar(0, 0, 255), 1, 8, 0);

		cout << "The co-ordinates of points on circle are: " << endl;

		for (double theta = 0; theta <= 360 - step; theta += step)
		{
			//iter++;
			double x_on_circle, y_on_circle;
			x_on_circle = cvRound(cvRound(circles[i][0]) + radius*1.5*cos((M_PI*theta) / 180));
			y_on_circle = cvRound(cvRound(circles[i][1]) + radius*1.5*sin((M_PI*theta) / 180));
			largest_contour.push_back(Point(x_on_circle, y_on_circle));
			//cout << "x:" << x_on_circle << ", " << "y:" << y_on_circle << endl;
		}
	}


	/// Show your results

	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);

	imshow("Hough Circle Transform Demo", src);

	setMouseCallback("Hough Circle Transform Demo", CallBackFunc, NULL);

	/*CvPoint* point = NULL;
	point = new CvPoint[largest_contour.size()];
	int length = largest_contour.size();
	float alpha = 0.05f, beta = 0.1f, gamma = 1.0f, kappa = 2.0f, flag = 0.0f, t;
	point = cvSnakeImageGVF(&src, point, &length, alpha, beta, gamma, kappa, 50, 10, CV_REINITIAL, CV_GVF);
	for (int i = 0; i<length; i++) {
		int j = (i + 1) % length;
		cvLine(&clone, point[i], point[j], CV_RGB(255, 0, 0), 2, 8, 0);
	}
	cvShowImage("Input Image", &clone);
*/

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat im_gray, im_bw;
	cvtColor(clone, im_gray, CV_RGB2GRAY);
	threshold(im_gray, im_bw, 50.0, 255.0,THRESH_BINARY);
	threshold(im_gray, im_bw, 10, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", im_bw);

	findContours(im_bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat drawing = Mat::zeros(im_bw.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	setMouseCallback("Contours", CallBackFunc, NULL);

	vector<Point> approx;
	vector<vector<Point> > contour_list;
		for (int i = 0; i <= contours.size(); i++)
		{
			//cout << Mat(contours[i]) << endl;
			approxPolyDP(Mat(contours[i]), approx, 0.01*(arcLength(contours[i],true)), true);
			double area = contourArea(Mat(contours[i]));
				if ((approx.size() > 8) & (area > 30))
					contour_list.insert(contour_list.end(), approx.begin(), approx.end());
		}
		drawContours(src, contour_list, -1, (255, 0, 0), 2);
		imshow("Objects Detected", src);
		waitKey(30);

	return 0;
}
