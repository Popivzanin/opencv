#include <opencv2/opencv.hpp>
#include <cmath>
using namespace cv;
int main() {
	Mat src_gray_img = imread("cross_0256x0256.png", IMREAD_GRAYSCALE);
	Mat src_rgb_img = imread("cross_0256x0256.png");
	imwrite("lab03_gre.png", src_gray_img);
	imwrite("lab03_rgb.png", src_rgb_img);

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = 128 * (sin(i * 0.125) + 1);
	Mat img;
	LUT(src_rgb_img, lookUpTable, img);
	imwrite("lab03_rgb_res.png", img);

	LUT(src_gray_img, lookUpTable, img);
	imwrite("lab03_gre_res.png", img);

	Mat src_func(512, 512, CV_8UC1, 255);
	for (int i = 1; i < 256; i++)
	{
		line(src_func,Point(i - 1 << 1, 511 - (((int)p[i - 1]) << 1)),Point(i << 1, 511 - (((int)p[i]) << 1)), 0, 1, 0);
	}
	imwrite("lab03_viz_func.png", src_func);
	waitKey();
}