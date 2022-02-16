#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
  cv::Mat img(180,768,CV_8UC1);
  // draw dummy image
  img = 0;
  cv::Rect2d rc = {0, 0, 768, 60 };
  for (int y = 0; y < 180; y++) {
	  for (int x = 0; x < 768; x++) {
		  img.at<uchar>(y, x) = x / 3;
	  }
  }
  cv::rectangle(img, rc, { 100 }, 1);
  rc.y += rc.height;
  cv::Mat img1(img);
  auto start = std::chrono::high_resolution_clock::now();
  img1.convertTo(img1, CV_32F,1.0/255.0);
  cv::pow(img1, 2.3F, img1);
  img1.convertTo(img1, CV_8UC1,255.0);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "time1 = " << duration.count() << " ms" << std::endl;
  img1(rc).copyTo(img(rc));
    cv::rectangle(img, rc, { 250 }, 1);
  rc.y += rc.height;
  start = std::chrono::high_resolution_clock::now();
  for (int y = rc.y; y < 180; y++) {
	  for (int x = 0; x < 768; x++) {
		  img.at<uchar>(y, x) = pow(( (img.at<uchar>(y,x) / 255.0)), 2.4)*255.0;
	  }
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "time2 = " << duration.count() << " ms" << std::endl;

  cv::rectangle(img, rc, { 150 }, 1);
  // save result
  cv::imwrite("lab01.png", img);
}
