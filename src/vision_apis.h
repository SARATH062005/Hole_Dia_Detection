#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using VType = std::tuple<std::string, std::string, float, float>; //QR, HOLE, DIA, THREAD
float detectHoleDiameter(const std::string& imagePath, cv::Mat img, int station_num, const std::string& holeId);
int find_threads(const std::string& imagePath, cv::Mat img, int station_num, const std::string& holeId);
