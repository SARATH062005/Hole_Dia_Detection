#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <chrono>
#include "HoleParameters.hpp"
#include <string>
#include <iostream>
#include <filesystem>
#include "vision_apis.h"
#include <unordered_set>
#include <tuple>
#include <filesystem>
#include <regex>
#include <map>
#include <opencv2/opencv.hpp>
// #include "camera_manager.h"

#include <fstream>   // For std::ofstream (fixes C2079)
#include <numeric>   // For std::accumulate (fixes C2039)
#include <cfloat>    // For FLT_MIN (fixes C2065)

using namespace cv;
using namespace std;

// namespace VisionApis{
std::mutex h_mtx;
std::map<std::string, std::pair<double, double>> holeRadiusLimits = {
    {"RC1", {800.0, 1000.0}},
    {"RC2", {800.0, 1200.0}},
    {"RC3", {800.0, 1100.0}},
    {"ML2", {630.0, 970.0}},
    {"LA2", {630.0, 970.0}},
    {"SP5", {660.0, 970.0}},
    {"SP1", {630.0, 810.0}},
    {"SP9", {630.0, 970.0}},
    {"LA3", {560.0, 600.0}},
    {"LA4", {630.0, 970.0}},
    {"LA5", {630.0, 970.0}},
    {"LA6", {555.0, 590.0}},
    {"RS4", {700.0, 806.0}},
    {"LS4", {700.0, 806.0}},
    {"UA1", {700.0, 806.0}},
    {"UA6", {510.0, 630.0}},
};


std::map<std::string, std::pair<double, double>> contourAreaLimits = {
    {"RC1", {671276.0, 2641810.0}},
    {"RC2", {671276.0, 2641810.0}},
    {"RC3", {671276.0, 2641810.0}},
    {"ML2", {671276.0, 2205940.0}},
    {"LA2", {671276.0, 2205940.0}},
    {"SP5", {671276.0, 2205940.0}},
    {"SP1", {671276.0, 2205940.0}},
    {"SP9", {671276.0, 2205940.0}},
    {"LA3", {671276.0, 2205940.0}},
    {"LA4", {671276.0, 2205940.0}},
    {"LA5", {671276.0, 2205940.0}},
    {"LA6", {671276.0, 2205940.0}},
    {"RS4", {671276.0, 2205940.0}},
    {"LS4", {671276.0, 2205940.0}},
    {"UA1", {671276.0, 2205940.0}},
    {"UA6", {671276.0, 2205940.0}},
};

std::map<std::string, std::tuple<int, int, int, int>> diaCoordinates = {
    {"LA3", {940, 624, 3048, 2464}},
    {"LA6", {1000, 392, 3120, 2285}},
    {"LS4", {300, 0, 3000, 2420}},
    {"RS4", {332, 496, 3080, 3000}},
    {"UA1", {480, 544, 3024, 2700}},
    {"LA2", {952, 708, 3420, 3000}},
    {"ML2", {1296, 860, 3456, 3000}},
    {"RC1", {1024, 500, 3564, 2712}},
    {"RC2", {760, 792, 3384, 2888}},
    {"RC3", {1120, 800, 3352, 3000}},
    {"UA6", {960, 480, 3260, 2500}},
    {"LA5", {540, 365, 3084, 2444}},
    {"LA4", {894, 420, 3090, 2324}},
    {"SP1", {724, 552, 3156, 2788}},
    {"SP5", {1068, 928, 3212, 2800}},
    {"SP9", {1540, 860, 4000, 3000}},

};

std::map<std::string, std::pair<float, float>> radius_range_map = {
    {"RC1", {800.0f, 1000.0f}},
    {"RC2", {800.0f, 1200.0f}},
    {"RC3", {800.0f, 1100.0f}},
    {"ML2", {630.0f, 970.0f}},
    {"LA2", {630.0f, 970.0f}},
    {"SP5", {660.0f, 970.0f}},
    {"SP1", {630.0f, 810.0f}},
    {"SP9", {630.0f, 970.0f}},
    {"LA3", {560.0f, 600.0f}},
    {"LA4", {630.0f, 970.0f}},
    {"LA5", {630.0f, 970.0f}},
    {"LA6", {555.0f, 590.0f}},
    {"RS4", {700.0f, 806.0f}},
    {"LS4", {700.0f, 806.0f}},
    {"UA1", {700.0f, 806.0f}},
    {"UA6", {510.0f, 630.0f}},
};

std::map<std::string, std::tuple<int, int, int, int>> holeCropCoords = {
    {"LA6", {800, 400, 3000, 2500}},
    {"LA3", {400, 600, 2600, 2600}},
    {"LS4", {900, 20, 4000, 2600}},
    {"RS4", {600, 20, 4000, 2700}},
    {"UA1", {772, 20, 4000, 2700}}};

// A struct to hold the results from the edge search
struct EdgeResult
{
    double radius = -1.0;               // The detected edge radius
    std::vector<cv::Point> edge_points; // The (x, y) coordinates on the edge arc
};

double averageOfTopKMostFrequentRadii(const std::vector<double> &radii, int top_k = 3, int precision = 2)
{
    if (radii.empty() || top_k <= 0)
        return 0.0;

    std::unordered_map<long long, int> histogram;
    const double scale = std::pow(10.0, precision);

    // Step 1: Build histogram with rounding
    for (double r : radii)
    {
        long long key = static_cast<long long>(std::round(r * scale));
        histogram[key]++;
    }

    // Step 2: Copy to vector for sorting
    std::vector<std::pair<long long, int>> sorted_freq(histogram.begin(), histogram.end());

    // Step 3: Sort descending by frequency
    std::sort(sorted_freq.begin(), sorted_freq.end(),
        [](const auto &a, const auto &b)
        {
            return a.second > b.second; // sort by count
        });

    // Step 4: Compute average of top K values
    int count = std::min(top_k, static_cast<int>(sorted_freq.size()));
    double sum = 0.0;

    for (int i = 0; i < count; ++i)
    {
        double radius = static_cast<double>(sorted_freq[i].first) / scale;
        sum += radius;
    }

    return sum / count;
}

std::pair<std::vector<double>, std::vector<cv::Point2f>> trimOutliersByRadius(
    const std::vector<double> &radii,
    const std::vector<cv::Point2f> &points,
    double trim_fraction = 0.1) // trim 10% from top and bottom by default
{
    if (radii.size() != points.size())
    {
        throw std::invalid_argument("Size mismatch between radii and points.");
    }

    size_t total_size = radii.size();
    if (total_size < 4 || trim_fraction <= 0.0 || trim_fraction >= 0.5)
    {
        // Nothing to trim or unsafe to trim
        return {radii, points};
    }

    // Zip radii and points
    std::vector<std::pair<double, cv::Point2f>> zipped;
    zipped.reserve(total_size);
    for (size_t i = 0; i < total_size; ++i)
    {
        zipped.emplace_back(radii[i], points[i]);
    }

    // Sort based on radius
    std::sort(zipped.begin(), zipped.end(),
        [](const auto &a, const auto &b)
        {
            return a.first < b.first;
        });

    // Calculate trim count
    size_t trim_count = static_cast<size_t>(trim_fraction * total_size);
    if (2 * trim_count >= total_size)
    {
        // Not enough to trim safely
        return {radii, points};
    }

    // Trim
    auto trimmed_begin = zipped.begin() + trim_count;
    auto trimmed_end = zipped.end() - trim_count;

    // Unzip back
    std::vector<double> trimmed_radii;
    std::vector<cv::Point2f> trimmed_points;
    trimmed_radii.reserve(trimmed_end - trimmed_begin);
    trimmed_points.reserve(trimmed_end - trimmed_begin);

    for (auto it = trimmed_begin; it != trimmed_end; ++it)
    {
        trimmed_radii.push_back(it->first);
        trimmed_points.push_back(it->second);
    }

    return {trimmed_radii, trimmed_points};
}

// Returns std::vector<cv::Point2f> with the four trapezoid corners in order
std::vector<cv::Point2f> computeTrapezoidPoints(
    const cv::Point2f &center, // Center point (x, y)
    double angle_rad,          // Angle of the ray in radians
    double delta_deg,          // Angular offset in degrees (±delta)
    double start_radius,       // Start radius
    double end_radius          // End radius
)
{
    double delta_rad = delta_deg * CV_PI / 180.0;
    std::vector<cv::Point2f> points(4);

    // FIX: Explicitly cast the results of math functions (which are double) to float.
    // Inner arc points (start_radius, at angle ± delta)
    points[0].x = static_cast<float>(center.x + start_radius * std::cos(angle_rad - delta_rad));
    points[0].y = static_cast<float>(center.y + start_radius * std::sin(angle_rad - delta_rad));

    points[1].x = static_cast<float>(center.x + start_radius * std::cos(angle_rad + delta_rad));
    points[1].y = static_cast<float>(center.y + start_radius * std::sin(angle_rad + delta_rad));

    // Outer arc points (end_radius, at angle ± delta)
    points[2].x = static_cast<float>(center.x + end_radius * std::cos(angle_rad + delta_rad));
    points[2].y = static_cast<float>(center.y + end_radius * std::sin(angle_rad + delta_rad));

    points[3].x = static_cast<float>(center.x + end_radius * std::cos(angle_rad - delta_rad));
    points[3].y = static_cast<float>(center.y + end_radius * std::sin(angle_rad - delta_rad));


    return points;
}

cv::Mat drawEdgeAndTrapezoid(
    const cv::Mat &image,
    const cv::Point2f &center,
    double edge_radius,
    double start_radius,
    double end_radius,
    double angle_rad,
    double delta_deg,
    const cv::Scalar &trapezoid_color = cv::Scalar(0, 255, 0), // Green
    const cv::Scalar &arc_color = cv::Scalar(0, 0, 255),       // Red
    int thickness = 2)
{
    // Create a copy to draw on
    cv::Mat annotated = image.clone();
    double delta_rad = delta_deg * CV_PI / 180.0;

    // 1. Draw the Trapezoid
    // Calculate the four corner points of the trapezoid
    std::vector<cv::Point2f> trapezoid_points(4);
    // FIX: Explicitly cast double results to float for Point2f constructor
    trapezoid_points[0] = cv::Point2f(static_cast<float>(center.x + start_radius * std::cos(angle_rad - delta_rad)), static_cast<float>(center.y + start_radius * std::sin(angle_rad - delta_rad)));
    trapezoid_points[1] = cv::Point2f(static_cast<float>(center.x + start_radius * std::cos(angle_rad + delta_rad)), static_cast<float>(center.y + start_radius * std::sin(angle_rad + delta_rad)));
    trapezoid_points[2] = cv::Point2f(static_cast<float>(center.x + end_radius * std::cos(angle_rad + delta_rad)), static_cast<float>(center.y + end_radius * std::sin(angle_rad + delta_rad)));
    trapezoid_points[3] = cv::Point2f(static_cast<float>(center.x + end_radius * std::cos(angle_rad - delta_rad)), static_cast<float>(center.y + end_radius * std::sin(angle_rad - delta_rad)));

    // Convert points to integer for drawing
    std::vector<cv::Point> int_points;
    for (const auto &pt : trapezoid_points)
    {
        int_points.push_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
    }

    // Draw the closed polygon
    const cv::Point *pts = int_points.data();
    int npts = 4;
    cv::polylines(annotated, &pts, &npts, 1, true, trapezoid_color, thickness, cv::LINE_AA);

    // 2. Draw the Edge Arc
    if (edge_radius > 0)
    {
        // OpenCV's ellipse function requires angles in degrees
        double start_angle_deg = (angle_rad - delta_rad) * 180.0 / CV_PI;
        double end_angle_deg = (angle_rad + delta_rad) * 180.0 / CV_PI;

        cv::ellipse(
            annotated,
            center,
            cv::Size(cvRound(edge_radius), cvRound(edge_radius)),
            0,
            start_angle_deg,
            end_angle_deg,
            arc_color,
            thickness,
            cv::LINE_AA);
    }

    return annotated;
}


// Function to draw a trapezoid and return the annotated image
cv::Mat drawTrapezoid(
    const cv::Mat &image,
    const std::vector<cv::Point2f> &trapezoid_points,
    const cv::Scalar &color = cv::Scalar(0, 255, 0), // Default: Green
    int thickness = 1)
{
    // Make a copy to avoid modifying the input image
    cv::Mat annotated = image.clone();

    // Convert to integer points as required by OpenCV drawing functions
    std::vector<cv::Point> int_points;
    for (const auto &pt : trapezoid_points)
        int_points.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));

    // Draw the trapezoid (closed polygon) on the annotated image
    const cv::Point *pts = int_points.data();
    int npts = static_cast<int>(int_points.size());
    cv::polylines(annotated, &pts, &npts, 1, true, color, thickness, cv::LINE_AA);

    // Return the annotated image
    return annotated;
}
void saveDiameterToCSV(const std::string &filename, const float &diameter, const std::string &csvPath)
{
    std::ofstream csvFile(csvPath, std::ios::app); // Open in append mode
    if (!csvFile.is_open())
    {
        std::cerr << "Could not open CSV file at: " << csvPath << "\n";
        return;
    }

    csvFile << filename << "," << diameter << "\n";
    csvFile.close();

    std::cout << "Saved: " << filename << " with diameter: " << diameter << " to " << csvPath << "\n";
}

// void saveDiameterToCSV_cycle_data(
//     const std::string& csvPath,
//     const std::map<std::string, std::map<std::string, float>>& cycleHoleMap,
//     const std::vector<std::string>& holeOrder)
// {
//     std::ofstream csvFile(csvPath);
//     if (!csvFile.is_open()) {
//         std::cerr << "Failed to open CSV file: " << csvPath << "\n";
//         return;
//     }

//     // Header row
//     csvFile << "cycle_id";
//     for (const auto& hole : holeOrder) {
//         csvFile << "\t" << hole;
//     }
//     csvFile << "\n";

//     // Data rows
//     // for (const auto& [cycleID, holeMap] : cycleHoleMap) {
//         csvFile << cycleID;
//         for (const auto& hole : holeOrder) {
//             auto it = holeMap.find(hole);
//             if (it != holeMap.end())
//                 csvFile << "\t" << it->second;  // -1 or valid
//             else
//                 csvFile << "\t0.0";  // missing hole: write 0.0
//         }
//         csvFile << "\n";
//     // }

//     csvFile.close();
//     std::cout << "✅ Saved diameter data to CSV: " << csvPath << "\n";
// }

const std::unordered_set<std::string> rc_group_ids = {"SP9"};
const std::unordered_set<std::string> ml_la_group_ids = {"SP5", "SP1", "RC3", "ML2", "LA2", "RC1", "RC2", "LA3", "LA6", "RS4", "LS4", "UA1", "UA6", "LA4", "LA5"};

//Mask function for Only in border of the image ----->>>>>>>>>>>>>>>
cv::Mat maskImageEdges(const cv::Mat &image, int border_size)
{
    // Check if the input image is empty
    if (image.empty())
    {
        std::cerr << "Warning: Input image to maskImageEdges is empty." << std::endl;
        return cv::Mat();
    }

    // Work on a copy to avoid modifying the original image
    cv::Mat maskedImage = image.clone();

    // Determine the fill color (255 for single-channel, white for three-channel)
    cv::Scalar fillColor = (image.channels() == 3) ? cv::Scalar(255, 255, 255) : cv::Scalar(255);

    // Get image dimensions
    int rows = image.rows;
    int cols = image.cols;

    // Define the four edge Regions of Interest (ROIs) using cv::Rect
    // Top edge
    cv::Rect top_edge(0, 0, cols, border_size);
    // Bottom edge
    cv::Rect bottom_edge(0, rows - border_size, cols, border_size);
    // Left edge
    cv::Rect left_edge(0, 0, border_size, rows);
    // Right edge
    cv::Rect right_edge(cols - border_size, 0, border_size, rows);

    // Draw a filled white rectangle on each of the four edges
    // The cv::FILLED flag ensures the rectangles are filled with the specified color
    cv::rectangle(maskedImage, top_edge, fillColor, cv::FILLED);
    cv::rectangle(maskedImage, bottom_edge, fillColor, cv::FILLED);
    cv::rectangle(maskedImage, left_edge, fillColor, cv::FILLED);
    cv::rectangle(maskedImage, right_edge, fillColor, cv::FILLED);

    return maskedImage;
}


cv::Mat maskImage_handleMlLaGroup(const cv::Mat &inputImage, int x1, int y1, int x2, int y2)
{
    try
    {
        if (inputImage.empty())
        {
            throw std::runtime_error("Input image is empty.");
        }

        // Make sure coordinates are in valid range
        x1 = std::clamp(x1, 0, inputImage.cols);
        y1 = std::clamp(y1, 0, inputImage.rows);
        x2 = std::clamp(x2, 0, inputImage.cols);
        y2 = std::clamp(y2, 0, inputImage.rows);

        if (x2 <= x1 || y2 <= y1)
        {
            throw std::runtime_error("Invalid crop coordinates (x2 <= x1 or y2 <= y1).");
        }

        int width = x2 - x1;
        int height = y2 - y1;

        // Determine fill color based on image type
        cv::Scalar fillColor;
        if (inputImage.channels() == 1)
        {
            fillColor = cv::Scalar(255); // white for grayscale
        }
        else if (inputImage.channels() == 3)
        {
            fillColor = cv::Scalar(255, 255, 255); // white for BGR
        }
        else
        {
            throw std::runtime_error("Unsupported number of channels.");
        }

        // Create white image
        cv::Mat maskedImage(inputImage.size(), inputImage.type(), fillColor);

        // Define valid ROI
        cv::Rect roi(x1, y1, width, height);

        // Copy region from input to masked
        inputImage(roi).copyTo(maskedImage(roi));

        return maskedImage;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in maskImage_handleMlLaGroup: " << e.what() << std::endl;
        return cv::Mat(); // Return empty matrix on error
    }
}

struct Edge
{
    float x, y, strength;
};

// Circle structure (from Program 1)
struct Circle
{
    float x, y, r;
};

// Timer utility (from Program 1)
class Timer
{
public:
    Timer(const std::string &msg) : message(msg), start(std::chrono::high_resolution_clock::now()) {}
    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "[Time] " << message << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    }

private:
    std::string message;
    std::chrono::high_resolution_clock::time_point start;
};

// cropImage function (common to both, slight merge for robustness)
std::tuple<cv::Mat, int, int, int, int, float> cropImage(const cv::Mat &image, int x1, int y1, float radius_passthrough)
{
    int cropSize = 2300; // Fixed crop window size

    int startX = std::max(x1 - cropSize / 2, 0);
    int startY = std::max(y1 - cropSize / 2, 0);
    int endX = std::min(x1 + cropSize / 2, image.cols);
    int endY = std::min(y1 + cropSize / 2, image.rows);

    if (startX >= endX || startY >= endY)
    {
        std::cerr << "Warning: Invalid crop dimensions in cropImage. "
                  << "Request: center(" << x1 << "," << y1 << ") size " << cropSize
                  << " on image " << image.cols << "x" << image.rows
                  << ". Resulted in: startX=" << startX << ", endX=" << endX
                  << ", startY=" << startY << ", endY=" << endY << std::endl;
        return std::make_tuple(cv::Mat(), 0, 0, 0, 0, radius_passthrough);
    }

    int newX1 = x1 - startX; // Center X in cropped image
    int newY1 = y1 - startY; // Center Y in cropped image

    cv::Rect roi(startX, startY, endX - startX, endY - startY);
    cv::Mat croppedImage = image(roi).clone();
    // delete image;
    return std::make_tuple(croppedImage, newX1, newY1, startX, startY, radius_passthrough);
}

// Compute inner circle mean
float computeInnerCircleMean(const cv::Mat &grayImage, const cv::Point2f &center, float radius)
{
    cv::Mat insideMask = cv::Mat::zeros(grayImage.size(), CV_8U);
    cv::Mat outsideMask = cv::Mat::zeros(grayImage.size(), CV_8U);

    // Draw filled circle for inside
    cv::circle(insideMask, center, static_cast<int>(radius * 0.99), cv::Scalar(255), -1);
    cv::circle(insideMask, center, static_cast<int>(radius * 0.9), cv::Scalar(0), -1);

    // Draw ring (annulus) for outside
    cv::circle(outsideMask, center, static_cast<int>(radius * 1.06), cv::Scalar(255), -1);
    cv::circle(outsideMask, center, static_cast<int>(radius * 1.01), cv::Scalar(0), -1);

    // Mean intensity inside and outside
    double meanInside = cv::mean(grayImage, insideMask)[0];
    double meanOutside = cv::mean(grayImage, outsideMask)[0];

    return static_cast<float>(meanInside - meanOutside);
}

std::tuple<float, float, float, cv::Mat, cv::Mat> computeInnerCircleMean_min_max(const cv::Mat &grayImage, const cv::Point2f &center, float radius)
{
    cv::Mat insideMask = cv::Mat::zeros(grayImage.size(), CV_8U);
    cv::Mat outsideMask = cv::Mat::zeros(grayImage.size(), CV_8U);

    // Inner ring
    cv::circle(insideMask, center, static_cast<int>(radius * 0.99), cv::Scalar(255), -1);
    cv::circle(insideMask, center, static_cast<int>(radius * 0.9), cv::Scalar(0), -1);

    // Outer ring
    cv::circle(outsideMask, center, static_cast<int>(radius * 1.06), cv::Scalar(255), -1);
    cv::circle(outsideMask, center, static_cast<int>(radius * 1.01), cv::Scalar(0), -1);

    double meanInside = cv::mean(grayImage, insideMask)[0];
    double meanOutside = cv::mean(grayImage, outsideMask)[0];
    float diff = static_cast<float>(meanInside - meanOutside);

    return std::make_tuple(static_cast<float>(meanInside),
                           static_cast<float>(meanOutside),
                           diff, insideMask, outsideMask);
}
// subpixelEdges function (from Program 1)
std::vector<Edge> subpixelEdges(const cv::Mat &image, float minEdgeThreshold, float maxEdgeThreshold)
{
    Timer timer("Subpixel edge detection"); // Optional timer

    cv::Mat gx, gy;
    // Ensure image is CV_8U for Sobel if not already guaranteed
    cv::Mat gray_for_sobel = image;
    if (image.depth() != CV_8U)
    {
        image.convertTo(gray_for_sobel, CV_8U); // Or handle error
    }

    cv::Sobel(gray_for_sobel, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray_for_sobel, gy, CV_32F, 0, 1, 3);

    std::vector<Edge> edges;
    edges.reserve(image.rows * image.cols / 10); // Pre-allocate roughly

    for (int y = 1; y < image.rows - 1; ++y)
    { // Avoid borders where Sobel is ill-defined
        for (int x = 1; x < image.cols - 1; ++x)
        {
            float dx = gx.at<float>(y, x);
            float dy = gy.at<float>(y, x);
            float magnitude = std::sqrt(dx * dx + dy * dy);

            if ((magnitude > minEdgeThreshold) && (magnitude < maxEdgeThreshold))
            {
                edges.push_back({static_cast<float>(x), static_cast<float>(y), magnitude});
            }
        }
    }
    return edges;
}

// fitCircleRANSAC function (from Program 1, with minor robustness enhancement)
Circle fitCircleRANSAC(const std::vector<cv::Point2f> &points, cv::Mat gradient_image, int iterations = 300, float RANSAC_threshold = 2.0f, int MIN_points = 80)
{
    Timer timer("RANSAC circle fit"); // Optional timer

    if (points.size() < 3)
        return {0, 0, 0}; // Cannot fit a circle

    std::mt19937 rng(42); // P1 used fixed seed 42 for repeatability
    // std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count()); // Alternative: time-based seed
    std::uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

    Circle bestCircle = {0, 0, 0};
    int bestInliers = 0;
    // float best_mse = __FLT_MIN__;
    float best_mse = -FLT_MAX;

    for (int i = 0; i < iterations; ++i)
    {
        int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);

        // Ensure 3 distinct points are chosen, especially if points.size() is small
        if (points.size() > 3)
        { // If many points, simple retry is fine
            int retries = 0;
            while ((i1 == i2 || i1 == i3 || i2 == i3) && retries < 10)
            { // Prevent infinite loop
                i2 = dist(rng);
                i3 = dist(rng); // Re-pick two
                retries++;
            }
            if (i1 == i2 || i1 == i3 || i2 == i3)
                continue; // Skip if still not distinct
        }
        else
        { // If points.size() == 3, indices must be 0,1,2 in some order
            if (i1 == i2 || i1 == i3 || i2 == i3)
                continue; // Should not happen if dist is correct for N=3
        }

        cv::Point2f A = points[i1], B = points[i2], C = points[i3];

        // Math for circle from 3 points (ax + by = c form)
        float a_val = B.x - A.x, b_val = B.y - A.y;
        float c_val = C.x - A.x, d_val = C.y - A.y;
        float e_val = a_val * (A.x + B.x) + b_val * (A.y + B.y); // 2* (a*midX1 + b*midY1)
        float f_val = c_val * (A.x + C.x) + d_val * (A.y + C.y); // 2* (c*midX2 + d*midY2)
        float g_val = 2.0f * (a_val * (C.y - B.y) - b_val * (C.x - B.x));

        if (std::abs(g_val) < 1e-6)
            continue; // Collinear points or very close to it

        float cx = (d_val * e_val - b_val * f_val) / g_val;
        float cy = (a_val * f_val - c_val * e_val) / g_val;
        float r_sq = (A.x - cx) * (A.x - cx) + (A.y - cy) * (A.y - cy);
        float r = std::sqrt(r_sq);

        // Additional check for extremely large radius (often from near-collinear points)
        if (r > (std::max(A.x, std::max(B.x, C.x)) - std::min(A.x, std::min(B.x, C.x)) +
                 std::max(A.y, std::max(B.y, C.y)) - std::min(A.y, std::min(B.y, C.y))) *
                    10 &&
            r > 1000)
        { // Heuristic
            continue;
        }

        int inliers = 0;
        float current_mse = 0.0f;
        int x, y;
        for (double deg = 0; deg < 360; deg += 0.5)
        {
            x = static_cast<int>(std::round(cx + r * std::cos(deg)));
            y = static_cast<int>(std::round(cy + r * std::sin(deg)));

            if ((x >= 0 && x < gradient_image.cols && y >= 0 && y < gradient_image.rows))
            {
                current_mse += std::abs(static_cast<int>(gradient_image.at<uchar>(y, x)));
            }
        }

        /*
        for (const auto& p : points) {
            float dist_to_center_sq = (p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy);
            // Compare distance to radius ( |sqrt(dist_sq) - r| < threshold )
            // More stable: |dist_sq - r_sq| < (2*r*threshold - threshold^2) approx 2*r*threshold for small threshold
            float error = std::abs(std::sqrt(dist_to_center_sq) - r);
            if (error < RANSAC_threshold) {
                ++inliers;
                current_mse += error;
            }
        }

        current_mse = current_mse / inliers;
        */

        /*
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestCircle = {cx, cy, r};
        }

       std::cout << "Iteration: " << i << " Inliers: " << inliers << " Radius : " << r << "\n";
       if (inliers > MIN_points){
            if (current_mse < best_mse){
                best_mse = current_mse;
                bestCircle = {cx, cy, r};
            }
       }
        */
        if (current_mse > best_mse)
        {
            best_mse = current_mse;
            bestCircle = {cx, cy, r};
        }
    }

    return bestCircle;
}

Circle fitCircleRANSAC_final(const std::vector<cv::Point2f> &points, cv::Mat gradient_image,
                             float expected_radius, // Required: expected radius in pixels
                             int iterations = 300,
                             float radius_tol = 0.1f,          // 10% allowed deviation
                             float arc_coverage_thresh = 0.5f, // Min arc coverage ratio (e.g., 50%)
                             int edge_threshold = 0.0)
{ // Sobel threshold for arc coverage
    // Optional timer for debug; remove if Timer not defined for your project
    // Timer timer("RANSAC circle fit");

    if (points.size() < 3)
        return {0, 0, 0}; // Need at least 3 points to fit a circle

    std::mt19937 rng(42); // Fixed seed for repeatability
    std::uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

    Circle bestCircle = {0, 0, 0};
    float bestScore = -1.0f;

    for (int i = 0; i < iterations; ++i)
    {
        // Randomly select 3 distinct points
        int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
        int retries = 0;
        while ((i1 == i2 || i1 == i3 || i2 == i3) && retries < 10)
        {
            i2 = dist(rng);
            i3 = dist(rng);
            retries++;
        }
        if (i1 == i2 || i1 == i3 || i2 == i3)
            continue;

        cv::Point2f A = points[i1], B = points[i2], C = points[i3];

        // Calculate circle from 3 points (analytic method)
        float a_val = B.x - A.x, b_val = B.y - A.y;
        float c_val = C.x - A.x, d_val = C.y - A.y;
        float e_val = a_val * (A.x + B.x) + b_val * (A.y + B.y);
        float f_val = c_val * (A.x + C.x) + d_val * (A.y + C.y);
        float g_val = 2.0f * (a_val * (C.y - B.y) - b_val * (C.x - B.x));
        if (std::abs(g_val) < 1e-6)
            continue; // Collinear points

        float cx = (d_val * e_val - b_val * f_val) / g_val;
        float cy = (a_val * f_val - c_val * e_val) / g_val;
        float r_sq = (A.x - cx) * (A.x - cx) + (A.y - cy) * (A.y - cy);
        float r = std::sqrt(r_sq);

        // Radius constraint (±radius_tol of expected_radius)
        if (std::abs(r - expected_radius) > expected_radius * radius_tol)
            continue;

        // Heuristic for extremely large radius (rare, safe guard)
        if (r > (std::max(A.x, std::max(B.x, C.x)) - std::min(A.x, std::min(B.x, C.x)) +
                 std::max(A.y, std::max(B.y, C.y)) - std::min(A.y, std::min(B.y, C.y))) *
                    10 &&
            r > 1000)
        {
            continue;
        }

        // Evaluate candidate circle: accumulate Sobel edges around perimeter,
        // calculate normalized score and arc coverage
        float score = 0.0f;
        int valid_points = 0, edge_points = 0;
        int x, y;
        for (double deg = 0; deg < 360; deg += 0.5)
        {
            double rad = deg * CV_PI / 180.0;
            x = static_cast<int>(std::round(cx + r * std::cos(rad)));
            y = static_cast<int>(std::round(cy + r * std::sin(rad)));
            if (x >= 0 && x < gradient_image.cols && y >= 0 && y < gradient_image.rows)
            {
                int sobel = static_cast<int>(std::abs(gradient_image.at<uchar>(y, x)));
                score += sobel;
                valid_points++;
                if (sobel >= edge_threshold)
                    edge_points++;
            }
        }
        if (valid_points == 0)
            continue; // Shouldn't happen, but avoid divide by zero.
        float normalized_score = score / valid_points;
        float arc_coverage = static_cast<float>(edge_points) / valid_points;
        if (arc_coverage < arc_coverage_thresh)
            continue; // Not enough real edge on perimeter

        if (normalized_score > bestScore)
        {
            bestScore = normalized_score;
            bestCircle = {cx, cy, r};
        }
    }
    return bestCircle;
}

// fitCircleRANSAC function (from Program 1, with minor robustness enhancement)
Circle fitCircleRANSAC_RC_group(const std::vector<cv::Point2f> &points, int iterations = 300, float RANSAC_threshold = 2.0f)
{
    Timer timer("RANSAC circle fit"); // Optional timer

    if (points.size() < 3)
        return {0, 0, 0}; // Cannot fit a circle

    std::mt19937 rng(42); // P1 used fixed seed 42 for repeatability
    // std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count()); // Alternative: time-based seed
    std::uniform_int_distribution<int> dist(0, static_cast<int>(0.9 * points.size()) - 1);

    Circle bestCircle = {0, 0, 0};
    int bestInliers = 0;

    for (int i = 0; i < iterations; ++i)
    {
        int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);

        // Ensure 3 distinct points are chosen, especially if points.size() is small
        if (points.size() > 3)
        { // If many points, simple retry is fine
            int retries = 0;
            while ((i1 == i2 || i1 == i3 || i2 == i3) && retries < 10)
            { // Prevent infinite loop
                i2 = dist(rng);
                i3 = dist(rng); // Re-pick two
                retries++;
            }
            if (i1 == i2 || i1 == i3 || i2 == i3)
                continue; // Skip if still not distinct
        }
        else
        { // If points.size() == 3, indices must be 0,1,2 in some order
            if (i1 == i2 || i1 == i3 || i2 == i3)
                continue; // Should not happen if dist is correct for N=3
        }

        cv::Point2f A = points[i1], B = points[i2], C = points[i3];

        // Math for circle from 3 points (ax + by = c form)
        float a_val = B.x - A.x, b_val = B.y - A.y;
        float c_val = C.x - A.x, d_val = C.y - A.y;
        float e_val = a_val * (A.x + B.x) + b_val * (A.y + B.y); // 2* (a*midX1 + b*midY1)
        float f_val = c_val * (A.x + C.x) + d_val * (A.y + C.y); // 2* (c*midX2 + d*midY2)
        float g_val = 2.0f * (a_val * (C.y - B.y) - b_val * (C.x - B.x));

        if (std::abs(g_val) < 1e-6)
            continue; // Collinear points or very close to it

        float cx = (d_val * e_val - b_val * f_val) / g_val;
        float cy = (a_val * f_val - c_val * e_val) / g_val;
        float r_sq = (A.x - cx) * (A.x - cx) + (A.y - cy) * (A.y - cy);
        float r = std::sqrt(r_sq);

        // Additional check for extremely large radius (often from near-collinear points)
        if (r > (std::max(A.x, std::max(B.x, C.x)) - std::min(A.x, std::min(B.x, C.x)) +
                 std::max(A.y, std::max(B.y, C.y)) - std::min(A.y, std::min(B.y, C.y))) *
                    10 &&
            r > 1000)
        { // Heuristic
            continue;
        }

        int inliers = 0;
        for (const auto &p : points)
        {
            float dist_to_center_sq = (p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy);
            // Compare distance to radius ( |sqrt(dist_sq) - r| < threshold )
            // More stable: |dist_sq - r_sq| < (2*r*threshold - threshold^2) approx 2*r*threshold for small threshold
            if (std::abs(std::sqrt(dist_to_center_sq) - r) < RANSAC_threshold)
            {
                ++inliers;
            }
        }

        if (inliers > bestInliers)
        {
            bestInliers = inliers;
            bestCircle = {cx, cy, r};
        }
    }
    return bestCircle;
}

double findSubpixelEdge_new(
    const cv::Mat &grad_magnitude, // Should be CV_8U now
    cv::Point2f center_in_crop,
    const double angle_rad,
    const double search_start_radius,
    const double search_end_radius)
{
    double max_rsan = -1.0;
    int max_grad_val = 0;
    double step = 0.5; // Scan step along the radius
    int x1, y1, x2, y2, cnt = 0;

    if (grad_magnitude.depth() != CV_8U || grad_magnitude.channels() != 1)
    {
        std::cerr << "Error: grad_magnitude is not CV_8U single channel!\n";
        return max_rsan;
    }

    for (double r_scan = search_start_radius; r_scan <= search_end_radius; r_scan += step)
    {
        if (!(r_scan > 0 && std::isfinite(r_scan) && std::isfinite(angle_rad)))
        {
            std::cout << "Invalid values  RScan, ANGLE ---> " << r_scan << " | " << angle_rad << "\n";
            break;
        }

        x1 = static_cast<int>(std::round(center_in_crop.x + r_scan * std::cos(angle_rad)));
        y1 = static_cast<int>(std::round(center_in_crop.y + r_scan * std::sin(angle_rad)));
        x2 = static_cast<int>(std::round(center_in_crop.x + (r_scan + 10 * step) * std::cos(angle_rad)));
        y2 = static_cast<int>(std::round(center_in_crop.y + (r_scan + 10 * step) * std::sin(angle_rad)));

        ++cnt;

        if ((x1 >= 0 && x1 < grad_magnitude.cols && y1 >= 0 && y1 < grad_magnitude.rows) &&
            (x2 >= 0 && x2 < grad_magnitude.cols && y2 >= 0 && y2 < grad_magnitude.rows))
        {

            //   std::cout << "\033[1;32m"
            //           << "\n\t current_grad -->" << static_cast<int>(grad_magnitude.at<uchar>(y1, x1))
            //           << "max_grad_val-->" << static_cast<int>(grad_magnitude.at<uchar>(y2, x2))
            //           << "\033[0m" << "\n";

            int current_grad = std::abs(static_cast<int>(grad_magnitude.at<uchar>(y1, x1)) - static_cast<int>(grad_magnitude.at<uchar>(y2, x2)));

            if (current_grad > max_grad_val)
            {
                max_grad_val = current_grad;

                max_rsan = r_scan;

                // std::cout << "\033[1;32m"
                //       << "\n\t current_grad------------- -->" << current_grad
                //       << "max_grad_val---------------------->" << max_grad_val
                //       << "\033[0m" << "\n";
            }
        }
        else
        {
            std::cerr << "Out of bounds: (" << x2 << ", " << y2 << ")\n";
            break;
        }
    }
    // std::cout << " Edge at : " << max_rsan << " , " << max_grad_val << std::endl;

    return max_rsan;
}

EdgeResult findEdgeInTrapezoid_withPoints(
    const cv::Mat &grad_magnitude,
    const cv::Point2f &center,
    double angle_rad,
    double delta_deg,
    double start_radius,
    double end_radius,
    double scan_step = 0.25)
{
    EdgeResult best_result;
    double max_avg_grad = -1.0;

    double delta_rad = delta_deg * CV_PI / 180.0;
    double angle_start = angle_rad - delta_rad;
    double angle_end = angle_rad + delta_rad;

    // Scan radially
    for (double r = start_radius; r <= end_radius; r += scan_step)
    {
        double current_arc_grad_sum = 0;
        std::vector<cv::Point> current_arc_points, mid_arc_point;

        double current_angle = angle_start + 0.5 * (angle_end - angle_start);
        int x = static_cast<int>(std::round(center.x + r * std::cos(current_angle)));
        int y = static_cast<int>(std::round(center.y + r * std::sin(current_angle)));
        if (x >= 0 && x < grad_magnitude.cols && y >= 0 && y < grad_magnitude.rows)
        {
            mid_arc_point.push_back(cv::Point(x, y));
        }

        // Sample points along the current arc
        int angular_steps = std::max(10, static_cast<int>(r * (2 * delta_rad) / 2));

        for (int i = 0; i <= angular_steps; ++i)
        {
            double current_angle = angle_start + (double(i) / angular_steps) * (angle_end - angle_start);
            int x = static_cast<int>(std::round(center.x + r * std::cos(current_angle)));
            int y = static_cast<int>(std::round(center.y + r * std::sin(current_angle)));

            if (x >= 0 && x < grad_magnitude.cols && y >= 0 && y < grad_magnitude.rows)
            {
                current_arc_grad_sum += grad_magnitude.at<uchar>(y, x);
                current_arc_points.push_back(cv::Point(x, y));
            }
        }

        if (!current_arc_points.empty())
        {
            double current_avg_grad = current_arc_grad_sum / current_arc_points.size();
            // std::cout << "\033[1;32m"
            //       << "\n\t current_grad--->" << current_avg_grad
            //       << "\033[0m" << "\n";

            // If this arc has a stronger gradient, update our best result
            if (current_avg_grad > max_avg_grad)
            {
                max_avg_grad = current_avg_grad;
                best_result.radius = r;
                // best_result.edge_points = current_arc_points; // Store the points
                best_result.edge_points = mid_arc_point;
            }
        }
        // std::cout << "\033[1;32m"

        //           << "max_grad_val---------------------->" << max_avg_grad
        //           << "\033[0m" << "\n";
    }
    return best_result; // Return the best result found

    // if (max_avg_grad < 50.0) {
    //     EdgeResult empty_result;
    //     empty_result.radius = -1.0; // Indicate no edge found
    //     empty_result.edge_points.clear(); // Clear points
    //     return empty_result; // Return empty result if no edge was found
    // }
    // else{
    //     return best_result; // Return the best result found
    // }
}

EdgeResult findEdgeInTrapezoid_withPoints_new(
    const cv::Mat &grad_magnitude,
    const cv::Point2f &center,
    double angle_rad,
    double delta_deg,
    double start_radius,
    double end_radius,
    double scan_step = 0.25)
{
    EdgeResult best_result;
    double max_grad_diff = -1.0;
    double previous_avg_grad = -1.0; // Initialize to track the last gradient value

    double delta_rad = delta_deg * CV_PI / 180.0;
    double angle_start = angle_rad - delta_rad;
    double angle_end = angle_rad + delta_rad;

    // Scan radially from the inner to the outer boundary
    for (double r = start_radius; r <= end_radius; r += scan_step)
    {
        double current_arc_grad_sum = 0;
        std::vector<cv::Point> current_arc_points;
        int angular_steps = std::max(10, static_cast<int>(r * (2 * delta_rad) / 2));

        for (int i = 0; i <= angular_steps; ++i)
        {
            double current_angle = angle_start + (double(i) / angular_steps) * (angle_end - angle_start);
            int x = static_cast<int>(std::round(center.x + r * std::cos(current_angle)));
            int y = static_cast<int>(std::round(center.y + r * std::sin(current_angle)));

            if (x >= 0 && x < grad_magnitude.cols && y >= 0 && y < grad_magnitude.rows)
            {
                current_arc_grad_sum += grad_magnitude.at<uchar>(y, x);
                current_arc_points.push_back(cv::Point(x, y));
            }
        }

        if (!current_arc_points.empty())
        {
            double current_avg_grad = current_arc_grad_sum / current_arc_points.size();

            // Check if we have a previous gradient to compare against
            if (previous_avg_grad >= 0)
            {
                double grad_diff = std::abs(current_avg_grad - previous_avg_grad);

                // If the current increase is the largest found so far, update our result
                if (grad_diff > max_grad_diff)
                {
                    max_grad_diff = grad_diff;
                    best_result.radius = r; // This is the radius of the sharpest increase
                    best_result.edge_points = current_arc_points;
                }
            }
            // Update the previous gradient for the next iteration
            previous_avg_grad = current_avg_grad;
        }
    }

    return best_result;

    // if (max_grad_diff < 7.0) {
    //     EdgeResult empty_result;
    //     empty_result.radius = -1.0; // Indicate no edge found
    //     empty_result.edge_points.clear(); // Clear points
    //     return empty_result; // Return empty result if no edge was found
    // }
    // else{
    //     return best_result; // Return the best result found
    // }
}

// findSubpixelEdge function (from Program 2)
double findSubpixelEdge(
    const cv::Mat &grad_magnitude, // Should be CV_32F
    cv::Point2f center_in_crop,    // Center for radial scan, in grad_magnitude's coordinate system
    const double angle_rad,
    const double search_start_radius,
    const double search_end_radius)
{
    double max_grad_val = 0.0;
    int max_idx_relative = -1;               // Index relative to the start of the scan profile
    std::vector<double> grad_profile;        // Stores gradient values along the ray
    std::vector<double> actual_radii_points; // Stores corresponding radius for each grad_profile point

    double step = 0.5; // Scan step along the radius (P2 used 0.5)
    int smooth_count = 5;

    for (double r_scan = search_start_radius; r_scan <= search_end_radius; r_scan += step)
    {
        int x1 = static_cast<int>(std::round(center_in_crop.x + r_scan * std::cos(angle_rad)));
        int y1 = static_cast<int>(std::round(center_in_crop.y + r_scan * std::sin(angle_rad)));
        int x2 = static_cast<int>(std::round(center_in_crop.x + (r_scan + 2 * step) * std::cos(angle_rad)));
        int y2 = static_cast<int>(std::round(center_in_crop.y + (r_scan + 2 * step) * std::sin(angle_rad)));

        if ((x1 >= 0 && x1 < grad_magnitude.cols && y1 >= 0 && y1 < grad_magnitude.rows) &&
            (x2 >= 0 && x2 < grad_magnitude.cols && y2 >= 0 && y2 < grad_magnitude.rows))
        {

            double current_grad = std::abs(grad_magnitude.at<float>(y1, x1) - grad_magnitude.at<float>(y2, x2));
            grad_profile.push_back(current_grad);
            actual_radii_points.push_back(r_scan);

            if (current_grad > max_grad_val)
            {
                smooth_count = 0;
                max_grad_val = current_grad;
                max_idx_relative = static_cast<int>(grad_profile.size()) - 1;
            }
            /*else{
                smooth_count += 1;
                if(smooth_count > 15){
                    break;
                }
            }*/
        }
        else
        {
            break; // Stop scanning if ray goes out of bounds
        }
    }

    // Check if a peak was found and if it's not at the very edge of the scanned profile
    if (max_idx_relative <= 0 || max_idx_relative >= static_cast<int>(grad_profile.size()) - 1)
    {
        if (max_idx_relative != -1 && !actual_radii_points.empty())
        {
            return actual_radii_points[max_idx_relative]; // Return boundary or isolated max without interpolation
            // return -1.0;
        }
        return -1.0; // No suitable peak for interpolation
    }

    // peak_offset_steps is in units of 'step'. Convert to radius units.
    return actual_radii_points[max_idx_relative];
}

// Calibration factor: pixels per millimeter
const float PIXELS_PER_MM = 1840.62664043f / 1.1254f;

float processSingleContourWithRansac(
    const cv::Mat &gray_img_for_rc,
    const std::vector<cv::Point> &contour,
    const std::string &holeId,
    const std::string &log_prefix,
    const std::string &save_path_if_success,
    bool is_fallback_call)
{
    float diameter_mm = -1.0f;

    auto [minArea, maxArea] = contourAreaLimits.count(holeId) ? contourAreaLimits[holeId] : std::make_pair(0.0, std::numeric_limits<double>::max());

    double contour_area = cv::contourArea(contour);
    if (contour_area < minArea || contour_area > maxArea)
        return -1.0f;

    cv::Point2f mec_center;
    float mec_radius;
    cv::minEnclosingCircle(contour, mec_center, mec_radius);

    auto [minR, maxR] = holeRadiusLimits.count(holeId) ? holeRadiusLimits[holeId] : std::make_pair(0.0, std::numeric_limits<double>::max());

    if (!is_fallback_call && (mec_radius < minR || mec_radius > maxR))
    {
        std::cout << log_prefix << " MEC R=" << mec_radius << "px out of bounds.\n";
        return -1.0f;
    }
    std::cout << log_prefix << " MEC R=" << mec_radius << "px.\n";

    auto [cropGray, cropX, cropY, _, __, ___] = cropImage(gray_img_for_rc,
                                                          static_cast<int>(std::round(mec_center.x)),
                                                          static_cast<int>(std::round(mec_center.y)), mec_radius);
    if (cropGray.empty())
        return -1.0f;

    const cv::Point2f mec_in_crop(cropX, cropY);
    const float tolerance = std::max(mec_radius * 0.001f, 2.0f);

    std::vector<cv::Point2f> inliers;
    for (const auto &e : subpixelEdges(cropGray, 70.0f, 110.0f))
    {
        float dist = std::abs(cv::norm(cv::Point2f(e.x, e.y) - mec_in_crop) - mec_radius);
        if (dist < tolerance)
            inliers.emplace_back(e.x, e.y);
    }

    std::cout << log_prefix << " RANSAC inliers: " << inliers.size() << "\n";
    if (inliers.size() < 30)
        return -1.0f;

    if (inliers.size() > 2000)
    {
        std::shuffle(inliers.begin(), inliers.end(), std::mt19937(42));
        inliers.resize(100);
    }

    Circle circle = fitCircleRANSAC_RC_group(inliers, 300, 2.0f);
    if (circle.r == 0)
    {
        std::cout << log_prefix << " RANSAC failed.\n";
        return -1.0f;
    }

    bool r_in_limits = (circle.r >= minR && circle.r <= maxR);
    std::ostringstream oss;

    if (!r_in_limits)
    {
        if (is_fallback_call)
        {
            oss << "WARNING: RANSAC (Fallback) R=" << std::fixed << std::setprecision(2) << circle.r
                << "px out of limits [" << (int)minR << ", " << (int)maxR << "]";
            std::cout << "\033[1;31m" << oss.str() << "\033[0m\n";
        }
        else
        {
            std::cout << log_prefix << " RANSAC R=" << circle.r << "px out of bounds.\n";
            return -1.0f;
        }
    }

    // diameter_mm = std::round((2.0f * circle.r / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
    diameter_mm = static_cast<float>(std::round((2.0f * circle.r / PIXELS_PER_MM) * 1000.0f) / 1000.0f);
    std::cout << "\033[1;32m" << log_prefix << " RANSAC Fit: R=" << circle.r
              << "px -> D=" << diameter_mm << "mm [" << (r_in_limits ? "OK" : "OUT") << "]\033[0m\n";

    // Drawing & Saving
    cv::Mat cropDisplay;
    cv::cvtColor(cropGray, cropDisplay, cv::COLOR_GRAY2BGR);
    cv::Scalar circle_color = (is_fallback_call && !r_in_limits) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
    cv::circle(cropDisplay, cv::Point2f(circle.x, circle.y), static_cast<int>(circle.r), circle_color, 2);
    cv::circle(cropDisplay, cv::Point2f(circle.x, circle.y), 3, cv::Scalar(0, 0, 255), -1);


    int text_bg_height = r_in_limits ? 150 : 180;
    cv::rectangle(cropDisplay, cv::Rect(0, 0, cropDisplay.cols, text_bg_height), cv::Scalar(200, 200, 200), cv::FILLED);

    auto drawText = [&](const std::string &txt, int y, double scale = 2.0)
    {
        cv::putText(cropDisplay, txt, {10, y}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), 2);
    };

    drawText("D" + std::string(is_fallback_call ? "(fallback)" : "") + ": " +
                 std::to_string(diameter_mm) + "mm",
             50, is_fallback_call ? 1.5 : 2);
    drawText("R" + std::string(is_fallback_call ? "(fallback)" : "") + ": " +
                 std::to_string(circle.r) + "px",
             110, is_fallback_call ? 1.5 : 2);
    if (!r_in_limits)
        drawText("Radius Out Of Limits!", 160, 0.8);

    cv::imwrite(save_path_if_success, cropDisplay);
    std::cout << log_prefix << " Saved: " << save_path_if_success << "\n";

    return diameter_mm;
}

// --- END: Definitions ---

float handleRcGroup(
    cv::Mat &img, // Main image, passed by ref for potential gray conversion
    const std::string &holeId,
    const std::string &save_path)
{
    std::cout << "-> Using RC_GROUP (Contour/RANSAC) logic for " << holeId << "." << std::endl;
    Timer group_timer("RC_GROUP Total for " + holeId);
    float detected_diameter_mm = -1.0f;

    try
    {
        if (img.empty())
        {
            std::cerr << "ERROR (RC): Empty input image.\n";
            return -1.0f;
        }

        // Convert to grayscale only if necessary (in-place reuse of gray_img)
        cv::Mat gray_img;
        if (img.channels() == 3 || img.channels() == 4)
        {
            cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
        }
        else if (img.channels() == 1)
        {
            gray_img = img; // No copy — just a reference
        }
        else
        {
            std::cerr << "ERROR (RC): Unsupported number of channels: " << img.channels() << std::endl;
            return -1.0f;
        }

        // Load parameters
        HoleParams params;
        int threshold_val = (holeId == "RC2" || holeId == "RC3" || holeId == "ML2" || holeId == "LA2") ? 135 : 180;
        auto h_params_it = holeParameters.find(holeId);
        if (h_params_it != holeParameters.end())
        {
            params = h_params_it->second;
            threshold_val = params.threshold;
            std::cout << "Using threshold " << threshold_val << " from HoleParameters (RC) for " << holeId << "\n";
        }
        else
        {
            std::cout << "Warning (RC): HoleParams not found. Using default/specific threshold " << threshold_val << ".\n";
        }

        // Preprocessing (in-place reuse)
        cv::Mat binary;
        cv::GaussianBlur(gray_img, binary, cv::Size(7, 7), 0);                // binary now holds blurred image
        cv::blur(binary, binary, cv::Size(3, 3));                             // further blur in-place
        cv::threshold(binary, binary, threshold_val, 255, cv::THRESH_BINARY); // in-place threshold

        // Morphology (in-place)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::erode(binary, binary, kernel, cv::Point(-1, -1), 2);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty())
        {
            std::cerr << "No contours (RC) for " << holeId << ".\n";
            return -1.0f;
        }

        std::cout << "Contours (RC): " << contours.size() << std::endl;
        std::sort(contours.begin(), contours.end(), [](const auto &a, const auto &b)
                  { return cv::contourArea(a) > cv::contourArea(b); });

        int numContoursToProcess = std::min(5, static_cast<int>(contours.size()));
        for (int i = 0; i < numContoursToProcess; ++i)
        {
            detected_diameter_mm = processSingleContourWithRansac(
                gray_img, contours[i], holeId,
                "(RC Contour " + std::to_string(i) + ")", save_path, false);
            if (detected_diameter_mm >= 0.0f)
            {
                return detected_diameter_mm; // Found a valid result
            }
        }

        std::cout << "No valid RANSAC diameter found after checking contours for " << holeId << ".\n";
    }
    catch (const cv::Exception &cv_e)
    {
        std::cerr << "OpenCV Exception in RC_GROUP for " << holeId << ": " << cv_e.what() << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Generic Exception in RC_GROUP for " << holeId << ": " << e.what() << "\n";
    }

    return -1.0f; // Default failure
}

// Processes the ML_LA_GROUP logic (including fallback)
static float handleMlLaGroup(
    const cv::Mat &img_input_orig,
    cv::Mat &img,
    const std::string &holeId,
    const std::string &primary_save_path,
    const std::string &output_dir_base,
    const std::string &image_file_base_name,
    const int station_num)
// CameraManager *cam
{
    std::unique_lock<std::mutex> iplock(h_mtx);
    // iplock.unlock();
    float detected_diameter_px = -1.0f;
    double detected_diameter_mm = -1.0f;
    cv::Point2f final_center;
    bool attempt_rc_fallback = false;

    if (img_input_orig.empty() || img.empty())
    {
        std::cerr << "ERROR: Invalid input image data (empty Mat)" << std::endl;
        return -1.0f;
    }

    auto range_map_it = radius_range_map.find(holeId);
    if (range_map_it == radius_range_map.end())
    {
        std::cerr << "ERROR: No radius range found for " << holeId << std::endl;
        if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" || holeId == "RS4")
        {
            attempt_rc_fallback = true;
        }
        else
        {
            return -1.0f;
        }
    }

    cv::Mat best_cropped_display_img_ml_la;
    cv::Point2f best_final_center_in_crop_ml_la_primary;
    double best_average_refined_radius_ml_la = 0.0;

    try
    {
        float lower_hough_radius = (range_map_it != radius_range_map.end()) ? range_map_it->second.first : 0.0f;
        float upper_hough_radius = (range_map_it != radius_range_map.end()) ? range_map_it->second.second : 0.0f;

        cv::Mat gray_image_for_ml_la, color_image_for_drawing_base;
        if (img.channels() == 3 || img.channels() == 4)
        {
            cv::cvtColor(img, gray_image_for_ml_la, cv::COLOR_BGR2GRAY);
            color_image_for_drawing_base = img.clone();
        }
        else if (img.channels() == 1)
        {
            gray_image_for_ml_la = img.clone();
            cv::cvtColor(img, color_image_for_drawing_base, cv::COLOR_GRAY2BGR);
        }
        else
        {
            std::cerr << "ERROR (ML_LA): Unsupported channels: " << img.channels() << std::endl;
            if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" || holeId == "RS4")
            {
                attempt_rc_fallback = true;
                throw std::runtime_error("Channel error, attempting fallback");
            }
            return -1.0f;
        }

        if (gray_image_for_ml_la.empty() || color_image_for_drawing_base.empty())
        {
            std::cerr << "ERROR (ML_LA): Critical image (gray or color base) is empty for " << holeId << std::endl;
            if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" || holeId == "RS4")
            {
                attempt_rc_fallback = true;
                throw std::runtime_error("Critical image empty, attempting fallback");
            }
            return -1.0f;
        }

        // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(40, 40));

        cv::Mat blurred_image_full_scale;
        // cv::GaussianBlur(gray_image_for_ml_la, blurred_image_full_scale, cv::Size(9, 9), 2.0, 2.0);
        cv::bilateralFilter(gray_image_for_ml_la, blurred_image_full_scale, 11, 150, 150);

        std::vector<double> result_diameters, result_votes;
        std::vector<cv::Point2f> result_center;

        cv::Mat current_cropped_display_img_ml_la;
        cv::Point2f current_final_center_in_crop_ml_la_primary;
        double current_average_refined_radius_ml_la = 0.0;
        cv::Mat annotated;
        std::string inner_mask_save_path = image_file_base_name + "_trapezoid.jpg";

        // for (double val : {0.45, 0.75, 0.85}) {
        for (double val : {0.95})
        {

            cv::Mat resized_blurred_image;
            float r_mec;
            double hough_scale_factor = val;
            cv::resize(blurred_image_full_scale, resized_blurred_image, cv::Size(),
                       hough_scale_factor, hough_scale_factor, cv::INTER_AREA);

            // clahe->apply(resized_blurred_image, resized_blurred_image);

            std::vector<cv::Vec4f> circles_hough;
            int hough_minR_scaled = static_cast<int>(lower_hough_radius * hough_scale_factor);
            int hough_maxR_scaled = static_cast<int>(upper_hough_radius * hough_scale_factor);

            // cv::Mat tmp_hough_canny, thresholdedImage_hough_canny;
            // cv::bitwise_not(resized_blurred_image, tmp_hough_canny);
            // double otsuThreshold = cv::threshold(tmp_hough_canny, thresholdedImage_hough_canny,
            //                                   0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            cv::Mat tmp_hough_canny, thresholdedImage_hough_canny;
            // cv::bitwise_not(resized_blurred_image, tmp_hough_canny);

            cv::Mat gx, gy;
            cv::Sobel(resized_blurred_image, gx, CV_32F, 1, 0, 3);
            cv::Sobel(resized_blurred_image, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, tmp_hough_canny);
            cv::convertScaleAbs(tmp_hough_canny, tmp_hough_canny);
            double otsuThreshold = cv::threshold(tmp_hough_canny, thresholdedImage_hough_canny,
                                                 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            // iplock.lock();
            cv::HoughCircles(resized_blurred_image, circles_hough, cv::HOUGH_GRADIENT,
                             1, resized_blurred_image.rows / 4.0, 2.0*otsuThreshold,
                             20, hough_minR_scaled, hough_maxR_scaled);
            // iplock.unlock();

            if (circles_hough.empty())
            {
                std::cerr << "No coarse circles (Hough) for " << holeId << " at scale " << hough_scale_factor << std::endl;
                result_diameters.push_back(0.0);
                continue;
            }

            std::cout << "Hough circle size: " << circles_hough.size() << std::endl;

            float max_score = -1.0f;
            cv::Point2f best_center;
            float best_radius = 0.0f;
            float best_mean_inside = 0.0f;
            float best_mean_outside = 0.0f;
            cv::Mat best_inside_mask;
            cv::Mat best_outside_mask;

            int range = std::min(10, static_cast<int>(circles_hough.size()));

            for (int i = 0; i < range; ++i)
            {
                cv::Point2f center(circles_hough[i][0], circles_hough[i][1]);
                float radius = circles_hough[i][2];

                // Get mean_inside, mean_outside, and score (difference)
                auto [mean_inside, mean_outside, score, inside_mask, outside_mask] = computeInnerCircleMean_min_max(resized_blurred_image, center, radius);

                std::cout << "Center: (" << center.x << ", " << center.y << "), "
                          << "Index: " << i << ", "
                          << "Mean Inside: " << mean_inside << ", "
                          << "Mean Outside: " << mean_outside << ", "
                          << "Score (diff): " << score << std::endl;

                if (score > max_score)
                {
                    max_score = score;
                    best_center = center;
                    best_radius = radius;
                    best_mean_inside = mean_inside;
                    best_mean_outside = mean_outside;
                    best_inside_mask = inside_mask;
                    best_outside_mask = outside_mask;
                }
            }

            // After loop
            std::cout << "Best center: (" << best_center.x << ", " << best_center.y
                      << ") with radius: " << best_radius
                      << "\nBest Mean Inside: " << best_mean_inside
                      << ", Best Mean Outside: " << best_mean_outside
                      << ", Best Score: " << max_score
                      << "\n,innermask:" << best_inside_mask.size() << "outermask" << best_outside_mask.size() << std::endl;

            std::string outer_mask_save_path = image_file_base_name + "_outside_mask.jpg";
            std::string threshold_image_save_path = image_file_base_name + "_threshold.jpg";
            std::string resize_image_save_path = image_file_base_name + "_points.jpg";

            cv::Mat thresh_img_temp;
            cv::threshold(resized_blurred_image, thresh_img_temp, static_cast<int>(round(best_mean_inside) - 1), 255, cv::THRESH_BINARY);
            cv::putText(thresh_img_temp, "Mean Inside: " + std::to_string(static_cast<int>(round(best_mean_inside) - 1)), cv::Point(200, 400),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 10);
            cv::putText(resized_blurred_image, "scale_factor: " + std::to_string(hough_scale_factor), cv::Point(200, 400),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 10);

            // cv::imwrite(threshold_image_save_path, thresh_img_temp);
            // cv::imwrite(resize_image_save_path, resized_blurred_image);
            // cv::imwrite(inner_mask_save_path, best_inside_mask);
            // cv::imwrite(outer_mask_save_path, best_outside_mask);

            // cv::HoughCircles(thresh_img_temp, circles_hough, cv::HOUGH_GRADIENT,
            //                1, 5.0, 110,
            //                20, hough_minR_scaled, hough_maxR_scaled);

            cv::Point2f coarse_center_orig_scale(best_center.x / hough_scale_factor,
                                                 best_center.y / hough_scale_factor);
            float coarse_radius_orig_scale = best_radius / hough_scale_factor;

            // cv::Point2f coarse_center_orig_scale(circles_hough[0][0] / hough_scale_factor,
            //                          circles_hough[0][1] / hough_scale_factor);
            // float coarse_radius_orig_scale = circles_hough[0][2] / hough_scale_factor;

            std::cout << "Hough center radius:: " << coarse_radius_orig_scale << "hough center x" << static_cast<int>(round(coarse_center_orig_scale.x)) << "hough center y" << static_cast<int>(round(coarse_center_orig_scale.y)) << std::endl;

            auto crop_tuple_display = cropImage(color_image_for_drawing_base,
                                                static_cast<int>(round(coarse_center_orig_scale.x)),
                                                static_cast<int>(round(coarse_center_orig_scale.y)),
                                                coarse_radius_orig_scale);

            current_cropped_display_img_ml_la = std::get<0>(crop_tuple_display);
            current_final_center_in_crop_ml_la_primary =
                cv::Point2f(std::get<1>(crop_tuple_display), std::get<2>(crop_tuple_display));

            if (current_cropped_display_img_ml_la.empty())
            {
                std::cerr << "Cropping (ML_LA display) empty for " << holeId << " at scale " << hough_scale_factor << std::endl;
                result_diameters.push_back(0.0);
                continue;
            }

            cv::Mat cropped_blurred_for_grad = std::get<0>(cropImage(
                blurred_image_full_scale,
                static_cast<int>(round(coarse_center_orig_scale.x)),
                static_cast<int>(round(coarse_center_orig_scale.y)),
                coarse_radius_orig_scale));

            if (cropped_blurred_for_grad.empty())
            {
                std::cerr << "Cropping (ML_LA grad) empty for " << holeId << " at scale " << hough_scale_factor << std::endl;
                result_diameters.push_back(0.0);
                continue;
            }

            cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad_magnitude_cropped, edge_image_for_scan;
            cv::Sobel(cropped_blurred_for_grad, grad_x, CV_32F, 1, 0, 3);
            cv::Sobel(cropped_blurred_for_grad, grad_y, CV_32F, 0, 1, 3);
            cv::magnitude(grad_x, grad_y, grad_magnitude_cropped);
            cv::convertScaleAbs(grad_magnitude_cropped, grad_magnitude_cropped, 1.0, 0);

            // if (holeId == "LS4" || holeId == "RS4" || holeId == "UA1") {
            if (false)
            {
                // edge_image_for_scan = cropped_blurred_for_grad;
                cv::bitwise_not(cropped_blurred_for_grad, edge_image_for_scan);
            }
            else
            {
                edge_image_for_scan = grad_magnitude_cropped;
            }

            std::vector<double> refined_radii_samples;
            std::vector<cv::Point2f> refined_edge_points_in_crop;
            std::vector<double> edge_radii;
            std::vector<cv::Point2f> refined_edge_points_in_crop_trapezoid;
            int num_rays = 180;
            double search_margin = 10.0;

            double radial_search_start_r = std::max(1.0,
                                                    static_cast<double>(coarse_radius_orig_scale) - search_margin);
            double radial_search_end_r = static_cast<double>((coarse_radius_orig_scale) +
                                                             search_margin);
            cout << "Range :: " << radial_search_start_r << " , " << radial_search_end_r << std::endl;

            annotated = current_cropped_display_img_ml_la.clone();

            double delta_deg = 2.0;
            double start_radius = lower_hough_radius - 100;
            double end_radius = upper_hough_radius + 100;
            for (double deg = 0; deg < 360; deg += 0.5)
            {
                double angle_rad = deg * CV_PI / 180.0;

                EdgeResult result = findEdgeInTrapezoid_withPoints(
                    edge_image_for_scan, // This should be the gradient image
                    current_final_center_in_crop_ml_la_primary,
                    angle_rad,
                    delta_deg,    // delta_deg
                    start_radius, // start_radius
                    end_radius    // end_radius
                );

                // 2. If an edge was found, draw the corresponding arc
                // delta_deg = 1.0;
                if (result.radius > 0)
                {
                    annotated = drawEdgeAndTrapezoid(
                        annotated,
                        current_final_center_in_crop_ml_la_primary,
                        result.radius,
                        start_radius,
                        end_radius,
                        angle_rad,
                        delta_deg,
                        cv::Scalar(0, 255, 0), // Trapezoid color: Green
                        cv::Scalar(0, 0, 0),   // Arc color: Red
                        2                      // Thickness                    // Thickness
                    );
                }
                   for (const cv::Point& pt : result.edge_points) {
                        // Do something with each point, e.g., draw a small circle
                        cv::circle(annotated, pt, 1, cv::Scalar(255, 0, 0), -1);
                        // continue;

                    }

                if (result.radius > 0)
                {
                    // std::cout<< "Edge Radius from trapezoid "<<result.radius << std::endl;

                    refined_edge_points_in_crop_trapezoid.insert(refined_edge_points_in_crop_trapezoid.end(), result.edge_points.begin(), result.edge_points.end());
                    edge_radii.push_back(result.radius);
                }

                // std::cout<< "Edge points from trapezoid  "<<result.edge_points.size() << " Total points "<< refined_edge_points_in_crop_trapezoid.size()<< std::endl;

                // double r_refined_sample = findSubpixelEdge_new(cropped_blurred_for_grad,
                //     current_final_center_in_crop_ml_la_primary, angle_rad,
                //     500, 800);

                // std::cout<<"radius from ray_no | radius "<< k << " | "<< r_refined_sample << std::endl;

                // if (r_refined_sample > 0) {
                //     refined_radii_samples.push_back(r_refined_sample);

                //     refined_edge_points_in_crop.push_back(
                //         cv::Point2f(current_final_center_in_crop_ml_la_primary.x +
                //         static_cast<float>(r_refined_sample * std::sin(angle_rad)),
                //         current_final_center_in_crop_ml_la_primary.y +
                //         static_cast<float>(r_refined_sample * std::cos(angle_rad))));
                // }
            }

            // iplock.unlock();
            // std::cout<<"Radii and point vector size " <<refined_radii_samples.size() <<" | "<<refined_edge_points_in_crop.size() <<std::endl;

            // if (refined_radii_samples.size() < num_rays * 0.25) {
            //     std::cerr << "Too few refined edge points (ML_LA) for " << holeId
            //              << " at scale " << hough_scale_factor << " (" << refined_radii_samples.size() << ")" << std::endl;
            //     result_diameters.push_back(0.0);
            //     result_votes.push_back(0.0);
            //     result_center.push_back(current_final_center_in_crop_ml_la_primary);
            //     continue;
            // }
            // auto [filtered_radii, filtered_points] = trimOutliersByRadius(refined_radii_samples, refined_edge_points_in_crop, 0.1);

            // std::cout << "Trimmed samples size: " << filtered_radii.size()<< " | " <<filtered_points.size()<< std::endl;

            // std::sort(refined_radii_samples.begin(), refined_radii_samples.end());

            // size_t trim_count = refined_radii_samples.size() / 10;
            // if (refined_radii_samples.size() > 2 * trim_count +
            //     std::min((size_t)5, refined_radii_samples.size() / 4)) {
            //     refined_radii_samples.assign(refined_radii_samples.begin() + trim_count, refined_radii_samples.end() - trim_count);
            // }

            int top_k = 10; // Number of top most frequent radii to consider
            int precision = 2;

            double tuned_radius = averageOfTopKMostFrequentRadii(edge_radii, top_k, precision);

            // std::cout << "Average of top " << top_k << " most common radii: " << tuned_radius << std::endl;

            // auto [mode_radius, mode_count] = getMostFrequentRadius(refined_radii_samples, 2);

            // std::cout << "Most common radius: " << mode_radius << ", Count: " << mode_count << std::endl;

            current_average_refined_radius_ml_la =
                std::accumulate(edge_radii.begin(), edge_radii.end(), 0.0) / edge_radii.size();

            // current_average_refined_radius_ml_la = tuned_radius;
            std::cout << "refined_edge_points_in_crop_trapezoid----------> " << refined_edge_points_in_crop_trapezoid.size() << " avg Radius---> " << current_average_refined_radius_ml_la << std::endl;

            if (refined_edge_points_in_crop_trapezoid.size() >= 3)
            {
                std::vector<cv::Point2f> points_for_center_fit;
                double tol_center_fit = current_average_refined_radius_ml_la * 0.10;
                cv::Point2f initial_center_for_selection = current_final_center_in_crop_ml_la_primary;

                for (auto &pt : refined_edge_points_in_crop_trapezoid)
                {
                    if (std::abs(cv::norm(pt - initial_center_for_selection) - current_average_refined_radius_ml_la) < tol_center_fit)
                    {
                        points_for_center_fit.push_back(pt);
                    }
                }
                std::cout << "RANSAC Points for best fit------> " << points_for_center_fit.size() << " avg Radius---> " << std::endl;

                for (const auto &point : points_for_center_fit)
                {
                    cv::circle(annotated, point, 2, cv::Scalar(0, 0, 255), cv::FILLED);
                }
                // cv::imwrite(resize_image_save_path, current_cropped_display_img_ml_la);
                cv::imwrite(inner_mask_save_path, annotated);

                // Circle circle_1 = fitCircleRANSAC(refined_edge_points_in_crop_trapezoid, 300, 2.0f);
                //  Circle circle_1 = fitCircleRANSAC(points_for_center_fit, 600, 2.0f);
                //  Circle circle_1 = fitCircleRANSAC(points_for_center_fit, edge_image_for_scan, 600, 5.0f);

                // float expected_radius_pixels = 750.0f; // Set this for your exact camera/magnification
                Circle circle_1 = fitCircleRANSAC_final(points_for_center_fit, edge_image_for_scan, current_average_refined_radius_ml_la);
                float diameter_mm_new = std::round((2.0f * circle_1.r / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
                std::cout << "\033[1;32m" << " RANSAAAAAAAAAAAAAC Fit: R=" << circle_1.r
                          << "px -> D=" << diameter_mm_new << "mm [" << "]\033[0m\n";

                // if (filtered_points.size() >= 3) {

                //     cv::minEnclosingCircle(filtered_points, current_final_center_in_crop_ml_la_primary, r_mec);
                //     std::cout << "Refined Center (ML_LA): ("
                //              << current_final_center_in_crop_ml_la_primary.x << ", "
                //              << current_final_center_in_crop_ml_la_primary.y << ")px "
                //              << " Radius :: " << current_average_refined_radius_ml_la << " at scale "
                //              << hough_scale_factor
                //              << std::endl;
                // }
                // }

                // double diameter_mm_ml_la_val_primary =
                //     std::round((2.0 * current_average_refined_radius_ml_la / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
                result_diameters.push_back(circle_1.r * 2.0f);
                result_votes.push_back(computeInnerCircleMean(current_cropped_display_img_ml_la,
                                                              {circle_1.x, circle_1.y}, circle_1.r));
                result_center.push_back(cv::Point2f(circle_1.x + static_cast<float>(std::get<3>(crop_tuple_display)),
                                                    circle_1.y + static_cast<float>(std::get<4>(crop_tuple_display))));

                // Store the current successful result as the 'best' for drawing
                best_cropped_display_img_ml_la = current_cropped_display_img_ml_la.clone();
                best_final_center_in_crop_ml_la_primary = {circle_1.x, circle_1.y};
                best_average_refined_radius_ml_la = circle_1.r;
            }
        } // End of loop for scaling factors

        if (result_diameters.size() != 1)
        {
            attempt_rc_fallback = true;
        }
        else
        {

            if (!result_votes.empty())
            {
                auto max_votes = std::max_element(result_votes.begin(), result_votes.end());
                int max_idx = std::distance(result_votes.begin(), max_votes);
                detected_diameter_px = result_diameters[max_idx];
                final_center = result_center[max_idx];
            }
            else
            {
                attempt_rc_fallback = true;
            }
        }

        if (!attempt_rc_fallback)
        {
            detected_diameter_mm = std::round((detected_diameter_px / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
            std::cout << "\033[1;32m" << "Final D (ML_LA Primary): "
                      << detected_diameter_mm << " mm\033[0m" << std::endl;

            auto crop_tuple_display = cropImage(color_image_for_drawing_base,
                                                static_cast<int>(round(final_center.x)),
                                                static_cast<int>(round(final_center.y)),
                                                detected_diameter_px / 2.0f);

            best_cropped_display_img_ml_la = std::get<0>(crop_tuple_display);

            if (!best_cropped_display_img_ml_la.empty())
            {
                cv::circle(best_cropped_display_img_ml_la,
                           cv::Point2f(std::get<1>(crop_tuple_display), std::get<2>(crop_tuple_display)),
                           static_cast<int>(round(detected_diameter_px / 2.0f)),
                           cv::Scalar(0, 255, 0), 2);
                cv::circle(best_cropped_display_img_ml_la,
                           cv::Point2f(std::get<1>(crop_tuple_display), std::get<2>(crop_tuple_display)), 3,
                           cv::Scalar(0, 0, 255), -1);

                cv::Rect txt_bg(0, 0, best_cropped_display_img_ml_la.cols, 150);
                cv::rectangle(best_cropped_display_img_ml_la, txt_bg,
                              cv::Scalar(200, 200, 200), -1);

                std::ostringstream txt;
                txt << "D: " << std::fixed << std::setprecision(3)
                    << detected_diameter_mm << "mm";
                cv::putText(best_cropped_display_img_ml_la, txt.str(),
                            cv::Point(10, 50), 0, 2, cv::Scalar(0, 0, 0), 2);

                txt.str("");
                txt.clear();
                txt << "R: " << std::fixed << std::setprecision(2)
                    << detected_diameter_px << "px";
                cv::putText(best_cropped_display_img_ml_la, txt.str(),
                            cv::Point(10, 110), 0, 2, cv::Scalar(0, 0, 0), 2);

                auto img_ptr_to_save =
                    std::make_shared<cv::Mat>(best_cropped_display_img_ml_la);
                // cam->async_imwrite(primary_save_path, img_ptr_to_save);
                cv::imwrite(primary_save_path, best_cropped_display_img_ml_la);
                cv::imwrite(inner_mask_save_path, annotated);
            }
            else
            {
                std::cerr << "WARNING: ML_LA detection successful, but drawing image was empty for " << holeId << std::endl;
            }

            return detected_diameter_mm;
        }
    }
    catch (const cv::Exception &cv_e)
    {
        std::cerr << "OpenCV Exception in ML_LA_GROUP (Primary) for "
                  << holeId << ": " << cv_e.what() << std::endl;
        if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" ||
            holeId == "RS4")
        {
            attempt_rc_fallback = true;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Generic Exception in ML_LA_GROUP (Primary) for "
                  << holeId << ": " << e.what() << std::endl;
        if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" ||
            holeId == "RS4")
        {
            attempt_rc_fallback = true;
        }
    }

    if (attempt_rc_fallback)
    {
        std::cout << "--> Executing RC_GROUP (RANSAC) fallback for "
                  << holeId << "..." << std::endl;

        try
        {
            cv::Mat gray_rc_fallback;
            if (img.empty())
            {
                std::cerr << "ERR(Fallback): Input image 'img' is empty before fallback conversion."
                          << std::endl;
                return -1.0f;
            }

            if (img.channels() == 3 || img.channels() == 4)
            {
                cv::cvtColor(img, gray_rc_fallback, cv::COLOR_BGR2GRAY);
            }
            else if (img.channels() == 1)
            {
                gray_rc_fallback = img.clone();
            }
            else
            {
                std::cerr << "ERR(Fallback): Unsupported channels for fallback: " << img.channels() << std::endl;
                return -1.0f;
            }

            if (!gray_rc_fallback.empty())
            {
                HoleParams p_rc;
                auto hp_it_rc = holeParameters.find(holeId);
                int fallback_thresh_val = (holeId == "LA3" || holeId == "LA6" ||
                                           holeId == "LS4" || holeId == "RS4")
                                              ? 135
                                              : 180;

                if (hp_it_rc != holeParameters.end())
                {
                    p_rc = hp_it_rc->second;
                    fallback_thresh_val = p_rc.threshold;
                }

                std::cout << "Using threshold " << fallback_thresh_val
                          << " for Fallback RC.\n";

                cv::Mat blur_rc, bin_rc;
                cv::GaussianBlur(gray_rc_fallback, blur_rc, cv::Size(7, 7), 0);
                cv::blur(blur_rc, blur_rc, cv::Size(3, 3));
                cv::threshold(blur_rc, bin_rc, fallback_thresh_val, 255,
                              cv::THRESH_BINARY);

                cv::Mat kern_rc = cv::getStructuringElement(cv::MORPH_RECT,
                                                            cv::Size(5, 5));
                cv::erode(bin_rc, bin_rc, kern_rc, cv::Point(-1, -1), 2);

                std::vector<std::vector<cv::Point>> cont_rc;
                cv::findContours(bin_rc, cont_rc, cv::RETR_TREE,
                                 cv::CHAIN_APPROX_SIMPLE);

                if (cont_rc.empty())
                {
                    std::cerr << "No contours found for Fallback RC for " << holeId << std::endl;
                }
                else
                {
                    std::sort(cont_rc.begin(), cont_rc.end(),
                              [](const auto &a, const auto &b)
                              {
                                  return cv::contourArea(a) > cv::contourArea(b);
                              });

                    int nProcess_rc = std::min(5, static_cast<int>(cont_rc.size()));
                    for (int i_rc = 0; i_rc < nProcess_rc; ++i_rc)
                    {
                        float fallback_diam = processSingleContourWithRansac(
                            gray_rc_fallback, cont_rc[i_rc], holeId,
                            "(Fallback RC Contour " + std::to_string(i_rc) + ")",
                            primary_save_path, true);

                        if (fallback_diam >= 0.0f)
                        {
                            std::cout << "\033[1;36m" << "Final D (RANSAC Fallback): "
                                      << fallback_diam << " mm\033[0m" << std::endl;
                            return fallback_diam;
                        }
                    }
                    std::cout << "RC_GROUP (Fallback) also failed to find a valid diameter "
                              << "after checking all contours.\n";
                }
            }
            else
            {
                std::cerr << "ERR(Fallback): Grayscale image for fallback is empty for " << holeId << std::endl;
            }
        }
        catch (const cv::Exception &cv_e)
        {
            std::cerr << "OpenCV Exception in RC_GROUP (Fallback) for "
                      << holeId << ": " << cv_e.what() << std::endl;
        }
        catch (const std::exception &e_rc)
        {
            std::cerr << "Generic Exception in RC_GROUP (Fallback) for "
                      << holeId << ": " << e_rc.what() << std::endl;
        }
    }

    std::cout << "All detection attempts failed for " << holeId << ". Resulting diameter: " << detected_diameter_mm << std::endl;

    cv::Mat final_fail_img = img_input_orig.clone();
    if (!final_fail_img.empty())
    {
        if (final_fail_img.channels() == 1)
        {
            cv::cvtColor(final_fail_img, final_fail_img, cv::COLOR_GRAY2BGR);
        }

        int text_box_width_final = std::min(final_fail_img.cols, 700);
        cv::Rect text_bg_rect_final(0, 0, text_box_width_final, 70);
        cv::rectangle(final_fail_img, text_bg_rect_final,
                      cv::Scalar(128, 128, 128), cv::FILLED);

        cv::putText(final_fail_img, "ALL DETECTION ATTEMPTS FAILED",
                    cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                    final_fail_img.cols > 400 ? 1.0 : 0.7,
                    cv::Scalar(0, 0, 255), 2);

        auto final_fail_img_ptr = std::make_shared<cv::Mat>(final_fail_img);
        // cam->async_imwrite(primary_save_path, final_fail_img_ptr);
        cv::imwrite(primary_save_path, final_fail_img);

        std::cout << "Saved 'ALL ATTEMPTS FAILED' image to primary path: "
                  << primary_save_path << std::endl;
    }

    return -1.0f;
}

// The MODIFIED detectHoleDiameter function
// float detectHoleDiameter(
//     const std::string &imagePath_for_saving_name,
//     cv::Mat img_input,
//     // std::shared_ptr<cv::Mat> img_input,
//     const int station_num,
//     // const std::string& holeId, CameraManager *cam)
//     const std::string &holeId)

// {
//     float detected_diameter_mm = -1.0f;
//     auto startTotal = std::chrono::high_resolution_clock::now();

//     std::cout << "Processing Hole ID: " << holeId << " for station " << station_num
//               << " (Output base: " << imagePath_for_saving_name << ")" << std::endl;

//     std::filesystem::path pathObjBase(imagePath_for_saving_name);
//     std::string output_dir_base = ".";
//     if (pathObjBase.has_parent_path() && !pathObjBase.parent_path().string().empty())
//     {
//         output_dir_base = pathObjBase.parent_path().string();
//     }
//     std::string image_file_base_name = pathObjBase.stem().string();
//     if (image_file_base_name.empty())
//         image_file_base_name = "processed_image";

//     std::string results_sub_dir = output_dir_base;
//     if (!std::filesystem::exists(results_sub_dir))
//     {
//         std::filesystem::create_directories(results_sub_dir);
//     }
//     // std::string primary_save_path = results_sub_dir + "/"+"result"+"/" + image_file_base_name + "_S" +
//     std::string primary_save_path = results_sub_dir + "/" + image_file_base_name + "_S" +
//                                     std::to_string(station_num) + "_" + holeId + "_annotated.jpg";
//     // std::string primary_save_path = results_sub_dir + "/" +"result"+"/"+ image_file_base_name + "_S" +
//     //         std::to_string(station_num) + "_" + holeId + "_annotated.jpg";

//     std::cout << "Output annotated image will be saved to: " << primary_save_path << std::endl;

//     if (img_input.empty())
//     {
//         std::cerr << "ERROR: Input image is empty for holeId: " << holeId << ".\n";
//         // Fall through to final timer and return -1.0f
//     }
//     else
//     {
//         cv::Mat img_processing_copy = img_input.clone(); // Work on a copy

//         if (rc_group_ids.count(holeId))
//         {
//             detected_diameter_mm = handleRcGroup(img_processing_copy, holeId, primary_save_path);
//         }
//         else if (ml_la_group_ids.count(holeId))
//         {

//             detected_diameter_mm = handleMlLaGroup(img_input, img_processing_copy, holeId, primary_save_path, output_dir_base, image_file_base_name, station_num);
//         }
//         else
//         {
//             std::cout << "Warning: Hole ID " << holeId << " does not belong to any known processing group." << std::endl;
//             // detected_diameter_mm remains -1.0f
//         }
//     }

//     auto stopTotal = std::chrono::high_resolution_clock::now();
//     auto durationTotal = std::chrono::duration_cast<std::chrono::milliseconds>(stopTotal - startTotal);
//     std::cout << "Total processing time for " << holeId << ": " << durationTotal.count() << " ms. Final diameter: ";
//     if (detected_diameter_mm >= 0.0f)
//     {
//         std::cout << std::fixed << std::setprecision(3) << detected_diameter_mm << " mm." << std::endl;
//     }
//     else
//     {
//         std::cout << "NOT FOUND." << std::endl;
//     }
//     return detected_diameter_mm;
// }

//Modified code For masking
// The MODIFIED detectHoleDiameter function
float detectHoleDiameter(
    const std::string &imagePath_for_saving_name,
    cv::Mat img_input,
    const int station_num,
    const std::string &holeId)

{
    float detected_diameter_mm = -1.0f;
    auto startTotal = std::chrono::high_resolution_clock::now();

    std::cout << "Processing Hole ID: " << holeId << " for station " << station_num
              << " (Output base: " << imagePath_for_saving_name << ")" << std::endl;

    std::filesystem::path pathObjBase(imagePath_for_saving_name);
    std::string output_dir_base = ".";
    if (pathObjBase.has_parent_path() && !pathObjBase.parent_path().string().empty())
    {
        output_dir_base = pathObjBase.parent_path().string();
    }
    std::string image_file_base_name = pathObjBase.stem().string();
    if (image_file_base_name.empty())
        image_file_base_name = "processed_image";

    std::string results_sub_dir = output_dir_base;
    if (!std::filesystem::exists(results_sub_dir))
    {
        std::filesystem::create_directories(results_sub_dir);
    }
    std::string primary_save_path = results_sub_dir + "/" + image_file_base_name + "_S" +
                                    std::to_string(station_num) + "_" + holeId + "_annotated.jpg";

    std::cout << "Output annotated image will be saved to: " << primary_save_path << std::endl;

    if (img_input.empty())
    {
        std::cerr << "ERROR: Input image is empty for holeId: " << holeId << ".\n";
    }
    else
    {
        // <<< MODIFIED: Block to apply the image mask and save it <<<
        auto it = diaCoordinates.find(holeId);
        if (it != diaCoordinates.end())
        {
            auto [x1, y1, x2, y2] = it->second;
            std::cout << "Applying mask for " << holeId << " using ROI: (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << ")." << std::endl;
            cv::Mat masked_img = maskImage_handleMlLaGroup(img_input, x1, y1, x2, y2); // Apply mask to original image

            // Check if masking was successful
            if (masked_img.empty())
            {
                std::cerr << "ERROR: Masking resulted in an empty image for holeId: " << holeId << ".\n";
            }
            else
            {
                // Assign the masked image as the new input image for all further processing
                img_input = masked_img;

                // Create the "masked_image" directory and save the masked image
                std::string masked_images_dir = output_dir_base + "/masked_image";
                if (!std::filesystem::exists(masked_images_dir))
                {
                    std::filesystem::create_directories(masked_images_dir);
                }
                std::string masked_save_path = masked_images_dir + "/" + image_file_base_name + "_S" +
                                               std::to_string(station_num) + "_" + holeId + "_masked.jpg";
                std::cout << "Masked image will be saved to: " << masked_save_path << std::endl;
                cv::imwrite(masked_save_path, img_input);
            }
        }
        else
        {
            std::cout << "Warning: No mask coordinates found in diaCoordinates for " << holeId << ". Using original image." << std::endl;
        }
        // >>> END MODIFIED BLOCK <<<

        // The img_input variable now holds the masked image (or the original if no mask was applied)
        if (img_input.empty())
        {
            std::cerr << "ERROR: Image for holeId " << holeId << " is empty after mask step.\n";
        }
        else
        {
            // img_processing_copy will now be a clone of the (potentially masked) img_input
            cv::Mat img_processing_copy = img_input.clone();

            if (rc_group_ids.count(holeId))
            {
                detected_diameter_mm = handleRcGroup(img_processing_copy, holeId, primary_save_path);
            }
            else if (ml_la_group_ids.count(holeId))
            {
                detected_diameter_mm = handleMlLaGroup(img_input, img_processing_copy, holeId, primary_save_path, output_dir_base, image_file_base_name, station_num);
            }
            else
            {
                std::cout << "Warning: Hole ID " << holeId << " does not belong to any known processing group." << std::endl;
            }
        }
    }

    auto stopTotal = std::chrono::high_resolution_clock::now();
    auto durationTotal = std::chrono::duration_cast<std::chrono::milliseconds>(stopTotal - startTotal);
    std::cout << "Total processing time for " << holeId << ": " << durationTotal.count() << " ms. Final diameter: ";
    if (detected_diameter_mm >= 0.0f)
    {
        std::cout << std::fixed << std::setprecision(3) << detected_diameter_mm << " mm." << std::endl;
    }
    else
    {
        std::cout << "NOT FOUND." << std::endl;
    }
    return detected_diameter_mm;
}


static std::map<std::string, double> min_thread_count_per_hole = {
    {"RC2", 4},
    {"RC3", 5},
    {"SP5", 1.5},
    {"SP9", 2},
    {"SP1", 2},
    {"LA2", 3},
    {"ML2", 4},
    {"RC1", 4},
    {"LA3", 3},
    {"LA4", 8},
    {"LA5", 8},
    {"LA6", 5},
    {"LS4", 3},
    {"RS4", 3.5},
    {"UA1", 4},
    {"UA6", 4},

};

bool checkBlackRegionBothSides(const Mat &image, const Rect &bbox, int threshold = 10)
{
    // Define sampling y positions (top, middle, bottom)
    vector<int> ySamples = {
        bbox.y + bbox.height / 4,
        bbox.y + bbox.height / 2,
        bbox.y + 3 * bbox.height / 4};

    int leftX = max(bbox.x - 5, 0);
    int rightX = min(bbox.x + bbox.width + 5, image.cols - 1);

    int blackLeft = 0, blackRight = 0;

    for (int y : ySamples)
    {
        if (y < 0 || y >= image.rows)
            continue;

        // Count black pixels to the left
        for (int x = leftX; x >= 0; --x)
        {
            if (image.at<uchar>(y, x) == 0)
                blackLeft++;
        }

        // Count black pixels to the right
        for (int x = rightX; x < image.cols; ++x)
        {
            if (image.at<uchar>(y, x) == 0)
                blackRight++;
        }
    }

    return (blackLeft >= threshold && blackRight >= threshold);
}

cv::Mat rotateImage(const cv::Mat &src, double angle)
{
    Point2f center(src.cols / 2.0F, src.rows / 2.0F);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    Rect2f bbox = RotatedRect(Point2f(), src.size(), angle).boundingRect2f();

    rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;

    Mat dst;
    warpAffine(src, dst, rot, bbox.size());
    return dst;
}

int find_threads(const std::string &imagePath,
                 // std::shared_ptr<cv::Mat> raw_img_p,
                 cv::Mat raw_img_p,
                 int station_num,
                 const std::string &hole_no)
// CameraManager* cam)
{

    // cv::Mat raw_img = *raw_img_p;
    cv::Mat raw_img = raw_img_p.clone();
    std::filesystem::path pathObj(imagePath);
    // Create new path with "_processed.jpg"
    std::string thread_new_path = pathObj.replace_extension("").string() + "_thread_processed.jpg";
    std::cout << "thread_new_path" << thread_new_path << std::endl;
    std::cout << "Thread count function called hole number -->> " << hole_no << std::endl;

    if (holeRotationAngles.find(hole_no) != holeRotationAngles.end())
    {
        double angle = holeRotationAngles[hole_no];
        raw_img = rotateImage(raw_img, angle);
        // imwrite("rotated_" + hole_no + ".jpg", raw_img);
        // cam->async_imwrite("rotated_" + hole_no + ".jpg", raw_img);

        std::cout << "Rotated image by " << angle << " degrees." << std::endl;
    }

    if (threadCoordinates.find(hole_no) != threadCoordinates.end())
    {
        auto [x1_r, y1_r, x2_r, y2_r] = threadCoordinates[hole_no];
        cv::Rect roi(x1_r, y1_r, x2_r - x1_r, y2_r - y1_r);
        std::cout << "x1_r: " << x1_r << ", y1_r: " << y1_r
                  << ", x2_r: " << x2_r << ", y2_r: " << y2_r << std::endl;

        if (raw_img.empty())
        {
            std::cerr << "Failed to load image." << std::endl;
            return -1;
        }

        roi = roi & cv::Rect(0, 0, raw_img.cols, raw_img.rows);
        Mat cropped_image = raw_img(roi);
        Mat thresh;

        if (station_num == 6)
        {

            threshold(cropped_image, thresh, 250, 255, THRESH_BINARY);
            // Define a structuring element (kernel) for erosion
            Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));

            // Apply erosion
            erode(thresh, thresh, kernel);
            // imwrite("thresh" + hole_no + ".jpg", thresh);
            // cam->async_imwrite("thresh" + hole_no + ".jpg", thresh);

            vector<vector<Point>> contours;
            findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (contours.empty())
            {
                std::cerr << "No contours found." << std::endl;
                return 0;
            }

            vector<vector<Point>> filtered_contours;
            for (const auto &contour : contours)
            {
                if (contourArea(contour) > 3500.0)
                {
                    filtered_contours.push_back(contour);
                }
            }


            if (filtered_contours.empty())
            {
                std::cerr << "No significant contours after filtering." << std::endl;
                return 0;
            }

            Mat vis_img_raw;
            cvtColor(raw_img, vis_img_raw, COLOR_GRAY2BGR);
            roi = roi & cv::Rect(0, 0, raw_img.cols, raw_img.rows);

            // Draw the ROI box
            cv::rectangle(vis_img_raw, roi, cv::Scalar(0, 255, 0), 2); // Green, thickness 2
            int thread_count = 0;

            std::vector<cv::Point2f> centers;

            // Step 1: Collect all centers first
            for (size_t i = 0; i < filtered_contours.size(); ++i)
            {
                RotatedRect minRect = minAreaRect(filtered_contours[i]);
                Point2f center = minRect.center;
                center.x += roi.x;
                center.y += roi.y;
                centers.push_back(center);
            }

            // Step 2: Fit a line to all centers
            Vec4f line_params;
            fitLine(centers, line_params, DIST_L2, 0, 0.01, 0.01);
            Point2f line_dir(line_params[0], line_params[1]);
            Point2f point_on_line(line_params[2], line_params[3]);

            // Step 3: Draw only boxes near the fitted line
            for (size_t i = 0; i < filtered_contours.size(); ++i)
            {
                RotatedRect minRect = minAreaRect(filtered_contours[i]);

                // Compute actual center of the box (adjusted for ROI)
                Point2f center = minRect.center;
                center.x += roi.x;
                center.y += roi.y;

                // Distance from point to line using cross product method
                Point2f diff = center - point_on_line;
                float distance = std::abs(diff.x * line_dir.y - diff.y * line_dir.x);

                // Accept if distance is below a threshold (tune 10.0 as needed)
                if (distance < 50.0f)
                {
                    Point2f rect_points[4];
                    minRect.points(rect_points);

                    for (int j = 0; j < 4; j++)
                    {
                        rect_points[j].x += roi.x;
                        rect_points[j].y += roi.y;
                    }

                    for (int j = 0; j < 4; j++)
                    {
                        line(vis_img_raw, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 6);
                    }

                    thread_count++;
                }
            }

            // Optional: Draw the fitted line across the image
            Point2f pt1 = point_on_line - 1000 * line_dir;
            Point2f pt2 = point_on_line + 1000 * line_dir;
            line(vis_img_raw, pt1, pt2, Scalar(0, 255, 0), 3);

            Scalar padding_color(200, 200, 200);
            Mat padded_image;
            copyMakeBorder(vis_img_raw, padded_image, 1000, 0, 0, 0, BORDER_CONSTANT, padding_color);
            double min_required = 0;
            bool status = false;
            min_required = min_thread_count_per_hole[hole_no];
            status = thread_count >= min_required;
            std::ostringstream oss_count;
            oss_count.precision(1);
            oss_count << std::fixed << thread_count;
            std::string text = "Thread Count: " + oss_count.str();

            // Format min_required
            std::ostringstream oss_min;
            oss_min.precision(1);
            oss_min << std::fixed << min_required;
            std::string min_text = "Min Required: " + oss_min.str();
            std::string status_text = "Status: " + std::string(status ? "PASS" : "FAIL");
            Scalar status_color = status ? Scalar(255, 0, 0) : Scalar(0, 0, 255); // Green or Red

            // Draw texts
            putText(padded_image, text, Point(20, 300), FONT_HERSHEY_COMPLEX, 8, Scalar(0, 0, 0), 6);
            putText(padded_image, min_text, Point(20, 600), FONT_HERSHEY_COMPLEX, 8, Scalar(0, 0, 0), 6);
            putText(padded_image, status_text, Point(20, 900), FONT_HERSHEY_COMPLEX, 8, status_color, 6);
            imwrite(thread_new_path, padded_image);
            return thread_count;
        }

        else
        {

            threshold(cropped_image, thresh, 160, 255, THRESH_BINARY);
            // threshold(cropped_image, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

            // Define a kernel for erosion (e.g., 3x3 rectangular structuring element)
            Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
            // Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
            //  Apply erosion
            erode(thresh, thresh, kernel);
            // dilate(thresh, thresh, kernel2);

            // imwrite("dial" + hole_no + ".jpg", thresh);
            Mat drawImg;

            cvtColor(cropped_image, drawImg, COLOR_GRAY2BGR); // Convert for color drawing

            vector<vector<Point>> contours;
            findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            double minArea = 5000;
            double maxArea = 24000;

            vector<pair<vector<Point>, Point2f>> valid;

            for (size_t i = 0; i < contours.size(); ++i)
            {
                double area = contourArea(contours[i]);
                if (area < minArea || area > maxArea)
                    continue;

                Rect bbox = boundingRect(contours[i]);
                if (!checkBlackRegionBothSides(thresh, bbox))
                    continue;

                vector<Point> poly;
                approxPolyDP(contours[i], poly, 5, true);

                Moments M = moments(poly);
                if (M.m00 == 0)
                    continue;
                Point2f center(M.m10 / M.m00, M.m01 / M.m00);

                valid.push_back({poly, center});
            }

            // Compute mean center
            Point2f meanCenter(0, 0);
            for (const auto &item : valid)
            {
                meanCenter += item.second;
            }
            if (!valid.empty())
            {
                meanCenter.x /= valid.size();
                meanCenter.y /= valid.size();
            }

            // Filter based on distance from mean center
            double maxDistance = 400.0;
            vector<pair<vector<Point>, Point2f>> filteredValid;
            for (const auto &item : valid)
            {
                Point2f c = item.second;
                double distance = norm(c - meanCenter);
                if (distance <= maxDistance)
                {
                    filteredValid.push_back(item);
                }
            }

            valid = filteredValid;

            // Sort by X center
            sort(valid.begin(), valid.end(), [](const auto &a, const auto &b)
                 { return a.second.x < b.second.x; });

            // Merge close-by contours (X difference <= 25 pixels)
            vector<pair<vector<Point>, Point2f>> mergedValid;
            const double xThreshold = 40.0;
            vector<Point> currentGroup;
            vector<Point2f> currentCenters;

            for (size_t i = 0; i < valid.size(); ++i)
            {
                if (currentGroup.empty())
                {
                    currentGroup.insert(currentGroup.end(), valid[i].first.begin(), valid[i].first.end());
                    currentCenters.push_back(valid[i].second);
                    continue;
                }

                double xDiff = abs(valid[i].second.x - currentCenters.back().x);
                if (xDiff <= xThreshold)
                {
                    currentGroup.insert(currentGroup.end(), valid[i].first.begin(), valid[i].first.end());
                    currentCenters.push_back(valid[i].second);
                }
                else
                {
                    vector<Point> approx;
                    approxPolyDP(currentGroup, approx, 5, true);
                    Moments M = moments(approx);
                    if (M.m00 != 0)
                    {
                        Point2f center(M.m10 / M.m00, M.m01 / M.m00);
                        mergedValid.push_back({approx, center});
                    }
                    currentGroup = valid[i].first;
                    currentCenters = {valid[i].second};
                }
            }

            // Add last group
            if (!currentGroup.empty())
            {
                vector<Point> approx;
                approxPolyDP(currentGroup, approx, 5, true);
                Moments M = moments(approx);
                if (M.m00 != 0)
                {
                    Point2f center(M.m10 / M.m00, M.m01 / M.m00);
                    mergedValid.push_back({approx, center});
                }
            }

            valid = mergedValid; // Replace with merged

            // Draw and label
            for (size_t i = 0; i < valid.size(); ++i)
            {
                drawContours(drawImg, vector<vector<Point>>{valid[i].first}, -1, Scalar(0, 255, 0), 2);
                putText(drawImg, to_string(i + 1), valid[i].second, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
            }

            // Total count label
            string totalText = "Total: " + to_string(valid.size());
            putText(drawImg, totalText, Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

            cout << "Mean center: [" << meanCenter.x << ", " << meanCenter.y << "]" << endl;
            cout << "Final valid count: " << valid.size() << endl;

            // Output result
            imwrite(thread_new_path, drawImg);
            // imwrite(thread_new_path, padded_image);
            // imwrite(thread_new_path, thresh);

            // delete cropped_image;
            // delete raw_img;
            // delete raw_img_p;
            return static_cast<int>(valid.size());
            // return valid.size();
        }
    }
    else
    {
        std::cerr << "Hole number not found: " << hole_no << std::endl;
        return -1;
    }
}

namespace fs = std::filesystem;

// const std::vector<std::string> expectedHoleOrder = {
//     "LA2", "LA3", "LA4", "LA5", "LA6",
//     "LS4", "ML2", "RC1", "RC2", "RC3",
//     "RS4", "SP1", "SP5", "SP9", "UA1", "UA6"};

const std::vector<std::string> expectedHoleOrder = {
    "LA2", "LA3", "LA4", "LA5", "LA6",
    "ML2", "RC1", "RC2", "RC3",
    "SP1", "SP5", "SP9", "UA1", "UA6"};

const size_t expectedHoleCount = expectedHoleOrder.size();

void appendCycleToCSV_with_fix_id(
    const std::string &csvPath,
    const std::string &cycleID,
    const std::string &fixtureID,
    const std::map<std::string, float> &holeMap,
    bool writeHeader = false)
{
    std::ofstream csvFile(csvPath, std::ios::app);
    if (!csvFile.is_open())
    {
        std::cerr << "❌ Failed to open CSV file: " << csvPath << "\n";
        return;
    }

    if (writeHeader)
    {
        csvFile << "cycle_id\tfixture_id";
        for (const auto &hole : expectedHoleOrder)
            csvFile << "\t" << hole;
        csvFile << "\n";
    }

    csvFile << cycleID << "\t" << fixtureID;
    for (const auto &hole : expectedHoleOrder)
    {
        auto it = holeMap.find(hole);
        if (it != holeMap.end())
            csvFile << "\t" << it->second;
        else
            csvFile << "\t0.0";
    }
    csvFile << "\n";
    csvFile.close();
    std::cout << "✅ Saved cycle " << cycleID << " (fixture " << fixtureID << ") to CSV\n";
}

void appendCycleToCSV(
    const std::string &csvPath,
    const std::string &cycleID,
    const std::map<std::string, float> &holeMap,
    bool writeHeader = false)
{
    std::ofstream csvFile(csvPath, std::ios::app);
    if (!csvFile.is_open())
    {
        std::cerr << "❌ Failed to open CSV file: " << csvPath << "\n";
        return;
    }

    if (writeHeader)
    {
        csvFile << "cycle_id";
        for (const auto &hole : expectedHoleOrder)
            csvFile << "\t" << hole;
        csvFile << "\n";
    }

    csvFile << cycleID;
    for (const auto &hole : expectedHoleOrder)
    {
        auto it = holeMap.find(hole);
        if (it != holeMap.end())
            csvFile << "\t" << it->second;
        else
            csvFile << "\t0.0";
    }
    csvFile << "\n";
    csvFile.close();
    std::cout << "✅ Saved cycle " << cycleID << " to CSV\n";
}

int main()
{
    namespace fs = std::filesystem;
    std::string parentFolder = "D:/Test_imgs(Aug_6)/";
    std::string csvPath = "D:/Test_imgs(Aug_6)/Sarath_backrun_on_data_6_aug_25_updated.csv";
    // Aug_4_exposures
    const std::map<std::string, int> Diameter_hole_expo = {
        {"RS4", 45000}, {"LA3", 17677}, {"LS4", 20000}, {"LA6", 17677}, {"UA1", 45000}, {"RC2", 8000}, {"RC1", 36108}, {"RC3", 8000}, {"ML2", 5000}, {"LA2", 6000}, {"SP5", 5000}, {"SP1", 18000}, {"SP9", 10000}, {"LA4", 2000}, {"LA5", 2000}, {"UA6", 17000}};
    // Aug_2_exposures
    // const std::map<std::string, int> Diameter_hole_expo = {
    //     {"RS4", 455887}, {"LA3", 30000}, {"LS4", 455887}, {"LA6", 10000}, {"UA1", 30000}, {"RC2", 8000}, {"RC1", 86108}, {"RC3", 8000}, {"ML2", 2000}, {"LA2", 2000}, {"SP5", 8000}, {"SP1", 18000}, {"SP9", 10000}, {"LA4", 4188}, {"LA5", 4188}, {"UA6", 10000}};

    // cycle_id_26202_station_num_2_LA3_1_exposure_20000_fixtureNo_6_X_position_51.2_Y_position_41_Z_position_38.35
    // std::regex pattern(R"(cycle_id_(\d+)station_num_(\d+)_([A-Z]+\d+(?:_\d+)?)_exposure_(\d+)_fixtureNo_(\d+)(?:_.*)?)", std::regex::icase);
    
    // std::regex pattern(R"(cycle_id_(\d+)_station_num_(\d+)_([A-Z]+\d+)(?:_\d+)?_exposure_(\d+)_fixtureNo_(\d+)_X_position_.*)", std::regex::icase);
    //std::regex pattern(R"(cycle_id_(\d+)station_num_(\d+)_([A-Z]+\d+)(?:_\d+)?_exposure_(\d+)_fixtureNo_(\d+)_X_position_.*)", std::regex::icase);
    std::regex pattern(R"(cycle_id_(\d+)_station_num_(\d+)_([A-Z]+\d+)(?:_\d+)?_exposure_(\d+)_fixtureNo_(\d+)_X_position_.*)", std::regex::icase);

    std::smatch match;

    std::map<std::string, std::map<std::string, float>> cycleHoleMap;
    std::set<std::string> savedCycles;

    bool headerWritten = false;

    for (const auto &entry : fs::recursive_directory_iterator(parentFolder))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;

        std::string filename = entry.path().filename().string();
        std::string imagePath = entry.path().string();

        if (!std::regex_search(filename, match, pattern))
        {
            std::cout << "⚠️ Skipping unmatched file: " << filename << "\n";
            continue;
        }

        std::string cycleID = match[1];
        int stationNumber = std::stoi(match[2]);
        std::string holeNumber = match[3];
        int exposure = std::stoi(match[4]);
        std::string fixtureID = match[5];
        // std::cout<<"Fixture id ----------->"<<fixtureID<< holeNumber<<std::endl;

        // if (holeNumber != "LS4" && fixtureID!="3" && exposure!= 25000){ //&& holeNumber != "LS4" && holeNumber != "UA1") {
        //     std::cout << "Continue called ! " << std::endl;
        //     continue; // Skip holes not in target set
        //     }
        //     "RC3", "ML2","LA2","SP5","SP1","SP9","LA4","LA5"

        auto it = Diameter_hole_expo.find(holeNumber);
        if (it == Diameter_hole_expo.end() || exposure != it->second)
        {
            std::cout << "⚠️ Skipping " << holeNumber << ": exposure mismatch or unknown hole in " << filename << "\n";
            continue;
        }

        cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            std::cerr << "❌ Failed to read image: " << imagePath << "\n";
            continue;
        }

        std::cout << "🔧 Processing: " << filename << " | Cycle: " << cycleID << ", Hole: " << holeNumber << "\n";

        float diameter = detectHoleDiameter(imagePath, img, stationNumber, holeNumber);
        cycleHoleMap[cycleID][holeNumber] = diameter;
        std::cout << "cycleHoleMap[cycleID].size() --------> cycleID" << cycleID << "| " << cycleHoleMap[cycleID].size() << " | " << expectedHoleCount << std::endl;
        if (cycleHoleMap[cycleID].size() == expectedHoleCount)
        {
            // Save CSV row for this cycle
            if (!headerWritten)
            {
                appendCycleToCSV_with_fix_id(csvPath, cycleID, fixtureID, cycleHoleMap[cycleID], true);
                headerWritten = true;
            }
            else
            {
                appendCycleToCSV_with_fix_id(csvPath, cycleID, fixtureID, cycleHoleMap[cycleID], false);
            }

            savedCycles.insert(cycleID);
            cycleHoleMap.erase(cycleID); // Free memory
        }
    }

    return 0;
}

// int main() {
//     namespace fs = std::filesystem;
//     std::string parentFolder = "/mnt/4TB/home/griffyn/Music/images/RAW_june_13";
//     std::string csvPath = "KK.csv";

//     // Only interested in these 3 holes
//     const std::vector<std::string> expectedHoleOrder = {"RS4", "LS4", "UA1"};

//     // Exposure lookup
//     const std::map<std::string, int> Diameter_hole_expo = {
//         {"RS4", 40000}, {"LS4", 25000}, {"UA1", 30000}
//     };

//     // Extended regex to capture fixtureNo
//     std::regex pattern(R"(cycle_id_(\d+)station_num_(\d+)_([A-Z]+\d+)_exposure_(\d+)_fixtureNo_(\d+))", std::regex::icase);
//     std::smatch match;

//     // Map: cycleID -> (hole -> diameter)
//     std::map<std::string, std::map<std::string, float>> cycleHoleMap;
//     std::map<std::string, std::string> fixtureMap;  // cycleID -> fixtureID

//     bool headerWritten = false;

//     for (const auto& entry : fs::recursive_directory_iterator(parentFolder)) {
//         if (!entry.is_regular_file() || entry.path().extension() != ".png")
//             continue;

//         std::string filename = entry.path().filename().string();
//         std::string imagePath = entry.path().string();

//         if (!std::regex_search(filename, match, pattern)) {
//             std::cout << "⚠️ Skipping unmatched file: " << filename << "\n";
//             continue;
//         }

//         std::string cycleID = match[1];
//         int stationNumber = std::stoi(match[2]);
//         std::string holeNumber = match[3];
//         int exposure = std::stoi(match[4]);
//         std::string fixtureID = match[5];

//         // Process only RS4, LS4, UA1
//         // if (holeNumber != "RS4" && holeNumber != "LS4" && holeNumber != "UA1") {
//         //     continue;
//         // }

//         auto it = Diameter_hole_expo.find(holeNumber);
//         if (it == Diameter_hole_expo.end() || exposure != it->second) {
//             std::cout << "⚠️ Exposure mismatch or unknown hole: " << holeNumber << " in " << filename << "\n";
//             continue;
//         }

//         cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
//         if (img.empty()) {
//             std::cerr << "❌ Failed to read image: " << imagePath << "\n";
//             continue;
//         }

//         std::cout << "🔧 Processing: " << filename << " | Cycle: " << cycleID << ", Hole: " << holeNumber << "\n";

//         float diameter = detectHoleDiameter(imagePath, img, stationNumber, holeNumber);

//         // Accumulate
//         cycleHoleMap[cycleID][holeNumber] = diameter;
//         fixtureMap[cycleID] = fixtureID;

//         // If all 3 holes are present, save this row
//         if (cycleHoleMap[cycleID].size() == 3) {
//             if (!headerWritten) {
//                 appendCycleToCSV_with_fix_id(csvPath, cycleID, fixtureMap[cycleID], cycleHoleMap[cycleID], true);
//                 headerWritten = true;
//             } else {
//                 appendCycleToCSV_with_fix_id(csvPath, cycleID, fixtureMap[cycleID], cycleHoleMap[cycleID], false);
//             }

//             // Clear from memory
//             cycleHoleMap.erase(cycleID);
//             fixtureMap.erase(cycleID);
//         }
//     }

//     return 0;
// }

// int main(int argc, char* argv[]) {
//     std::string folderPath = "/mnt/4TB/home/griffyn/Music/images/RAW_Amrut_GRR/IT_1_problem/UA6";  // Replace with your folder name
//     std::string holeNumber = "UA6";
//     std::string csvPath = "/mnt/4TB/home/griffyn/Music/images/RAW_Amrut_GRR/IT_1_problem/UA6/UA6.csv";  // or any desired path

//     int stationNumber = 2;
//     // int counter=0;
//     for (const auto& entry : fs::directory_iterator(folderPath)) {
//         if (entry.is_regular_file()&& entry.path().extension() == ".png") {
//             // counter=counter+1;
//             std::string imagePath = entry.path().string();
//             std::string partName= entry.path().filename().string() ;
//             cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
//             std::cout << "Processing image: " << imagePath << "\n";
//             float diameters = detectHoleDiameter(imagePath, img, stationNumber, holeNumber);
//             // std::cout<<counter << "***********" << std::endl;

//             //std::vector<float> diameters = detectHoleDiameter(imagePath, stationNumber, holeNumber,partName);

//             // if (!diameters.empty()) {float detectHoleDiameter(

//             //     for (size_t i = 0; i < diameters.size(); ++i) {
//             //         std::cout << "Diameter " << i + 1 << " of hole " << holeNumber << ": " << diameters[i] << " mm\n";
//                     saveDiameterToCSV(entry.path().filename().string(), diameters, csvPath);
//             //     }
//             // } else {
//             //     std::cout << "Failed to detect diameter for hole " << holeNumber << " in image " << imagePath << "\n";
//             //     saveDiameterToCSV(entry.path().filename().string(), {0.0f}, csvPath);
//             // }
//         }
//     }

//     return 0;
// }
