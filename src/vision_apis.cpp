#include <array>
#include <string>
#include <opencv2/opencv.hpp> // or at least core.hpp

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <mutex>
#include <chrono>
#include "HoleParameters.hpp"
#include <string>
#include <iostream>
#include <filesystem>
#include "vision_apis.h"
#include <unordered_set>
#include "camera_manager.h"
#include "logger_factory.h"
#include <thread>
#include "logger_macros.h"


using namespace cv;
using namespace std;

namespace VisionApis
{
    std::mutex h_mtx;  // Mutex for Hough fit
    std::mutex r_mtx;  // Mutex for RANSAC fit
    std::mutex ft_mtx; // Mutex for find_thread fit

    std::map<std::string, std::pair<double, double>> holeRadiusLimits = {
        {"RC1", {800, 1000}},
        {"RC2", {800, 1200}},
        {"RC3", {800, 1100}},
        {"ML2", {630, 970}},
        {"LA2", {500, 970}},
        {"SP5", {660, 970}},
        {"SP1", {630, 810}},
        {"SP9", {630, 970}},
        {"LA3", {560, 600}},
        {"LA4", {630, 970}},
        {"LA5", {630, 970}},
        {"LA6", {555, 590}},
        {"RS4", {700, 806}},
        {"LS4", {700, 806}},
        {"UA1", {700, 806}},
        {"UA6", {510, 630}},
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
        {"LA3", {816, 756, 2656, 2436}},
        {"LA6", {1228, 532, 3296, 2308}},
        {"LS4", {800, 280, 3152, 2540}},
        {"RS4", {700, 32, 3272, 2296}},
        {"UA1", {1000, 184, 3468, 2712}},
        {"LA2", {920, 424, 3260, 2664}},
        {"ML2", {500, 644, 3284, 2836}},
        {"RC1", {900, 500, 3432, 2980}},
        {"RC2", {552, 452, 3216, 2776}},
        {"RC3", {550, 392, 3044, 2756}},
        {"UA6", {1392, 964, 3184, 2628}},
        {"LA5", {820, 656, 3340, 2916}},
        {"LA4", {724, 728, 3168, 2864}},
        {"SP1", {1096, 528, 3480, 2352}},
        {"SP5", {600, 392, 3268, 2552}},
        {"SP9", {648, 752, 2828, 2640}},

    };

    std::map<std::string, std::pair<float, float>> radius_range_map = {

        {"LA3", {500, 650}},
        {"LA4", {500, 600}},
        {"LA5", {650, 800}},
        {"LA6", {500, 650}},
        {"RS4", {700, 806}},
        {"LS4", {700, 806}},
        {"UA1", {700, 806}},
        {"UA6", {500, 600}},
        {"RC1", {800, 1100}},
        {"RC2", {800, 1000}}};

    std::map<std::string, std::tuple<int, int, int, int>> holeCropCoords = {
        {"LA6", {800, 400, 3000, 2500}},
        {"LA3", {400, 600, 2600, 2600}},
        {"LS4", {900, 20, 4000, 2600}},
        {"RS4", {600, 20, 4000, 2700}},
        {"UA1", {772, 20, 4000, 2700}}};

    const std::unordered_set<std::string> rc_group_ids = {"RC3", "ML2", "LA2", "SP5", "SP1", "SP9", "LA4", "LA5"};
    const std::unordered_set<std::string> ml_la_group_ids = {"RC1", "RC2", "LA3", "LA6", "RS4", "LS4", "UA1", "UA6"};

    /**
     * @brief Computes the mean value of the L-channel within a central ROI of an image.
     * A lower return value indicates a darker region.
     *
     * @param img The input image (assumed to be in BGR format).
     * @param roi_ratio The ratio of the image dimensions to use for the central ROI (e.g., 0.5 for 50%).
     * @return double The mean lightness (L-channel) value in the central ROI.
     */
    double computeCenterRegionMeanL(const cv::Mat &img, double roi_ratio)
    {
        // Ensure the input is a single-channel grayscale image
        if (img.channels() != 1)
        {
            throw std::invalid_argument("Input image must be a single-channel grayscale image.");
        }

        // Define the central Region of Interest (ROI)
        int roi_width = static_cast<int>(img.cols * roi_ratio);
        int roi_height = static_cast<int>(img.rows * roi_ratio);
        int x_start = (img.cols - roi_width) / 2;
        int y_start = (img.rows - roi_height) / 2;

        cv::Rect center_roi(x_start, y_start, roi_width, roi_height);

        // Calculate the mean brightness in the ROI
        return cv::mean(img(center_roi))[0];
    }

    /**
     * @brief Compares two images and returns the one that is darker in the central ROI, along with its filename.
     *
     * @param raw_imgs A vector containing exactly two cv::Mat images to compare.
     * @param fnames A vector containing the corresponding filenames for the images in raw_imgs.
     * @param roi_scaling_factor A factor between 0.0 and 1.0 to determine the size of the central ROI.
     * @return std::pair<cv::Mat, std::string> A pair containing the darker image and its filename.
     * @throws std::invalid_argument if input vectors do not contain exactly two elements or if images are empty.
     */
    std::pair<cv::Mat, std::string> findDarkerImage(
        const std::vector<cv::Mat> &raw_imgs,
        const std::vector<std::string> &fnames,
        double roi_scaling_factor, const std::string &holeId)
    {
        // 1. Input Validation
        if (raw_imgs.size() != 2 || fnames.size() != 2)
        {
            throw std::invalid_argument("Input vectors for images and filenames must both contain exactly two elements.");
        }
        int darker = 0;
        const cv::Mat &img1 = raw_imgs[0];
        const cv::Mat &img2 = raw_imgs[1];

        if (img1.empty() || img2.empty())
        {
            throw std::invalid_argument("One or both input images are empty.");
        }

        // 2. Calculate the mean lightness for the central ROI of each image
        double score1 = computeCenterRegionMeanL(img1, roi_scaling_factor);
        double score2 = computeCenterRegionMeanL(img2, roi_scaling_factor);

        std::cout << "Image '" << fnames[0] << "' central ROI brightness score: " << score1 << std::endl;
        std::cout << "Image '" << fnames[1] << "' central ROI brightness score: " << score2 << std::endl;

        // 3. Compare scores and return the darker image and its filename as a pair
        // A lower L-channel score means the image is darker.

        if (score2 > score1)
        {
            darker = 1;
        }

        // if (holeId == "LA3" || holeId == "LA6" || holeId == "SP5" || holeId == "LA4" || holeId == "LA5")
        if (holeId == "LA3" || holeId == "LA6" || holeId == "ML2" || holeId == "SP9" || holeId == "SP1" || holeId == "RC2" ||
            holeId == "RC3" || holeId == "LA2" || holeId == "LS4" || holeId == "RS4" || holeId == "UA1" || holeId == "UA6")

        {
            // return darker
            std::cout << "Current hole - " << holeId << " Returning " << fnames[darker] << " as the darker image. for holes LA3, LA6, SP5, LA4, LA5" << std::endl;
            return {raw_imgs[darker], fnames[darker]};
        }
        else
        {
            // return another
            std::cout << "Current hole - " << holeId << " Returning " << fnames[1 - darker] << " as the brigher image. for holes other than LA3, LA6, SP5, LA4, LA5" << std::endl;
            return {raw_imgs[1 - darker], fnames[1 - darker]};
        }

        // if (score1 <= score2) {

        //     std::cout << "Returning '" << fnames[0] << "' as the darker image." << std::endl;
        //     return {img1, fnames[0]}; // C++17 aggregate initialization for std::pair
        // } else {
        //     std::cout << "Returning '" << fnames[1] << "' as the darker image." << std::endl;
        //     return {img2, fnames[1]};
        // }
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
    std::tuple<cv::Mat, int, int, int, int, float> cropImage(const cv::Mat &image, int x1, int y1, float radius_passthrough, std::shared_ptr<spdlog::logger> logger)
    {
        if (image.empty())
        {
            throw std::runtime_error("Input image is empty in crop image.");
        }
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
            LOG_ERROR_FT(logger, "----> Warning: Invalid crop dimensions in cropImage ");
            return std::make_tuple(cv::Mat(), 0, 0, 0, 0, radius_passthrough);
        }

        int newX1 = x1 - startX; // Center X in cropped image
        int newY1 = y1 - startY; // Center Y in cropped image

        try
        {
            cv::Rect roi(startX, startY, endX - startX, endY - startY);
            printf("\t\t\t\t\tline number 209 crop image\n\n\n\n");
            cv::Mat croppedImage = image(roi).clone();
            LOG_INFO_FT(logger, "----> Image cropped and cloned, size {}x{} ", croppedImage.size().width, croppedImage.size().height);
            return std::make_tuple(croppedImage, newX1, newY1, startX, startY, radius_passthrough);
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "OpenCV Exception during cropping: " << e.what() << std::endl;
            LOG_INFO_FT(logger, "----> {} ", e.what());
            return std::make_tuple(cv::Mat(), 0, 0, 0, 0, radius_passthrough);
        }
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
    Circle fitCircleRANSAC(const std::vector<cv::Point2f> &points, int iterations = 300, float RANSAC_threshold = 2.0f)
    {
        Timer timer("RANSAC circle fit"); // Optional timer

        if (points.size() < 3)
            return {0, 0, 0}; // Cannot fit a circle

        std::mt19937 rng(42); // P1 used fixed seed 42 for repeatability
        // std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count()); // Alternative: time-based seed
        std::uniform_int_distribution<int> dist(0, static_cast<int>(points.size()) - 1);

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

    // findSubpixelEdge function (from Program 2)
    double findSubpixelEdge(
        const cv::Mat &grad_magnitude, // Should be CV_8U now
        cv::Point2f center_in_crop,
        const double angle_rad,
        const double search_start_radius,
        const double search_end_radius)
    {
        double max_rsan = -1.0;
        uchar max_grad_val = 0;
        double step = 0.5; // Scan step along the radius
        int x, y, cnt = 0;

        // std::cout << std::dec;
        //  std::cout << "grad_magnitude size: " << grad_magnitude.cols << " x " << grad_magnitude.rows << std::endl;
        //  std::cout << "grad_magnitude type: " << grad_magnitude.type() << std::endl;
        //  std::cout << "Depth: " << grad_magnitude.depth() << ", Channels: " << grad_magnitude.channels() << std::endl;

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

            x = static_cast<int>(std::round(center_in_crop.x + r_scan * std::cos(angle_rad)));
            y = static_cast<int>(std::round(center_in_crop.y + r_scan * std::sin(angle_rad)));
            ++cnt;

            if (x >= 0 && x < grad_magnitude.cols && y >= 0 && y < grad_magnitude.rows)
            {
                uchar current_grad = grad_magnitude.at<uchar>(y, x);
                // std::cout << "\033[1;32m"
                //           << "\n\t current X value -->" << x
                //           << " current y value -->" << y
                //           << " current grad is ->" << static_cast<int>(current_grad)
                //           << "\033[0m" << "\n";

                if (current_grad > max_grad_val)
                {
                    max_grad_val = current_grad;
                    max_rsan = r_scan;
                    // std::cout << "\t\tItr -> " << cnt
                    //           << " current X value -->" << x
                    //           << " current y value -->" << y
                    //           << " R_max | G_max ---->" << max_rsan << " | " << static_cast<int>(max_grad_val)
                    //           << "\n";
                }
            }
            else
            {
                std::cerr << "Out of bounds: (" << x << ", " << y << ")\n";
                break;
            }
        }

        return max_rsan;
    }

    // double findSubpixelEdge(
    //         const cv::Mat &grad_magnitude,
    //         cv::Point2f center_in_crop,
    //         const double angle_rad,
    //         const double search_start_radius,
    //         const double search_end_radius)
    // {
    //     CV_Assert(!grad_magnitude.empty() && grad_magnitude.type() == CV_32F);

    //     double max_grad_val = 0.0;
    //     int max_idx_relative = -1;

    //     std::vector<double> grad_profile;
    //     std::vector<double> actual_radii_points;

    //     double step = 0.5;
    //     const int max_steps_without_peak = 15;
    //     int no_peak_counter = 0;

    //     for (double r_scan = search_start_radius; r_scan <= search_end_radius; r_scan += step)
    //     {
    //         int x = static_cast<int>(std::round(center_in_crop.x + r_scan * std::cos(angle_rad)));
    //         int y = static_cast<int>(std::round(center_in_crop.y + r_scan * std::sin(angle_rad)));

    //         if (x >= 0 && x < grad_magnitude.cols && y >= 0 && y < grad_magnitude.rows)
    //         {
    //             double current_grad = grad_magnitude.at<float>(y, x);
    //             grad_profile.push_back(current_grad);
    //             actual_radii_points.push_back(r_scan);
    //             std::cout <<" \n\n\n\t\t current X value -->"<<x <<" current y value -->"<<y <<"current grad->"<<current_grad <<"\n\n";
    //             if (current_grad > max_grad_val)
    //             {
    //                 max_grad_val = current_grad;
    //                 max_idx_relative = static_cast<int>(grad_profile.size()) - 1;
    //                 no_peak_counter = 0;
    //             }
    //             else
    //             {
    //                 ++no_peak_counter;
    //                 if (no_peak_counter > max_steps_without_peak)
    //                 {
    //                     break;
    //                 }
    //             }
    //         }
    //         else
    //         {
    //             break;
    //         }
    //     }

    //     if (actual_radii_points.empty())
    //         return -1.0;

    //     if (max_idx_relative <= 0 || max_idx_relative >= static_cast<int>(grad_profile.size()) - 1)
    //     {
    //         return actual_radii_points[max_idx_relative]; // Return boundary point
    //     }

    //     // Optional: Subpixel interpolation here using parabola fit

    //     return actual_radii_points[max_idx_relative];
    // }
    // Calibration factor: pixels per millimeter
    const float PIXELS_PER_MM = 1840.62664043f / 1.1254f;

    float processSingleContourWithRansac(
        const cv::Mat &gray_img_for_rc,
        const std::vector<cv::Point> &contour,
        const std::string &holeId,
        const std::string &log_prefix,
        const std::string &save_path_if_success,
        bool is_fallback_call, CameraManager *cam, const int station_num)
    {
        std::ostringstream logger_name;
        logger_name << "Station" << station_num;
        auto loggers = Logger::LoggerFactory::instance().get_all_loggers();
        auto logger = loggers.at(logger_name.str());

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
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> return -1",
                        holeId, station_num);
            return -1.0f;
        }
        std::cout << log_prefix << " MEC R=" << mec_radius << "px.\n";

        auto [cropGray, cropX, cropY, _, __, ___] = cropImage(gray_img_for_rc,
                                                              static_cast<int>(std::round(mec_center.x)),
                                                              static_cast<int>(std::round(mec_center.y)), mec_radius, logger);
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> Cropping Done, image size {}x{} ",
                    holeId, station_num, cropGray.size().width, cropGray.size().height);
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

        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> No of RANSAC inliers are {}",
                    holeId, station_num, inliers.size());
        if (inliers.size() < 30)
            return -1.0f;

        if (inliers.size() > 2000)
        {
            std::shuffle(inliers.begin(), inliers.end(), std::mt19937(42));
            inliers.resize(100);
        }

        Circle circle = fitCircleRANSAC(inliers, 300, 2.0f);
        if (circle.r == 0)
        {
            std::cout << log_prefix << " RANSAC failed.\n";
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> RANSAC Failed",
                        holeId, station_num);
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

        diameter_mm = std::round((2.0f * circle.r / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
        std::cout << "\033[1;32m" << log_prefix << " RANSAC Fit: R=" << circle.r
                  << "px -> D=" << diameter_mm << "mm [" << (r_in_limits ? "OK" : "OUT") << "]\033[0m\n";

        // Drawing & Saving
        cv::Mat cropDisplay;
        cv::cvtColor(cropGray, cropDisplay, cv::COLOR_GRAY2BGR);
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> GRAY TO BGR",
                    holeId, station_num);
        cv::Scalar circle_color = (is_fallback_call && !r_in_limits) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);

        cv::circle(cropDisplay, {circle.x, circle.y}, static_cast<int>(circle.r), circle_color, 2);
        cv::circle(cropDisplay, {circle.x, circle.y}, 3, cv::Scalar(0, 0, 255), -1);

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

        // cv::imwrite(save_path_if_success, cropDisplay);
        auto cropDisplay_ptr = std::make_shared<cv::Mat>(cropDisplay);
        cam->async_imwrite(save_path_if_success, cropDisplay_ptr);
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> Image write at {} ",
                    holeId, station_num, save_path_if_success);

        std::cout << log_prefix << " Saved: " << save_path_if_success << "\n";

        return diameter_mm;
    }

    // --- END: Definitions ---

    float handleRcGroup(
        cv::Mat &img, // Main image, passed by ref for potential gray conversion
        const std::string &holeId,
        const std::string &save_path, CameraManager *cam, const int station_num)
    {
        std::ostringstream logger_name;
        logger_name << "Station" << station_num;
        auto loggers = Logger::LoggerFactory::instance().get_all_loggers();
        auto logger = loggers.at(logger_name.str());

        std::unique_lock<std::mutex> rclock(r_mtx);
        std::cout << "-> Using RC_GROUP (Contour/RANSAC) logic for " << holeId << "." << std::endl;
        Timer group_timer("RC_GROUP Total for " + holeId);
        float detected_diameter_mm = -1.0f;
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> handleRcGroup() started",
                    holeId, station_num);

        try
        {
            if (img.empty())
            {
                std::cerr << "ERROR (RC): Empty input image.\n";
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> input image empty",
                             holeId, station_num);
                return -1.0f;
            }

            // Convert to grayscale only if necessary (in-place reuse of gray_img)
            cv::Mat gray_img;
            if (img.channels() == 3 || img.channels() == 4)
            {
                cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> BGR to GRAY",
                            holeId, station_num);
            }
            else if (img.channels() == 1)
            {
                gray_img = img; // No copy — just a reference
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> No copy — just a reference",
                             holeId, station_num);
            }
            else
            {
                std::cerr << "ERROR (RC): Unsupported number of channels: " << img.channels() << std::endl;
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> Unsupported number of channels return -1",
                             holeId, station_num);
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
            cv::GaussianBlur(gray_img, binary, cv::Size(7, 7), 0); // binary now holds blurred image
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Gaussian Blur Done",
                        holeId, station_num);
            cv::blur(binary, binary, cv::Size(3, 3)); // further blur in-place
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Only Blur Done",
                        holeId, station_num);
            cv::threshold(binary, binary, threshold_val, 255, cv::THRESH_BINARY); // in-place threshold
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Binary Done with value {}",
                        holeId, station_num, threshold_val);
            // Morphology (in-place)
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::erode(binary, binary, kernel, cv::Point(-1, -1), 2);
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Erode Done",
                        holeId, station_num);

            // Find contours
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty())
            {
                std::cerr << "No contours (RC) for " << holeId << ".\n";
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> No contours found return -1",
                             holeId, station_num);
                return -1.0f;
            }

            std::cout << "Contours (RC): " << contours.size() << std::endl;
            std::sort(contours.begin(), contours.end(), [](const auto &a, const auto &b)
                      { return cv::contourArea(a) > cv::contourArea(b); });

            int numContoursToProcess = std::min(5, static_cast<int>(contours.size()));
            for (int i = 0; i < numContoursToProcess; ++i)
            {
                detected_diameter_mm = processSingleContourWithRansac(gray_img, contours[i], holeId,
                                                                      "(RC Contour " + std::to_string(i) + ")", save_path, false, cam, station_num);
                if (detected_diameter_mm >= 0.0f)
                {
                    return detected_diameter_mm; // Found a valid result
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Contour Processed with diameter {}",
                                holeId, station_num, detected_diameter_mm);
                }
            }

            std::cout << "No valid RANSAC diameter found after checking contours for " << holeId << ".\n";
            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> No valid RANSAC diameter found after checking all contours",
                         holeId, station_num);
        }
        catch (const cv::Exception &cv_e)
        {
            std::cerr << "OpenCV Exception in RC_GROUP for " << holeId << ": " << cv_e.what() << "\n";
            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> {}",
                         holeId, station_num, cv_e.what());
        }
        catch (const std::exception &e)
        {
            std::cerr << "Generic Exception in RC_GROUP for " << holeId << ": " << e.what() << "\n";
            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> {}",
                         holeId, station_num, e.what());
        }
        LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> return -1",
                     holeId, station_num);
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
        const int station_num,
        CameraManager *cam)
    {
        std::ostringstream logger_name;
        logger_name << "Station" << station_num;
        auto loggers = Logger::LoggerFactory::instance().get_all_loggers();
        auto logger = loggers.at(logger_name.str());
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
            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> input image empty",
                         holeId, station_num);
        }

        auto range_map_it = radius_range_map.find(holeId);
        if (range_map_it == radius_range_map.end())
        {
            std::cerr << "ERROR: No radius range found for " << holeId << std::endl;
            if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" || holeId == "RS4")
            {
                attempt_rc_fallback = true;
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> No radius found set fallback",
                             holeId, station_num);
            }
            else
            {
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> return -1",
                             holeId, station_num);
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
                printf("\t\t\t\t\tline number 633  image\n\n\n\n");
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> BGR to GRAY",
                            holeId, station_num);

                color_image_for_drawing_base = img.clone();
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Image clone done",
                            holeId, station_num);
            }
            else if (img.channels() == 1)
            {
                printf("\t\t\t\t\tline number 637  image\n\n\n\n");

                gray_image_for_ml_la = img.clone();
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Image clone done",
                            holeId, station_num);

                cv::cvtColor(img, color_image_for_drawing_base, cv::COLOR_GRAY2BGR);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> GRAY to  BGR",
                            holeId, station_num);
            }
            else
            {
                std::cerr << "ERROR (ML_LA): Unsupported channels: " << img.channels() << std::endl;
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> Unsupported channels",
                             holeId, station_num);

                if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" || holeId == "RS4")
                {
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> set fallback",
                                 holeId, station_num);
                    attempt_rc_fallback = true;
                    throw std::runtime_error("Channel error, attempting fallback");
                }
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> return -1",
                             holeId, station_num);
                return -1.0f;
            }

            if (gray_image_for_ml_la.empty() || color_image_for_drawing_base.empty())
            {
                std::cerr << "ERROR (ML_LA): Critical image (gray or color base) is empty for " << holeId << std::endl;
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> input image empty",
                             holeId, station_num);

                if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" || holeId == "RS4")
                {
                    attempt_rc_fallback = true;
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> set fallback",
                                 holeId, station_num);
                    throw std::runtime_error("Critical image empty, attempting fallback");
                }
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> return -1",
                             holeId, station_num);
                return -1.0f;
            }

            // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(40, 40));

            cv::Mat blurred_image_full_scale;
            if (holeId == "LA3" || holeId == "LA6" || holeId == "RC2")
            {
                cv::GaussianBlur(gray_image_for_ml_la, blurred_image_full_scale, cv::Size(9, 9), 2.0, 2.0);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Gaussian Blur Done",
                            holeId, station_num);
            }
            else
            {
                cv::GaussianBlur(gray_image_for_ml_la, gray_image_for_ml_la, cv::Size(5, 5), 1.0, 1.0);
                cv::bilateralFilter(gray_image_for_ml_la, blurred_image_full_scale, 11, 150, 150);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> bilateralFilter done",
                            holeId, station_num);
            }

            std::vector<double> result_diameters, result_votes;
            std::vector<cv::Point2f> result_center;

            cv::Mat current_cropped_display_img_ml_la;
            cv::Point2f current_final_center_in_crop_ml_la_primary;
            double current_average_refined_radius_ml_la = 0.0;

            for (double val : {0.45, 0.75, 0.85})
            {
                cv::Mat resized_blurred_image;
                float r_mec;
                double hough_scale_factor = val;
                cv::resize(blurred_image_full_scale, resized_blurred_image, cv::Size(),
                           hough_scale_factor, hough_scale_factor, cv::INTER_AREA);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> resize done at scaling factor {}",
                            holeId, station_num, hough_scale_factor);

                // clahe->apply(resized_blurred_image, resized_blurred_image);

                std::vector<cv::Vec4f> circles_hough;
                int hough_minR_scaled = static_cast<int>(lower_hough_radius * hough_scale_factor);
                int hough_maxR_scaled = static_cast<int>(upper_hough_radius * hough_scale_factor);

                cv::Mat tmp_hough_canny, thresholdedImage_hough_canny;
                // cv::bitwise_not(resized_blurred_image, tmp_hough_canny);

                cv::Mat gx, gy;
                cv::Sobel(resized_blurred_image, gx, CV_32F, 1, 0, 3);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Sobel X",
                            holeId, station_num);
                cv::Sobel(resized_blurred_image, gy, CV_32F, 0, 1, 3);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Sobel Y",
                            holeId, station_num);
                cv::magnitude(gx, gy, tmp_hough_canny);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> magnitude x, y, temp",
                            holeId, station_num);
                cv::convertScaleAbs(tmp_hough_canny, tmp_hough_canny);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> convertScaleAbs for otsu threshold",
                            holeId, station_num);
                double otsuThreshold = cv::threshold(tmp_hough_canny, thresholdedImage_hough_canny,
                                                     0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> ostu thresold calculated {}",
                            holeId, station_num, otsuThreshold);
                // int thr = 110;

                // if(holeId == "UA6"){
                //     int thr = 90;
                // }else{
                //     int thr = 110;
                // }

                // iplock.lock();
                if (otsuThreshold < 20)
                    otsuThreshold = 20;
                cv::HoughCircles(resized_blurred_image, circles_hough, cv::HOUGH_GRADIENT,
                                 3, 5.0, 1.5 * otsuThreshold,
                                 20, hough_minR_scaled, hough_maxR_scaled);
                // iplock.unlock();
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Hough algo applied",
                            holeId, station_num);

                if (circles_hough.empty())
                {
                    std::cerr << "No coarse circles (Hough) for " << holeId << " at scale " << hough_scale_factor << std::endl;

                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> No Hough circles found, returned 0",
                                 holeId, station_num);
                    result_diameters.push_back(0.0);
                    continue;
                }
                // std::cout << "Hough circle sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" << circles_hough.size() << std::endl;

                float max_score = -1.0f; // Or use std::numeric_limits<float>::lowest();
                cv::Point2f best_center;
                float best_radius = 0.0f;
                int range = 10;

                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> number of circles {}",
                            holeId, station_num, circles_hough.size());

                if (circles_hough.size() < 10)
                    range = circles_hough.size();

                for (int i = 0; i < range; ++i)
                {
                    cv::Point2f center(circles_hough[i][0], circles_hough[i][1]);
                    float radius = circles_hough[i][2];

                    float score = computeInnerCircleMean(resized_blurred_image, center, radius);

                    // std::cout << "centerrrrr------------------>"
                    //           << center.x << "," << center.y
                    //           << " index--" << i
                    //           << " Score---->" << score << std::endl;

                    if (score > max_score)
                    {
                        max_score = score;
                        best_center = center;
                        best_radius = radius;
                    }
                }

                // After loop
                std::cout << "Best center: (" << best_center.x << ", " << best_center.y
                          << ") with radius: " << best_radius
                          << " and max score: " << max_score << std::endl;

                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Best radius {} center x {} center y {} ",
                            holeId, station_num, best_radius, best_center.x, best_center.y);

                cv::Point2f coarse_center_orig_scale(best_center.x / hough_scale_factor,
                                                     best_center.y / hough_scale_factor);
                float coarse_radius_orig_scale = best_radius / hough_scale_factor;

                std::cout << "Hough center radius:: " << coarse_radius_orig_scale << "hough center x" << static_cast<int>(round(coarse_center_orig_scale.x)) << "hough center y" << static_cast<int>(round(coarse_center_orig_scale.y)) << std::endl;

                auto crop_tuple_display = cropImage(color_image_for_drawing_base,
                                                    static_cast<int>(round(coarse_center_orig_scale.x)),
                                                    static_cast<int>(round(coarse_center_orig_scale.y)),
                                                    coarse_radius_orig_scale, logger);

                current_cropped_display_img_ml_la = std::get<0>(crop_tuple_display);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> cropping done according to hough, crop size {}X{}",
                            holeId, station_num, current_cropped_display_img_ml_la.size().width, current_cropped_display_img_ml_la.size().height);
                current_final_center_in_crop_ml_la_primary =
                    cv::Point2f(std::get<1>(crop_tuple_display), std::get<2>(crop_tuple_display));

                if (current_cropped_display_img_ml_la.empty())
                {
                    std::cerr << "Cropping (ML_LA display) empty for " << holeId << " at scale " << hough_scale_factor << std::endl;
                    result_diameters.push_back(0.0);
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> cropping failed returned 0",
                                holeId, station_num);
                    continue;
                }

                cv::Mat cropped_blurred_for_grad = std::get<0>(cropImage(
                    blurred_image_full_scale,
                    static_cast<int>(round(coarse_center_orig_scale.x)),
                    static_cast<int>(round(coarse_center_orig_scale.y)),
                    coarse_radius_orig_scale, logger));

                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Grayscale image cropping done according to hough, crop size {}X{}",
                            holeId, station_num, cropped_blurred_for_grad.size().width, cropped_blurred_for_grad.size().height);

                if (cropped_blurred_for_grad.empty())
                {
                    std::cerr << "Cropping (ML_LA grad) empty for " << holeId << " at scale " << hough_scale_factor << std::endl;
                    result_diameters.push_back(0.0);
                    continue;
                }

                cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad_magnitude_cropped, edge_image_for_scan;
                cv::Sobel(cropped_blurred_for_grad, grad_x, CV_32F, 1, 0, 5);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Sobel X",
                            holeId, station_num);
                cv::Sobel(cropped_blurred_for_grad, grad_y, CV_32F, 0, 1, 5);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Sobel Y",
                            holeId, station_num);
                cv::magnitude(grad_x, grad_y, grad_magnitude_cropped);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Magnitude of X, Y and gradient image",
                            holeId, station_num);
                cv::convertScaleAbs(grad_magnitude_cropped, grad_magnitude_cropped, 1.0, 0);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> convertScaleAbs for radial scan",
                            holeId, station_num);
                // if (holeId == "LS4" || holeId == "RS4" || holeId == "UA1") {
                if (false)
                {
                    // edge_image_for_scan = cropped_blurred_for_grad;
                    cv::bitwise_not(cropped_blurred_for_grad, edge_image_for_scan);
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ---->Bitwise not for cropped blureed",
                                holeId, station_num);
                }
                else
                {
                    edge_image_for_scan = grad_magnitude_cropped;
                }

                std::vector<double> refined_radii_samples;
                std::vector<cv::Point2f> refined_edge_points_in_crop;
                int num_rays = 1080;
                double search_margin = 10.0;

                double radial_search_start_r = std::max(1.0,
                                                        static_cast<double>(coarse_radius_orig_scale) - search_margin);
                double radial_search_end_r = static_cast<double>((coarse_radius_orig_scale) +
                                                                 search_margin);
                std::cout << "Range :: " << radial_search_start_r << " , " << radial_search_end_r << std::endl;
                // iplock.lock();
                for (int k = 0; k < num_rays; ++k)
                {
                    double angle_rad = k * (CV_PI * 2.0 / num_rays);

                    std::cout << "\033[1;31m" << "\t\t-----------  will invoke findSubpixelEdge with Rstart, REnd, Angle :: "
                              << radial_search_start_r << " | " << radial_search_end_r << " | " << angle_rad << "\033[0m" << "\n";

                    double r_refined_sample = findSubpixelEdge(edge_image_for_scan,
                                                               current_final_center_in_crop_ml_la_primary, angle_rad,
                                                               radial_search_start_r, radial_search_end_r);

                    std::cout << "\033[1;31m" << "\t\t-----------  completed findSubpixelEdge with Rsample :: " << r_refined_sample << "\033[0m" << "\n";

                    if (r_refined_sample > 0)
                    {
                        // std::cout << "R output :: "<< r_refined_sample << std::endl;
                        refined_radii_samples.push_back(r_refined_sample);
                        refined_edge_points_in_crop.push_back(
                            cv::Point2f(current_final_center_in_crop_ml_la_primary.x +
                                            static_cast<float>(r_refined_sample * std::sin(angle_rad)),
                                        current_final_center_in_crop_ml_la_primary.y +
                                            static_cast<float>(r_refined_sample * std::cos(angle_rad))));
                    }
                }
                // iplock.unlock();
                std::cout << "refined_radii_samples size :: " << refined_radii_samples.size() << std::endl;

                if (refined_radii_samples.size() < num_rays * 0.25)
                {
                    std::cerr << "Too few refined edge points (ML_LA) for " << holeId
                              << " at scale " << hough_scale_factor << " (" << refined_radii_samples.size() << ")" << std::endl;
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ---->Too few refined edge points (ML_LA)",
                                 holeId, station_num);

                    result_diameters.push_back(0.0);
                    result_votes.push_back(0.0);
                    result_center.push_back(current_final_center_in_crop_ml_la_primary);
                    continue;
                }

                std::sort(refined_radii_samples.begin(), refined_radii_samples.end());

                int total = refined_radii_samples.size();
                int omit_cnt = total / 10;

                if (total >= 20)
                {
                    // Remove 10% from front
                    refined_radii_samples.erase(refined_radii_samples.begin(),
                                                refined_radii_samples.begin() + omit_cnt);

                    // Remove 10% from end
                    refined_radii_samples.erase(refined_radii_samples.end() - omit_cnt,
                                                refined_radii_samples.end());
                }

                // size_t trim_count = refined_radii_samples.size() / 10;
                // if (refined_radii_samples.size() > 2 * trim_count +
                //                                        std::min((size_t)5, refined_radii_samples.size() / 4))
                // {
                //     refined_radii_samples.assign(refined_radii_samples.begin() + trim_count, refined_radii_samples.end() - trim_count);
                // }

                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ---->Trimmed top 10 outerliers",
                            holeId, station_num);

                current_average_refined_radius_ml_la =
                    std::accumulate(refined_radii_samples.begin(), refined_radii_samples.end(), 0.0) / refined_radii_samples.size();

                if (refined_edge_points_in_crop.size() >= 3)
                {
                    std::vector<cv::Point2f> points_for_center_fit;
                    double tol_center_fit = current_average_refined_radius_ml_la * 0.20;
                    cv::Point2f initial_center_for_selection = current_final_center_in_crop_ml_la_primary;

                    for (auto &pt : refined_edge_points_in_crop)
                    {
                        if (std::abs(cv::norm(pt - initial_center_for_selection) - current_average_refined_radius_ml_la) < tol_center_fit)
                        {
                            points_for_center_fit.push_back(pt);
                        }
                    }

                    if (points_for_center_fit.size() >= 3)
                    {

                        cv::minEnclosingCircle(points_for_center_fit, current_final_center_in_crop_ml_la_primary, r_mec);
                        std::cout << "Refined Center (ML_LA): ("
                                  << current_final_center_in_crop_ml_la_primary.x << ", "
                                  << current_final_center_in_crop_ml_la_primary.y << ")px "
                                  << " Radius :: " << current_average_refined_radius_ml_la << " at scale "
                                  << hough_scale_factor
                                  << std::endl;
                        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ---->minEnclosingCircle drawn ",
                                    holeId, station_num);
                    }
                }

                // double diameter_mm_ml_la_val_primary =
                //     std::round((2.0 * current_average_refined_radius_ml_la / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
                result_diameters.push_back(current_average_refined_radius_ml_la);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> ",
                            holeId, station_num, current_average_refined_radius_ml_la);
                result_votes.push_back(computeInnerCircleMean(current_cropped_display_img_ml_la,
                                                              current_final_center_in_crop_ml_la_primary, current_average_refined_radius_ml_la));
                result_center.push_back(cv::Point2f(current_final_center_in_crop_ml_la_primary.x + static_cast<float>(std::get<3>(crop_tuple_display)),
                                                    current_final_center_in_crop_ml_la_primary.y + static_cast<float>(std::get<4>(crop_tuple_display))));

                // Store the current successful result as the 'best' for drawing
                best_cropped_display_img_ml_la = current_cropped_display_img_ml_la.clone();
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Image clone done size {}X{}",
                            holeId, station_num, best_cropped_display_img_ml_la.size().width, best_cropped_display_img_ml_la.size().height);
                best_final_center_in_crop_ml_la_primary = current_final_center_in_crop_ml_la_primary;
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Image reference best_final_center_in_crop_ml_la_primary point {}, {}",
                            holeId, station_num, best_final_center_in_crop_ml_la_primary.x, best_final_center_in_crop_ml_la_primary.y);
                best_average_refined_radius_ml_la = current_average_refined_radius_ml_la;
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Image reference best_average_refined_radius_ml_la {}",
                            holeId, station_num, best_average_refined_radius_ml_la);

            } // End of loop for scaling factors
            if (result_diameters.size() != 3)
            {
                attempt_rc_fallback = true;
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Set fallback after scale size not equal 3",
                            holeId, station_num);
            }
            else
            {

                if (!result_votes.empty())
                {
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Results vote not empty",
                                holeId, station_num);
                    auto max_votes = std::max_element(result_votes.begin(), result_votes.end());
                    int max_idx = std::distance(result_votes.begin(), max_votes);
                    detected_diameter_px = result_diameters[max_idx];
                    final_center = result_center[max_idx];
                }
                else
                {
                    attempt_rc_fallback = true;
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> set fallback",
                                holeId, station_num);
                }
            }

            if (!attempt_rc_fallback)
            {
                detected_diameter_mm = std::round((2.0 * detected_diameter_px / PIXELS_PER_MM) * 1000.0f) / 1000.0f;
                std::cout << "\033[1;32m" << "Final D (ML_LA Primary): "
                          << detected_diameter_mm << " mm\033[0m" << std::endl;

                auto crop_tuple_display = cropImage(color_image_for_drawing_base,
                                                    static_cast<int>(round(final_center.x)),
                                                    static_cast<int>(round(final_center.y)),
                                                    detected_diameter_px, logger);

                best_cropped_display_img_ml_la = std::get<0>(crop_tuple_display);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Cropping done,best_cropped_display_img_ml_la size{}X{} ",
                            holeId, station_num, best_cropped_display_img_ml_la.size().width, best_cropped_display_img_ml_la.size().height);

                if (!best_cropped_display_img_ml_la.empty())
                {
                    cv::circle(best_cropped_display_img_ml_la,
                               cv::Point2f(std::get<1>(crop_tuple_display), std::get<2>(crop_tuple_display)),
                               static_cast<int>(round(detected_diameter_px)),
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

                    auto img_ptr_to_save = std::make_shared<cv::Mat>(best_cropped_display_img_ml_la);
                    cam->async_imwrite(primary_save_path, img_ptr_to_save);
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> handleMlLaGroup write best_cropped_display_img_ml_la",
                                holeId, station_num);
                }
                else
                {
                    std::cerr << "WARNING: ML_LA detection successful, but drawing image was empty for " << holeId << std::endl;
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> WARNING: ML_LA detection successful, but drawing image was empty",
                                 holeId, station_num);
                }
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> returned diameter {}",
                            holeId, station_num, detected_diameter_mm);
                return detected_diameter_mm;
            }
        }
        catch (const cv::Exception &cv_e)
        {

            std::cerr << "OpenCV Exception in ML_LA_GROUP (Primary) for "
                      << holeId << ": " << cv_e.what() << std::endl;
            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> {}",
                         holeId, station_num, cv_e.what());
            if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" ||
                holeId == "RS4")
            {
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Set fallback",
                            holeId, station_num);
                attempt_rc_fallback = true;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Generic Exception in ML_LA_GROUP (Primary) for "
                      << holeId << ": " << e.what() << std::endl;
            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> {}",
                         holeId, station_num, e.what());
            if (holeId == "LA3" || holeId == "LA6" || holeId == "LS4" ||
                holeId == "RS4")
            {
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Set fallback",
                            holeId, station_num);
                attempt_rc_fallback = true;
            }
        }

        if (attempt_rc_fallback)
        {
            std::cout << "--> Executing RC_GROUP (RANSAC) fallback for "
                      << holeId << "..." << std::endl;
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Executing RC_GROUP (RANSAC) fallback",
                        holeId, station_num);

            try
            {
                cv::Mat gray_rc_fallback;
                if (img.empty())
                {
                    std::cerr << "ERR(Fallback): Input image 'img' is empty before fallback conversion."
                              << std::endl;
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> Image empty return -1",
                                 holeId, station_num);
                    return -1.0f;
                }

                if (img.channels() == 3 || img.channels() == 4)
                {
                    cv::cvtColor(img, gray_rc_fallback, cv::COLOR_BGR2GRAY);
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> BGR TO GRAY",
                                holeId, station_num);
                }
                else if (img.channels() == 1)
                {
                    printf("\t\t\t\t\tline number 885  image\n\n\n\n");

                    gray_rc_fallback = img.clone();
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Image clone done",
                                holeId, station_num);
                }
                else
                {
                    std::cerr << "ERR(Fallback): Unsupported channels for fallback: " << img.channels() << std::endl;
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> Unsupported channels return -1",
                                 holeId, station_num);
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
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Gaussian Blur Done",
                                holeId, station_num);
                    cv::blur(blur_rc, blur_rc, cv::Size(3, 3));
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Only Blur Done",
                                holeId, station_num);
                    cv::threshold(blur_rc, bin_rc, fallback_thresh_val, 255,
                                  cv::THRESH_BINARY);
                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Binary Done",
                                holeId, station_num);

                    cv::Mat kern_rc = cv::getStructuringElement(cv::MORPH_RECT,
                                                                cv::Size(5, 5));
                    cv::erode(bin_rc, bin_rc, kern_rc, cv::Point(-1, -1), 2);

                    LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> Erode Done",
                                holeId, station_num);

                    std::vector<std::vector<cv::Point>> cont_rc;
                    cv::findContours(bin_rc, cont_rc, cv::RETR_TREE,
                                     cv::CHAIN_APPROX_SIMPLE);

                    if (cont_rc.empty())
                    {
                        std::cerr << "No contours found for Fallback RC for " << holeId << std::endl;
                        LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> Contours empty",
                                     holeId, station_num);
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
                                primary_save_path, true, cam, station_num);

                            if (fallback_diam >= 0.0f)
                            {
                                std::cout << "\033[1;36m" << "Final D (RANSAC Fallback): "
                                          << fallback_diam << " mm\033[0m" << std::endl;
                                return fallback_diam;
                            }
                        }
                        std::cout << "RC_GROUP (Fallback) also failed to find a valid diameter "
                                  << "after checking all contours.\n";
                        LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> Fallback also failed",
                                     holeId, station_num);
                    }
                }
                else
                {
                    std::cerr << "ERR(Fallback): Grayscale image for fallback is empty for " << holeId << std::endl;
                    LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> ERR(Fallback)",
                                 holeId, station_num);
                }
            }
            catch (const cv::Exception &cv_e)
            {
                std::cerr << "OpenCV Exception in RC_GROUP (Fallback) for "
                          << holeId << ": " << cv_e.what() << std::endl;
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> {}",
                             holeId, station_num, cv_e.what());
            }
            catch (const std::exception &e_rc)
            {
                std::cerr << "Generic Exception in RC_GROUP (Fallback) for "
                          << holeId << ": " << e_rc.what() << std::endl;
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> {}",
                             holeId, station_num, e_rc.what());
            }
        }

        std::cout << "All detection attempts failed for " << holeId << ". Resulting diameter: " << detected_diameter_mm << std::endl;
        printf("\t\t\t\t\tline number 958 image\n\n\n\n");
        LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ---->All detection attempts failed ML-LA group ",
                     holeId, station_num);

        cv::Mat final_fail_img = img_input_orig.clone();
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ---->Image clone Done ",
                    holeId, station_num);
        if (!final_fail_img.empty())
        {
            if (final_fail_img.channels() == 1)
            {
                cv::cvtColor(final_fail_img, final_fail_img, cv::COLOR_GRAY2BGR);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ---->GRAY TO BGR",
                            holeId, station_num);
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
            cam->async_imwrite(primary_save_path, final_fail_img_ptr);
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> write done {}",
                        holeId, station_num, primary_save_path);

            std::cout << "Saved 'ALL ATTEMPTS FAILED' image to primary path: "
                      << primary_save_path << std::endl;
        }
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ----> return -1",
                    holeId, station_num);
        return -1.0f;
    }

    // The MODIFIED detectHoleDiameter function
    float detectHoleDiameter(
        std::vector<std::string> fnames, std::vector<cv::Mat> raw_imgs, const std::string &cur_process,
        const int station_num,
        const std::string &holeId, CameraManager *cam)
    {
        std::ostringstream logger_name;
        logger_name << "Station" << station_num;
        auto loggers = Logger::LoggerFactory::instance().get_all_loggers();
        auto logger = loggers.at(logger_name.str());

        float detected_diameter_mm = -1.0f;
        auto startTotal = std::chrono::high_resolution_clock::now();

        std::cout << "\n Before functionnnnnnnnnnnnnnnnnnnnnnn: " << std::endl;
        std::string imagePath_for_saving_name;
        cv::Mat img_input;

        // std::cout << "Image shape: ("
        //   << raw_imgs[0].rows << ", "         // height
        //   << raw_imgs[0].cols << ", "
        //   << raw_imgs[0].channels() << ")"   // number of channels
        //   << std::endl;

        if (cur_process == "POST-CNC")
        {
            imagePath_for_saving_name = fnames[0];
            img_input = raw_imgs[0];
        }
        else
        {
            double roi_factor = 0.50;
            // Use C++17 structured bindings to easily unpack the returned pair
            auto [darker_img, darker_fname] = findDarkerImage(raw_imgs, fnames, roi_factor, holeId);

            std::cout << "\nFunction selected: " << darker_fname << std::endl;
            // cv::Mat darker_result = findDarkerImage(raw_imgs, roi_factor);
            std::cout << "\n After functionnnnnnnnnnnnnnnnnnnnnnn: " << std::endl;

            imagePath_for_saving_name = darker_fname;
            img_input = darker_img;
        }

        std::cout << "Processing Hole ID: " << holeId << " for station " << station_num
                  << " (Output base: " << imagePath_for_saving_name << ")" << std::endl;

        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} Output base: {}",
                    holeId, station_num, imagePath_for_saving_name);

        // LOG_ERROR_FT(logger,"Processing Hole ID: {} for station {} (Output base: {})",
        //             holeId, station_num, imagePath_for_saving_name);

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
        // std::string primary_save_path = results_sub_dir + "/"+"result"+"/" + image_file_base_name + "_S" +
        std::string primary_save_path = results_sub_dir + "/" + image_file_base_name + "_S" +
                                        std::to_string(station_num) + "_" + holeId + "_annotated.jpg";
        std::cout << "Output annotated image will be saved to: " << primary_save_path << std::endl;
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} ---->Output annotated image will be saved to: {}",
                    holeId, station_num, primary_save_path);

        if (img_input.empty())
        {
            std::cerr << "ERROR: Input image is empty for holeId: " << holeId << ".\n";

            LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} ----> input image empty",
                         holeId, station_num);
            // Fall through to final timer and return -1.0f
        }
        else
        {
            printf("\t\t\t\t\tline number 1022  image\n\n\n\n");
            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> input image available",
                        holeId, station_num);

            cv::Mat img_processing_copy = img_input.clone(); // Work on a copy

            LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> input image cloned",
                        holeId, station_num);

            if (rc_group_ids.count(holeId))
            {
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> handleRcGroup() called",
                            holeId, station_num);
                detected_diameter_mm = handleRcGroup(img_processing_copy, holeId, primary_save_path, cam, station_num);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> handleRcGroup() diameter returned {}",
                            holeId, station_num, detected_diameter_mm);
            }
            else if (ml_la_group_ids.count(holeId))
            {
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> handleMlLaGroup() called",
                            holeId, station_num);
                detected_diameter_mm = handleMlLaGroup(img_input, img_processing_copy, holeId, primary_save_path, output_dir_base, image_file_base_name, station_num, cam);
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> handleMlLaGroup() diameter returned {}",
                            holeId, station_num, detected_diameter_mm);
            }
            else
            {
                std::cout << "Warning: Hole ID " << holeId << " does not belong to any known processing group." << std::endl;
                // detected_diameter_mm remains -1.0f
                LOG_ERROR_FT(logger, "Processing Hole ID: {} for station {} -----> Invalid Hole ID unable to process both functions",
                             holeId, station_num);
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
        LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> final diamtere from detectHoleDiameter() is {} ",
                    holeId, station_num, detected_diameter_mm);
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

    // int find_threads(const std::string& imagePath, cv::Mat raw_img, int station_num, const std::string& hole_no){

    // int find_threads(Mat raw_img, const string& hole_no) {
    // cv::imwrite(imagePath, raw_img);
    int find_threads(std::vector<std::string> fnames, std::vector<cv::Mat> raw_imgs, const std::string &cur_process,
                     int station_num,
                     const std::string &hole_no,
                     CameraManager *cam)
    {
        std::ostringstream logger_name;
        logger_name << "Station" << station_num;
        auto loggers = Logger::LoggerFactory::instance().get_all_loggers();
        auto logger = loggers.at(logger_name.str());

        std::string imagePath = fnames.at(0);
        cv::Mat raw_img = raw_imgs.at(0);

        std::unique_lock<std::mutex> ftlock(ft_mtx);
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

                for (size_t i = 0; i < filtered_contours.size(); ++i)
                {
                    RotatedRect minRect = minAreaRect(filtered_contours[i]);

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
                auto padded_image_ptr = std::make_shared<cv::Mat>(padded_image);

                cam->async_imwrite(thread_new_path, padded_image_ptr);
                // LOG_INFO_FT( logger, "-----> find_threads write padded_image_ptr");
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> Thread image write {} ",
                            hole_no, station_num, thread_new_path);
                // imwrite(thread_new_path, padded_image);
                return thread_count;
            }

            else
            {
                threshold(cropped_image, thresh, 127, 255, THRESH_BINARY);

                // Define a kernel for dilation (e.g., 11x11 rectangular)
                Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
                dilate(thresh, thresh, kernel);

                // Save thresholded image for debugging
                // imwrite("thresh_erode.jpg", thresh);

                int height = thresh.rows;
                int width = thresh.cols;

                // Convert grayscale to color for visualization
                Mat cropped_rgba;
                cvtColor(cropped_image, cropped_rgba, COLOR_GRAY2BGR);

                int avg_thread_count = 0;
                bool status = false;
                const int min_distance = 600;
                const int max_attempts = 2;
                double min_required = 0; // Replace this with actual minimum requirement

                // Filtering parameters for contour area
                const double min_area = 1000.0;

                double max_area = 4000.0;
                if (hole_no == "SP5")
                {
                    max_area = 5000.0;
                }

                int y_line = height / 2 - 25; // Initial y_line for cropping

                for (int attempt = 0; attempt < max_attempts; ++attempt)
                {
                    // Step 1: Crop horizontal strip
                    int crop_top = max(y_line, 0);
                    int crop_bottom = min(y_line + 50, height);
                    Rect crop_rect(0, crop_top, width, crop_bottom - crop_top);
                    Mat horizontal_strip = thresh(crop_rect);

                    // Step 2: Find contours
                    vector<vector<Point>> contours;
                    findContours(horizontal_strip, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                    // Step 3: Filter contours by area
                    vector<vector<Point>> valid_contours;
                    for (const auto &contour : contours)
                    {
                        double area = contourArea(contour);
                        // Draw the contour (green line)
                        drawContours(cropped_rgba, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 1); // green contour

                        // Calculate contour center for placing text
                        Moments m = moments(contour);
                        Point center;
                        if (m.m00 != 0)
                        {
                            center = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00) + crop_top);
                        }
                        else
                        {
                            center = Point(0, 0);
                        }

                        // Put area text near the center (red text)
                        putText(cropped_rgba, to_string(static_cast<int>(area)), center, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);

                        if (area >= min_area && area <= max_area)
                        {
                            valid_contours.push_back(contour);
                        }
                    }

                    // Step 4: Draw cropped region
                    rectangle(cropped_rgba, crop_rect, Scalar(0, 0, 255), 1); // Red rectangle

                    // Step 5: Calculate centers of valid contours
                    vector<Point> centers;
                    for (const auto &contour : valid_contours)
                    {
                        Rect bbox = boundingRect(contour);
                        Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2 + crop_top);
                        centers.push_back(center);
                    }

                    // Step 6: Filter centers by distance within 100 pixels
                    vector<bool> is_valid(centers.size(), true);
                    for (size_t i = 0; i < centers.size(); ++i)
                    {
                        for (size_t j = i + 1; j < centers.size(); ++j)
                        {
                            double dist = norm(centers[i] - centers[j]);
                            std::cout << "Hole distname: " << dist << std::endl;
                            if (dist > min_distance)
                            {
                                is_valid[j] = false;
                            }
                        }
                    }

                    // Step 7: Count only valid centers and draw them
                    // Sort valid centers by x (rightmost first)
                    vector<pair<Point, size_t>> centers_with_index;
                    for (size_t i = 0; i < centers.size(); ++i)
                    {
                        if (is_valid[i])
                        {
                            centers_with_index.push_back({centers[i], i});
                        }
                    }

                    sort(centers_with_index.begin(), centers_with_index.end(),
                         [](const pair<Point, size_t> &a, const pair<Point, size_t> &b)
                         {
                             return a.first.x > b.first.x;
                         });

                    // Determine number to exclude based on hole_no
                    size_t exclude_count = 0;
                    if (hole_no == "RC1" || hole_no == "RC2" || hole_no == "RC3")
                    {
                        exclude_count = min(static_cast<size_t>(1), centers_with_index.size());
                    }
                    if (hole_no == "SP5")
                    {
                        exclude_count = min(static_cast<size_t>(1), centers_with_index.size());
                    }

                    // Final count
                    int filtered_count = centers_with_index.size() - exclude_count;

                    // Draw only valid centers not excluded
                    for (size_t idx = exclude_count; idx < centers_with_index.size(); ++idx)
                    {
                        Point center = centers_with_index[idx].first;
                        circle(cropped_rgba, center, 3, Scalar(255, 0, 0), -1); // Blue dot
                    }

                    // Update thread count
                    avg_thread_count = filtered_count;

                    // Check if minimum required threads are met
                    min_required = min_thread_count_per_hole[hole_no];
                    status = avg_thread_count >= min_required;

                    if (status)
                    {
                        break; // Found enough threads, stop
                    }

                    // Shift y_line for next attempt
                    y_line += 100;
                    if (y_line + 50 >= height)
                    {
                        break; // Avoid going out of bounds
                    }
                }
                Scalar padding_color(200, 200, 200);
                Mat padded_image;
                copyMakeBorder(cropped_rgba, padded_image, 200, 0, 0, 0, BORDER_CONSTANT, padding_color);

                // Format detected_count
                std::ostringstream oss_count;
                oss_count.precision(1);
                oss_count << std::fixed << avg_thread_count;
                std::string text = "Thread Count: " + oss_count.str();

                // Format min_required
                std::ostringstream oss_min;
                oss_min.precision(1);
                oss_min << std::fixed << min_required;
                std::string min_text = "Min Required: " + oss_min.str();
                std::string status_text = "Status: " + std::string(status ? "PASS" : "FAIL");
                Scalar status_color = status ? Scalar(255, 0, 0) : Scalar(0, 0, 255); // Green or Red

                // Draw texts
                putText(padded_image, text, Point(20, 30), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 2);
                putText(padded_image, min_text, Point(20, 70), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 2);
                putText(padded_image, status_text, Point(20, 120), FONT_HERSHEY_COMPLEX, 1, status_color, 2);
                // imwrite(thread_new_path, padded_image);
                auto padded_image_ptr = std::make_shared<cv::Mat>(padded_image);

                cam->async_imwrite(thread_new_path, padded_image_ptr);
                // LOG_INFO_FT( logger, "-----> find_threads write padded_image_ptr 2");
                LOG_INFO_FT(logger, "Processing Hole ID: {} for station {} -----> Thread image write {} ",
                            hole_no, station_num, thread_new_path);

                // imwrite(thread_new_path, thresh);

                std::cout << "\033[1;35m" << " Hole id = " << hole_no << ", Final Avg Thread Count = " << avg_thread_count << "\033[0m" << std::endl;

                return avg_thread_count;
            }
        }
        else
        {
            std::cerr << "Hole number not found: " << hole_no << std::endl;
            return -1;
        }
    }
}
