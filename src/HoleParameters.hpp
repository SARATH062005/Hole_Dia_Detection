#ifndef HOLE_PARAMETERS_HPP
#define HOLE_PARAMETERS_HPP

#include <map>
#include <string>
#include <opencv2/core.hpp>

struct HoleParams {
    float threshold;
    int exposure;
};

std::map<std::string, double> holeRotationAngles = {
    {"RC2", 0},
    {"RC3", 0},
    {"SP5", 0},
    {"SP9", 0},
    {"SP1", 0},
    {"LA2", 0},
    {"ML2", 0},
    {"RC1", 0},
    {"LA3", 70.0},
    {"LA4", 25.0},
    {"LA5", 25.0},
    {"LA6", 200.0},
    {"LS4", -10.0},
    {"RS4", -75.0},
    {"UA1", 15.0},
    {"UA6", 20.0},

};
// Hole-specific threshold and exposure (using string keys)
inline std::map<std::string, HoleParams> holeParameters = {
    {"RC1", {127.0f, 12000}},
    {"RC2", {130.0f, 11000}},
    {"RC3", {127.0f, 10000}},
    {"ML2", {127.0f, 10000}},
    {"LA2", {127.0f, 10000}},
    {"SP5", {127.0f, 10000}},
    {"RS4", {127.0f, 10000}},
    {"LA4", {127.0f, 10000}},
    {"LS4", {127.0f, 10000}},
    {"UA6", {127.0f, 10000}},
    {"SP1", {127.0f, 10000}},
    {"SP9", {127.0f, 10000}},
    {"UA1", {127.0f, 10000}},
    {"LA3", {127.0f, 10000}},
    {"LA5", {127.0f, 10000}},  
    // Add more hole configs here
};

// Store (x1, y1, x2, y2)
inline std::map<std::string, std::tuple<int, int, int, int>> threadCoordinates = {
    {"RC2", {1576, 1144, 2060, 1828}},
    {"RC3", {1544, 1156, 2120, 1896}},
    {"SP5", {1520, 1240, 1936, 1800}},
    {"SP9", {1124, 1336, 1560, 1800}},
    {"SP1", {1875, 1186, 2501, 1624}},
    {"LA2", {1580, 1240, 1980, 1880}},
    {"ML2", {1656, 1288, 2068, 1856}},
    {"RC1", {1632, 1196, 2144, 1784}},
    {"LA3", {1136, 1880, 2424, 2536}},
    {"LA4", {1628, 1985, 3305, 2387}},
    {"LA5", {1930, 1963, 3762, 2365}},
    {"LA6", {1776, 1930, 3294, 2167}},
    {"LS4", {2200, 1622, 3151, 2046}},
    {"RS4", {1568, 2304, 2912, 2848}},
    {"UA1", {1644, 1996, 3080, 2260}},
    {"UA6", {1694, 1859, 2975, 2365}},
    // Add more here
};

#endif // HOLE_PARAMETERS_HPP

