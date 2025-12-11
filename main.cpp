// #include "opencv2/core.hpp"
// #include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <print>
#include <ctime>
#include <numeric>
#include <complex>
#include <ranges>
using namespace cv;

int MAX_ITERATIONS = 4;

const double RE_START = -2.0;
const double RE_END = 2.0;
const double IM_START = -2.0;
const double IM_END = 2.0;

bool calculate_pixel(double y, double x, double C) {
    std::complex<double> z(x, y);

    for (int i : std::views::iota(0, MAX_ITERATIONS))
    {
        if (std::norm(z) > 4.0)
            return false;

        z = z * z - C;
    }
    return true;
}

int main( int argc, char** argv )
{

    int SIZE_X = 640, SIZE_Y = 640;
    cv::namedWindow("Winda", cv::WINDOW_NORMAL);


    cv::Mat_<double> display(SIZE_X, SIZE_Y);
    for (double C = -3.28; C <= 4.0; C+=0.005)
    {

        for (auto y: std::views::iota(0, SIZE_Y))
        {
            for (auto x: std::views::iota(0, SIZE_X))
            {
                double map_x = RE_START + (x / (double)SIZE_X) * (RE_END - RE_START);
                double map_y = IM_START + (y / (double)SIZE_Y) * (IM_END - IM_START);

                if (calculate_pixel(map_y, map_x, C)) {
                    display(y, x) = 0.0; // Точка внутри множества (черный)
                } else {
                    display(y, x) = 1.0; // Точка улетела (белый)
                }

            }
        }
        std::print("displaying... {}", C);
        cv::imshow("Winda", display);
        auto key = cv::waitKey(1);

        if (key == 'q') {
            break;
        }

    }

    destroyAllWindows();





    //
    // srand(time(0));
    // int size[] = {10, 10};
    // cv::SparseMat sm(2, size, CV_32F);
    // for(int i = 0; i < 10; i++)
    // {
    //     int idx[2];
    //     idx[0] = rand() % size[0];
    //     idx[1] = rand() % size[1];
    //
    //     sm.ref<float>(idx) += 1.0f;
    // }
    //
    // cv::SparseMatConstIterator_<float> it = sm.begin<float>();
    // cv::SparseMatConstIterator_<float> it_end = sm.end<float>();
    //
    // for(; it != it_end; it++)
    // {
    //     const cv::SparseMat::Node* node = it.node();
    //     std::print(" ({}, {}) = {}\n", node->idx[0], node->idx[1], *it);
    // }
}