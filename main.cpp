// #include "opencv2/core.hpp"
// #include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <print>
#include <numeric>
#include <complex>
#include <ranges>

using namespace cv;

int MAX_ITERATIONS = 100;

const double RE_START = -2.0;
const double RE_END = 2.0;
const double IM_START = -2.0;
const double IM_END = 2.0;



class JuliaCalculator: public ParallelLoopBody
{
    Mat &m_img;
    float m_x1;
    float m_y1;
    float m_scaleX;
    float m_scaleY;
    float m_C;
public:
    JuliaCalculator(Mat &img, const float& x1, const float& y1, const float& scaleX, const float& scaleY, const float& C)
    : m_img(img), m_x1(x1), m_y1(y1), m_scaleX(scaleX), m_scaleY(scaleY), m_C(C)
    {}

    virtual void operator()(const Range &range) const CV_OVERRIDE
    {
        for (int r : std::views::iota(range.start, range.end))
        {
            int y = r / m_img.size().height;
            int x = r % m_img.size().width;

            double map_x = RE_START + (x / (double)m_img.cols) * (RE_END - RE_START);
            double map_y = IM_START + (y / (double)m_img.rows) * (IM_END - IM_START);

            if (calculate_pixel(map_y, map_x, m_C)) {
               m_img.at<uchar>(y, x) = 0; // Точка внутри множества (черный)
            } else {
               m_img.at<uchar>(y, x) = 255; // Точка улетела (белый)
            }
        }
    }


    bool calculate_pixel(const double& y, const double& x, const double& C) const{
        std::complex<double> z(x, y);

        for (int i : std::views::iota(0, MAX_ITERATIONS))
        {
            if (std::norm(z) > 4.0)
                return false;

            z = std::pow(z, 5) - C + std::complex<double>(0.0, 0.003);
        }
        return true;
    }
};



int main( int argc, char** argv )
{
    int SIZE_X = 4280, SIZE_Y = 4280;
    cv::namedWindow("Winda", cv::WINDOW_NORMAL);
    cv::Mat_<uchar> juliaImg(SIZE_X, SIZE_Y);

    // for (double C = -0.6099999999999996 ; C <= -0.4999999; C+=0.005)
    double C = 0.549653;
    {

        JuliaCalculator parallelJulia(juliaImg, 0, 0, SIZE_X, SIZE_Y, C);
        parallel_for_(Range(0, juliaImg.rows*juliaImg.cols), parallelJulia);


        std::print("displaying... {} \n.", C);
        cv::imshow("Winda", juliaImg);
        auto key = cv::waitKey(1);

        if (key == 'q') {
            //break;
        }
    }
    waitKey(0);
    cv::destroyAllWindows();

}