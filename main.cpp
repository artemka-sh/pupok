#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/optim.hpp>

using namespace cv;
using namespace std;

// Функция для получения матрицы дрейфа
Matx22d getDrift(double L) {
    return Matx22d(1.0, L,
                   0.0, 1.0);
}

// Функция для получения матрицы квадруполя
// is_x = true для горизонтальной плоскости, false для вертикальной
Matx22d getQuad(double L, double K, bool is_x) {
    if (abs(K) < 1e-9) return getDrift(L);

    // В одной плоскости квадруполь фокусирует (K>0), в другой - дефокусирует (K<0)
    double k_val = is_x ? K : -K;
    double sqrtK = sqrt(abs(k_val));

    if (k_val > 0) { // Фокусировка
        return Matx22d(cos(sqrtK * L),          (1.0 / sqrtK) * sin(sqrtK * L),
                       -sqrtK * sin(sqrtK * L), cos(sqrtK * L));
    } else {         // Дефокусировка
        return Matx22d(cosh(sqrtK * L),         (1.0 / sqrtK) * sinh(sqrtK * L),
                       sqrtK * sinh(sqrtK * L), cosh(sqrtK * L));
    }
}

// Трансформация матрицы Твисса (эллипса) через матрицу элемента: T_out = M * T_in * M^T
Matx22d transformTwiss(const Matx22d& T_in, const Matx22d& M) {
    return M * T_in * M.t();
}

// Целевая функция для оптимизатора OpenCV
class BeamMatchingObjective : public MinProblemSolver::Function {
public:
    int getDims() const override { return 4; } // У нас 4 квадруполя (4 переменных K)

    double calc(const double* x) const override {
        // Силы квадруполей (наши переменные для оптимизации)
        double K1 = x[0], K2 = x[1], K3 = x[2], K4 = x[3];

        // Длины элементов (в метрах). Можно менять под ваши габариты.
        double L_drift = 1.0;
        double L_quad = 0.5;

        // Создаем матрицы для X и Y плоскостей

        Matx22d Mx = Matx22d::eye();
        Matx22d My = Matx22d::eye();

        // Собираем линию: Дрейф -> Квад1 -> Дрейф -> Квад2 -> Дрейф -> Квад3 -> Дрейф -> Квад4 -> Дрейф
        double K_vals[4] = {K1, K2, K3, K4};
        for (int i = 0; i < 4; ++i) {
            Mx = getDrift(L_drift) * Mx;
            My = getDrift(L_drift) * My;

            Mx = getQuad(L_quad, K_vals[i], true) * Mx;
            My = getQuad(L_quad, K_vals[i], false) * My;
        }
        Mx = getDrift(L_drift) * Mx;
        My = getDrift(L_drift) * My;

        // Начальные параметры Твисса (Таблица 1)
        double bx0 = 5.0, ax0 = -0.5, gx0 = (1 + ax0*ax0) / bx0;
        double by0 = 2.5, ay0 = 0.3,  gy0 = (1 + ay0*ay0) / by0;
        Matx22d Tx0(bx0, -ax0, -ax0, gx0);
        Matx22d Ty0(by0, -ay0, -ay0, gy0);

        // Прогоняем пучок через систему
        Matx22d Tx_out = transformTwiss(Tx0, Mx);
        Matx22d Ty_out = transformTwiss(Ty0, My);

        // Целевые параметры (Таблица 2)
        double bx_target = 8.0, ax_target = 0.0;
        double by_target = 4.0, ay_target = 0.0;

        // Вычисляем ошибку (Loss). Чем ближе к нулю, тем лучше.
        double loss = pow(Tx_out(0,0) - bx_target, 2) +
                      pow(-Tx_out(0,1) - ax_target, 2) +
                      pow(Ty_out(0,0) - by_target, 2) +
                      pow(-Ty_out(0,1) - ay_target, 2);

        return loss;
    }
};

void drawBeamEnvelope(double K1, double K2, double K3, double K4) {
    // Параметры для шага
    double ds = 0.01; // Шаг 1 см
    double L_drift = 1.0;
    double L_quad = 0.5;

    // Эмиттансы (из таблиц, в нм*рад)
    double eps_x = 10e-9;
    double eps_y = 2e-9;

    // Начальные параметры Твисса
    double bx = 5.0, ax = -0.5, gx = (1 + ax*ax)/bx;
    double by = 2.5, ay = 0.3,  gy = (1 + ay*ay)/by;
    Matx22d Tx(bx, -ax, -ax, gx);
    Matx22d Ty(by, -ay, -ay, gy);

    // Массивы для хранения точек графика: (координата S, размер пучка Sigma)
    vector<Point2d> env_x, env_y;
    double current_s = 0.0;

    // Лямбда-функция для прохода по элементу (дрейф или квадруполь)
    auto trackElement = [&](double length, double K) {
        int steps = round(length / ds);
        for (int i = 0; i < steps; ++i) {
            // Считаем размер пучка в текущей точке
            double sigma_x = sqrt(eps_x * Tx(0,0)); // Tx(0,0) это beta_x
            double sigma_y = sqrt(eps_y * Ty(0,0));

            env_x.push_back(Point2d(current_s, sigma_x));
            env_y.push_back(Point2d(current_s, sigma_y));

            // Продвигаем матрицу Твисса на шаг ds
            Matx22d Mx_step = getQuad(ds, K, true);
            Matx22d My_step = getQuad(ds, K, false);

            Tx = transformTwiss(Tx, Mx_step);
            Ty = transformTwiss(Ty, My_step);

            current_s += ds;
        }
    };

    // Собираем линию с найденными параметрами K
    trackElement(L_drift, 0.0);
    trackElement(L_quad, K1);
    trackElement(L_drift, 0.0);
    trackElement(L_quad, K2);
    trackElement(L_drift, 0.0);
    trackElement(L_quad, K3);
    trackElement(L_drift, 0.0);
    trackElement(L_quad, K4);
    trackElement(L_drift, 0.0);

    // --- Отрисовка в OpenCV ---
    int img_w = 1000, img_h = 600;
    Mat plot = Mat::ones(img_h, img_w, CV_8UC3) * 255; // Белый фон

    // Поиск максимальных значений для масштабирования
    double max_sigma = 0;
    for(auto& p : env_x) max_sigma = max(max_sigma, p.y);
    for(auto& p : env_y) max_sigma = max(max_sigma, p.y);
    max_sigma *= 1.2; // Отступ сверху

    // Функция перевода физических координат в пиксели
    auto toPixel = [&](Point2d pt) {
        int px = cvRound((pt.x / current_s) * (img_w - 100)) + 50;
        int py = img_h - 50 - cvRound((pt.y / max_sigma) * (img_h - 100));
        return Point(px, py);
    };

    // Отрисовка осей
    line(plot, Point(50, img_h-50), Point(img_w-50, img_h-50), Scalar(0,0,0), 2); // X
    line(plot, Point(50, img_h-50), Point(50, 50), Scalar(0,0,0), 2); // Y

    // Отрисовка графиков
    for (size_t i = 1; i < env_x.size(); ++i) {
        // X - красным (горизонтальная плоскость)
        line(plot, toPixel(env_x[i-1]), toPixel(env_x[i]), Scalar(0, 0, 255), 2);
        // Y - синим (вертикальная плоскость)
        line(plot, toPixel(env_y[i-1]), toPixel(env_y[i]), Scalar(255, 0, 0), 2);
    }

    // Добавим легенду
    putText(plot, "X Envelope (Red)", Point(70, 70), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
    putText(plot, "Y Envelope (Blue)", Point(70, 100), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);

    imshow("Beam Envelope", plot);
    waitKey(0);
}

int main() {
    cv::namedWindow("Beam Envelope", cv::WINDOW_NORMAL);
    Ptr<DownhillSolver> solver = DownhillSolver::create();
    Ptr<MinProblemSolver::Function> objFunc = makePtr<BeamMatchingObjective>();
    solver->setFunction(objFunc);

    // Начальное приближение для сил квадруполей K1, K2, K3, K4
    Mat x = (Mat_<double>(1, 4) << 1.0, -1.0, 1.0, -1.0);

    // Шаг для симплекса
    Mat step = (Mat_<double>(1, 4) << 0.1, 0.1, 0.1, 0.1);
    solver->setInitStep(step);

    // Критерии остановки (точность)
    TermCriteria termcrit(TermCriteria::MAX_ITER + TermCriteria::EPS, 10000, 1e-6);
    solver->setTermCriteria(termcrit);

    // Запуск оптимизации
    double res = solver->minimize(x);

    cout << "Оптимизация завершена." << endl;
    cout << "Финальная ошибка (Loss): " << res << endl;
    cout << "Подобранные параметры квадруполей (K):" << endl;
    cout << "K1 = " << x.at<double>(0, 0) << " m^-2" << endl;
    cout << "K2 = " << x.at<double>(0, 1) << " m^-2" << endl;
    cout << "K3 = " << x.at<double>(0, 2) << " m^-2" << endl;
    cout << "K4 = " << x.at<double>(0, 3) << " m^-2" << endl;

    drawBeamEnvelope(x.at<double>(0,0), x.at<double>(0,1), x.at<double>(0,2), x.at<double>(0,3));
    return 0;
}