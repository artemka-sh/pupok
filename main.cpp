#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
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

int main() {
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

    return 0;
}