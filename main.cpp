#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/optim.hpp>

using namespace cv;
using namespace std;

// --- СТРУКТУРЫ ---
struct Particle {
    double x, xp;
    double y, yp;
};

struct BeamState {
    double s;
    double bx, ax, by, ay;
    bool is_magnet;
    vector<Particle> particles; // Облако частиц в конкретной точке
};

// Глобальные переменные для интерактива
vector<BeamState> history;
double eps_x = 10e-9;
double eps_y = 2e-9;

// --- МАТЕМАТИКА УСКОРИТЕЛЯ ---
Matx22d getDrift(double L) {
    return Matx22d(1.0, L, 0.0, 1.0);
}

Matx22d getQuad(double L, double K, bool is_x) {
    if (std::abs(K) < 1e-9) return getDrift(L);
    double k_val = is_x ? K : -K;
    double r = std::sqrt(std::abs(k_val));
    if (k_val > 0) {
        return Matx22d(cos(r * L), (1.0 / r) * sin(r * L),
                      -r * sin(r * L), cos(r * L));
    } else {
        return Matx22d(cosh(r * L), (1.0 / r) * sinh(r * L),
                       r * sinh(r * L), cosh(r * L));
    }
}

Matx22d transformTwiss(const Matx22d& T_in, const Matx22d& M) {
    return M * T_in * M.t();
}

// --- ЧАСТИЦЫ (Код Богдана) ---
vector<Particle> generateBeam(int N, double betaX, double alphaX, double betaY, double alphaY) {
    vector<Particle> beam;
    RNG rng;
    for (int i = 0; i < N; i++) {
        double u = rng.uniform(0.0, 1.0);
        double v = rng.uniform(0.0, 1.0);
        double r = sqrt(u);
        double theta = 2 * M_PI * v;

        double x = sqrt(eps_x * betaX) * r * cos(theta);
        double xp = -sqrt(eps_x / betaX) * (alphaX * r * cos(theta) + r * sin(theta));

        double y = sqrt(eps_y * betaY) * r * sin(theta);
        double yp = -sqrt(eps_y / betaY) * (alphaY * r * sin(theta) + r * cos(theta));

        beam.push_back({x, xp, y, yp});
    }
    return beam;
}

void propagateParticles(vector<Particle>& beam, Matx22d Mx, Matx22d My) {
    for (auto& p : beam) {
        double x_new = Mx(0, 0) * p.x + Mx(0, 1) * p.xp;
        double xp_new = Mx(1, 0) * p.x + Mx(1, 1) * p.xp;
        p.x = x_new; p.xp = xp_new;

        double y_new = My(0, 0) * p.y + My(0, 1) * p.yp;
        double yp_new = My(1, 0) * p.y + My(1, 1) * p.yp;
        p.y = y_new; p.yp = yp_new;
    }
}

// Проекция для 3D вида
Point project3D(double x, double y) {
    // Увеличиваем масштаб для наглядности (м -> пиксели)
    int px = 400 + x * 500000;
    int py = 300 - y * 500000;
    return Point(px, py);
}

// --- ОПТИМИЗАТОР (Твой код) ---
class BeamMatchingObjective : public MinProblemSolver::Function {
public:
    int getDims() const override { return 4; }
    double calc(const double* x) const override {
        double L_drift = 1.0, L_quad = 0.5;
        Matx22d Mx = Matx22d::eye(), My = Matx22d::eye();

        for (int i = 0; i < 4; ++i) {
            Mx = getQuad(L_quad, x[i], true) * getDrift(L_drift) * Mx;
            My = getQuad(L_quad, x[i], false) * getDrift(L_drift) * My;
        }
        Mx = getDrift(L_drift) * Mx; My = getDrift(L_drift) * My;

        Matx22d Tx(5.0, 0.5, 0.5, (1 + 0.5 * 0.5) / 5.0);
        Matx22d Ty(2.5, -0.3, -0.3, (1 + 0.3 * 0.3) / 2.5);

        Matx22d Tx_out = transformTwiss(Tx, Mx);
        Matx22d Ty_out = transformTwiss(Ty, My);

        return pow((Tx_out(0, 0) - 8.0), 2) + pow(-Tx_out(0, 1) - 0.0, 2) +
               pow((Ty_out(0, 0) - 4.0), 2) + pow(-Ty_out(0, 1) - 0.0, 2);
    }
};

// --- ВИЗУАЛИЗАЦИЯ (Отрисовка в окнах) ---
void drawPhaseEllipse(Mat& img, double beta, double alpha, double epsilon, Scalar color, string label) {
    int cx = img.cols / 2, cy = img.rows / 2;
    double scale_pos = 150.0 / 3e-4; // Жесткий масштаб для честного вида
    double scale_ang = 150.0 / 1e-4;

    vector<Point> pts;
    for (double t = 0; t < 2 * CV_PI; t += 0.05) {
        double x = sqrt(epsilon * beta) * cos(t);
        double xp = -sqrt(epsilon / beta) * (alpha * cos(t) + sin(t));
        pts.push_back(Point(cx + x * scale_pos, cy + xp * scale_ang));
    }
    line(img, Point(0, cy), Point(img.cols, cy), Scalar(40, 40, 40), 1);
    line(img, Point(cx, 0), Point(cx, img.rows), Scalar(40, 40, 40), 1);
    polylines(img, pts, true, color, 2, LINE_AA);
    putText(img, label, Point(10, (color[2] > 0 ? 30 : 60)), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

// Callback ползунка: обновляет 2D фазовое пространство и 3D вид
void onTrackbar(int pos, void*) {
    if (history.empty()) return;
    BeamState st = history[pos];

    // 1. Окно фазового пространства (эллипсы X-x', Y-y')
    Mat phase = Mat::zeros(400, 400, CV_8UC3);
    drawPhaseEllipse(phase, st.bx, st.ax, eps_x, Scalar(0, 0, 255), "X plane: " + std::to_string(st.bx));
    drawPhaseEllipse(phase, st.by, st.ay, eps_y, Scalar(255, 0, 0), "Y plane: " + std::to_string(st.by));
    imshow("Phase Space (X-x', Y-y')", phase);

    // 2. Окно 3D среза (частицы в трубе)
    Mat view3d(600, 800, CV_8UC3, Scalar(255, 255, 255));

    // Рисуем кольцо трубы/магнита
    int ring_radius = 250;
    if (st.is_magnet) {
        circle(view3d, Point(400, 300), ring_radius, Scalar(200, 255, 200), -1); // Зеленая заливка
        circle(view3d, Point(400, 300), ring_radius, Scalar(0, 150, 0), 4);      // Зеленый контур
        putText(view3d, "QUADRUPOLE MAGNET", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 100, 0), 2);
    } else {
        circle(view3d, Point(400, 300), ring_radius, Scalar(240, 240, 240), -1); // Серая заливка
        circle(view3d, Point(400, 300), ring_radius, Scalar(150, 150, 150), 2);  // Серый контур
        putText(view3d, "DRIFT SPACE", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 100, 100), 2);
    }

    // Рисуем частицы
    for (const auto& p : st.particles) {
        circle(view3d, project3D(p.x, p.y), 2, Scalar(50, 50, 50), -1, LINE_AA);
    }

    putText(view3d, "Position S = " + to_string(st.s).substr(0,4) + " m", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    imshow("3D Beam Cross-Section", view3d);
}

int main() {
    // Окна
    namedWindow("Beam Envelope", WINDOW_NORMAL);
    namedWindow("Phase Space (X-x', Y-y')", WINDOW_NORMAL);
    namedWindow("3D Beam Cross-Section", WINDOW_NORMAL);

    // 1. Оптимизация
    Ptr<DownhillSolver> solver = DownhillSolver::create();
    solver->setFunction(makePtr<BeamMatchingObjective>());
    Mat x = (Mat_<double>(1, 4) << 0.5, -0.5, 0.5, -0.5);
    Mat step = (Mat_<double>(1, 4) << 0.05, 0.05, 0.01, 0.01);
    solver->setInitStep(step);
    solver->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50000, 1e-10));

    solver->minimize(x);
    double* K = x.ptr<double>();

    // 2. Генерация данных (Tracking)
    double ds = 0.005, L_drift = 1.0, L_quad = 0.5, s_curr = 0;

    // Начальные параметры
    Matx22d Tx(5.0, 0.5, 0.5, 1.25/5.0);
    Matx22d Ty(2.5, -0.3, -0.3, 1.09/2.5);
    auto beam = generateBeam(800, Tx(0,0), -Tx(0,1), Ty(0,0), -Ty(0,1)); // 800 частиц

    auto track = [&](double L, double k) {
        for (double t = 0; t < L; t += ds) {
            // Сохраняем всё текущее состояние
            history.push_back({s_curr, Tx(0, 0), -Tx(0, 1), Ty(0, 0), -Ty(0, 1), std::abs(k) > 1e-6, beam});

            // Шагаем вперед на ds
            Matx22d Mx_step = getQuad(ds, k, true);
            Matx22d My_step = getQuad(ds, k, false);
            Tx = transformTwiss(Tx, Mx_step);
            Ty = transformTwiss(Ty, My_step);
            propagateParticles(beam, Mx_step, My_step);

            s_curr += ds;
        }
    };

    // Строим линию
    for(int i=0; i<4; ++i) { track(L_drift, 0); track(L_quad, K[i]); }
    track(L_drift, 0);

    // 3. Главная отрисовка огибающей (Твоя)
    int w = 1200, h = 400;
    Mat plot = Mat::ones(h, w, CV_8UC3) * 255;
    double max_b = 0;
    for(auto& st : history) max_b = max({max_b, st.bx, st.by});

    auto toPx = [&](double s, double b) {
        return Point(50 + (s/s_curr)*(w-100), h - 50 - (b/max_b)*(h-100));
    };

    for(size_t i=1; i<history.size(); ++i) {
        if(history[i].is_magnet) {
            // Рисуем магниты серым фоном
            line(plot, toPx(history[i].s, 0), Point(toPx(history[i].s, 0).x, 50), Scalar(240,240,240), 2);
        }
        // X - красным, Y - синим
        line(plot, toPx(history[i-1].s, history[i-1].bx), toPx(history[i].s, history[i].bx), Scalar(0,0,255), 2, LINE_AA);
        line(plot, toPx(history[i-1].s, history[i-1].by), toPx(history[i].s, history[i].by), Scalar(255,0,0), 2, LINE_AA);
    }

    // Оси и подписи
    line(plot, Point(50, h-50), Point(w-50, h-50), Scalar(0,0,0), 2);
    putText(plot, "Beam Envelope: Red (X), Blue (Y)", Point(60, 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,0), 1);

    int slider = 0;
    createTrackbar("Position S", "Beam Envelope", &slider, history.size()-1, onTrackbar);
    imshow("Beam Envelope", plot);

    // Инициируем первый кадр
    onTrackbar(0, 0);

    cout << "Optimization Done." << endl;
    cout << "K1=" << K[0] << " K2=" << K[1] << " K3=" << K[2] << " K4=" << K[3] << endl;

    waitKey(0);
    return 0;
}