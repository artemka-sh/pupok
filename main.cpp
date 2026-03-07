#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/optim.hpp>

using namespace cv;
using namespace std;

// --- СТРУКТУРЫ ---
struct Particle {
    double x, xp;
    double y, yp;
};

// Дипольный магнит (коррекция орбиты)
struct Dipole {
    double s_pos;       // позиция вдоль линии [м]
    double angle_mrad;  // угол отклонения [мрад]
    string name;        // название магнита
};

struct BeamState {
    double s;
    double bx, ax, by, ay;
    double centroid_x;   // смещение центра пучка [м]
    double centroid_xp;  // угол центра пучка [рад]
    bool is_magnet;
    bool is_dipole;
    vector<Particle> particles;
};

// Глобальные переменные
vector<BeamState> history;
vector<Dipole>    dipoles_applied; // диполи, применённые в симуляции
double eps_x = 10e-9;
double eps_y = 2e-9;

// --- МАТЕМАТИКА ---
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

// --- ЧАСТИЦЫ ---
vector<Particle> generateBeam(int N, double betaX, double alphaX, double betaY, double alphaY) {
    vector<Particle> beam;
    RNG rng(12345);
    for (int i = 0; i < N; i++) {
        double u = rng.uniform(0.0, 1.0);
        double v = rng.uniform(0.0, 1.0);
        double r = sqrt(u);
        double theta = 2 * CV_PI * v;

        double x  = sqrt(eps_x * betaX) * r * cos(theta);
        double xp = -sqrt(eps_x / betaX) * (alphaX * r * cos(theta) + r * sin(theta));
        double y  = sqrt(eps_y * betaY) * r * sin(theta);
        double yp = -sqrt(eps_y / betaY) * (alphaY * r * sin(theta) + r * cos(theta));

        beam.push_back({x, xp, y, yp});
    }
    return beam;
}

void propagateParticles(vector<Particle>& beam, Matx22d Mx, Matx22d My) {
    for (auto& p : beam) {
        double x_new  = Mx(0,0)*p.x  + Mx(0,1)*p.xp;
        double xp_new = Mx(1,0)*p.x  + Mx(1,1)*p.xp;
        p.x = x_new; p.xp = xp_new;

        double y_new  = My(0,0)*p.y  + My(0,1)*p.yp;
        double yp_new = My(1,0)*p.y  + My(1,1)*p.yp;
        p.y = y_new; p.yp = yp_new;
    }
}

// Применить угловой удар диполя ко всему пучку
void applyDipoleKick(vector<Particle>& beam, double angle_rad) {
    for (auto& p : beam) {
        p.xp += angle_rad;
    }
}

Point project3D(double x, double y) {
    int px = 400 + (int)(x * 500000);
    int py = 300 - (int)(y * 500000);
    return Point(px, py);
}

// --- ОПТИМИЗАТОР ---
class BeamMatchingObjective : public MinProblemSolver::Function {
public:
    int getDims() const override { return 4; }
    double calc(const double* x) const override {
        double L_drift = 1.0, L_quad = 0.5;
        Matx22d Mx = Matx22d::eye(), My = Matx22d::eye();

        for (int i = 0; i < 4; ++i) {
            Mx = getQuad(L_quad, x[i], true)  * getDrift(L_drift) * Mx;
            My = getQuad(L_quad, x[i], false) * getDrift(L_drift) * My;
        }
        Mx = getDrift(L_drift) * Mx;
        My = getDrift(L_drift) * My;

        Matx22d Tx(5.0, 0.5, 0.5, (1 + 0.5*0.5)/5.0);
        Matx22d Ty(2.5, -0.3, -0.3, (1 + 0.3*0.3)/2.5);

        Matx22d Tx_out = transformTwiss(Tx, Mx);
        Matx22d Ty_out = transformTwiss(Ty, My);

        return pow((Tx_out(0,0) - 8.0), 2) + pow(-Tx_out(0,1) - 0.0, 2) +
               pow((Ty_out(0,0) - 4.0), 2) + pow(-Ty_out(0,1) - 0.0, 2);
    }
};

void drawPhaseEllipse(Mat& img, double beta, double alpha, double epsilon,
                      Scalar color, string label, int y_offset = 30) {
    int cx = img.cols / 2, cy = img.rows / 2;
    double scale_pos = 200.0 / 4e-4;
    double scale_ang = 200.0 / 1.5e-4;

    vector<Point> pts;
    for (double t = 0; t < 2 * CV_PI; t += 0.05) {
        double x  = sqrt(epsilon * beta) * cos(t);
        double xp = -sqrt(epsilon / beta) * (alpha * cos(t) + sin(t));
        pts.push_back(Point(cx + (int)(x * scale_pos), cy + (int)(xp * scale_ang)));
    }
    line(img, Point(0, cy), Point(img.cols, cy), Scalar(40,40,40), 1);
    line(img, Point(cx, 0), Point(cx, img.rows), Scalar(40,40,40), 1);
    polylines(img, pts, true, color, 2, LINE_AA);
    putText(img, label, Point(10, y_offset), FONT_HERSHEY_SIMPLEX, 0.5, color, 1, LINE_AA);
}

// --- ОТРИСОВКА ИНТЕРАКТИВА ---
void onTrackbar(int pos, void*) {
    if (history.empty()) return;
    BeamState st = history[pos];

    // 1. Окно фазового пространства
    Mat phase = Mat::zeros(400, 400, CV_8UC3);
    drawPhaseEllipse(phase, st.bx, st.ax, eps_x, Scalar(180,105,255),
                     cv::format("X: Beta=%.2f, Alpha=%.2f", st.bx, st.ax));
    drawPhaseEllipse(phase, st.by, st.ay, eps_y, Scalar(0,255,255),
                     cv::format("Y: Beta=%.2f, Alpha=%.2f", st.by, st.ay), 60);
    imshow("Current Phase Space", phase);

    // 2. Окно 3D среза
    Mat view3d(600, 800, CV_8UC3, Scalar(255,255,255));
    int ring_radius = 250;
    if (st.is_dipole) {
        circle(view3d, Point(400,300), ring_radius, Scalar(220,240,255), -1);
        circle(view3d, Point(400,300), ring_radius, Scalar(200,100,0), 4);
        putText(view3d, "DIPOLE MAGNET", Point(20,60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(150,80,0), 2);
    } else if (st.is_magnet) {
        circle(view3d, Point(400,300), ring_radius, Scalar(200,255,200), -1);
        circle(view3d, Point(400,300), ring_radius, Scalar(0,150,0), 4);
        putText(view3d, "QUADRUPOLE", Point(20,60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,100,0), 2);
    } else {
        circle(view3d, Point(400,300), ring_radius, Scalar(240,240,240), -1);
        circle(view3d, Point(400,300), ring_radius, Scalar(150,150,150), 2);
        putText(view3d, "DRIFT", Point(20,60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100,100,100), 2);
    }

    // Центроид пучка (жёлтая точка)
    Point centroid_pt = project3D(st.centroid_x, 0);
    circle(view3d, centroid_pt, 10, Scalar(0,220,255), -1, LINE_AA);

    for (const auto& p : st.particles) {
        circle(view3d, project3D(p.x, p.y), 2, Scalar(50,50,50), -1, LINE_AA);
    }
    putText(view3d, cv::format("S = %.3f m  |  Centroid X = %.3f mm  |  Angle = %.2f mrad",
            st.s, st.centroid_x*1000, st.centroid_xp*1000),
            Point(20,30), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,0,0), 2);
    imshow("3D Beam Cross-Section", view3d);

    // 3. Вход в коллайдер (Эмиттансы)
    Mat injection = Mat::zeros(400, 800, CV_8UC3);
    Mat roiX = injection(Rect(0,   0, 400, 400));
    Mat roiY = injection(Rect(400, 0, 400, 400));
    drawPhaseEllipse(roiX, 8.0, 0.0, eps_x, Scalar(0,255,0),  "TARGET X", 30);
    drawPhaseEllipse(roiY, 4.0, 0.0, eps_y, Scalar(0,255,0),  "TARGET Y", 30);
    drawPhaseEllipse(roiX, st.bx, st.ax, eps_x, Scalar(180,105,255), "BEAM X", 60);
    drawPhaseEllipse(roiY, st.by, st.ay, eps_y, Scalar(0,255,255),   "BEAM Y", 60);
    line(injection, Point(400,0), Point(400,400), Scalar(255,255,255), 2);
    imshow("Collider Acceptance Matching", injection);

    // 4. Вид сверху — траектория с диполями
    int tw = 1200, th = 350;
    Mat trajPlot = Mat::ones(th, tw, CV_8UC3) * 255;
    double max_s = history.back().s;

    // Труба коллайдера (серая полоса = целевая орбита)
    rectangle(trajPlot, Point(50, 155), Point(tw-50, 195), Scalar(210,210,210), -1);
    line(trajPlot, Point(50, 175), Point(tw-50, 175), Scalar(150,150,150), 1, LINE_AA);
    putText(trajPlot, "COLLIDER ORBIT (x=0)", Point(tw-300, 148),
            FONT_HERSHEY_SIMPLEX, 0.45, Scalar(120,120,120), 1);

    // Траектория центроида (синяя линия)
    for (size_t i = 1; i <= (size_t)pos; ++i) {
        int scale = 600; // пикселей на метр
        Point p1(50 + (int)(history[i-1].s / max_s * (tw-100)),
                 175 + (int)(history[i-1].centroid_x * scale));
        Point p2(50 + (int)(history[i].s   / max_s * (tw-100)),
                 175 + (int)(history[i].centroid_x   * scale));
        line(trajPlot, p1, p2, Scalar(200,50,0), 2, LINE_AA);
    }

    // Отмечаем позиции диполей
    for (const auto& d : dipoles_applied) {
        int sx = 50 + (int)(d.s_pos / max_s * (tw-100));
        // Цвет: септум - синий, кикер - зелёный, корректоры - оранжевый
        Scalar col;
        if (d.name == "SEPTUM")       col = Scalar(200, 50,  0);
        else if (d.name == "KICKER")  col = Scalar(0,  180, 20);
        else                          col = Scalar(0,  130, 220);

        rectangle(trajPlot, Point(sx-5, 130), Point(sx+5, 220), col, -1);
        putText(trajPlot, d.name, Point(sx-20, 120),
                FONT_HERSHEY_SIMPLEX, 0.38, col, 1, LINE_AA);
        putText(trajPlot, cv::format("%.1fmr", d.angle_mrad),
                Point(sx-20, 235), FONT_HERSHEY_SIMPLEX, 0.35, col, 1, LINE_AA);
    }

    // Текущая позиция пучка
    Point curPt(50 + (int)(st.s / max_s * (tw-100)),
                175 + (int)(st.centroid_x * 600));
    circle(trajPlot, curPt, 7, Scalar(0,0,200), -1, LINE_AA);

    putText(trajPlot, "Top-Down Injection Trajectory  (centroid path)",
            Point(20, 28), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0,0,0), 1, LINE_AA);
    putText(trajPlot, cv::format("S=%.2fm  X=%.2fmm  X'=%.2fmrad",
            st.s, st.centroid_x*1000, st.centroid_xp*1000),
            Point(20, 320), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0,0,200), 1, LINE_AA);

    imshow("Trajectory View (Top-Down)", trajPlot);
}

// =====================================================================
int main() {
    namedWindow("Beam Envelope",              WINDOW_NORMAL);
    namedWindow("Current Phase Space",        WINDOW_NORMAL);
    namedWindow("3D Beam Cross-Section",      WINDOW_NORMAL);
    namedWindow("Collider Acceptance Matching", WINDOW_NORMAL);
    namedWindow("Trajectory View (Top-Down)", WINDOW_NORMAL);

    // --- ОПТИМИЗАЦИЯ КВАДРУПОЛЕЙ ---
    Ptr<DownhillSolver> solver = DownhillSolver::create();
    solver->setFunction(makePtr<BeamMatchingObjective>());
    Mat x = (Mat_<double>(1,4) << 0.5, -0.5, 0.5, -0.5);
    Mat step = (Mat_<double>(1,4) << 0.05, 0.05, 0.01, 0.01);
    solver->setInitStep(step);
    solver->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 50000, 1e-10));
    solver->minimize(x);
    double* K = x.ptr<double>();

    // --- ПАРАМЕТРЫ ЛИНИИ ---
    double ds     = 0.005;
    double L_drift = 1.0, L_quad = 0.5;
    double total_L = 4*(L_drift + L_quad) + L_drift; // 7.0 м

    // Начальные Twiss-параметры (на входе в согласующую линию)
    Matx22d Tx(5.0,  0.5,  0.5,  1.25/5.0);
    Matx22d Ty(2.5, -0.3, -0.3,  1.09/2.5);
    auto beam = generateBeam(1000, Tx(0,0), -Tx(0,1), Ty(0,0), -Ty(0,1));

    double s_curr = 0;

    // ============================================================
    // ДИПОЛЬНАЯ ФИЗИКА:
    //   centroid_x  = поперечное смещение центра пучка [м]
    //   centroid_xp = угол траектории центра пучка [рад]
    //
    //   СЕПТУМ (s=0):   начальное смещение 10 см, угол -14 мрад
    //                   (разворачивает пучок в сторону оси коллайдера)
    //   КОРРЕКТОРЫ:     расставлены между квадруполями,
    //                   небольшие поправки ~1-2 мрад
    //   КИКЕР (конец): обнуляет оставшийся угол (рассчитывается автоматически)
    // ============================================================
    double centroid_x  =  0.10;    // начальное смещение: 10 см
    double centroid_xp = -0.014;   // угол септума: -14.0 мрад

    // Регистрируем септум
    dipoles_applied.push_back({0.0, centroid_xp * 1000.0, "SEPTUM"});

    // Позиции корректирующих диполей (между квадруполями)
    // и их углы (небольшие поправки орбиты)
    struct CorrectorDef { double s_pos; double angle_rad; string name; };
    vector<CorrectorDef> correctors = {
        { 1.75, +0.002, "CORR1" },   // после Q1 (+2 мрад)
        { 3.25, +0.0015,"CORR2" },   // после Q2 (+1.5 мрад)
        { 4.75, +0.001, "CORR3" },   // после Q3 (+1 мрад)
    };
    size_t corr_idx = 0;

    // ====== ОСНОВНОЙ TRACKING ======
    auto track = [&](double L, double k, bool is_q = false) {
        for (double t = 0; t < L; t += ds) {

            // Применяем корректор, если добрались до его позиции
            if (corr_idx < correctors.size() &&
                s_curr >= correctors[corr_idx].s_pos) {
                double ca = correctors[corr_idx].angle_rad;
                centroid_xp += ca;
                applyDipoleKick(beam, ca);
                dipoles_applied.push_back({s_curr, ca*1000.0,
                                           correctors[corr_idx].name});
                corr_idx++;
            }

            // КИКЕР — в конце линии (за 0.15 м до финиша)
            // Вычисляем нужный угол, чтобы пучок стал параллелен оси
            if (s_curr >= total_L - 0.15 && s_curr < total_L - 0.14) {
                double kicker_angle = -centroid_xp;  // компенсируем оставшийся угол
                centroid_xp += kicker_angle;
                centroid_x   = 0.0;
                applyDipoleKick(beam, kicker_angle);
                dipoles_applied.push_back({s_curr, kicker_angle*1000.0, "KICKER"});
            }

            // Дрейф центроида (баллистическое движение)
            centroid_x += centroid_xp * ds;

            bool is_dipole_here = false;
            for (auto& d : dipoles_applied)
                if (std::abs(d.s_pos - s_curr) < ds * 2) is_dipole_here = true;

            history.push_back({s_curr,
                                Tx(0,0), -Tx(0,1),
                                Ty(0,0), -Ty(0,1),
                                centroid_x, centroid_xp,
                                (is_q && std::abs(k) > 1e-6),
                                is_dipole_here,
                                beam});

            Matx22d Mx_step = getQuad(ds, k, true);
            Matx22d My_step = getQuad(ds, k, false);
            Tx = transformTwiss(Tx, Mx_step);
            Ty = transformTwiss(Ty, My_step);
            propagateParticles(beam, Mx_step, My_step);

            s_curr += ds;
        }
    };

    track(L_drift, 0);
    for (int i = 0; i < 4; ++i) { track(L_quad, K[i], true); track(L_drift, 0); }

    // ============================================================
    // ВЫВОД НАСТРОЕК МАГНИТОВ В КОНСОЛЬ
    // ============================================================
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════╗\n";
    cout << "║          INJECTION LINE — MAGNET SETTINGS               ║\n";
    cout << "╠══════════════════════════════════════════════════════════╣\n";
    cout << "║  QUADRUPOLES (beam focusing / matching)                 ║\n";
    for (int i = 0; i < 4; ++i) {
        cout << "║    Q" << (i+1) << "  s=" << fixed << setprecision(2) << (L_drift + i*(L_drift+L_quad))
             << " m   K = " << setw(8) << setprecision(4) << K[i]
             << " m⁻²  (" << (K[i]>0?"focussing X":"defocussing X") << ")\n";
    }
    cout << "╠══════════════════════════════════════════════════════════╣\n";
    cout << "║  DIPOLES (orbit correction / injection)                 ║\n";
    for (auto& d : dipoles_applied) {
        cout << "║    " << left << setw(8) << d.name
             << "  s=" << setw(5) << setprecision(2) << fixed << d.s_pos << " m"
             << "   θ = " << setw(8) << setprecision(3) << d.angle_mrad << " mrad";
        if (d.name == "SEPTUM")
            cout << "   ← начальный разворот пучка";
        else if (d.name == "KICKER")
            cout << "   ← финальная коррекция (вход в кольцо)";
        else
            cout << "   ← коррекция орбиты";
        cout << "\n";
    }
    cout << "╠══════════════════════════════════════════════════════════╣\n";
    cout << "║  РЕЗУЛЬТАТ ИНЖЕКЦИИ                                     ║\n";
    cout << "║    Centroid X  (финал): " << setw(8) << setprecision(4)
         << centroid_x*1000 << " mm  (цель: 0 мм)\n";
    cout << "║    Centroid X' (финал): " << setw(8) << setprecision(4)
         << centroid_xp*1000 << " mrad  (цель: 0 мрад)\n";
    double loss_x = pow((history.back().bx - 8.0), 2) + pow(history.back().ax, 2);
    double loss_y = pow((history.back().by - 4.0), 2) + pow(history.back().ay, 2);
    cout << "║    Matching error:       " << setprecision(6)
         << sqrt(loss_x + loss_y) << " (норма отклонения Twiss)\n";
    cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ====== ОТРИСОВКА ОГИБАЮЩЕЙ ======
    int w = 1200, h = 400;
    Mat plot = Mat::ones(h, w, CV_8UC3) * 255;
    double max_b = 0;
    for (auto& st : history) max_b = max({max_b, st.bx, st.by});

    auto toPx = [&](double s, double b) {
        return Point(50 + (int)(s/s_curr*(w-100)),
                     h - 50 - (int)(b/max_b*(h-100)));
    };

    // Фон квадруполей и диполей
    for (size_t i = 1; i < history.size(); ++i) {
        if (history[i].is_magnet) {
            line(plot, toPx(history[i].s, 0),
                 Point(toPx(history[i].s,0).x, 50), Scalar(230,250,230), 2);
        }
        if (history[i].is_dipole) {
            line(plot, toPx(history[i].s, 0),
                 Point(toPx(history[i].s,0).x, 50), Scalar(255,220,180), 3);
        }
    }
    // Огибающие X и Y
    for (size_t i = 1; i < history.size(); ++i) {
        line(plot, toPx(history[i-1].s, history[i-1].bx),
             toPx(history[i].s,   history[i].bx),   Scalar(180,105,255), 2, LINE_AA);
        line(plot, toPx(history[i-1].s, history[i-1].by),
             toPx(history[i].s,   history[i].by),   Scalar(0,255,255),   2, LINE_AA);
    }

    // Легенда диполей
    for (auto& d : dipoles_applied) {
        int sx = 50 + (int)(d.s_pos / s_curr * (w-100));
        Scalar col = (d.name=="SEPTUM") ? Scalar(200,100,0) :
                     (d.name=="KICKER") ? Scalar(0,180,20)  : Scalar(0,130,220);
        line(plot, Point(sx, h-50), Point(sx, 55), col, 2);
        putText(plot, d.name, Point(sx-15, 52),
                FONT_HERSHEY_SIMPLEX, 0.35, col, 1, LINE_AA);
    }

    line(plot, Point(50, h-50), Point(w-50, h-50), Scalar(0,0,0), 2);
    putText(plot, "Beam Envelope:  Pink = Beta_X,  Yellow = Beta_Y  |  Green bg = Quad  |  Orange line = Dipole",
            Point(60, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1, LINE_AA);

    int slider = 0;
    createTrackbar("Position S", "Beam Envelope", &slider,
                   (int)history.size()-1, onTrackbar);
    imshow("Beam Envelope", plot);
    onTrackbar(0, 0);
    waitKey(0);
    return 0;
}