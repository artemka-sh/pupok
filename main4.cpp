#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct Particle {
    double x, xp, y, yp, dp;
};

struct BeamState {
    double s;
    double bx, ax, by, ay;
    double dx, dpx;
    int elem_type; // 0=Drift, 1=Quad, 2=Dipole
    vector<Particle> particles;
};

vector<BeamState> history;
double eps_x = 10e-9;
double eps_y = 2e-9;

Matx33d getDrift(double L) {
    return Matx33d(1.0, L, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

Matx33d getDipole(double L, double angle, bool is_x) {
    if (!is_x || std::abs(angle) < 1e-9) return getDrift(L);
    double rho = L / angle;
    return Matx33d(cos(angle), rho * sin(angle), rho * (1.0 - cos(angle)),
                   -(1.0/rho) * sin(angle), cos(angle), sin(angle),
                   0.0, 0.0, 1.0);
}

Matx33d getQuad(double L, double K, bool is_x) {
    if (std::abs(K) < 1e-9) return getDrift(L);
    double k_val = is_x ? K : -K;
    double r = std::sqrt(std::abs(k_val));
    if (k_val > 0) {
        return Matx33d(cos(r * L), (1.0 / r) * sin(r * L), 0.0,
                       -r * sin(r * L), cos(r * L), 0.0, 0.0, 0.0, 1.0);
    } else {
        return Matx33d(cosh(r * L), (1.0 / r) * sinh(r * L), 0.0,
                       r * sinh(r * L), cosh(r * L), 0.0, 0.0, 0.0, 1.0);
    }
}

Matx22d transformTwiss(const Matx22d& T_in, const Matx33d& M3) {
    Matx22d M(M3(0,0), M3(0,1), M3(1,0), M3(1,1));
    return M * T_in * M.t();
}

vector<Particle> generateBeam(int N, double betaX, double alphaX, double betaY, double alphaY) {
    vector<Particle> beam;
    RNG rng(12345);
    for (int i = 0; i < N; i++) {
        double r = sqrt(rng.uniform(0.0, 1.0));
        double theta = 2 * M_PI * rng.uniform(0.0, 1.0);
        double x = sqrt(eps_x * betaX) * r * cos(theta);
        double xp = -sqrt(eps_x / betaX) * (alphaX * r * cos(theta) + r * sin(theta));

        double r_y = sqrt(rng.uniform(0.0, 1.0));
        double theta_y = 2 * M_PI * rng.uniform(0.0, 1.0);
        double y = sqrt(eps_y * betaY) * r_y * sin(theta_y);
        double yp = -sqrt(eps_y / betaY) * (alphaY * r_y * sin(theta_y) + r_y * cos(theta_y));

        double dp = rng.gaussian(1e-3);
        beam.push_back({x, xp, y, yp, dp});
    }
    return beam;
}

void propagateParticles(vector<Particle>& beam, Matx33d Mx, Matx33d My) {
    for (auto& p : beam) {
        double x_new = Mx(0, 0) * p.x + Mx(0, 1) * p.xp + Mx(0, 2) * p.dp;
        double xp_new = Mx(1, 0) * p.x + Mx(1, 1) * p.xp + Mx(1, 2) * p.dp;
        p.x = x_new; p.xp = xp_new;

        double y_new = My(0, 0) * p.y + My(0, 1) * p.yp + My(0, 2) * p.dp;
        double yp_new = My(1, 0) * p.y + My(1, 1) * p.yp + My(1, 2) * p.dp;
        p.y = y_new; p.yp = yp_new;
    }
}

Point project3D(double x, double y) {
    return Point(400 + x * 500000, 300 - y * 500000);
}

class BeamMatchingObjective : public MinProblemSolver::Function {
public:
    int getDims() const override { return 8; } // 6 Квадруполей + 2 Дрейфа
    double calc(const double* x) const override {
        double K[6] = {x[0], x[1], x[2], x[3], x[4], x[5]};
        // Динамические дрейфы (от 0.1 до ~2 метров)
        double D1 = 0.1 + std::abs(x[6]);
        double D2 = 0.1 + std::abs(x[7]);

        Matx33d Mx = Matx33d::eye(), My = Matx33d::eye();
        Vec3d Dx_vec(0.0, 0.0, 1.0);

        Matx22d Tx_curr(5.0, 0.5, 0.5, 1.25/5.0);
        Matx22d Ty_curr(2.5, -0.3, -0.3, 1.09/2.5);
        double max_b = 0;

        auto apply_step = [&](Matx33d Mx_s, Matx33d My_s) {
            Mx = Mx_s * Mx; My = My_s * My;
            Dx_vec = Mx_s * Dx_vec;
            Tx_curr = transformTwiss(Tx_curr, Mx_s);
            Ty_curr = transformTwiss(Ty_curr, My_s);
            max_b = std::max({max_b, Tx_curr(0,0), Ty_curr(0,0)});
        };

        auto step_drift = [&](double L) { apply_step(getDrift(L), getDrift(L)); };
        auto step_quad = [&](double L, double k) { apply_step(getQuad(L, k, true), getQuad(L, k, false)); };
        auto step_dipole = [&](double L, double angle) { apply_step(getDipole(L, angle, true), getDipole(L, angle, false)); };

        double L_q = 0.5, L_dip = 1.0, ang = 0.05;

        // Архитектура Double-Bend Achromat
        step_drift(D1);  step_quad(L_q, K[0]); step_drift(D2);   step_quad(L_q, K[1]);
        step_drift(0.5); step_dipole(L_dip, ang); step_drift(0.5);
        step_quad(L_q, K[2]); step_drift(D1);  step_quad(L_q, K[3]);
        step_drift(0.5); step_dipole(L_dip, ang); step_drift(0.5);
        step_quad(L_q, K[4]); step_drift(D2);  step_quad(L_q, K[5]); step_drift(D1);

        double error = 0.0;
        // 1. Целевые параметры (BetaX=8, BetaY=4, Alpha=0)
        error += pow(Tx_curr(0, 0) - 8.0, 2) + pow(Tx_curr(0, 1) - 0.0, 2);
        error += pow(Ty_curr(0, 0) - 4.0, 2) + pow(Ty_curr(0, 1) - 0.0, 2);

        // 2. Жесткое обнуление дисперсии
        error += pow(Dx_vec(0), 2) * 50000.0;
        error += pow(Dx_vec(1), 2) * 50000.0;

        // 3. Ограничения на магниты (3 Тесла ~ K <= 10)
        for(int i=0; i<6; ++i) {
            if (std::abs(K[i]) > 10.0) error += pow(std::abs(K[i]) - 10.0, 2) * 10000.0;
        }

        // 4. ШТРАФ ЗА РАЗДУВАНИЕ: Не даем бета-функции превысить 40 м внутри канала
        if (max_b > 40.0) {
            error += pow(max_b - 40.0, 2) * 1000.0;
        }

        return error;
    }
};

void drawPhaseEllipse(Mat& img, double beta, double alpha, double epsilon, Scalar color, string label, int offset) {
    int cx = img.cols / 2, cy = img.rows / 2;
    double scale_pos = 150.0 / 3e-4;
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
    putText(img, label, Point(10, offset), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
}

void onTrackbar(int pos, void*) {
    if (history.empty()) return;
    BeamState st = history[pos];

    Mat phase = Mat::zeros(400, 400, CV_8UC3);
    drawPhaseEllipse(phase, st.bx, st.ax, eps_x, Scalar(0, 0, 255), "X: " + to_string(st.bx).substr(0,5), 30);
    drawPhaseEllipse(phase, st.by, st.ay, eps_y, Scalar(255, 0, 0), "Y: " + to_string(st.by).substr(0,5), 60);
    drawPhaseEllipse(phase, std::abs(st.dx)+1e-9, st.dpx, 1e-9, Scalar(0, 255, 0), "D: " + to_string(st.dx).substr(0,5), 90);
    imshow("Phase Space", phase);

    Mat view3d(600, 800, CV_8UC3, Scalar(255, 255, 255));
    if (st.elem_type == 1) putText(view3d, "QUADRUPOLE", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 100, 0), 2);
    else if (st.elem_type == 2) putText(view3d, "DIPOLE", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 0, 0), 2);
    else putText(view3d, "DRIFT", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 100, 100), 2);

    for (const auto& p : st.particles) circle(view3d, project3D(p.x, p.y), 2, Scalar(50, 50, 50), -1, LINE_AA);
    putText(view3d, "S = " + to_string(st.s).substr(0,5) + " m", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    imshow("3D Beam", view3d);
}

int main() {
    Ptr<DownhillSolver> solver = DownhillSolver::create();
    solver->setFunction(makePtr<BeamMatchingObjective>());

    // Начальное приближение
    Mat x = (Mat_<double>(1, 8) << 1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 0.5, 0.5);
    Mat step = (Mat_<double>(1, 8) << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1);
    solver->setInitStep(step);
    solver->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 150000, 1e-12));

    cout << "Optimizing DBA lattice... This might take a few seconds." << endl;
    solver->minimize(x);
    double* K_opt = x.ptr<double>();

    double K[6] = {K_opt[0], K_opt[1], K_opt[2], K_opt[3], K_opt[4], K_opt[5]};
    double D1 = 0.1 + std::abs(K_opt[6]);
    double D2 = 0.1 + std::abs(K_opt[7]);

    double ds = 0.005, L_quad = 0.5, L_dip = 1.0, ang = 0.05, s_curr = 0;
    Matx22d Tx(5.0, 0.5, 0.5, 1.25/5.0), Ty(2.5, -0.3, -0.3, 1.09/2.5);
    auto beam = generateBeam(800, Tx(0,0), -Tx(0,1), Ty(0,0), -Ty(0,1));
    Vec3d D_vec(0.0, 0.0, 1.0);

    auto track = [&](double L, int type, double param) {
        for (double t = 0; t < L; t += ds) {
            history.push_back({s_curr, Tx(0,0), -Tx(0,1), Ty(0,0), -Ty(0,1), D_vec(0), D_vec(1), type, beam});
            Matx33d Mx_s = (type==0) ? getDrift(ds) : (type==1) ? getQuad(ds, param, true) : getDipole(ds, param*(ds/L), true);
            Matx33d My_s = (type==0) ? getDrift(ds) : (type==1) ? getQuad(ds, param, false) : getDipole(ds, param*(ds/L), false);
            Tx = transformTwiss(Tx, Mx_s); Ty = transformTwiss(Ty, My_s);
            D_vec = Mx_s * D_vec;
            propagateParticles(beam, Mx_s, My_s);
            s_curr += ds;
        }
    };

    // Точная копия структуры из Objective Function
    track(D1, 0, 0);  track(L_quad, 1, K[0]); track(D2, 0, 0);   track(L_quad, 1, K[1]);
    track(0.5, 0, 0); track(L_dip, 2, ang);   track(0.5, 0, 0);
    track(L_quad, 1, K[2]); track(D1, 0, 0);  track(L_quad, 1, K[3]);
    track(0.5, 0, 0); track(L_dip, 2, ang);   track(0.5, 0, 0);
    track(L_quad, 1, K[4]); track(D2, 0, 0);  track(L_quad, 1, K[5]); track(D1, 0, 0);

    int w = 1400, h = 400;
    Mat plot = Mat::ones(h, w, CV_8UC3) * 255;
    double max_b = 0;
    for(auto& st : history) max_b = max({max_b, st.bx, st.by});

    auto toPx = [&](double s, double b) { return Point(50 + (s/s_curr)*(w-100), h - 50 - (b/max_b)*(h-100)); };

    for(size_t i=1; i<history.size(); ++i) {
        if(history[i].elem_type == 1) line(plot, toPx(history[i].s, 0), Point(toPx(history[i].s, 0).x, 50), Scalar(230,245,230), 2);
        if(history[i].elem_type == 2) line(plot, toPx(history[i].s, 0), Point(toPx(history[i].s, 0).x, 50), Scalar(255,230,210), 2);
        line(plot, toPx(history[i-1].s, history[i-1].bx), toPx(history[i].s, history[i].bx), Scalar(0,0,255), 2, LINE_AA);
        line(plot, toPx(history[i-1].s, history[i-1].by), toPx(history[i].s, history[i].by), Scalar(0,255,255), 2, LINE_AA);
        line(plot, toPx(history[i-1].s, history[i-1].dx * 100), toPx(history[i].s, history[i].dx * 100), Scalar(0,200,0), 2, LINE_AA);
    }
    line(plot, Point(50, h-50), Point(w-50, h-50), Scalar(0,0,0), 2);

    namedWindow("Beam Envelope", WINDOW_NORMAL);
    namedWindow("Phase Space", WINDOW_NORMAL);
    namedWindow("3D Beam", WINDOW_NORMAL);

    int slider = 0;
    createTrackbar("S", "Beam Envelope", &slider, history.size()-1, onTrackbar);
    imshow("Beam Envelope", plot);
    onTrackbar(0, 0);

    cout << "\nOptimization Complete." << endl;
    cout << "Drift 1: " << D1 << " m | Drift 2: " << D2 << " m" << endl;
    cout << "Final Dx: " << history.back().dx << " m" << endl;

    waitKey(0);
    return 0;
}