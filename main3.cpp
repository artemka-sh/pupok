#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct Particle {
    double x, xp;
    double y, yp;
    double dp; // Отклонение по импульсу \Delta p / p
};

struct BeamState {
    double s;
    double bx, ax, by, ay;
    double dx, dpx; // Дисперсия и её производная
    int elem_type;
    vector<Particle> particles;
};

vector<BeamState> history;
double eps_x = 10e-9;
double eps_y = 2e-9;

// Теперь матрицы 3x3 для учета дисперсии (x, x', \delta)
Matx33d getDrift(double L) {
    return Matx33d(1.0, L, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0);
}

void applyKick(vector<Particle>& beam, double theta)
{
    for(auto& p : beam)
        p.xp += theta;
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
                       -r * sin(r * L), cos(r * L), 0.0,
                       0.0, 0.0, 1.0);
    } else {
        return Matx33d(cosh(r * L), (1.0 / r) * sinh(r * L), 0.0,
                       r * sinh(r * L), cosh(r * L), 0.0,
                       0.0, 0.0, 1.0);
    }
}

// Преобразование Твисса работает с подматрицей 2x2
Matx22d transformTwiss(const Matx22d& T_in, const Matx33d& M3) {
    Matx22d M(M3(0,0), M3(0,1), M3(1,0), M3(1,1));
    return M * T_in * M.t();
}

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

        // Разброс по импульсу (например, сигма = 0.1%)
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
    int px = 400 + x * 500000;
    int py = 300 - y * 500000;
    return Point(px, py);
}

class BeamMatchingObjective : public MinProblemSolver::Function {
public:
    int getDims() const override { return 4; }
    double calc(const double* x) const override {
        double L_drift = 1.0, L_quad = 0.5, L_dipole = 1.0, dip_angle = 0.1;
        Matx33d Mx = Matx33d::eye(), My = Matx33d::eye();

        for (int i = 0; i < 4; ++i) {
            Mx = getQuad(L_quad, x[i], true) * getDrift(L_drift) * Mx;
            My = getQuad(L_quad, x[i], false) * getDrift(L_drift) * My;
        }
        Mx = getDrift(L_drift) * Mx;
        My = getDrift(L_drift) * My;
        Mx = getDipole(L_dipole, dip_angle, true) * Mx;
        My = getDipole(L_dipole, dip_angle, false) * My;

        Matx22d Tx(5.0, 0.5, 0.5, (1 + 0.5 * 0.5) / 5.0);
        Matx22d Ty(2.5, -0.3, -0.3, (1 + 0.3 * 0.3) / 2.5);

        Matx22d Tx_out = transformTwiss(Tx, Mx);
        Matx22d Ty_out = transformTwiss(Ty, My);

        return pow((Tx_out(0, 0) - 8.0), 2) + pow(-Tx_out(0, 1) - 0.0, 2) +
               pow((Ty_out(0, 0) - 4.0), 2) + pow(-Ty_out(0, 1) - 0.0, 2);
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
    drawPhaseEllipse(phase, st.bx, st.ax, eps_x, Scalar(0, 0, 255), "X: " + to_string(st.bx) + "; X': " + to_string(st.ax), 30);
    drawPhaseEllipse(phase, st.by, st.ay, eps_y, Scalar(255, 0, 0), "Y: " + to_string(st.by) + "; Y': " + to_string(st.ay), 60);
    drawPhaseEllipse(phase, st.dx, st.dpx, 0, Scalar(0, 255, 0), "D: " + to_string(st.dx) + "; D': " + to_string(st.dpx), 90);
    imshow("Phase Space (X-x', Y-y')", phase);

    Mat view3d(600, 800, CV_8UC3, Scalar(255, 255, 255));
    int ring_radius = 250;

    if (st.elem_type == 1) {
        circle(view3d, Point(400, 300), ring_radius, Scalar(200, 255, 200), -1);
        circle(view3d, Point(400, 300), ring_radius, Scalar(0, 150, 0), 4);
        putText(view3d, "QUADRUPOLE MAGNET", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 100, 0), 2);
    } else if (st.elem_type == 2) {
        circle(view3d, Point(400, 300), ring_radius, Scalar(255, 220, 200), -1);
        circle(view3d, Point(400, 300), ring_radius, Scalar(255, 0, 0), 4);
        putText(view3d, "INJECTION DIPOLE", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 0, 0), 2);
    } else {
        circle(view3d, Point(400, 300), ring_radius, Scalar(240, 240, 240), -1);
        circle(view3d, Point(400, 300), ring_radius, Scalar(150, 150, 150), 2);
        putText(view3d, "DRIFT SPACE", Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(100, 100, 100), 2);
    }

    for (const auto& p : st.particles) {
        circle(view3d, project3D(p.x, p.y), 2, Scalar(50, 50, 50), -1, LINE_AA);
    }

    putText(view3d, "Position S = " + to_string(st.s).substr(0,4) + " m", Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,0), 2);
    imshow("3D Beam Cross-Section", view3d);
}

int main() {
    namedWindow("Beam Envelope", WINDOW_NORMAL);
    namedWindow("Phase Space (X-x', Y-y')", WINDOW_NORMAL);
    namedWindow("3D Beam Cross-Section", WINDOW_NORMAL);

    Ptr<DownhillSolver> solver = DownhillSolver::create();
    solver->setFunction(makePtr<BeamMatchingObjective>());
    Mat x = (Mat_<double>(1, 4) << 0.5, -0.5, 0.5, -0.5);
    Mat step = (Mat_<double>(1, 4) << 0.05, 0.05, 0.05, 0.05);
    solver->setInitStep(step);
    solver->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50000, 1e-16));

    solver->minimize(x);
    double* K = x.ptr<double>();

    double ds = 0.005, L_drift = 1.0, L_quad = 0.5, L_dipole = 1.0, dip_angle = 0.1;
    double s_curr = 0;

    Matx22d Tx(5.0, 0.5, 0.5, 1.25/5.0);
    Matx22d Ty(2.5, -0.3, -0.3, 1.09/2.5);
    auto beam = generateBeam(800, Tx(0,0), -Tx(0,1), Ty(0,0), -Ty(0,1));

    Vec3d D_vec(0.0, 0.0, 1.0);

    auto track_drift = [&](double L) {
        for (double t = 0; t < L; t += ds) {
            history.push_back({s_curr, Tx(0, 0), -Tx(0, 1), Ty(0, 0), -Ty(0, 1), D_vec(0), D_vec(1), 0, beam});
            Matx33d M_step = getDrift(ds);
            Tx = transformTwiss(Tx, M_step);
            Ty = transformTwiss(Ty, M_step);
            D_vec = M_step * D_vec;
            propagateParticles(beam, M_step, M_step);
            s_curr += ds;
        }
    };

    auto track_quad = [&](double L, double k) {
        for (double t = 0; t < L; t += ds) {
            history.push_back({s_curr, Tx(0, 0), -Tx(0, 1), Ty(0, 0), -Ty(0, 1), D_vec(0), D_vec(1), 1, beam});
            Matx33d Mx_step = getQuad(ds, k, true);
            Matx33d My_step = getQuad(ds, k, false);
            Tx = transformTwiss(Tx, Mx_step);
            Ty = transformTwiss(Ty, My_step);
            D_vec = Mx_step * D_vec;
            propagateParticles(beam, Mx_step, My_step);
            s_curr += ds;
        }
    };

    auto track_dipole = [&](double L, double angle) {
        for (double t = 0; t < L; t += ds) {
            history.push_back({s_curr, Tx(0, 0), -Tx(0, 1), Ty(0, 0), -Ty(0, 1), D_vec(0), D_vec(1), 2, beam});
            double step_angle = angle * (ds / L);
            Matx33d Mx_step = getDipole(ds, step_angle, true);
            Matx33d My_step = getDipole(ds, step_angle, false);
            Tx = transformTwiss(Tx, Mx_step);
            Ty = transformTwiss(Ty, My_step);
            D_vec = Mx_step * D_vec;
            propagateParticles(beam, Mx_step, My_step);
            s_curr += ds;
        }
    };

    for(int i=0; i<4; ++i) {
        track_drift(L_drift);
        track_quad(L_quad, K[i]);
    }
    track_drift(L_drift);
    track_dipole(L_dipole, dip_angle);
    track_drift(L_drift);

    int w = 1200, h = 400;
    Mat plot = Mat::ones(h, w, CV_8UC3) * 255;
    double max_b = 0;
    for(auto& st : history) max_b = max({max_b, st.bx, st.by});

    auto toPx = [&](double s, double b) {
        return Point(50 + (s/s_curr)*(w-100), h - 50 - (b/max_b)*(h-100));
    };

    for(size_t i=1; i<history.size(); ++i) {
        if(history[i].elem_type == 1) {
            line(plot, toPx(history[i].s, 0), Point(toPx(history[i].s, 0).x, 50), Scalar(230,245,230), 2);
        } else if (history[i].elem_type == 2) {
            line(plot, toPx(history[i].s, 0), Point(toPx(history[i].s, 0).x, 50), Scalar(255,230,210), 2);
        }

        line(plot, toPx(history[i-1].s, history[i-1].bx), toPx(history[i].s, history[i].bx), Scalar(0,0,255), 2, LINE_AA);
        line(plot, toPx(history[i-1].s, history[i-1].by), toPx(history[i].s, history[i].by), Scalar(0,255,255), 2, LINE_AA);

        line(plot, toPx(history[i-1].s, history[i-1].dx * 100), toPx(history[i].s, history[i].dx * 100), Scalar(0,200,0), 2, LINE_AA);
    }

    line(plot, Point(50, h-50), Point(w-50, h-50), Scalar(0,0,0), 2);
    putText(plot, "Envelope: Red(X), Yel(Y). Green Line = Dispersion Dx * 100", Point(60, 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50,50,50), 1);

    int slider = 0;
    createTrackbar("Position S", "Beam Envelope", &slider, history.size()-1, onTrackbar);
    imshow("Beam Envelope", plot);

    onTrackbar(0, 0);

    cout << "Optimization Done." << endl;
    cout << "K1=" << K[0] << " K2=" << K[1] << " K3=" << K[2] << " K4=" << K[3] << endl;
    cout << "Final Dispersion Dx = " << history.back().dx << " m" << endl;

    waitKey(0);
    return 0;
}