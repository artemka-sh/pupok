#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/optim.hpp>

using namespace std;
using namespace cv;

struct Element {
    int type; // 0 drift, 1 quad
    double L;
    double K;
};

Matx22d getDrift(double L) {
    return Matx22d(
        1.0, L,
        0.0, 1.0
    );
}

Matx22d getQuad(double L, double K, bool is_x) {

    if (abs(K) < 1e-9)
        return getDrift(L);

    double k = is_x ? K : -K;
    double r = sqrt(abs(k));

    if (k > 0) {

        return Matx22d(
            cos(r*L), sin(r*L)/r,
            -r*sin(r*L), cos(r*L)
        );

    } else {

        return Matx22d(
            cosh(r*L), sinh(r*L)/r,
            r*sinh(r*L), cosh(r*L)
        );
    }
}

Matx22d transformTwiss(Matx22d T, Matx22d M) {
    return M * T * M.t();
}

class BeamMatchingObjective : public MinProblemSolver::Function {

public:

    int getDims() const override { return 4; }

    double calc(const double* x) const override {

        double K1 = x[0];
        double K2 = x[1];
        double K3 = x[2];
        double K4 = x[3];

        double Ld = 1.0;
        double Lq = 0.5;

        Matx22d Mx = Matx22d::eye();
        Matx22d My = Matx22d::eye();

        double Kvals[4] = {K1,K2,K3,K4};

        for(int i=0;i<4;i++){

            Mx = getDrift(Ld) * Mx;
            My = getDrift(Ld) * My;

            Mx = getQuad(Lq,Kvals[i],true) * Mx;
            My = getQuad(Lq,Kvals[i],false) * My;
        }

        Mx = getDrift(Ld) * Mx;
        My = getDrift(Ld) * My;

        double bx0=5.0, ax0=-0.5;
        double by0=2.5, ay0=0.3;

        double gx0=(1+ax0*ax0)/bx0;
        double gy0=(1+ay0*ay0)/by0;

        Matx22d Tx(bx0,-ax0,-ax0,gx0);
        Matx22d Ty(by0,-ay0,-ay0,gy0);

        Tx = transformTwiss(Tx,Mx);
        Ty = transformTwiss(Ty,My);

        double bx=Tx(0,0);
        double ax=-Tx(0,1);

        double by=Ty(0,0);
        double ay=-Ty(0,1);

        double loss =
            pow(bx-8.0,2)+
            pow(ax-0.0,2)+
            pow(by-4.0,2)+
            pow(ay-0.0,2);

        return loss;
    }
};

struct TwissPoint{
    double s,bx,ax,by,ay;
};

vector<TwissPoint> propagate(vector<Element> lat){

    vector<TwissPoint> res;

    double bx=5, ax=-0.5;
    double by=2.5, ay=0.3;

    double gx=(1+ax*ax)/bx;
    double gy=(1+ay*ay)/by;

    Matx22d Tx(bx,-ax,-ax,gx);
    Matx22d Ty(by,-ay,-ay,gy);

    double s=0;

    for(auto &e:lat){

        Matx22d Mx,My;

        if(e.type==0){
            Mx=getDrift(e.L);
            My=getDrift(e.L);
        }else{
            Mx=getQuad(e.L,e.K,true);
            My=getQuad(e.L,e.K,false);
        }

        Tx=transformTwiss(Tx,Mx);
        Ty=transformTwiss(Ty,My);

        bx=Tx(0,0);
        ax=-Tx(0,1);

        by=Ty(0,0);
        ay=-Ty(0,1);

        s+=e.L;

        res.push_back({s,bx,ax,by,ay});
    }

    return res;
}

void plotBeta(vector<TwissPoint> data){

    int W=900;
    int H=500;

    Mat img(H,W,CV_8UC3,Scalar(255,255,255));

    double smax=data.back().s;
    double bmax=0;

    for(auto &p:data)
        bmax=max(bmax,max(p.bx,p.by));

    for(int i=1;i<data.size();i++){

        Point p1(
            data[i-1].s/smax*W,
            H-data[i-1].bx/bmax*H
        );

        Point p2(
            data[i].s/smax*W,
            H-data[i].bx/bmax*H
        );

        line(img,p1,p2,Scalar(255,0,0),2);

        p1=Point(
            data[i-1].s/smax*W,
            H-data[i-1].by/bmax*H
        );

        p2=Point(
            data[i].s/smax*W,
            H-data[i].by/bmax*H
        );

        line(img,p1,p2,Scalar(0,0,255),2);
    }

    imshow("beta functions",img);
}

void drawEllipse(Mat &img,Matx22d T){

    double beta=T(0,0);
    double alpha=-T(0,1);

    double eps=1.0;

    for(double t=0;t<2*M_PI;t+=0.01){

        double x=sqrt(eps*beta)*cos(t);
        double xp=-sqrt(eps/beta)*(alpha*cos(t)+sin(t));

        int px=400+x*80;
        int py=300-xp*80;

        circle(img,Point(px,py),1,Scalar(0,0,0),-1);
    }
}

void animate(vector<Element> lat){

    double bx=5, ax=-0.5;
    double gx=(1+ax*ax)/bx;

    Matx22d T(bx,-ax,-ax,gx);

    for(auto &e:lat){

        Matx22d M;

        if(e.type==0)
            M=getDrift(e.L);
        else
            M=getQuad(e.L,e.K,true);

        T=transformTwiss(T,M);

        Mat img(600,800,CV_8UC3,Scalar(255,255,255));

        drawEllipse(img,T);

        imshow("phase ellipse",img);
        waitKey(120);
    }
}

Point project3D(double x,double y,double z)
{
    double cam=8.0;

    double scale=cam/(cam+z);

    int px = 400 + x*120*scale;
    int py = 300 - y*120*scale;

    return Point(px,py);
}

void drawEllipse3D(Mat& img,double beta,double alpha,double z)
{
    double eps=1.0;

    vector<Point> pts;

    for(double t=0;t<2*M_PI;t+=0.05)
    {
        double x = sqrt(eps*beta)*cos(t);

        double xp =
            -sqrt(eps/beta)*
            (alpha*cos(t)+sin(t));

        pts.push_back(
            project3D(x,xp,z)
        );
    }

    for(int i=1;i<pts.size();i++)
        line(img,pts[i-1],pts[i],Scalar(0,0,0),1);
}

void drawQuad(Mat& img,double z)
{
    int r=20;

    circle(img,
           project3D(0,0,z),
           r,
           Scalar(0,150,0),
           2);
}

void animate3D(vector<Element> lat)
{
    double bx=5, ax=-0.5;
    double gx=(1+ax*ax)/bx;

    Matx22d T(bx,-ax,-ax,gx);

    double z=0;

    for(auto &e:lat)
    {
        Mat img(600,800,CV_8UC3,Scalar(255,255,255));

        Matx22d M;

        if(e.type==0)
            M=getDrift(e.L);
        else
            M=getQuad(e.L,e.K,true);

        T=transformTwiss(T,M);

        double beta=T(0,0);
        double alpha=-T(0,1);

        drawEllipse3D(img,beta,alpha,z);

        if(e.type==1)
            drawQuad(img,z);

        z+=e.L;

        imshow("3D Beam",img);

        waitKey(120*5);
    }
}

struct Particle
{
    double x, xp;
    double y, yp;
};

vector<Particle> generateBeam3D(int N,double betaX,double alphaX,double betaY,double alphaY)
{
    vector<Particle> beam;
    double eps = 1.0;   // эмиттанс
    RNG rng;

    for(int i=0;i<N;i++)
    {
        // Сначала генерируем точку внутри единичного круга
        double u = rng.uniform(0.0,1.0);
        double v = rng.uniform(0.0,1.0);

        double r = sqrt(u);       // радиус для равномерного распределения по площади
        double theta = 2*M_PI*v;  // угол

        // X–X' эллипс
        double x  = sqrt(eps*betaX) * r * cos(theta);
        double xp = -sqrt(eps/betaX) * (alphaX*r*cos(theta) + r*sin(theta));

        // Y–Y' эллипс
        double y  = sqrt(eps*betaY) * r * sin(theta);  // используем тот же r и theta
        double yp = -sqrt(eps/betaY) * (alphaY*r*sin(theta) + r*cos(theta));

        beam.push_back({x,xp,y,yp});
    }

    return beam;
}

void propagateParticles3D(vector<Particle>& beam, Matx22d Mx, Matx22d My)
{
    for(auto &p:beam)
    {
        double x_new  = Mx(0,0)*p.x + Mx(0,1)*p.xp;
        double xp_new = Mx(1,0)*p.x + Mx(1,1)*p.xp;
        p.x  = x_new;
        p.xp = xp_new;

        double y_new  = My(0,0)*p.y + My(0,1)*p.yp;
        double yp_new = My(1,0)*p.y + My(1,1)*p.yp;
        p.y  = y_new;
        p.yp = yp_new;
    }
}

void drawBeam3D(Mat& img, vector<Particle>& beam, double z, Matx22d Tx, Matx22d Ty)
{
    // частицы
//    for(auto &p:beam)
//    {
//        Point pt = project3D(p.x, p.y, z);
//        circle(img, pt, 1, Scalar(0,0,0), -1);
//    }

    // элипс X–X' (черный)
    double eps=1.0;
    for(double t=0;t<2*M_PI;t+=0.05)
    {
        double x  = sqrt(eps*Tx(0,0))*cos(t);
        double xp = -sqrt(eps/Tx(0,0))*( -Tx(0,1)*cos(t) + sin(t) );
        Point pt = project3D(x, xp, z);
        circle(img, pt, 1, Scalar(0,0,0), -1);
    }

    // элипс Y–Y' (красный)
    for(double t=0;t<2*M_PI;t+=0.05)
    {
        double y  = sqrt(eps*Ty(0,0))*cos(t);
        double yp = -sqrt(eps/Ty(0,0))*( -Ty(0,1)*cos(t) + sin(t) );
        Point pt = project3D(yp, y, z); // меняем местами для визуального различия
        circle(img, pt, 1, Scalar(0,0,255), -1);
    }
}

void animateBeam3D(vector<Element> lat)
{
    double bx0=5, ax0=-0.5;
    double by0=2.5, ay0=0.3;

    Matx22d Tx(bx0, -ax0, -ax0, (1+ax0*ax0)/bx0);
    Matx22d Ty(by0, -ay0, -ay0, (1+ay0*ay0)/by0);

    auto beam = generateBeam3D(800, bx0, ax0, by0, ay0);

    double z=0;
    const double step = 0.05;

    for(auto &e:lat)
    {
        double L = e.L;
        double s = 0;

        while(s < L)
        {
            double dl = min(step, L - s);

            Matx22d Mx = (e.type==0)? getDrift(dl) : getQuad(dl,e.K,true);
            Matx22d My = (e.type==0)? getDrift(dl) : getQuad(dl,e.K,false);

            Tx = transformTwiss(Tx, Mx);
            Ty = transformTwiss(Ty, My);

            propagateParticles3D(beam, Mx, My);

            Mat img(600,800,CV_8UC3,Scalar(255,255,255));
            drawBeam3D(img, beam, z+s, Tx, Ty);

            if(e.type==1)
            {
                int r=20;
                Point c = project3D(0,0,z+s);
                circle(img,c,r,Scalar(0,150,0),2);
            }

            imshow("Beam 3D", img);
            waitKey(20);

            s += dl;
        }

        z += L;
    }
}

int main(){

    Ptr<DownhillSolver> solver=DownhillSolver::create();
    Ptr<MinProblemSolver::Function> f=makePtr<BeamMatchingObjective>();

    solver->setFunction(f);

    Mat x=(Mat_<double>(1,4)<<1,-1,1,-1);
    Mat step=(Mat_<double>(1,4)<<0.1,0.1,0.1,0.1);

    solver->setInitStep(step);

    solver->setTermCriteria(
        TermCriteria(
            TermCriteria::MAX_ITER+TermCriteria::EPS,
            10000,
            1e-6
        )
    );

    double res=solver->minimize(x);

    double K1=x.at<double>(0);
    double K2=x.at<double>(1);
    double K3=x.at<double>(2);
    double K4=x.at<double>(3);

    cout<<"Loss: "<<res<<endl;

    cout<<"K1 "<<K1<<endl;
    cout<<"K2 "<<K2<<endl;
    cout<<"K3 "<<K3<<endl;
    cout<<"K4 "<<K4<<endl;

    vector<Element> lat={
        {0,1,0},
        {1,0.5,K1},
        {0,1,0},
        {1,0.5,K2},
        {0,1,0},
        {1,0.5,K3},
        {0,1,0},
        {1,0.5,K4},
        {0,1,0}
    };

    auto data=propagate(lat);

    plotBeta(data);
    animateBeam3D(lat);
    waitKey();
}