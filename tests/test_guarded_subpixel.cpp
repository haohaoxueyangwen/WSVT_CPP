#include <catch2/catch_test_macros.hpp>
#include "wsvt/guarded_subpixel.hpp"
#include "wsvt/wsvt_pipeline.hpp"
#include <random>

using namespace wsvt;
namespace {
std::array<float,9> quadratic(double dx,double dy,double a=4,double b=.5,double d=3,double base=1) {
    std::array<float,9> c{};
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        const double u=x-dx,v=y-dy;
        c[(y+1)*3+x+1]=static_cast<float>(base+(a*u*u+2*b*u*v+d*v*v)/2);
    }
    return c;
}
}
TEST_CASE("guarded subpixel recovers a positive definite quadratic", "[sp1]") {
    for(double dx:{-.3,0.,.3}) for(double dy:{-.2,0.,.2}) {
        const auto c=quadratic(dx,dy);
        auto f=guarded_subpixel(c,true,21);
        REQUIRE(f.reason==SubpixelReason::Accepted);
        REQUIRE(std::abs(f.dx-dx)<1e-6);
        REQUIRE(std::abs(f.dy-dy)<1e-6);
        auto scaled=c;for(auto& v:scaled) v*=8;
        auto g=guarded_subpixel(scaled,true,21);
        REQUIRE(g.reason==f.reason);
        REQUIRE(g.dx==f.dx);REQUIRE(g.dy==f.dy);
    }
}
TEST_CASE("guarded subpixel rejects invalid local minima", "[sp1]") {
    REQUIRE(guarded_subpixel(quadratic(0,0),false,21).reason==SubpixelReason::Boundary);
    REQUIRE(guarded_subpixel(quadratic(0,0,4,0,-3),true,21).reason==SubpixelReason::NotPositive);
    REQUIRE(guarded_subpixel(quadratic(0,0,0,0,0),true,21).reason==SubpixelReason::NotPositive);
    REQUIRE(guarded_subpixel(quadratic(.8,0),true,21).reason==SubpixelReason::OutsideCell);
    auto c=quadratic(0,0,4,0,.00001,100);
    REQUIRE(guarded_subpixel(c,true,21).reason==SubpixelReason::UnresolvedCurvature);
    c[1]=std::numeric_limits<float>::infinity();
    REQUIRE(guarded_subpixel(c,true,21).reason==SubpixelReason::Nonfinite);
    c[1]=std::numeric_limits<float>::quiet_NaN();
    REQUIRE(guarded_subpixel(c,true,21).reason==SubpixelReason::Nonfinite);
}
TEST_CASE("guarded subpixel preserves coarse search and legacy output", "[sp1]") {
    std::mt19937 rng(4321);
    std::uniform_real_distribution<float> uniform(.1f,2.f);
    std::vector<float> a(16*64*72);for(auto& v:a)v=uniform(rng);
    auto make=[&]() { return WSVT(a,a,16,64,72,0,16,0,4,1,1,
        14000,.65e-6,1,.425,1,1,1,false,true,0,false,1); };
    auto base=make(),off=make(),guard=make();
    off.configure_guarded_subpixel(false);guard.configure_guarded_subpixel(true);
    base.configure_search_topk_diagnostics({100},9,1);
    guard.configure_search_topk_diagnostics({100},9,1);
    const auto b=base.solver(),o=off.solver(),g=guard.solver();
    REQUIRE(b.displace_x==o.displace_x);REQUIRE(b.displace_y==o.displace_y);
    REQUIRE(b.search_candidate_count==g.search_candidate_count);
    REQUIRE(b.search_distance_terms_evaluated==g.search_distance_terms_evaluated);
    const auto bd=base.search_topk_diagnostics(),gd=guard.search_topk_diagnostics();
    REQUIRE(bd.size()==gd.size());
    for(std::size_t i=0;i<bd.size();++i) {
        REQUIRE(bd[i].descriptor_cost_ssd==gd[i].descriptor_cost_ssd);
        REQUIRE(bd[i].internal_candidate_x==gd[i].internal_candidate_x);
        REQUIRE(bd[i].internal_candidate_y==gd[i].internal_candidate_y);
    }
    REQUIRE(guard.guarded_subpixel_reasons().size()==64*72);
    REQUIRE(off.guarded_subpixel_reasons().empty());
    auto excluded=make();excluded.configure_guarded_subpixel(true);
    excluded.configure_fixed_set_transport();
    REQUIRE_THROWS(excluded.solver());
}

TEST_CASE("box subpixel solves a coupled edge not componentwise clipping", "[sp2]") {
    const auto c=quadratic(.8,.1);
    const auto r=guarded_subpixel(c,true,21,true);
    REQUIRE(r.reason==SubpixelReason::ConstrainedBoundary);
    REQUIRE(std::abs(r.dx-.5)<1e-12);
    REQUIRE(std::abs(r.dy-.15)<1e-6);
    REQUIRE(guarded_subpixel(c,true,21).reason==SubpixelReason::OutsideCell);
    REQUIRE(guarded_subpixel(c,false,21,true).reason==SubpixelReason::Boundary);
}

TEST_CASE("box subpixel satisfies KKT and beats dense grid feasible points", "[sp2]") {
    std::mt19937 rng(2026090910);
    std::uniform_real_distribution<double> random(-2,2);
    for(int trial=0;trial<200;++trial) {
        const double p=random(rng),q=random(rng),r=random(rng),s=random(rng);
        auto c=quadratic(random(rng),random(rng),
                         p*p+r*r+1,p*q+r*s,q*q+s*s+1);
        const double gx=(double(c[5])-c[3])/2,gy=(double(c[7])-c[1])/2;
        const double a=double(c[5])+c[3]-2.0*c[4];
        const double d=double(c[7])+c[1]-2.0*c[4];
        const double b=(double(c[8])-c[6]-c[2]+c[0])/4;
        auto result=guarded_subpixel(c,true,21,true);
        REQUIRE((result.reason==SubpixelReason::Accepted ||
                 result.reason==SubpixelReason::ConstrainedBoundary));
        const auto energy=[&](double x,double y) {
            return gx*x+gy*y+.5*(a*x*x+2*b*x*y+d*y*y);
        };
        const double value=energy(result.dx,result.dy);
        REQUIRE(value<=1e-12);
        REQUIRE(std::abs(result.dx)<=.5);REQUIRE(std::abs(result.dy)<=.5);
        const auto kkt=[](double pos,double grad) {
            if(pos==-.5)return grad>=-1e-10;
            if(pos== .5)return grad<= 1e-10;
            return std::abs(grad)<=1e-10;
        };
        REQUIRE(kkt(result.dx,gx+a*result.dx+b*result.dy));
        REQUIRE(kkt(result.dy,gy+b*result.dx+d*result.dy));
        for(int j=-10;j<=10;++j)for(int i=-10;i<=10;++i)
            REQUIRE(value<=energy(i*.05,j*.05)+1e-10);
        const double determinant=a*d-b*b;
        const double ux=-(d*gx-b*gy)/determinant,uy=-(a*gy-b*gx)/determinant;
        REQUIRE(value<=energy(std::clamp(ux,-.5,.5),std::clamp(uy,-.5,.5))+1e-10);
    }
}
