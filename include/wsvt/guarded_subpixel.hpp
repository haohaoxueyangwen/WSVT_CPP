#pragma once
#include "wsvt/fixed_spatial_cost.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace wsvt {
enum class SubpixelReason : unsigned char {
    Accepted=0, Boundary=1, Nonfinite=2, NotPositive=3,
    UnresolvedCurvature=4, OutsideCell=5, ConstrainedBoundary=6
};
struct GuardedSubpixel {
    double dx=0,dy=0;
    SubpixelReason reason=SubpixelReason::Boundary;
};
// Positive SSD, row-major 3x3 around integer winner. No synthetic edge stencil.
// A fallback is an integer estimate, NOT a certificate of tracking accuracy.
inline GuardedSubpixel guarded_subpixel(const std::array<float,9>& cost,
                                       bool complete, std::size_t depth,
                                       bool minimize_in_cell=false) {
    if(!complete) return {};
    double scale=1;
    for(float c:cost) {
        if(!finite_descriptor_value(c)) return {0,0,SubpixelReason::Nonfinite};
        scale=std::max(scale,std::abs(static_cast<double>(c)));
    }
    const double gx=(double(cost[5])-cost[3])/2;
    const double gy=(double(cost[7])-cost[1])/2;
    const double a=double(cost[5])+cost[3]-2.0*cost[4];
    const double b=(double(cost[8])-cost[6]-cost[2]+cost[0])/4;
    const double d=double(cost[7])+cost[1]-2.0*cost[4];
    const double determinant=a*d-b*b;
    if(!(a>0 && d>0 && determinant>0)) return {0,0,SubpixelReason::NotPositive};
    // For |cost error|<=e: diagonal Hessian error<=4e, cross<=e,
    // so ||Delta H||_2<=5e. Add a conservative double arithmetic allowance.
    const double q=(4.0*static_cast<double>(depth)+32)*std::numeric_limits<float>::epsilon();
    if(!(q<1)) return {0,0,SubpixelReason::UnresolvedCurvature};
    const double eta=(5*q/(1-q)+64*std::numeric_limits<double>::epsilon())*scale;
    const double lambda_max=(a+d+std::hypot(a-d,2*b))/2;
    const double lambda_min=determinant/lambda_max;
    if(!(lambda_min>eta)) return {0,0,SubpixelReason::UnresolvedCurvature};
    const double dx=-(d*gx-b*gy)/determinant;
    const double dy=-(a*gy-b*gx)/determinant;
    if(!(std::abs(dx)<=0.5 && std::abs(dy)<=0.5)) {
        if(minimize_in_cell) {
            // Strictly convex quadratic on a compact rectangle: interior
            // stationary point or a 1D edge minimum (including corners).
            const auto objective=[&](double x,double y) {
                return gx*x+gy*y+0.5*(a*x*x+2*b*x*y+d*y*y);
            };
            double best_x=0,best_y=0,best=0;
            const auto consider=[&](double x,double y) {
                const double value=objective(x,y);
                if(value<best) { best=value;best_x=x;best_y=y; }
            };
            for(double s:{-0.5,0.5}) {
                consider(s,std::clamp(-(gy+b*s)/d,-0.5,0.5));
                consider(std::clamp(-(gx+b*s)/a,-0.5,0.5),s);
            }
            return {best_x,best_y,SubpixelReason::ConstrainedBoundary};
        }
        return {0,0,SubpixelReason::OutsideCell};
    }
    return {dx,dy,SubpixelReason::Accepted};
}
}
