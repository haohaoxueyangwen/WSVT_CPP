#include <catch2/catch_test_macros.hpp>
#include "wsvt/fixed_spatial_cost.hpp"
#include "wsvt/wsvt_pipeline.hpp"
#include <cmath>
#include <numeric>
#include <random>

using namespace wsvt;
namespace {
WSVT make_solver(const std::vector<float>& a,int threads=1);
std::vector<float> texture(std::size_t n) {
    std::mt19937 gen(20260909);
    std::uniform_real_distribution<float> d(-2,2);
    std::vector<float> a(n);
    for (auto& v:a) v=d(gen);
    return a;
}

TEST_CASE("spatial reuse keys actual coordinates and preserves cost order", "[fixed-support][reuse]") {
    auto a=texture(7*9*17),b=texture(8*10*17);
    SpatialFeatureView av(a,7,9,17),bv(b,8,10,17);
    for(auto mode:{FixedSpatialSupport::Uniform3,FixedSpatialSupport::Hamming3}) {
        SpatialPointCache cache;
        cache.select_tile(0);
        FixedSpatialCost cost(mode);
        for(int y=0;y<7;++y) for(int x=0;x<9;++x)
        for(int uy:{-2,0,2}) for(int ux:{-1,1}) {
            // Both actual coordinate pairs matter, not equal residual indices.
            const auto actual=cost.evaluate(av,bv,y,x,y+uy,x+ux,cache);
            REQUIRE(actual==cost(av,bv,y,x,y+uy,x+ux));
        }
        REQUIRE(cache.stats().hits>0);
        REQUIRE(cache.stats().overflows==0);
        const auto misses=cache.stats().misses;
        cache.select_tile(1);
        REQUIRE(cost.evaluate(av,bv,2,2,2,3,cache)==cost(av,bv,2,2,2,3));
        REQUIRE(cache.stats().misses==misses+9);
    }
}

TEST_CASE("spatial reuse bounded table falls back exactly when full", "[fixed-support][reuse]") {
    auto a=texture(2*2*3);
    SpatialFeatureView view(a,2,2,3);
    SpatialPointCache cache;
    cache.select_tile(0);
    for(int i=0;i<32769;++i)
        REQUIRE(cache(view,view,0,0,i,0)==FixedSpatialCost::point_distance(view,view,0,0,i,0));
    REQUIRE(cache.stats().peak_entries==32768);
    REQUIRE(cache.stats().overflows==1);
}

TEST_CASE("spatial reuse preserves pipeline output across tile and worker boundaries", "[fixed-support][reuse]") {
    auto a=texture(16*64*72);
    for(auto support:{FixedSpatialSupport::Uniform3,FixedSpatialSupport::Hamming3})
    for(bool box:{false,true}) {
        auto direct=make_solver(a);
        direct.configure_fixed_spatial_support(support);
        direct.configure_constrained_subpixel(box);
        const auto d=direct.solver();
        for(int tile:{8,16,32}) for(int threads:{1,4}) {
            auto cached=make_solver(a,threads);
            cached.configure_fixed_spatial_support(support);
            cached.configure_constrained_subpixel(box);
            cached.configure_fixed_spatial_reuse(true,tile);
            const auto c=cached.solver();
            REQUIRE(c.displace_x==d.displace_x);
            REQUIRE(c.displace_y==d.displace_y);
            REQUIRE(c.darkfield_nd==d.darkfield_nd);
            REQUIRE(c.search_boundary_hit==d.search_boundary_hit);
            REQUIRE(c.search_geometry_valid==d.search_geometry_valid);
            REQUIRE(c.search_candidate_count==d.search_candidate_count);
            REQUIRE(c.search_distance_terms_possible==d.search_distance_terms_possible);
            REQUIRE(c.search_distance_terms_evaluated<d.search_distance_terms_evaluated);
            REQUIRE(cached.guarded_subpixel_reasons()==direct.guarded_subpixel_reasons());
            REQUIRE(cached.spatial_reuse_stats().peak_entries<=std::size_t(tile)*81*9);
            REQUIRE(cached.spatial_reuse_stats().overflows==0);
            REQUIRE(d.search_distance_terms_evaluated-c.search_distance_terms_evaluated==
                    cached.spatial_reuse_stats().hits*21);
        }
    }
}

TEST_CASE("spatial reuse profile remains opt in and revalidated", "[fixed-support][reuse]") {
    auto a=texture(16*64*72);
    auto solver=make_solver(a);
    REQUIRE_THROWS(solver.configure_fixed_spatial_reuse(true));
    solver.configure_fixed_spatial_support(FixedSpatialSupport::Hamming3);
    REQUIRE_THROWS(solver.configure_fixed_spatial_reuse(true,3));
    solver.configure_fixed_spatial_reuse(true);
    solver.configure_fixed_spatial_support(FixedSpatialSupport::Point);
    REQUIRE_THROWS(solver.solver());
}
WSVT make_solver(const std::vector<float>& a,int threads) {
    return WSVT(a,a,16,64,72,0,16,0,4,threads,1,
                14000,0.65e-6,1.0,0.425,1,1,1,false,true,0,false,1);
}
}

TEST_CASE("fixed support weights and point retain SSD", "[fixed-support]") {
    auto a=texture(7*9*21), b=texture(7*9*21);
    for (auto& v:b) v+=0.1f;
    SpatialFeatureView av(a,7,9,21),bv(b,7,9,21);
    for (auto s:{FixedSpatialSupport::Point,FixedSpatialSupport::Uniform3,
                 FixedSpatialSupport::Hamming3}) {
        FixedSpatialCost cost(s);
        REQUIRE(std::abs(std::accumulate(cost.weights().begin(),cost.weights().end(),0.0)-1)<1e-14);
        for(auto w:cost.weights()) REQUIRE(w>=0);
    }
    REQUIRE(FixedSpatialCost(FixedSpatialSupport::Point)(av,bv,2,4,3,5)==
            squared_distance_full_simd(av.pixel(2,4),bv.pixel(3,5),21));
    REQUIRE_THROWS(FixedSpatialCost(static_cast<FixedSpatialSupport>(99)));
}

TEST_CASE("fixed support matches independent double oracle including zero halo", "[fixed-support]") {
    auto a=texture(7*9*17),b=texture(8*10*17);
    SpatialFeatureView av(a,7,9,17),bv(b,8,10,17);
    for (auto s:{FixedSpatialSupport::Uniform3,FixedSpatialSupport::Hamming3}) {
        FixedSpatialCost cost(s);
        for (int y=0;y<7;++y) for (int x=0;x<9;++x)
        for (int uy:{-2,0,2}) for(int ux:{-1,1}) {
            double expected=0;
            for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox) {
                const auto* p=av.pixel(y+oy,x+ox);
                const auto* q=bv.pixel(y+uy+oy,x+ux+ox);
                double sum=0;
                for(std::size_t c=0;c<17;++c) {
                    double d=(p ? double(p[c]):0)-(q ? double(q[c]):0);
                    sum+=d*d;
                }
                const double hy=oy==0 ? 25.0/29 : 2.0/29;
                const double hx=ox==0 ? 25.0/29 : 2.0/29;
                expected+=(s==FixedSpatialSupport::Uniform3 ? 1.0/9:hy*hx)*sum;
            }
            REQUIRE(std::abs(cost(av,bv,y,x,y+uy,x+ux)-expected) <=
                    128*std::numeric_limits<float>::epsilon()*std::max(1.0,expected));
        }
    }
}

TEST_CASE("fixed support rejects invalid descriptors and coordinates", "[fixed-support]") {
    auto a=texture(2*3*4);
    REQUIRE_THROWS(SpatialFeatureView(a,2,3,5));
    SpatialFeatureView good(a,2,3,4);
    REQUIRE_THROWS(FixedSpatialCost(FixedSpatialSupport::Point)(
        good,good,std::numeric_limits<std::int64_t>::max(),0,0,0));
    for(float v:{std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN()}) {
        auto b=a;b[3]=v;
        REQUIRE_FALSE(finite_descriptor_value(v));
        REQUIRE_THROWS(SpatialFeatureView(b,2,3,4));
    }
}

TEST_CASE("point configuration leaves B0 output bit identical", "[fixed-support]") {
    auto a=texture(16*64*72);
    auto base=make_solver(a),point=make_solver(a);
    point.configure_fixed_spatial_support(FixedSpatialSupport::Point);
    const auto b=base.solver(),p=point.solver();
    REQUIRE(b.displace_x==p.displace_x);
    REQUIRE(b.displace_y==p.displace_y);
    REQUIRE(b.darkfield_nd==p.darkfield_nd);
    REQUIRE(b.phase==p.phase);
    REQUIRE(b.search_distance_terms_evaluated==p.search_distance_terms_evaluated);
}

TEST_CASE("fixed support preserves coarse candidates and diagnoses extra work", "[fixed-support]") {
    auto a=texture(16*64*72);
    std::vector<SearchTopKDiagnostic> baseline;
    std::uint64_t baseline_candidates=0,baseline_terms=0;
    for (auto s:{FixedSpatialSupport::Point,FixedSpatialSupport::Uniform3,
                 FixedSpatialSupport::Hamming3}) {
        auto solver=make_solver(a);
        solver.configure_fixed_spatial_support(s);
        solver.configure_search_topk_diagnostics({0,100},5,1);
        auto result=solver.solver();
        const auto d=solver.search_topk_diagnostics();
        if(s==FixedSpatialSupport::Point) {
            baseline=d;baseline_candidates=result.search_candidate_count;
            baseline_terms=result.search_distance_terms_evaluated;
        } else {
            REQUIRE(d.size()==baseline.size());
            for(std::size_t i=0;i<d.size();++i) {
                REQUIRE(d[i].internal_candidate_x==baseline[i].internal_candidate_x);
                REQUIRE(d[i].internal_candidate_y==baseline[i].internal_candidate_y);
                REQUIRE(d[i].descriptor_cost_ssd==baseline[i].descriptor_cost_ssd);
            }
            REQUIRE(result.search_candidate_count==baseline_candidates);
            REQUIRE(result.search_distance_terms_evaluated-baseline_terms==
                    8ULL*64*72*81*21);
        }
    }
}

TEST_CASE("fixed support is independent of OpenMP thread count", "[fixed-support]") {
    auto a=texture(16*64*72);
    auto single=make_solver(a,1),multi=make_solver(a,4);
    single.configure_fixed_spatial_support(FixedSpatialSupport::Hamming3);
    multi.configure_fixed_spatial_support(FixedSpatialSupport::Hamming3);
    const auto s=single.solver(),m=multi.solver();
    REQUIRE(s.displace_x==m.displace_x);
    REQUIRE(s.displace_y==m.displace_y);
    REQUIRE(s.darkfield_nd==m.darkfield_nd);
}

TEST_CASE("fixed support excludes old profiles regardless of setter order", "[fixed-support]") {
    auto a=texture(16*64*72);
    auto first=make_solver(a);
    first.configure_fixed_set_transport();
    REQUIRE_THROWS(first.configure_fixed_spatial_support(FixedSpatialSupport::Hamming3));
    auto second=make_solver(a);
    second.configure_fixed_spatial_support(FixedSpatialSupport::Hamming3);
    second.configure_fixed_set_transport();
    REQUIRE_THROWS(second.solver());
    auto nonfinite=a;nonfinite[0]=std::numeric_limits<float>::quiet_NaN();
    auto bad=make_solver(nonfinite);
    bad.configure_fixed_spatial_support(FixedSpatialSupport::Uniform3);
    REQUIRE_THROWS(bad.solver());
}
