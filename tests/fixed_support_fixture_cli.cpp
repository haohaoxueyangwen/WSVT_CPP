#include "wsvt/wsvt_pipeline.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace {
template<class T> void read(std::istream& f,T* data,std::size_t n=1) {
    f.read(reinterpret_cast<char*>(data),static_cast<std::streamsize>(sizeof(T)*n));
    if(!f) throw std::runtime_error("truncated fixed-support fixture");
}
template<class T> void write(std::ostream& f,const T* data,std::size_t n=1) {
    f.write(reinterpret_cast<const char*>(data),static_cast<std::streamsize>(sizeof(T)*n));
    if(!f) throw std::runtime_error("failed fixed-support output");
}
}
int main(int argc,char** argv) try {
    if(argc!=6&&argc!=8&&argc!=9&&argc!=10) {
        std::cerr<<"usage: wsvt_fixed_support_fixture INPUT.bin PREFIX point|uniform3|hamming3 THREADS LEVEL [Y X]\n";
        return 2;
    }
    if constexpr(std::endian::native!=std::endian::little)
        throw std::runtime_error("fixture requires little endian");
    const std::string prefix=argv[2],support=argv[3];
    const int threads=std::stoi(argv[4]),level=std::stoi(argv[5]);
    if((threads!=1&&threads!=4)||(level!=0&&level!=1))
        throw std::invalid_argument("bounded fixture uses threads 1/4 and level 0/1");
    wsvt::FixedSpatialSupport mode;
    if(support=="point"||support=="legacy_n1") mode=wsvt::FixedSpatialSupport::Point;
    else if(support=="uniform3") mode=wsvt::FixedSpatialSupport::Uniform3;
    else if(support=="hamming3") mode=wsvt::FixedSpatialSupport::Hamming3;
    else throw std::invalid_argument("invalid support");
    std::ifstream input(argv[1],std::ios::binary);
    char magic[4];read(input,magic,4);
    std::uint32_t dims[3];read(input,dims,3);
    if(std::string(magic,4)!="FSV1"||dims[0]<2||dims[0]>49||
       dims[1]<64||dims[1]>256||dims[2]<64||dims[2]>256)
        throw std::invalid_argument("invalid bounded fixture shape/magic");
    const std::size_t count=std::size_t(dims[0])*dims[1]*dims[2];
    std::vector<float> sample(count),ref(count);read(input,sample.data(),count);read(input,ref.data(),count);
    if(input.peek()!=std::char_traits<char>::eof()) throw std::invalid_argument("trailing fixture bytes");
    auto construct=[&]() {
        return std::make_unique<wsvt::WSVT>(sample,ref,dims[0],dims[1],dims[2],
            0,16,support=="legacy_n1" ? 1 : 0,4,threads,1,14000,0.65e-6,1.0,0.425,1,1,1,false,true,0,false,1);
    };
    // Separate object: production wavelet_data consumes its raw buffers.
    auto probe=construct();auto features=probe->wavelet_data();
    std::ofstream ff(prefix+".features.bin",std::ios::binary);
    std::uint32_t levels=static_cast<std::uint32_t>(features.img_levels.size());write(ff,&levels);
    for(std::size_t i=0;i<levels;++i) {
        const auto& a=features.img_levels[i];const auto& b=features.ref_levels[i];
        std::uint32_t shape[]{static_cast<std::uint32_t>(a.d0),
            static_cast<std::uint32_t>(a.d1),static_cast<std::uint32_t>(a.d2)};
        write(ff,shape,3);write(ff,a.data.data(),a.data.size());write(ff,b.data.data(),b.data.size());
    }
    ff.close();
    const auto& grid=features.img_levels[static_cast<std::size_t>(level)];
    bool guarded=false, constrained=false;
    std::vector<std::size_t> points;
    for(auto y:{grid.d1/2-8,grid.d1/2,grid.d1/2+8})
        for(auto x:{grid.d2/2-8,grid.d2/2,grid.d2/2+8}) points.push_back(y*grid.d2+x);
    if(argc==8) {
        const auto y=std::stoll(argv[6]),x=std::stoll(argv[7]);
        if(y<0||x<0||static_cast<std::size_t>(y)>=grid.d1||
           static_cast<std::size_t>(x)>=grid.d2)
            throw std::invalid_argument("diagnostic coordinate outside pyramid");
        points={static_cast<std::size_t>(y)*grid.d2+static_cast<std::size_t>(x)};
    }
    if(argc>=9) {
        if(std::string(argv[6])!="sp1" ||
           (std::string(argv[7])!="legacy" && std::string(argv[7])!="guarded" &&
            std::string(argv[7])!="box"))
            throw std::invalid_argument("syntax: sp1 legacy|guarded|box POINTS.txt");
        guarded=std::string(argv[7])=="guarded";
        constrained=std::string(argv[7])=="box";
        std::ifstream selected(argv[8]);
        if(!selected) throw std::invalid_argument("missing SP1 point list");
        points.clear();
        long long y,x;
        while(selected>>y>>x) {
            if(y<0||x<0||static_cast<std::size_t>(y)>=grid.d1||
               static_cast<std::size_t>(x)>=grid.d2||points.size()>=4096)
                throw std::invalid_argument("SP1 points outside bounds");
            points.push_back(static_cast<std::size_t>(y)*grid.d2+static_cast<std::size_t>(x));
        }
        if(points.empty()||!selected.eof()) throw std::invalid_argument("invalid SP1 points");
    }
    const std::size_t candidates=level==0 ? 81 : 289;
    auto solver=construct();
    solver->configure_fixed_spatial_support(mode);
    solver->configure_guarded_subpixel(guarded);
    if(constrained) solver->configure_constrained_subpixel(true);
    if(argc==10) solver->configure_fixed_spatial_reuse(true,std::stoi(argv[9]));
    // The bounded attribution audit requests every central pixel but needs
    // only the initial center. Keep the established 81/289 top-K contract for
    // the older <=64-point diagnostics and use rank-2 for the compact all-pixel
    // center capture.
    const std::size_t diagnostic_top_k=points.size()>64 ? 2 : candidates;
    solver->configure_search_topk_diagnostics(points,diagnostic_top_k,level);
    const auto out=solver->solver();
    std::ofstream df(prefix+".candidates.csv");
    df<<"level,y,x,rank,center_y,center_x,uy,ux,cost\n"<<std::setprecision(17);
    for(const auto& d:solver->search_topk_diagnostics())
        df<<d.pyramid_level<<','<<d.raw_y<<','<<d.raw_x<<','<<d.rank<<','
          <<d.initial_guess_y<<','<<d.initial_guess_x<<','
          <<d.internal_candidate_y<<','<<d.internal_candidate_x<<','
          <<d.descriptor_cost_ssd<<'\n';
    std::ofstream of(prefix+".output.bin",std::ios::binary);
    std::uint32_t shape[]{static_cast<std::uint32_t>(out.h),static_cast<std::uint32_t>(out.w)};
    write(of,shape,2);
    for(const auto* array:{&out.displace_x,&out.displace_y,&out.darkfield_nd,
                          &out.search_boundary_hit,&out.search_geometry_valid})
        write(of,array->data(),array->size());
    if(argc>=9) {
        std::ofstream sf(prefix+".reasons.bin",std::ios::binary);
        const std::uint32_t raw_shape[]{dims[1],dims[2]};
        write(sf,raw_shape,2);
        auto reasons=solver->guarded_subpixel_reasons();
        if(reasons.empty()) reasons.assign(std::size_t(dims[1])*dims[2],255);
        write(sf,reasons.data(),reasons.size());
        std::ofstream offset_file(prefix+".offsets.bin",std::ios::binary);
        write(offset_file,raw_shape,2);
        auto offsets=solver->guarded_subpixel_offsets();
        if(offsets.empty()) offsets.assign(2*std::size_t(dims[1])*dims[2],0.0);
        write(offset_file,offsets.data(),offsets.size());
    }
    std::ofstream jf(prefix+".json");
    jf<<std::setprecision(17)<<"{\"support\":\""<<support<<"\",\"threads\":"<<threads
      <<",\"h\":"<<out.h<<",\"w\":"<<out.w
      <<",\"candidate_count\":"<<out.search_candidate_count
      <<",\"descriptor_terms\":"<<out.search_distance_terms_evaluated
      <<",\"possible_terms\":"<<out.search_distance_terms_possible
      <<",\"cache_hits\":"<<solver->spatial_reuse_stats().hits
      <<",\"cache_misses\":"<<solver->spatial_reuse_stats().misses
      <<",\"cache_resets\":"<<solver->spatial_reuse_stats().resets
      <<",\"cache_overflows\":"<<solver->spatial_reuse_stats().overflows
      <<",\"cache_peak_entries_per_worker\":"<<solver->spatial_reuse_stats().peak_entries
      <<",\"cache_allocated_bytes_all_workers\":"<<solver->spatial_reuse_stats().allocated_bytes
      <<",\"guarded_subpixel\":"<<((guarded||constrained) ? "true" : "false")
      <<",\"constrained_subpixel\":"<<(constrained ? "true" : "false")
      <<",\"solver_time_s_diagnostic\":"<<out.time_cost_s
      <<",\"formal_timing\":false}\n";
    if(!df||!jf) throw std::runtime_error("failed fixture report");
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
