#include "speckle_native.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#ifndef SPECKLE_PLATFORM
#define SPECKLE_PLATFORM "desktop"
#endif
namespace fs=std::filesystem;
namespace {
const char* names[]={"displace_y","displace_x","DPC_y","DPC_x","phase","transmission","match_score_neg_ssd"};
struct Host {
    void* p=nullptr;size_t bytes=0;Host(size_t n,bool mapped):bytes(n){p=speckle_host_alloc(n,mapped);if(!p)throw std::runtime_error(speckle_error());}
    Host(const Host&)=delete;~Host(){speckle_host_free(p);}
};
struct Array {
    std::vector<int> shape;std::string dtype;std::unique_ptr<Host> data;
    Array(const fs::path& path,bool mapped){
        std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot open "+path.string());
        char magic[8];f.read(magic,8);if(f.gcount()!=8||std::memcmp(magic,"\x93NUMPY",6)||magic[6]<1||magic[6]>2)throw std::runtime_error("Expected little-endian NPY v1/v2");
        unsigned char lb[4]={};f.read(reinterpret_cast<char*>(lb),magic[6]==1?2:4);size_t len=lb[0]+(size_t(lb[1])<<8)+(size_t(lb[2])<<16)+(size_t(lb[3])<<24);
        if(len>65536)throw std::runtime_error("Invalid NPY header");std::string header(len,' ');f.read(header.data(),len);
        std::smatch match;if(!std::regex_search(header,match,std::regex("'descr'\\s*:\\s*'([^']+)'")))throw std::runtime_error("NPY dtype missing");dtype=match[1];
        if(dtype!="<f4"&&dtype!="<u2"&&dtype!="=f4"&&dtype!="=u2")throw std::runtime_error("Use float32/uint16 little-endian input");
        if(header.find("True")!=std::string::npos)throw std::runtime_error("Fortran-order input unsupported");
        if(!std::regex_search(header,match,std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)")))throw std::runtime_error("NPY shape missing");
        std::string dims=match[1];std::regex numbers("[0-9]+");size_t count=1;
        for(std::sregex_iterator i(dims.begin(),dims.end(),numbers),end;i!=end;++i){int v=std::stoi(i->str());if(v<=0||count>SIZE_MAX/size_t(v))throw std::runtime_error("Input shape overflow");shape.push_back(v);count*=size_t(v);}
        if(shape.empty()||shape.size()>3)throw std::runtime_error("Expected HxW or 100xHxW");size_t bytes=count*(dtype.back()=='2'?2:4);data=std::make_unique<Host>(bytes,mapped);f.read(static_cast<char*>(data->p),bytes);if(size_t(f.gcount())!=bytes)throw std::runtime_error("Truncated NPY input");
    }
};
void save_npy(const fs::path& path,const float* a,int h,int w){
    std::string header="{'descr': '<f4', 'fortran_order': False, 'shape': ("+std::to_string(h)+", "+std::to_string(w)+"), }";
    size_t pad=16-((10+header.size()+1)%16);header.append(pad,' ');header+='\n';unsigned short len=header.size();
    std::ofstream out(path,std::ios::binary);out.write("\x93NUMPY\x01\x00",8);out.put(char(len&255));out.put(char(len>>8));out.write(header.data(),header.size());out.write(reinterpret_cast<const char*>(a),size_t(h)*w*4);if(!out)throw std::runtime_error("Output write failed");
}
void ck(int code){if(code)throw std::runtime_error(speckle_error());}
double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
}
int main(int argc,char** argv){
    void* session=nullptr;
    try{
        std::map<std::string,std::string> args;
        for(int j=1;j<argc;++j){std::string k=argv[j];if(k=="--help"){std::cout<<"Native C++/CUDA/cuFFT WSVT100 and WXST7x7; no Python/PyTorch dependency.\nRequired: --mode wsvt|wxst --sample sample.npy --reference reference.npy --out NEW_DIR --pixel-m VALUE --distance-m VALUE --energy-ev VALUE\nOptional: --platform desktop|jetson --workspace speed|low-memory --input-policy copy|mapped --chunk 16 --warmup 1 --repeats 5 --device 0 --reference-policy each-pair|cache\nFixed half10/coarse5/fine4/L2/I1/cut1/no dark-field. Input: contiguous uint16/float32 WSVT100xHxW; float32 WXSTHxW.\n";return 0;}
            if(j+1>=argc||k.substr(0,2)!="--")throw std::runtime_error("Expected --key value");args[k]=argv[++j];}
        const std::vector<std::string> allowed={"--mode","--sample","--reference","--out","--pixel-m","--distance-m","--energy-ev","--platform","--workspace","--input-policy","--chunk","--warmup","--repeats","--device","--reference-policy"};
        for(auto& v:args)if(std::find(allowed.begin(),allowed.end(),v.first)==allowed.end())throw std::runtime_error("Unknown option "+v.first);
        auto get=[&](const char* k,const char* fallback=""){auto i=args.find(k);if(i!=args.end())return i->second;if(*fallback)return std::string(fallback);throw std::runtime_error(std::string("Missing ")+k);};
        std::string mode=get("--mode"),platform=get("--platform",SPECKLE_PLATFORM),workspace=get("--workspace","speed"),policy=get("--input-policy","copy"),reference_policy=get("--reference-policy","each-pair");
        if((mode!="wsvt"&&mode!="wxst")||(platform!="desktop"&&platform!="jetson")||(workspace!="speed"&&workspace!="low-memory")||(policy!="copy"&&policy!="mapped")||(reference_policy!="each-pair"&&reference_policy!="cache"))throw std::runtime_error("Unsupported option value");
        if(policy=="mapped"&&platform!="jetson")throw std::runtime_error("Mapped input is a Jetson preset");
        int warmup=std::stoi(get("--warmup","1")),repeats=std::stoi(get("--repeats","5"));if(warmup<0||repeats<1)throw std::runtime_error("Invalid repeat count");
        fs::path out=get("--out");if(fs::exists(out))throw std::runtime_error("Refuse to overwrite existing output directory");
        double start=now();Array sample(get("--sample"),policy=="mapped"),ref(get("--reference"),policy=="mapped");double read=now()-start;
        if(sample.shape!=ref.shape||sample.dtype!=ref.dtype)throw std::runtime_error("Sample/reference shape or dtype mismatch");
        bool wsvt=mode=="wsvt";if((wsvt&&(sample.shape.size()!=3||sample.shape[0]!=100))||(!wsvt&&sample.shape.size()!=2))throw std::runtime_error("WSVT requires100 frames; WXST requires a single pair");
        int h=sample.shape[sample.shape.size()-2],w=sample.shape.back();SpeckleConfig cfg{};cfg.height=h;cfg.width=w;cfg.mode=wsvt;cfg.input_u16=sample.dtype.back()=='2';cfg.low_memory=workspace=="low-memory";cfg.spatial_chunk=std::stoi(get("--chunk","16"));cfg.device=std::stoi(get("--device","0"));cfg.mapped_input=policy=="mapped";cfg.pixel_m=std::stod(get("--pixel-m"));cfg.distance_m=std::stod(get("--distance-m"));cfg.energy_ev=std::stod(get("--energy-ev"));
        start=now();session=speckle_create(&cfg);if(!session)throw std::runtime_error(speckle_error());double constructor=now()-start;
        std::vector<std::unique_ptr<Host>> host;void* outputs[7];for(int j=0;j<7;++j){host.push_back(std::make_unique<Host>(size_t(j==5?h:h-20)*(j==5?w:w-20)*4,false));outputs[j]=host.back()->p;}
        std::vector<SpeckleTiming> records;std::vector<double> apis;double reference=0.;
        if(reference_policy=="cache")ck(speckle_prepare(session,ref.data->p,ref.data->bytes,&reference));
        for(int j=0;j<warmup+repeats;++j){start=now();if(reference_policy=="each-pair")ck(speckle_prepare(session,ref.data->p,ref.data->bytes,&reference));SpeckleTiming t;ck(speckle_solve(session,sample.data->p,sample.data->bytes,outputs,&t));double api=now()-start;
            if(reference_policy=="cache")t.reference_s=0.;
            if(j>=warmup){records.push_back(t);apis.push_back(api);std::cout<<"measured "<<j-warmup<<" solver="<<t.reference_s+t.sample_solver_s<<" api="<<api<<std::endl;}}
        struct rusage usage{};getrusage(RUSAGE_SELF,&usage);char device[256];ck(speckle_device_info(cfg.device,device,sizeof(device)));
        fs::create_directories(out);start=now();for(int j=0;j<7;++j)save_npy(out/(std::string(names[j])+".npy"),static_cast<float*>(outputs[j]),j==5?h:h-20,j==5?w:w-20);double save=now()-start;
        std::ofstream csv(out/"timing.csv");csv.precision(17);csv<<"repeat,solver_s,api_s,reference_s,sample_s,download_s\n";
        double mean=0.,api_mean=0.;for(size_t j=0;j<records.size();++j){auto& t=records[j];double v=t.reference_s+t.sample_solver_s;mean+=v;api_mean+=apis[j];csv<<j<<','<<v<<','<<apis[j]<<','<<t.reference_s<<','<<t.sample_solver_s<<','<<t.download_s<<'\n';}mean/=records.size();api_mean/=records.size();double sd=0.,api_sd=0.;for(size_t j=0;j<records.size();++j){sd+=std::pow(records[j].reference_s+records[j].sample_solver_s-mean,2);api_sd+=std::pow(apis[j]-api_mean,2);}if(records.size()>1){sd=std::sqrt(sd/(records.size()-1));api_sd=std::sqrt(api_sd/(records.size()-1));}
        std::ofstream report(out/"report.json");report.precision(17);auto& t=records.back();report<<"{\n\"implementation\":\"native C++/CUDA/cuFFT\",\n\"device\":\""<<device<<"\",\n\"mode\":\""<<mode<<"\",\"platform\":\""<<platform<<"\",\"workspace\":\""<<workspace<<"\",\"input_policy\":\""<<policy<<"\",\"reference_policy\":\""<<reference_policy<<"\",\n\"half\":10,\"coarse\":5,\"fine\":4,\"height\":"<<h<<",\"width\":"<<w<<",\"frames\":"<<(wsvt?100:1)<<",\"input_u16\":"<<cfg.input_u16<<",\"chunk\":"<<cfg.spatial_chunk<<",\"warmup\":"<<warmup<<",\"repeats\":"<<repeats<<",\n\"pixel_m\":"<<cfg.pixel_m<<",\"distance_m\":"<<cfg.distance_m<<",\"energy_ev\":"<<cfg.energy_ev<<",\n\"solver_mean_s\":"<<mean<<",\"solver_sample_sd_s\":"<<sd<<",\"api_mean_s\":"<<api_mean<<",\"api_sample_sd_s\":"<<api_sd<<",\"read_s\":"<<read<<",\"constructor_s\":"<<constructor<<",\"save_s\":"<<save<<",\n\"explicit_device_bytes\":"<<t.device_allocation_bytes<<",\"fft_workspace_bytes\":"<<t.fft_workspace_bytes<<",\"peak_rss_bytes\":"<<uint64_t(usage.ru_maxrss)*1024<<",\n\"scope\":\"Solver includes reference rebuild unless explicit cache mode; API includes seven outputs; input file read, construction, save excluded. Host buffers pinned outside timing. RSS through measured solves includes input loading and initialization. Device bytes are explicit allocations, not NVML/physical peak.\"\n}\n";
        speckle_destroy(session);session=nullptr;std::cout<<"complete "<<out.string()<<std::endl;return 0;
    }catch(const std::exception& e){if(session)speckle_destroy(session);std::cerr<<"speckle: "<<e.what()<<std::endl;return 1;}
}
