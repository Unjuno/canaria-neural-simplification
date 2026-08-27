#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static constexpr int B=2,T=48,D=24,H=4,HD=6;
static constexpr size_t ARENA_BYTES=45056;
static constexpr size_t BLOCK_FLOATS=3696;
static constexpr size_t BLOCK_BYTES=BLOCK_FLOATS*4;
static constexpr size_t ACT_FLOATS=B*T*D;
static constexpr float EXPECTED_BLOCK0_SUM=57.7515073120594f;
static constexpr float EXPECTED_BLOCK1_SUM=56.73727474361658f;

static volatile int g_guard=0;
static volatile uint64_t g_heap_violations=0;
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_calloc(size_t,size_t);
extern "C" void* __real_realloc(void*,size_t);
extern "C" void* __wrap_malloc(size_t n){ if(g_guard) ++g_heap_violations; return __real_malloc(n); }
extern "C" void* __wrap_calloc(size_t a,size_t b){ if(g_guard) ++g_heap_violations; return __real_calloc(a,b); }
extern "C" void* __wrap_realloc(void* p,size_t n){ if(g_guard) ++g_heap_violations; return __real_realloc(p,n); }

struct Arena {
    alignas(64) unsigned char data[ARENA_BYTES];
    size_t off=0, high=0;
    void* alloc(size_t n,size_t align=64){
        size_t p=(off+align-1)&~(align-1);
        if(p+n>ARENA_BYTES) return nullptr;
        off=p+n; if(off>high) high=off; return data+p;
    }
};
static Arena arena;

static bool read_exact(const char* path, void* dst, size_t n){
    int fd=open(path,O_RDONLY); if(fd<0) return false;
    struct stat st{}; if(fstat(fd,&st)!=0 || (size_t)st.st_size!=n){ close(fd); return false; }
    unsigned char* p=(unsigned char*)dst; size_t got=0;
    while(got<n){ ssize_t r=read(fd,p+got,n-got); if(r<=0){ close(fd); return false; } got+=(size_t)r; }
    close(fd); return true;
}
static bool write_exact(const char* path,const void* src,size_t n){
    int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644); if(fd<0) return false;
    const unsigned char* p=(const unsigned char*)src; size_t put=0;
    while(put<n){ ssize_t w=write(fd,p+put,n-put); if(w<=0){ close(fd); return false; } put+=(size_t)w; }
    close(fd); return true;
}
static inline float dot24(const float* x,const float* w){ float s=0.f; for(int j=0;j<D;++j) s += x[j]*w[j]; return s; }
static inline void linear24(const float* x,const float* w,const float* b,float* y){ for(int o=0;o<D;++o) y[o]=dot24(x,w+o*D)+b[o]; }
static inline void layernorm_row(const float* x,const float* w,const float* b,float* y){
    float m=0.f; for(int j=0;j<D;++j)m+=x[j]; m/=D;
    float v=0.f; for(int j=0;j<D;++j){ float c=x[j]-m; v+=c*c; } v/=D;
    float inv=1.f/std::sqrt(v+1e-5f); for(int j=0;j<D;++j)y[j]=(x[j]-m)*inv*w[j]+b[j];
}
struct W {
    float *n1w,*n1b,*ipw,*ipb,*opw,*opb,*n2w,*n2b,*m0w,*m0b,*m2w,*m2b;
};
static W parse(float* p){
    W w{}; size_t o=0;
    w.n1w=p+o;o+=24; w.n1b=p+o;o+=24; w.ipw=p+o;o+=1728; w.ipb=p+o;o+=72;
    w.opw=p+o;o+=576; w.opb=p+o;o+=24; w.n2w=p+o;o+=24; w.n2b=p+o;o+=24;
    w.m0w=p+o;o+=576; w.m0b=p+o;o+=24; w.m2w=p+o;o+=576; w.m2b=p+o;o+=24;
    return w;
}
static float fsum(const float* p,size_t n){ double s=0.; for(size_t i=0;i<n;++i)s+=p[i]; return (float)s; }

static void run_block(float* x,float* k,float* v,float* score,float* ln,float* q,float* ctx,float* out,float* hid,const W& w){
    const float* wk=w.ipw+D*D; const float* wv=w.ipw+2*D*D; const float* bk=w.ipb+D; const float* bv=w.ipb+2*D;
    // Recompute/stream LayerNorm rows instead of retaining a full z[B,T,D].
    for(int r=0;r<B*T;++r){
        layernorm_row(x+r*D,w.n1w,w.n1b,ln);
        linear24(ln,wk,bk,k+r*D);
        linear24(ln,wv,bv,v+r*D);
    }
    const float* wq=w.ipw; const float* bq=w.ipb;
    const float scale=1.f/std::sqrt((float)HD);
    for(int b=0;b<B;++b){
        for(int ti=0;ti<T;++ti){
            float* xr=x+(b*T+ti)*D;
            layernorm_row(xr,w.n1w,w.n1b,ln);
            linear24(ln,wq,bq,q);
            for(int h=0;h<H;++h){
                float mx=-INFINITY;
                for(int tj=0;tj<=ti;++tj){
                    float s=0.f;
                    for(int d=0;d<HD;++d)s+=q[h*HD+d]*k[(b*T+tj)*D+h*HD+d];
                    s*=scale; score[h*T+tj]=s; if(s>mx)mx=s;
                }
                float den=0.f;
                for(int tj=0;tj<=ti;++tj){ float e=std::exp(score[h*T+tj]-mx); score[h*T+tj]=e; den+=e; }
                for(int d=0;d<HD;++d){
                    float c=0.f;
                    for(int tj=0;tj<=ti;++tj)c+=(score[h*T+tj]/den)*v[(b*T+tj)*D+h*HD+d];
                    ctx[h*HD+d]=c;
                }
            }
            linear24(ctx,w.opw,w.opb,out);
            for(int d=0;d<D;++d) xr[d]+=out[d];
        }
    }
    const float invsqrt2=0.7071067811865475244f;
    for(int r=0;r<B*T;++r){
        float* xr=x+r*D;
        layernorm_row(xr,w.n2w,w.n2b,ln);
        linear24(ln,w.m0w,w.m0b,hid);
        for(int j=0;j<D;++j) hid[j]=0.5f*hid[j]*(1.f+::erff(hid[j]*invsqrt2));
        linear24(hid,w.m2w,w.m2b,out);
        for(int j=0;j<D;++j) xr[j]+=out[j];
    }
}
static double secs(const timespec& a,const timespec& b){ return (b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)*1e-9; }

int main(){
    float* weights=(float*)arena.alloc(BLOCK_BYTES);
    float* x=(float*)arena.alloc(ACT_FLOATS*4);
    float* k=(float*)arena.alloc(ACT_FLOATS*4);
    float* v=(float*)arena.alloc(ACT_FLOATS*4);
    float* score=(float*)arena.alloc(H*T*4);
    float* ln=(float*)arena.alloc(D*4);
    float* q=(float*)arena.alloc(D*4);
    float* ctx=(float*)arena.alloc(D*4);
    float* out=(float*)arena.alloc(D*4);
    float* hid=(float*)arena.alloc(D*4);
    if(!weights||!x||!k||!v||!score||!ln||!q||!ctx||!out||!hid) return 90;

    g_guard=1;
    if(!read_exact("/data/input.bin",x,ACT_FLOATS*4)) return 91;
    timespec t0{},t1{}; clock_gettime(CLOCK_MONOTONIC,&t0);
    float bsum[2]{};
    for(int bi=0;bi<2;++bi){
        const char* path=bi==0?"/data/block0.bin":"/data/block1.bin";
        if(!read_exact(path,weights,BLOCK_BYTES)) return 92;
        bsum[bi]=fsum(weights,BLOCK_FLOATS);
        W w=parse(weights); run_block(x,k,v,score,ln,q,ctx,out,hid,w);
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    if(!write_exact("/data/output.bin",x,ACT_FLOATS*4)) return 93;
    g_guard=0;

    bool sums_ok=std::fabs(bsum[0]-EXPECTED_BLOCK0_SUM)<1e-4f && std::fabs(bsum[1]-EXPECTED_BLOCK1_SUM)<1e-4f;
    bool pass=arena.high<=ARENA_BYTES && g_heap_violations==0 && sums_ok;
    double osum=0.; for(size_t i=0;i<ACT_FLOATS;++i) osum+=x[i];
    char buf[2048];
    int n=snprintf(buf,sizeof(buf),"{\"status\":\"%s\",\"output_sum\":%.12g,\"arena_high_water_bytes\":%zu,\"arena_capacity_bytes\":%zu,\"heap_allocation_violations\":%llu,\"block0_sum\":%.9g,\"block1_sum\":%.9g,\"block_sums_ok\":%s,\"full_z_tensor_created\":false,\"full_score_tensor_created\":false,\"elapsed_seconds\":%.9g}\n",pass?"RUNTIME_PASS":"RUNTIME_FAIL",osum,arena.high,ARENA_BYTES,(unsigned long long)g_heap_violations,bsum[0],bsum[1],sums_ok?"true":"false",secs(t0,t1));
    if(n>0) write(STDOUT_FILENO,buf,(size_t)n);
    return pass?0:1;
}
