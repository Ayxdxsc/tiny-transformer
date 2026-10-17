// tiny.cpp - ai1 训练逻辑精修版
// 编译: clang++ -O3 -march=armv8.2-a+dotprod -ffast-math -fno-math-errno \
//   -funroll-loops -fomit-frame-pointer -fno-exceptions -fno-rtti -o tiny tiny.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <arm_neon.h>
using namespace std;
typedef vector<float> VF;
static inline float silu_f(float x){ if(x>30.f)return x; if(x<-30.f)return 0.f; return x/(1.f+expf(-x)); }
static inline float silu_g(float x){ if(x>30.f)return 1.f; if(x<-30.f)return 0.f; float s=1.f/(1.f+expf(-x)); return s*(1.f+x*(1.f-s)); }
static inline float hs4(float32x4_t v){ return vaddvq_f32(v); }

static vector<string> split_utf8(const string& s){
    vector<string> o; size_t i=0;
    while(i<s.size()){
        unsigned char c=(unsigned char)s[i]; int L=1;
        if(c<0x80)L=1;
        else if((c&0xE0)==0xC0)L=2;
        else if((c&0xF0)==0xE0)L=3;
        else if((c&0xF8)==0xF0)L=4;
        if(i+L>s.size())L=1;
        o.push_back(s.substr(i,L)); i+=L;
    }
    return o;
}
struct Vocab{
    vector<string> chars; unordered_map<string,int> c2i;
    void build(const string& text){
        chars.clear(); c2i.clear();
        for(auto& t:split_utf8(text)){
            if(c2i.find(t)==c2i.end()){ c2i[t]=(int)chars.size(); chars.push_back(t); }
        }
    }
    vector<int> encode(const string& s) const {
        vector<int> r;
        for(auto& t:split_utf8(s)){ auto it=c2i.find(t); if(it!=c2i.end())r.push_back(it->second); }
        return r;
    }
};

struct Config{
    string mode="train",data="data.txt",model="tiny.bin",ckpt="model.ckpt",prompt="",batch_file="",config_file="tiny.cfg";
    int D=128,L=4,nq=4,nkv=2,ctx=64,infer_maxT=0;
    float dropout=0.f,embed_dropout=0.f;
    int epochs=300,batch=16,grad_accum=1;
    float lr=1e-3f,min_lr=1e-5f,weight_decay=0.1f,b1=0.9f,b2=0.999f,eps=1e-8f;
    int warmup=100; string lr_sched="cosine";
    float stop_loss=0.f,val_split=0.1f,grad_clip=1.0f; bool lr_override=0;
    float temp=0.8f,top_p=0.9f; int topk=10,rep_window=0; float rep_pen=1.f;
    int maxlen=200; bool one_token=0;
    unsigned seed=42; bool resume=0,greedy=0; int ckpt_interval=10;
};
static string trm(const string& s){ size_t a=s.find_first_not_of(" \t\r\n"); if(a==string::npos)return ""; size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1); }
static bool s2i(const string& s,int& o){ if(s.empty())return 0; const char* p=s.c_str(); char* e=0; long v=strtol(p,&e,10); if(e==p||*e)return 0; o=(int)v; return 1; }
static bool s2f(const string& s,float& o){ if(s.empty())return 0; const char* p=s.c_str(); char* e=0; float v=strtof(p,&e); if(e==p||*e)return 0; o=v; return 1; }

static void load_cfg(Config& c,const string& p){
    ifstream f(p); if(!f)return; string ln; int iv; float fv;
    while(getline(f,ln)){
        auto h=ln.find('#'); if(h!=string::npos)ln=ln.substr(0,h);
        auto eq=ln.find('='); if(eq==string::npos)continue;
        string k=trm(ln.substr(0,eq)),v=trm(ln.substr(eq+1)); if(k.empty()||v.empty())continue;
        if(k=="D"){ if(s2i(v,iv))c.D=iv; }
        else if(k=="L"||k=="layers"){ if(s2i(v,iv))c.L=iv; }
        else if(k=="nq"){ if(s2i(v,iv))c.nq=iv; }
        else if(k=="nkv"){ if(s2i(v,iv))c.nkv=iv; }
        else if(k=="ctx"){ if(s2i(v,iv))c.ctx=iv; }
        else if(k=="infer_maxT"){ if(s2i(v,iv))c.infer_maxT=iv; }
        else if(k=="dropout"){ if(s2f(v,fv))c.dropout=fv; }
        else if(k=="embed_dropout"){ if(s2f(v,fv))c.embed_dropout=fv; }
        else if(k=="epochs"){ if(s2i(v,iv))c.epochs=iv; }
        else if(k=="batch"){ if(s2i(v,iv))c.batch=iv; }
        else if(k=="grad_accum"){ if(s2i(v,iv))c.grad_accum=iv; }
        else if(k=="lr"){ if(s2f(v,fv))c.lr=fv; }
        else if(k=="min_lr"){ if(s2f(v,fv))c.min_lr=fv; }
        else if(k=="weight_decay"){ if(s2f(v,fv))c.weight_decay=fv; }
        else if(k=="warmup_steps"){ if(s2i(v,iv))c.warmup=iv; }
        else if(k=="lr_schedule")c.lr_sched=v;
        else if(k=="stop_loss"){ if(s2f(v,fv))c.stop_loss=fv; }
        else if(k=="val_split"){ if(s2f(v,fv))c.val_split=fv; }
        else if(k=="grad_clip"){ if(s2f(v,fv))c.grad_clip=fv; }
        else if(k=="temp"){ if(s2f(v,fv))c.temp=fv; }
        else if(k=="top_p"){ if(s2f(v,fv))c.top_p=fv; }
        else if(k=="topk"){ if(s2i(v,iv))c.topk=iv; }
        else if(k=="rep_window"){ if(s2i(v,iv))c.rep_window=iv; }
        else if(k=="rep_penalty"){ if(s2f(v,fv))c.rep_pen=fv; }
        else if(k=="maxlen"){ if(s2i(v,iv))c.maxlen=iv; }
        else if(k=="seed")c.seed=(unsigned)stoul(v);
        else if(k=="ckpt_interval"){ if(s2i(v,iv))c.ckpt_interval=iv; }
    }
}
static bool parse_args(Config& c,int argc,char** argv){
    for(int i=1;i<argc;++i){
        string k=argv[i]; auto nx=[&](){ return (i+1<argc)?argv[++i]:""; };
        if(k=="--mode")c.mode=nx();
        else if(k=="--data")c.data=nx();
        else if(k=="--model")c.model=nx();
        else if(k=="--ckpt")c.ckpt=nx();
        else if(k=="--prompt"||k=="-p")c.prompt=nx();
        else if(k=="--batch-file")c.batch_file=nx();
        else if(k=="--config")c.config_file=nx();
        else if(k=="--D"||k=="-D"){ int x; if(s2i(nx(),x))c.D=x; }
        else if(k=="--L"||k=="--layers"){ int x; if(s2i(nx(),x))c.L=x; }
        else if(k=="--nq"){ int x; if(s2i(nx(),x))c.nq=x; }
        else if(k=="--nkv"){ int x; if(s2i(nx(),x))c.nkv=x; }
        else if(k=="--ctx"){ int x; if(s2i(nx(),x))c.ctx=x; }
        else if(k=="--infer-maxt"){ int x; if(s2i(nx(),x))c.infer_maxT=x; }
        else if(k=="--dropout"){ float x; if(s2f(nx(),x))c.dropout=x; }
        else if(k=="--epochs"||k=="-e"){ int x; if(s2i(nx(),x))c.epochs=x; }
        else if(k=="--batch"||k=="-b"){ int x; if(s2i(nx(),x))c.batch=x; }
        else if(k=="--grad-accum"){ int x; if(s2i(nx(),x))c.grad_accum=x; }
        else if(k=="--lr"){ float x; if(s2f(nx(),x)){ c.lr=x; c.lr_override=1; } }
        else if(k=="--min-lr"){ float x; if(s2f(nx(),x))c.min_lr=x; }
        else if(k=="--weight-decay"){ float x; if(s2f(nx(),x))c.weight_decay=x; }
        else if(k=="--warmup"){ int x; if(s2i(nx(),x))c.warmup=x; }
        else if(k=="--lr-schedule")c.lr_sched=nx();
        else if(k=="--stop-loss"){ float x; if(s2f(nx(),x))c.stop_loss=x; }
        else if(k=="--val-split"){ float x; if(s2f(nx(),x))c.val_split=x; }
        else if(k=="--grad-clip"){ float x; if(s2f(nx(),x))c.grad_clip=x; }
        else if(k=="--temp"||k=="-t"){ float x; if(s2f(nx(),x))c.temp=x; }
        else if(k=="--top-p"){ float x; if(s2f(nx(),x))c.top_p=x; }
        else if(k=="--topk"||k=="-k"){ int x; if(s2i(nx(),x))c.topk=x; }
        else if(k=="--rep-window"){ int x; if(s2i(nx(),x))c.rep_window=x; }
        else if(k=="--rep-penalty"){ float x; if(s2f(nx(),x))c.rep_pen=x; }
        else if(k=="--maxlen"||k=="-n"){ int x; if(s2i(nx(),x))c.maxlen=x; }
        else if(k=="--one-token")c.one_token=1;
        else if(k=="--seed"){ int x; if(s2i(nx(),x))c.seed=(unsigned)x; }
        else if(k=="--resume")c.resume=1;
        else if(k=="--greedy")c.greedy=1;
        else if(k=="--ckpt-interval"){ int x; if(s2i(nx(),x))c.ckpt_interval=x; }
        else if(k=="--help"||k=="-h"){ cout<<"mode: train|gen|next|chat|batch\n"; return 0; }
        else { cerr<<"未知: "<<k<<"\n"; return 0; }
    }
    return 1;
}

struct AdamW{
    VF m,v; float lr=1e-3f,base_lr=1e-3f,b1=0.9f,b2=0.999f,eps=1e-8f,wd=0.1f,bc1=1.f,bc2=1.f; long long t=0;
    void init(size_t n,float l,float B1,float B2,float E,float W){ m.assign(n,0.f); v.assign(n,0.f); b1=B1; b2=B2; eps=E; wd=W; bc1=1.f-b1; bc2=1.f-b2; t=0; lr=base_lr=l; }
    void step(float* p,const float* g,size_t n){
        ++t; float lt=lr/bc1, ib=1.f/bc2;
        bc1=b1*bc1+(1.f-b1); bc2=b2*bc2+(1.f-b2);
        size_t i=0;
        float32x4_t b1v=vdupq_n_f32(b1),b2v=vdupq_n_f32(b2),o1=vdupq_n_f32(1.f-b1),o2=vdupq_n_f32(1.f-b2),lv=vdupq_n_f32(lt),ic=vdupq_n_f32(ib),ev=vdupq_n_f32(eps),wv=vdupq_n_f32(wd*lr);
        for(;i+4<=n;i+=4){
            float32x4_t m4=vld1q_f32(&m[i]),v4=vld1q_f32(&v[i]),g4=vld1q_f32(&g[i]);
            m4=vfmaq_f32(vmulq_f32(b1v,m4),o1,g4); v4=vfmaq_f32(vmulq_f32(b2v,v4),o2,vmulq_f32(g4,g4));
            vst1q_f32(&m[i],m4); vst1q_f32(&v[i],v4);
            float32x4_t p4=vld1q_f32(&p[i]),den=vaddq_f32(vsqrtq_f32(vmulq_f32(v4,ic)),ev);
            float32x4_t u=vmulq_f32(lv,vdivq_f32(m4,den)); u=vfmaq_f32(u,wv,p4);
            vst1q_f32(&p[i],vsubq_f32(p4,u));
        }
        for(;i<n;++i){ m[i]=b1*m[i]+(1.f-b1)*g[i]; v[i]=b2*v[i]+(1.f-b2)*g[i]*g[i]; p[i]-=lt*m[i]/(sqrtf(v[i]*ib)+eps)+wd*lr*p[i]; }
    }
};
struct LRSched{
    float base_lr=1e-3f,min_lr=1e-5f; long long warmup=100,total=1000; string kind="cosine";
    float get(long long s) const {
        if(s<warmup){ if(warmup<=0)return base_lr; return base_lr*(float)(s+1)/warmup; }
        long long x=s-warmup,T=total-warmup; if(T<=0)return base_lr;
        float p=(float)x/T; if(p>1.f)p=1.f;
        if(kind=="cosine"){ float c=0.5f*(1.f+cosf(3.14159265f*p)); return min_lr+(base_lr-min_lr)*c; }
        if(kind=="linear")return min_lr+(base_lr-min_lr)*(1.f-p);
        if(kind=="exp")return base_lr*powf(min_lr/base_lr,p);
        return base_lr;
    }
};

static void rms_f(const float* x,const float* g,int T,int D,float* y,float* xh,float* rr){
    for(int t=0;t<T;++t){
        const float* xp=x+(size_t)t*D; float* xhp=xh+(size_t)t*D; float* yp=y+(size_t)t*D;
        float32x4_t a=vdupq_n_f32(0); int d=0;
        for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(xp+d); a=vfmaq_f32(a,v,v); }
        float m=hs4(a); for(;d<D;++d)m+=xp[d]*xp[d]; m/=D;
        float r=1.f/sqrtf(m+1e-6f); rr[t]=r;
        for(int i=0;i<D;++i){ xhp[i]=xp[i]*r; yp[i]=xhp[i]*g[i]; }
    }
}
static void rms_b(const float* dy,const float* g,const float* xh,const float* rr,int T,int D,float* dx,float* dg){
    for(int t=0;t<T;++t){
        const float* xhp=xh+(size_t)t*D; const float* dyp=dy+(size_t)t*D; float* dxp=dx+(size_t)t*D;
        float r=rr[t]; float32x4_t ac=vdupq_n_f32(0); int d=0;
        for(;d+4<=D;d+=4){
            float32x4_t dyv=vld1q_f32(dyp+d),gv=vld1q_f32(g+d),xhv=vld1q_f32(xhp+d);
            float32x4_t ge=vmulq_f32(dyv,gv); vst1q_f32(dxp+d,ge); ac=vfmaq_f32(ac,ge,xhv);
            float32x4_t dgv=vld1q_f32(dg+d); vst1q_f32(dg+d,vfmaq_f32(dgv,dyv,xhv));
        }
        float s=hs4(ac);
        for(;d<D;++d){ float ge=dyp[d]*g[d]; dxp[d]=ge; s+=ge*xhp[d]; dg[d]+=dyp[d]*xhp[d]; }
        s/=D;
        for(int i=0;i<D;++i)dxp[i]=r*(dxp[i]-xhp[i]*s);
    }
}

struct RoPE{
    int Dh=0,maxT=0; float th=10000.f; VF c,s;
    void init(int D_,int M_,float t_){
        Dh=D_; maxT=M_; th=t_; int h=D_/2;
        c.assign((size_t)M_*h,0.f); s.assign((size_t)M_*h,0.f);
        for(int p=0;p<M_;++p)for(int i=0;i<h;++i){
            float fr=1.f/powf(th,2.f*i/Dh);
            float aa=(float)p*fr; c[(size_t)p*h+i]=cosf(aa); s[(size_t)p*h+i]=sinf(aa);
        }
    }
    void apply(float* x,int p) const {
        int h=Dh/2; if(p>=maxT)p=maxT-1;
        const float* cc=c.data()+(size_t)p*h; const float* ss=s.data()+(size_t)p*h;
        for(int i=0;i<h;++i){ float a=x[2*i],b=x[2*i+1]; x[2*i]=a*cc[i]-b*ss[i]; x[2*i+1]=a*ss[i]+b*cc[i]; }
    }
    void applyb(float* x,int p) const {
        int h=Dh/2; if(p>=maxT)p=maxT-1;
        const float* cc=c.data()+(size_t)p*h; const float* ss=s.data()+(size_t)p*h;
        for(int i=0;i<h;++i){ float a=x[2*i],b=x[2*i+1]; x[2*i]=a*cc[i]+b*ss[i]; x[2*i+1]=-a*ss[i]+b*cc[i]; }
    }
};

struct W{
    VF v,g; AdamW a;
    void init(size_t n,float lr,float b1,float b2,float e,float wd){ v.assign(n,0.f); g.assign(n,0.f); a.init(n,lr,b1,b2,e,wd); }
    void zg(){ fill(g.begin(),g.end(),0.f); }
    float* p(){ return v.data(); } const float* p() const { return v.data(); }
    float* gp(){ return g.data(); } const float* gp() const { return g.data(); }
    size_t n() const { return v.size(); }
};
struct Dropout{
    float p=0.f; VF mk;
    void fwd(float* x,int n,mt19937& g){ if(p<=0.f)return; if((int)mk.size()!=n)mk.resize(n);
        bernoulli_distribution b(p); float s=1.f/(1.f-p);
        for(int i=0;i<n;++i){ bool k=!b(g); mk[i]=k?s:0.f; x[i]*=mk[i]; } }
    void bwd(float* dx,int n){ if(p<=0.f)return; for(int i=0;i<n;++i)dx[i]*=mk[i]; }
};

struct SwiGLU{
    int D=0,Df=0; W W1,W3,W2; VF b1,b3,b2,db1,db3,db2,X,H1,H3,G,dGb,dH1b,dH3b;
    void init(int d,int df,mt19937& g,float lr,const Config& c){
        D=d; Df=df;
        W1.init((size_t)D*Df,lr,c.b1,c.b2,c.eps,0.f); W3.init((size_t)D*Df,lr,c.b1,c.b2,c.eps,0.f); W2.init((size_t)Df*D,lr,c.b1,c.b2,c.eps,0.f);
        normal_distribution<float> d1(0.f,sqrtf(2.f/D)),d2(0.f,sqrtf(2.f/Df));
        for(auto& x:W1.v)x=d1(g); for(auto& x:W3.v)x=d1(g); for(auto& x:W2.v)x=d2(g);
        b1.assign(Df,0.f); b3.assign(Df,0.f); b2.assign(D,0.f); db1.assign(Df,0.f); db3.assign(Df,0.f); db2.assign(D,0.f);
    }
    template<typename F> void fe(F f){ f(W1.a); f(W3.a); f(W2.a); }
    void fwd(const VF& x,VF& y,int T){
        X=x; H1.resize((size_t)T*Df); H3.resize((size_t)T*Df); G.resize((size_t)T*Df);
        for(int t=0;t<T;++t){
            const float* xp=x.data()+(size_t)t*D; float* h1=H1.data()+(size_t)t*Df; float* h3=H3.data()+(size_t)t*Df; float* gp=G.data()+(size_t)t*Df;
            memcpy(h1,b1.data(),Df*4); memcpy(h3,b3.data(),Df*4);
            for(int d=0;d<D;++d){ float xv=xp[d]; const float* w1=W1.p()+(size_t)d*Df; const float* w3=W3.p()+(size_t)d*Df; int j=0; float32x4_t x4=vdupq_n_f32(xv);
                for(;j+4<=Df;j+=4){ vst1q_f32(h1+j,vfmaq_f32(vld1q_f32(h1+j),x4,vld1q_f32(w1+j))); vst1q_f32(h3+j,vfmaq_f32(vld1q_f32(h3+j),x4,vld1q_f32(w3+j))); }
                for(;j<Df;++j){ h1[j]+=xv*w1[j]; h3[j]+=xv*w3[j]; } }
            for(int j=0;j<Df;++j)gp[j]=silu_f(h1[j])*h3[j];
        }
        y.resize((size_t)T*D);
        for(int t=0;t<T;++t){
            const float* gp=G.data()+(size_t)t*Df; float* yp=y.data()+(size_t)t*D; memcpy(yp,b2.data(),D*4);
            for(int j=0;j<Df;++j){ float gv=gp[j]; const float* w2=W2.p()+(size_t)j*D; int d=0; float32x4_t g4=vdupq_n_f32(gv);
                for(;d+4<=D;d+=4)vst1q_f32(yp+d,vfmaq_f32(vld1q_f32(yp+d),g4,vld1q_f32(w2+d)));
                for(;d<D;++d)yp[d]+=gv*w2[d]; }
        }
    }
    void bwd(const VF& dy,VF& dx,int T){
        float* dW2=W2.gp();
        for(int t=0;t<T;++t){
            const float* gp=G.data()+(size_t)t*Df; const float* dyp=dy.data()+(size_t)t*D;
            for(int j=0;j<Df;++j){ float gv=gp[j]; float* g2=dW2+(size_t)j*D; int d=0; float32x4_t g4=vdupq_n_f32(gv);
                for(;d+4<=D;d+=4)vst1q_f32(g2+d,vfmaq_f32(vld1q_f32(g2+d),g4,vld1q_f32(dyp+d)));
                for(;d<D;++d)g2[d]+=gv*dyp[d]; }
            for(int d=0;d<D;++d)db2[d]+=dyp[d];
        }
        if((int)dGb.size()!=T*Df)dGb.resize((size_t)T*Df); VF& dG=dGb;
        for(int t=0;t<T;++t){
            const float* dyp=dy.data()+(size_t)t*D; float* dgp=dG.data()+(size_t)t*Df;
            for(int j=0;j<Df;++j){ const float* w2=W2.p()+(size_t)j*D; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=D;d+=4)a=vfmaq_f32(a,vld1q_f32(dyp+d),vld1q_f32(w2+d));
                float s=hs4(a); for(;d<D;++d)s+=dyp[d]*w2[d]; dgp[j]=s; }
        }
        if((int)dH1b.size()!=T*Df)dH1b.resize((size_t)T*Df); if((int)dH3b.size()!=T*Df)dH3b.resize((size_t)T*Df);
        VF& dH1=dH1b; VF& dH3=dH3b;
        for(int t=0;t<T;++t){
            const float* h1=H1.data()+(size_t)t*Df; const float* h3=H3.data()+(size_t)t*Df; const float* dgp=dG.data()+(size_t)t*Df;
            float* dh1=dH1.data()+(size_t)t*Df; float* dh3=dH3.data()+(size_t)t*Df;
            for(int j=0;j<Df;++j){ dh3[j]=dgp[j]*silu_f(h1[j]); dh1[j]=dgp[j]*h3[j]*silu_g(h1[j]); }
        }
        float* dW1=W1.gp(); float* dW3=W3.gp();
        for(int t=0;t<T;++t){
            const float* xp=X.data()+(size_t)t*D; const float* dh1=dH1.data()+(size_t)t*Df; const float* dh3=dH3.data()+(size_t)t*Df;
            for(int d=0;d<D;++d){ float xv=xp[d]; float* g1=dW1+(size_t)d*Df; float* g3=dW3+(size_t)d*Df; int j=0; float32x4_t x4=vdupq_n_f32(xv);
                for(;j+4<=Df;j+=4){ vst1q_f32(g1+j,vfmaq_f32(vld1q_f32(g1+j),x4,vld1q_f32(dh1+j))); vst1q_f32(g3+j,vfmaq_f32(vld1q_f32(g3+j),x4,vld1q_f32(dh3+j))); }
                for(;j<Df;++j){ g1[j]+=xv*dh1[j]; g3[j]+=xv*dh3[j]; } }
            for(int j=0;j<Df;++j){ db1[j]+=dh1[j]; db3[j]+=dh3[j]; }
        }
        dx.resize((size_t)T*D);
        for(int t=0;t<T;++t){
            const float* dh1=dH1.data()+(size_t)t*Df; const float* dh3=dH3.data()+(size_t)t*Df; float* dxp=dx.data()+(size_t)t*D;
            for(int d=0;d<D;++d){ const float* w1=W1.p()+(size_t)d*Df; const float* w3=W3.p()+(size_t)d*Df; float32x4_t a=vdupq_n_f32(0); int j=0;
                for(;j+4<=Df;j+=4){ a=vfmaq_f32(a,vld1q_f32(dh1+j),vld1q_f32(w1+j)); a=vfmaq_f32(a,vld1q_f32(dh3+j),vld1q_f32(w3+j)); }
                float s=hs4(a); for(;j<Df;++j)s+=dh1[j]*w1[j]+dh3[j]*w3[j]; dxp[d]=s; }
        }
    }
    void zg(){ W1.zg(); W3.zg(); W2.zg(); fill(db1.begin(),db1.end(),0.f); fill(db3.begin(),db3.end(),0.f); fill(db2.begin(),db2.end(),0.f); }
    void ag(){ W1.a.step(W1.p(),W1.gp(),W1.n()); W3.a.step(W3.p(),W3.gp(),W3.n()); W2.a.step(W2.p(),W2.gp(),W2.n());
        for(size_t i=0;i<b1.size();++i)b1[i]-=W1.a.lr*db1[i];
        for(size_t i=0;i<b3.size();++i)b3[i]-=W3.a.lr*db3[i];
        for(size_t i=0;i<b2.size();++i)b2[i]-=W2.a.lr*db2[i]; }
    size_t pc() const { return W1.n()+W3.n()+W2.n()+b1.size()+b3.size()+b2.size(); }
};

struct Attn{
    int D=0,Hq=0,Hkv=0,Dh=0,Dkv=0,Dqkv=0;
    W Wqkv,Wo;
    VF bqkv,bo,dbqkv,dbo,qg,kg,dqg,dkg;
    VF X,Q,K,V,cc,qxh,kxh,qr,kr; vector<VF> ap,dab,dsb; VF dQb,dKb,dVb,dCb;
    void init(int d,int hq,int hkv,mt19937& g,float lr,const Config& c){
        D=d; Hq=hq; Hkv=hkv; Dh=D/Hq; Dkv=Hkv*Dh; Dqkv=(Hq+2*Hkv)*Dh;
        Wqkv.init((size_t)D*Dqkv,lr,c.b1,c.b2,c.eps,0.f);
        Wo.init((size_t)D*D,lr,c.b1,c.b2,c.eps,0.f);
        normal_distribution<float> dd(0.f,sqrtf(2.f/D));
        for(auto& x:Wqkv.v)x=dd(g); for(auto& x:Wo.v)x=dd(g);
        bqkv.assign(Dqkv,0.f); bo.assign(D,0.f); dbqkv.assign(Dqkv,0.f); dbo.assign(D,0.f);
        qg.assign(Dh,1.f); kg.assign(Dh,1.f); dqg.assign(Dh,0.f); dkg.assign(Dh,0.f);
    }
    template<typename F> void fe(F f){ f(Wqkv.a); f(Wo.a); }
    void fwd(const VF& x,VF& y,int T,const RoPE& rp){
        X=x; Q.resize((size_t)T*D); K.resize((size_t)T*Dkv); V.resize((size_t)T*Dkv);
        for(int t=0;t<T;++t){
            const float* xp=x.data()+(size_t)t*D;
            float* qp=Q.data()+(size_t)t*D; float* kp=K.data()+(size_t)t*Dkv; float* vp=V.data()+(size_t)t*Dkv;
            for(int j=0;j<Dqkv;++j){ qp[0]=0; }  // 先清零，下面累加
            for(int j=0;j<Dqkv;++j){ float s=0; (void)s; }
            // 初始化 Q/K/V 为零
            memset(qp,0,D*4); memset(kp,0,Dkv*4); memset(vp,0,Dkv*4);
            for(int d=0;d<D;++d){ float xv=xp[d];
                const float* wqkv=Wqkv.p()+(size_t)d*Dqkv;
                const float* wk=wqkv+D; const float* wv=wk+Dkv;
                int j=0; float32x4_t x4=vdupq_n_f32(xv);
                for(;j+4<=D;j+=4)vst1q_f32(qp+j,vfmaq_f32(vld1q_f32(qp+j),x4,vld1q_f32(wqkv+j)));
                for(;j<D;++j)qp[j]+=xv*wqkv[j];
                j=0;
                for(;j+4<=Dkv;j+=4){ vst1q_f32(kp+j,vfmaq_f32(vld1q_f32(kp+j),x4,vld1q_f32(wk+j))); vst1q_f32(vp+j,vfmaq_f32(vld1q_f32(vp+j),x4,vld1q_f32(wv+j))); }
                for(;j<Dkv;++j){ kp[j]+=xv*wk[j]; vp[j]+=xv*wv[j]; }
            }
            for(int j=0;j<D;++j)qp[j]+=bqkv[j];
            for(int j=0;j<Dkv;++j){ kp[j]+=bqkv[D+j]; vp[j]+=bqkv[D+Dkv+j]; }
        }
        qxh.resize((size_t)T*D); kxh.resize((size_t)T*Dkv); qr.assign((size_t)T*Hq,0.f); kr.assign((size_t)T*Hkv,0.f);
        for(int t=0;t<T;++t){
            for(int h=0;h<Hq;++h){ float* qp=Q.data()+(size_t)t*D+h*Dh; float* xh=qxh.data()+(size_t)t*D+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=Dh;d+=4){ float32x4_t v=vld1q_f32(qp+d); a=vfmaq_f32(a,v,v); }
                float m=hs4(a); for(;d<Dh;++d)m+=qp[d]*qp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f); qr[(size_t)t*Hq+h]=r;
                for(int i=0;i<Dh;++i){ xh[i]=qp[i]*r; qp[i]=xh[i]*qg[i]; } }
            for(int h=0;h<Hkv;++h){ float* kp=K.data()+(size_t)t*Dkv+h*Dh; float* xh=kxh.data()+(size_t)t*Dkv+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=Dh;d+=4){ float32x4_t v=vld1q_f32(kp+d); a=vfmaq_f32(a,v,v); }
                float m=hs4(a); for(;d<Dh;++d)m+=kp[d]*kp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f); kr[(size_t)t*Hkv+h]=r;
                for(int i=0;i<Dh;++i){ xh[i]=kp[i]*r; kp[i]=xh[i]*kg[i]; } }
        }
        for(int h=0;h<Hq;++h)for(int t=0;t<T;++t)rp.apply(Q.data()+(size_t)t*D+h*Dh,t);
        for(int h=0;h<Hkv;++h)for(int t=0;t<T;++t)rp.apply(K.data()+(size_t)t*Dkv+h*Dh,t);
        int gr=Hq/Hkv;
        if((int)ap.size()!=Hq)ap.resize(Hq);
        for(int h=0;h<Hq;++h){ if((size_t)ap[h].size()!=(size_t)T*T)ap[h].assign((size_t)T*T,0.f); }
        if((size_t)cc.size()!=(size_t)T*D)cc.assign((size_t)T*D,0.f); else fill(cc.begin(),cc.end(),0.f);
        float sc=1.f/sqrtf((float)Dh);
        for(int h=0;h<Hq;++h){
            int kh=h/gr; VF& s=ap[h];
            for(int i=0;i<T;++i){ const float* qi=Q.data()+(size_t)i*D+h*Dh;
                for(int j=0;j<=i;++j){ const float* kj=K.data()+(size_t)j*Dkv+kh*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                    for(;d+4<=Dh;d+=4)a=vfmaq_f32(a,vld1q_f32(qi+d),vld1q_f32(kj+d));
                    float ss=hs4(a); for(;d<Dh;++d)ss+=qi[d]*kj[d]; s[(size_t)i*T+j]=ss*sc; }
                for(int j=i+1;j<T;++j)s[(size_t)i*T+j]=-1e30f; }
            for(int i=0;i<T;++i){ float mx=-1e30f; for(int j=0;j<T;++j)mx=max(mx,s[(size_t)i*T+j]);
                float sm=0.f; for(int j=0;j<T;++j){ float e=expf(s[(size_t)i*T+j]-mx); s[(size_t)i*T+j]=e; sm+=e; }
                for(int j=0;j<T;++j)s[(size_t)i*T+j]/=sm; }
            for(int i=0;i<T;++i){ float* ci=cc.data()+(size_t)i*D+h*Dh; for(int d=0;d<Dh;++d)ci[d]=0;
                for(int j=0;j<=i;++j){ float p=s[(size_t)i*T+j]; const float* vj=V.data()+(size_t)j*Dkv+kh*Dh; for(int d=0;d<Dh;++d)ci[d]+=p*vj[d]; } }
        }
        y.resize((size_t)T*D);
        for(int t=0;t<T;++t){ const float* cp=cc.data()+(size_t)t*D; float* yp=y.data()+(size_t)t*D; memcpy(yp,bo.data(),D*4);
            for(int d=0;d<D;++d){ float cv=cp[d]; const float* wr=Wo.p()+(size_t)d*D; int j=0; float32x4_t c4=vdupq_n_f32(cv);
                for(;j+4<=D;j+=4)vst1q_f32(yp+j,vfmaq_f32(vld1q_f32(yp+j),c4,vld1q_f32(wr+j)));
                for(;j<D;++j)yp[j]+=cv*wr[j]; } }
    }
    void bwd(const VF& dy,VF& dx,int T,const RoPE& rp){
        int gr=Hq/Hkv; float* dWo=Wo.gp();
        for(int t=0;t<T;++t){ const float* cp=cc.data()+(size_t)t*D; const float* dyp=dy.data()+(size_t)t*D;
            for(int d=0;d<D;++d){ float cv=cp[d]; float* gw=dWo+(size_t)d*D; int j=0; float32x4_t c4=vdupq_n_f32(cv);
                for(;j+4<=D;j+=4)vst1q_f32(gw+j,vfmaq_f32(vld1q_f32(gw+j),c4,vld1q_f32(dyp+j)));
                for(;j<D;++j)gw[j]+=cv*dyp[j]; }
            for(int d=0;d<D;++d)dbo[d]+=dyp[d]; }
        if((int)dCb.size()!=T*D)dCb.resize((size_t)T*D); VF& dC=dCb;
        for(int t=0;t<T;++t){ const float* dyp=dy.data()+(size_t)t*D; float* dcp=dC.data()+(size_t)t*D;
            for(int d=0;d<D;++d){ const float* wr=Wo.p()+(size_t)d*D; float32x4_t a=vdupq_n_f32(0); int j=0;
                for(;j+4<=D;j+=4)a=vfmaq_f32(a,vld1q_f32(dyp+j),vld1q_f32(wr+j));
                float s=hs4(a); for(;j<D;++j)s+=dyp[j]*wr[j]; dcp[d]=s; } }
        if((int)dQb.size()!=T*D)dQb.assign((size_t)T*D,0.f); else fill(dQb.begin(),dQb.end(),0.f);
        if((int)dKb.size()!=T*Dkv)dKb.assign((size_t)T*Dkv,0.f); else fill(dKb.begin(),dKb.end(),0.f);
        if((int)dVb.size()!=T*Dkv)dVb.assign((size_t)T*Dkv,0.f); else fill(dVb.begin(),dVb.end(),0.f);
        VF& dQ=dQb; VF& dK=dKb; VF& dV=dVb;
        float sc=1.f/sqrtf((float)Dh);
        if((int)dab.size()<Hq)dab.resize(Hq); if((int)dsb.size()<Hq)dsb.resize(Hq);
        for(int h=0;h<Hq;++h){
            int kh=h/gr; const VF& s=ap[h];
            for(int j=0;j<T;++j){ float* dvj=dV.data()+(size_t)j*Dkv+kh*Dh;
                for(int i=j;i<T;++i){ float p=s[(size_t)i*T+j]; const float* dci=dC.data()+(size_t)i*D+h*Dh; for(int d=0;d<Dh;++d)dvj[d]+=p*dci[d]; } }
            if((size_t)dab[h].size()!=(size_t)T*T)dab[h].assign((size_t)T*T,0.f); else fill(dab[h].begin(),dab[h].end(),0.f);
            VF& dA=dab[h];
            for(int i=0;i<T;++i){ const float* dci=dC.data()+(size_t)i*D+h*Dh;
                for(int j=0;j<=i;++j){ const float* vj=V.data()+(size_t)j*Dkv+kh*Dh; float ss=0; for(int d=0;d<Dh;++d)ss+=dci[d]*vj[d]; dA[(size_t)i*T+j]=ss; } }
            if((size_t)dsb[h].size()!=(size_t)T*T)dsb[h].assign((size_t)T*T,0.f); else fill(dsb[h].begin(),dsb[h].end(),0.f);
            VF& dS=dsb[h];
            for(int i=0;i<T;++i){ float s2=0; for(int j=0;j<T;++j)s2+=dA[(size_t)i*T+j]*s[(size_t)i*T+j];
                for(int j=0;j<=i;++j)dS[(size_t)i*T+j]=s[(size_t)i*T+j]*(dA[(size_t)i*T+j]-s2)*sc; }
            for(int i=0;i<T;++i){ float* dqi=dQ.data()+(size_t)i*D+h*Dh; for(int j=0;j<=i;++j){ float ds=dS[(size_t)i*T+j]; const float* kj=K.data()+(size_t)j*Dkv+kh*Dh; for(int d=0;d<Dh;++d)dqi[d]+=ds*kj[d]; } }
            for(int j=0;j<T;++j){ float* dkj=dK.data()+(size_t)j*Dkv+kh*Dh; for(int i=j;i<T;++i){ float ds=dS[(size_t)i*T+j]; const float* qi=Q.data()+(size_t)i*D+h*Dh; for(int d=0;d<Dh;++d)dkj[d]+=ds*qi[d]; } }
        }
        for(int h=0;h<Hq;++h)for(int t=0;t<T;++t)rp.applyb(dQ.data()+(size_t)t*D+h*Dh,t);
        for(int h=0;h<Hkv;++h)for(int t=0;t<T;++t)rp.applyb(dK.data()+(size_t)t*Dkv+h*Dh,t);
        for(int t=0;t<T;++t){
            for(int h=0;h<Hq;++h){ float* dqp=dQ.data()+(size_t)t*D+h*Dh; const float* xh=qxh.data()+(size_t)t*D+h*Dh; float r=qr[(size_t)t*Hq+h];
                for(int i=0;i<Dh;++i)dqg[i]+=dqp[i]*xh[i];
                float s=0; for(int i=0;i<Dh;++i){ float g=dqp[i]*qg[i]; dqp[i]=g; s+=g*xh[i]; }
                float id=1.f/Dh; for(int i=0;i<Dh;++i)dqp[i]=r*(dqp[i]-xh[i]*s*id); }
            for(int h=0;h<Hkv;++h){ float* dkp=dK.data()+(size_t)t*Dkv+h*Dh; const float* xh=kxh.data()+(size_t)t*Dkv+h*Dh; float r=kr[(size_t)t*Hkv+h];
                for(int i=0;i<Dh;++i)dkg[i]+=dkp[i]*xh[i];
                float s=0; for(int i=0;i<Dh;++i){ float g=dkp[i]*kg[i]; dkp[i]=g; s+=g*xh[i]; }
                float id=1.f/Dh; for(int i=0;i<Dh;++i)dkp[i]=r*(dkp[i]-xh[i]*s*id); }
        }
        float* dWqkv=Wqkv.gp();
        for(int t=0;t<T;++t){ const float* xp=X.data()+(size_t)t*D;
            const float* dqp=dQ.data()+(size_t)t*D; const float* dkp=dK.data()+(size_t)t*Dkv; const float* dvp=dV.data()+(size_t)t*Dkv;
            for(int d=0;d<D;++d){ float xv=xp[d];
                float* gwqkv=dWqkv+(size_t)d*Dqkv; float* gk=gwqkv+D; float* gv=gk+Dkv;
                int j=0; float32x4_t x4=vdupq_n_f32(xv);
                for(;j+4<=D;j+=4)vst1q_f32(gwqkv+j,vfmaq_f32(vld1q_f32(gwqkv+j),x4,vld1q_f32(dqp+j)));
                for(;j<D;++j)gwqkv[j]+=xv*dqp[j];
                j=0;
                for(;j+4<=Dkv;j+=4){ vst1q_f32(gk+j,vfmaq_f32(vld1q_f32(gk+j),x4,vld1q_f32(dkp+j))); vst1q_f32(gv+j,vfmaq_f32(vld1q_f32(gv+j),x4,vld1q_f32(dvp+j))); }
                for(;j<Dkv;++j){ gk[j]+=xv*dkp[j]; gv[j]+=xv*dvp[j]; } }
            for(int d=0;d<D;++d)dbqkv[d]+=dqp[d];
            for(int d=0;d<Dkv;++d){ dbqkv[D+d]+=dkp[d]; dbqkv[D+Dkv+d]+=dvp[d]; }
        }
        dx.resize((size_t)T*D,0.f);
        for(int t=0;t<T;++t){ const float* dqp=dQ.data()+(size_t)t*D; const float* dkp=dK.data()+(size_t)t*Dkv; const float* dvp=dV.data()+(size_t)t*Dkv;
            float* dxp=dx.data()+(size_t)t*D;
            for(int d=0;d<D;++d){ const float* wqkv=Wqkv.p()+(size_t)d*Dqkv; const float* wk=wqkv+D; const float* wv=wk+Dkv;
                float32x4_t a=vdupq_n_f32(0); int j=0;
                for(;j+4<=D;j+=4)a=vfmaq_f32(a,vld1q_f32(dqp+j),vld1q_f32(wqkv+j));
                float s=hs4(a); for(;j<D;++j)s+=dqp[j]*wqkv[j];
                float32x4_t a2=vdupq_n_f32(0); int j2=0;
                for(;j2+4<=Dkv;j2+=4){ a2=vfmaq_f32(a2,vld1q_f32(dkp+j2),vld1q_f32(wk+j2)); a2=vfmaq_f32(a2,vld1q_f32(dvp+j2),vld1q_f32(wv+j2)); }
                s+=hs4(a2); for(;j2<Dkv;++j2)s+=dkp[j2]*wk[j2]+dvp[j2]*wv[j2];
                dxp[d]=s; } }
    }
    void zg(){ Wqkv.zg(); Wo.zg();
        fill(dbqkv.begin(),dbqkv.end(),0.f); fill(dbo.begin(),dbo.end(),0.f);
        fill(dqg.begin(),dqg.end(),0.f); fill(dkg.begin(),dkg.end(),0.f); }
    void ag(){ Wqkv.a.step(Wqkv.p(),Wqkv.gp(),Wqkv.n()); Wo.a.step(Wo.p(),Wo.gp(),Wo.n());
        for(size_t i=0;i<bqkv.size();++i)bqkv[i]-=Wqkv.a.lr*dbqkv[i];
        for(size_t i=0;i<bo.size();++i)bo[i]-=Wo.a.lr*dbo[i];
        for(size_t i=0;i<qg.size();++i)qg[i]-=Wqkv.a.lr*dqg[i];
        for(size_t i=0;i<kg.size();++i)kg[i]-=Wqkv.a.lr*dkg[i]; }
    size_t pc() const { return Wqkv.n()+Wo.n()+bqkv.size()+bo.size()+qg.size()+kg.size(); }
};

struct Block{
    int D=0; VF ln1,ln2,dln1,dln2; Attn at; SwiGLU ff;
    VF xn1,xh1,rr1,ao,aa,xn2,xh2,rr2,fo;
    void init(int d,int hq,int hkv,int df,mt19937& g,float lr,const Config& c){
        D=d;
        ln1.assign(D,1.f); dln1.assign(D,0.f); ln2.assign(D,1.f); dln2.assign(D,0.f);
        at.init(D,hq,hkv,g,lr,c); ff.init(D,df,g,lr,c);
    }
    template<typename F> void fe(F f){ at.fe(f); ff.fe(f); }
    void fwd(const VF& x,VF& y,int T,const RoPE& rp){
        xn1.resize((size_t)T*D); xh1.resize((size_t)T*D); rr1.resize(T);
        rms_f(x.data(),ln1.data(),T,D,xn1.data(),xh1.data(),rr1.data());
        at.fwd(xn1,ao,T,rp);
        aa.resize((size_t)T*D);
        for(size_t i=0;i<x.size();++i)aa[i]=x[i]+ao[i];
        xn2.resize((size_t)T*D); xh2.resize((size_t)T*D); rr2.resize(T);
        rms_f(aa.data(),ln2.data(),T,D,xn2.data(),xh2.data(),rr2.data());
        ff.fwd(xn2,fo,T);
        y.resize(x.size());
        for(size_t i=0;i<x.size();++i)y[i]=aa[i]+fo[i];
    }
    void bwd(const VF& dy,VF& dx,int T,const RoPE& rp){
        VF dff,dln2_in,da;
        ff.bwd(dy,dff,T);
        dln2_in.resize(dy.size());
        rms_b(dff.data(),ln2.data(),xh2.data(),rr2.data(),T,D,dln2_in.data(),dln2.data());
        da.resize(dy.size());
        for(size_t i=0;i<dy.size();++i)da[i]=dy[i]+dln2_in[i];
        VF dat,dln1_in;
        at.bwd(da,dat,T,rp);
        dln1_in.resize(dy.size());
        rms_b(dat.data(),ln1.data(),xh1.data(),rr1.data(),T,D,dln1_in.data(),dln1.data());
        dx.resize(dy.size());
        for(size_t i=0;i<dy.size();++i)dx[i]=da[i]+dln1_in[i];
    }
    void zg(){ fill(dln1.begin(),dln1.end(),0.f); fill(dln2.begin(),dln2.end(),0.f); at.zg(); ff.zg(); }
    void ag(){ float bl=at.Wqkv.a.lr;
        for(size_t i=0;i<ln1.size();++i)ln1[i]-=bl*dln1[i];
        for(size_t i=0;i<ln2.size();++i)ln2[i]-=bl*dln2[i];
        at.ag(); ff.ag(); }
    size_t pc() const { return ln1.size()+ln2.size()+at.pc()+ff.pc(); }
};

struct Model{
    int V=0,D=0,L=0,Hq=0,Hkv=0,Df=0,maxT=0;
    VF E,dE; AdamW aE; vector<Block> bl; VF lnf,dlnf;
    VF x,fh,lg,xhf,rrf;
    int kv=0; vector<VF> kK,kV;
    VF s_cur,s_xn1,s_xh1,s_q,s_k,s_v,s_ao,s_ay,s_aa,s_xn2,s_xh2,s_h1,s_h3,s_g,s_fo,s_hf,s_sc;
    RoPE rp; vector<Dropout> de,df; LRSched sch;
    void init(int v,int d,int l,int hq,int hkv,int mt,mt19937& g,const Config& c){
        V=v; D=d; L=l; Hq=hq; Hkv=hkv; maxT=mt;
        if(D%Hq||Hq%Hkv||(D/Hq)%2){ cerr<<"[fatal] D/Hq/Hkv\n"; exit(1); }
        Df=(int)(4.0*D*2.0/3.0); Df=(Df+3)&~3;
        E.assign((size_t)V*D,0.f); normal_distribution<float> dd(0.f,0.02f); for(auto& vv:E)vv=dd(g);
        dE.assign((size_t)V*D,0.f); aE.init(E.size(),c.lr,c.b1,c.b2,c.eps,c.weight_decay);
        bl.clear(); bl.reserve(L);
        for(int i=0;i<L;++i){ Block b; b.init(D,Hq,Hkv,Df,g,c.lr,c); bl.push_back(std::move(b)); }
        lnf.assign(D,1.f); dlnf.assign(D,0.f);
        rp.init(D/Hq,maxT,10000.f);
        de.assign(L,Dropout()); df.assign(L,Dropout());
        for(int i=0;i<L;++i){ de[i].p=c.embed_dropout; df[i].p=c.dropout; }
        sch.base_lr=c.lr; sch.min_lr=c.min_lr; sch.warmup=c.warmup; sch.kind=c.lr_sched;
    }
    size_t pc() const { size_t n=E.size(); for(auto& b:bl)n+=b.pc(); n+=lnf.size(); return n; }
    void set_total(long long t){ sch.total=t; }
    void upd_lr(long long s){ float lr=sch.get(s); aE.lr=lr; auto set=[&](AdamW& a){ a.lr=lr; }; for(auto& b:bl)b.fe(set); }
    void zg(){ fill(dE.begin(),dE.end(),0.f); for(auto& b:bl)b.zg(); fill(dlnf.begin(),dlnf.end(),0.f); }
    void ag(){ aE.step(E.data(),dE.data(),E.size()); for(auto& b:bl)b.ag(); for(size_t i=0;i<lnf.size();++i)lnf[i]-=aE.lr*dlnf[i]; }
    void fwd(const vector<int>& ids,mt19937& g){
        int T=(int)ids.size(); x.resize((size_t)T*D);
        for(int t=0;t<T;++t)memcpy(x.data()+(size_t)t*D,E.data()+(size_t)ids[t]*D,D*4);
        vector<float> cur=x;
        for(int l=0;l<L;++l){ vector<float> out; bl[l].fwd(cur,out,T,rp); if(df[l].p>0.f)df[l].fwd(out.data(),T*D,g); cur=out; }
        fh.resize((size_t)T*D); xhf.resize((size_t)T*D); rrf.resize(T);
        rms_f(cur.data(),lnf.data(),T,D,fh.data(),xhf.data(),rrf.data());
        lg.resize((size_t)T*V);
        for(int t=0;t<T;++t){ const float* hp=fh.data()+(size_t)t*D; float* lp=lg.data()+(size_t)t*V;
            for(int v=0;v<V;++v){ const float* ep=E.data()+(size_t)v*D; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=D;d+=4)a=vfmaq_f32(a,vld1q_f32(hp+d),vld1q_f32(ep+d));
                float s=hs4(a); for(;d<D;++d)s+=hp[d]*ep[d]; lp[v]=s; } }
    }
    float bwd(const vector<int>& ids,const vector<int>& tg){
        int T=(int)ids.size();
        vector<float> dl((size_t)T*V); float loss=0.f;
        for(int t=0;t<T;++t){ const float* lp=lg.data()+(size_t)t*V; float* dp=dl.data()+(size_t)t*V;
            float mx=-1e30f; for(int v=0;v<V;++v)mx=max(mx,lp[v]);
            float sm=0.f; for(int v=0;v<V;++v){ dp[v]=expf(lp[v]-mx); sm+=dp[v]; }
            loss+=-logf(max(dp[tg[t]]/sm,1e-9f));
            float iv=1.f/sm; for(int v=0;v<V;++v)dp[v]*=iv;
            dp[tg[t]]-=1.f; for(int v=0;v<V;++v)dp[v]/=T; }
        loss/=T;
        vector<float> dh((size_t)T*D,0.f);
        for(int t=0;t<T;++t){ const float* dp=dl.data()+(size_t)t*V; float* dhp=dh.data()+(size_t)t*D;
            for(int v=0;v<V;++v){ float dv=dp[v]; const float* ep=E.data()+(size_t)v*D; int d=0; float32x4_t dv4=vdupq_n_f32(dv);
                for(;d+4<=D;d+=4)vst1q_f32(dhp+d,vfmaq_f32(vld1q_f32(dhp+d),dv4,vld1q_f32(ep+d)));
                for(;d<D;++d)dhp[d]+=dv*ep[d]; } }
        for(int t=0;t<T;++t){ const float* hp=fh.data()+(size_t)t*D; const float* dp=dl.data()+(size_t)t*V;
            for(int v=0;v<V;++v){ float dv=dp[v]; float* dep=dE.data()+(size_t)v*D; int d=0; float32x4_t dv4=vdupq_n_f32(dv);
                for(;d+4<=D;d+=4)vst1q_f32(dep+d,vfmaq_f32(vld1q_f32(dep+d),dv4,vld1q_f32(hp+d)));
                for(;d<D;++d)dep[d]+=dv*hp[d]; } }
        VF dfin((size_t)T*D); rms_b(dh.data(),lnf.data(),xhf.data(),rrf.data(),T,D,dfin.data(),dlnf.data());
        vector<float> dcur=dfin;
        for(int l=L-1;l>=0;--l){
            if(df[l].p>0.f)df[l].bwd(dcur.data(),T*D);
            vector<float> dp; bl[l].bwd(dcur,dp,T,rp);
            dcur=dp;
        }
        if(de[0].p>0.f)de[0].bwd(dcur.data(),T*D);
        for(int t=0;t<T;++t){ const float* dp=dcur.data()+(size_t)t*D; float* dep=dE.data()+(size_t)ids[t]*D; for(int d=0;d<D;++d)dep[d]+=dp[d]; }
        return loss;
    }
    void kv_reset(){ kv=0; int Dkv=Hkv*(D/Hq); kK.assign(L,VF((size_t)maxT*Dkv,0.f)); kV.assign(L,VF((size_t)maxT*Dkv,0.f)); }
    const float* step(int id){
        int Dh=D/Hq,Dkv=Hkv*Dh,gr=Hq/Hkv; float sc=1.f/sqrtf((float)Dh);
        auto en=[&](VF& v,int n){ if((int)v.size()!=n)v.resize(n); };
        en(s_cur,D); en(s_xn1,D); en(s_xh1,D); en(s_q,D); en(s_k,Dkv); en(s_v,Dkv);
        en(s_ao,D); en(s_ay,D); en(s_aa,D); en(s_xn2,D); en(s_xh2,D); en(s_hf,D);
        VF& cur=s_cur; memcpy(cur.data(),E.data()+(size_t)id*D,D*4);
        for(int l=0;l<L;++l){
            Block& b=bl[l];
            VF& xn1=s_xn1; VF& xh1=s_xh1;
            { float32x4_t a=vdupq_n_f32(0); int d=0;
              for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(cur.data()+d); a=vfmaq_f32(a,v,v); }
              float m=hs4(a); for(;d<D;++d)m+=cur[d]*cur[d]; m/=D;
              float r=1.f/sqrtf(m+1e-6f);
              for(int i=0;i<D;++i){ xh1[i]=cur[i]*r; xn1[i]=xh1[i]*b.ln1[i]; } }
            VF& ao=s_ao; fill(ao.begin(),ao.end(),0.f);
            VF& q=s_q; VF& k=s_k; VF& v=s_v;
            memset(q.data(),0,D*4); memset(k.data(),0,Dkv*4); memset(v.data(),0,Dkv*4);
            for(int d=0;d<D;++d){ float xv=xn1[d];
                const float* wqkv=b.at.Wqkv.p()+(size_t)d*b.at.Dqkv;
                const float* wk=wqkv+D; const float* wv=wk+Dkv;
                int j=0; float32x4_t x4=vdupq_n_f32(xv);
                for(;j+4<=D;j+=4)vst1q_f32(q.data()+j,vfmaq_f32(vld1q_f32(q.data()+j),x4,vld1q_f32(wqkv+j)));
                for(;j<D;++j)q[j]+=xv*wqkv[j];
                j=0;
                for(;j+4<=Dkv;j+=4){ vst1q_f32(k.data()+j,vfmaq_f32(vld1q_f32(k.data()+j),x4,vld1q_f32(wk+j))); vst1q_f32(v.data()+j,vfmaq_f32(vld1q_f32(v.data()+j),x4,vld1q_f32(wv+j))); }
                for(;j<Dkv;++j){ k[j]+=xv*wk[j]; v[j]+=xv*wv[j]; } }
            for(int j=0;j<D;++j)q[j]+=b.at.bqkv[j];
            for(int j=0;j<Dkv;++j){ k[j]+=b.at.bqkv[D+j]; v[j]+=b.at.bqkv[D+Dkv+j]; }
            for(int h=0;h<Hq;++h){ float* qp=q.data()+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=Dh;d+=4){ float32x4_t vv=vld1q_f32(qp+d); a=vfmaq_f32(a,vv,vv); }
                float m=hs4(a); for(;d<Dh;++d)m+=qp[d]*qp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f);
                for(int i=0;i<Dh;++i)qp[i]=qp[i]*r*b.at.qg[i]; }
            for(int h=0;h<Hkv;++h){ float* kp=k.data()+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=Dh;d+=4){ float32x4_t vv=vld1q_f32(kp+d); a=vfmaq_f32(a,vv,vv); }
                float m=hs4(a); for(;d<Dh;++d)m+=kp[d]*kp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f);
                for(int i=0;i<Dh;++i)kp[i]=kp[i]*r*b.at.kg[i]; }
            for(int h=0;h<Hq;++h)rp.apply(q.data()+h*Dh,kv);
            for(int h=0;h<Hkv;++h)rp.apply(k.data()+h*Dh,kv);
            float* kc=kK[l].data()+(size_t)kv*Dkv; float* vc=kV[l].data()+(size_t)kv*Dkv;
            memcpy(kc,k.data(),Dkv*4); memcpy(vc,v.data(),Dkv*4);
            int Tf=kv+1;
            for(int h=0;h<Hq;++h){ int kh=h/gr;
                if((int)s_sc.size()<Tf)s_sc.resize(Tf); VF& s=s_sc; const float* qh=q.data()+h*Dh; const float* kb=kK[l].data(); const float* vb=kV[l].data();
                for(int j=0;j<Tf;++j){ const float* kj=kb+(size_t)j*Dkv+kh*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                    for(;d+4<=Dh;d+=4)a=vfmaq_f32(a,vld1q_f32(qh+d),vld1q_f32(kj+d));
                    float ss=hs4(a); for(;d<Dh;++d)ss+=qh[d]*kj[d]; s[j]=ss*sc; }
                float mx=-1e30f; for(int j=0;j<Tf;++j)mx=max(mx,s[j]);
                float sm=0.f; for(int j=0;j<Tf;++j){ s[j]=expf(s[j]-mx); sm+=s[j]; }
                for(int j=0;j<Tf;++j)s[j]/=sm;
                float* oh=ao.data()+h*Dh;
                for(int j=0;j<Tf;++j){ const float* vj=vb+(size_t)j*Dkv+kh*Dh; float p=s[j]; for(int d=0;d<Dh;++d)oh[d]+=p*vj[d]; } }
            VF& ay=s_ay; memcpy(ay.data(),b.at.bo.data(),D*4);
            for(int d=0;d<D;++d){ float cv=ao[d]; const float* wr=b.at.Wo.p()+(size_t)d*D; int j=0; float32x4_t c4=vdupq_n_f32(cv);
                for(;j+4<=D;j+=4)vst1q_f32(ay.data()+j,vfmaq_f32(vld1q_f32(ay.data()+j),c4,vld1q_f32(wr+j)));
                for(;j<D;++j)ay[j]+=cv*wr[j]; }
            for(int d=0;d<D;++d)ao[d]=ay[d];
            VF& aa=s_aa; for(int d=0;d<D;++d)aa[d]=cur[d]+ao[d];
            VF& xn2=s_xn2; VF& xh2=s_xh2;
            { float32x4_t a=vdupq_n_f32(0); int d=0;
              for(;d+4<=D;d+=4){ float32x4_t vv=vld1q_f32(aa.data()+d); a=vfmaq_f32(a,vv,vv); }
              float m=hs4(a); for(;d<D;++d)m+=aa[d]*aa[d]; m/=D;
              float r=1.f/sqrtf(m+1e-6f);
              for(int i=0;i<D;++i){ xh2[i]=aa[i]*r; xn2[i]=xh2[i]*b.ln2[i]; } }
            if((int)s_fo.size()!=D)s_fo.resize(D); VF& fo=s_fo;
            int Df=b.ff.Df;
            if((int)s_h1.size()!=Df)s_h1.resize(Df); if((int)s_h3.size()!=Df)s_h3.resize(Df); if((int)s_g.size()!=Df)s_g.resize(Df);
            VF& h1=s_h1; VF& h3=s_h3; VF& g_=s_g;
            memcpy(h1.data(),b.ff.b1.data(),Df*4); memcpy(h3.data(),b.ff.b3.data(),Df*4);
            for(int d=0;d<D;++d){ float xv=xn2[d];
                const float* w1=b.ff.W1.p()+(size_t)d*Df; const float* w3=b.ff.W3.p()+(size_t)d*Df;
                int j=0; float32x4_t x4=vdupq_n_f32(xv);
                for(;j+4<=Df;j+=4){ vst1q_f32(h1.data()+j,vfmaq_f32(vld1q_f32(h1.data()+j),x4,vld1q_f32(w1+j))); vst1q_f32(h3.data()+j,vfmaq_f32(vld1q_f32(h3.data()+j),x4,vld1q_f32(w3+j))); }
                for(;j<Df;++j){ h1[j]+=xv*w1[j]; h3[j]+=xv*w3[j]; } }
            for(int j=0;j<Df;++j)g_[j]=silu_f(h1[j])*h3[j];
            memcpy(fo.data(),b.ff.b2.data(),D*4);
            for(int j=0;j<Df;++j){ float gv=g_[j]; const float* w2=b.ff.W2.p()+(size_t)j*D;
                int d=0; float32x4_t g4=vdupq_n_f32(gv);
                for(;d+4<=D;d+=4)vst1q_f32(fo.data()+d,vfmaq_f32(vld1q_f32(fo.data()+d),g4,vld1q_f32(w2+d)));
                for(;d<D;++d)fo[d]+=gv*w2[d]; }
            for(int d=0;d<D;++d)cur[d]=aa[d]+fo[d];
        }
        float32x4_t a=vdupq_n_f32(0); int d=0;
        for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(cur.data()+d); a=vfmaq_f32(a,v,v); }
        float m=hs4(a); for(;d<D;++d)m+=cur[d]*cur[d]; m/=D;
        float r=1.f/sqrtf(m+1e-6f); VF& hf=s_hf;
        for(int i=0;i<D;++i)hf[i]=cur[i]*r*lnf[i];
        lg.resize(V);
        for(int v=0;v<V;++v){ const float* ep=E.data()+(size_t)v*D; float32x4_t ac=vdupq_n_f32(0); int dd=0;
            for(;dd+4<=D;dd+=4)ac=vfmaq_f32(ac,vld1q_f32(hf.data()+dd),vld1q_f32(ep+dd));
            float s=hs4(ac); for(;dd<D;++dd)s+=hf[dd]*ep[dd]; lg[v]=s; }
        kv++;
        return lg.data();
    }
};

struct Sampler{
    vector<int> idx; VF p,pk;
    Sampler(int V){ idx.resize(V); p.resize(V); pk.resize(V); }
    int sample(const float* l,int V,float t,int tk,float tp,const vector<int>& h,int rw,float rp,mt19937& g,bool gr){
        for(int v=0;v<V;++v)p[v]=l[v];
        if(rw>0&&rp>1.f){ float pen=logf(rp); int st=max(0,(int)h.size()-rw);
            for(int i=st;i<(int)h.size();++i){ int tt=h[i]; if(tt>=0&&tt<V)p[tt]-=pen; } }
        if(gr){ int b=0; float bv=p[0]; for(int v=1;v<V;++v)if(p[v]>bv){ bv=p[v]; b=v; } return b; }
        float mx=-1e30f; for(int v=0;v<V;++v)mx=max(mx,p[v]);
        float sm=0.f; for(int v=0;v<V;++v){ p[v]=expf((p[v]-mx)/t); sm+=p[v]; }
        if(!(sm>0.f))return 0;
        for(int v=0;v<V;++v)p[v]/=sm;
        if(tk>0&&tk<V){
            for(int v=0;v<V;++v)idx[v]=v;
            partial_sort(idx.begin(),idx.begin()+tk,idx.end(),[&](int a,int b){ return p[a]>p[b]; });
            int kp=tk;
            if(tp>0.f&&tp<1.f){
                float s2=0.f; for(int i=0;i<tk;++i)s2+=p[idx[i]];
                float cu=0.f; kp=0;
                for(int i=0;i<tk;++i){ cu+=p[idx[i]]/s2; ++kp; if(cu>=tp)break; }
                if(kp<1)kp=1;
            }
            float s3=0.f; for(int i=0;i<kp;++i){ pk[i]=p[idx[i]]; s3+=pk[i]; }
            for(int i=0;i<kp;++i)pk[i]/=s3;
            discrete_distribution<int> dist(pk.begin(),pk.begin()+kp);
            return idx[dist(g)];
        }
        discrete_distribution<int> dist(p.begin(),p.end());
        return dist(g);
    }
};

static const char MG[4]={'T','R','N','S'};
static const uint32_t VER=5;
static const char CM[4]={'T','C','K','P'};
static const uint32_t CV=7;
static void wv(ofstream& f,const VF& v){ int32_t n=(int32_t)v.size(); f.write((char*)&n,4); if(n>0)f.write((const char*)v.data(),n*4); }
static void rv(ifstream& f,VF& v){ int32_t n; f.read((char*)&n,4); v.resize(n); if(n>0)f.read((char*)v.data(),n*4); }
static void wW(ofstream& f,const W& w){ wv(f,w.v); wv(f,w.a.m); wv(f,w.a.v); f.write((char*)&w.a.t,8); f.write((char*)&w.a.bc1,4); f.write((char*)&w.a.bc2,4); f.write((char*)&w.a.base_lr,4); f.write((char*)&w.a.wd,4); }
static void rW(ifstream& f,W& w){ rv(f,w.v); rv(f,w.a.m); rv(f,w.a.v); f.read((char*)&w.a.t,8); f.read((char*)&w.a.bc1,4); f.read((char*)&w.a.bc2,4); f.read((char*)&w.a.base_lr,4); f.read((char*)&w.a.wd,4); w.a.lr=w.a.base_lr; w.g.assign(w.v.size(),0.f); }
static void wvstr(ofstream& f,const vector<string>& vs){ int32_t N=(int32_t)vs.size(); f.write((char*)&N,4);
    for(auto& s:vs){ int32_t L=(int32_t)s.size(); f.write((char*)&L,4); f.write(s.data(),L); } }
static void rvstr(ifstream& f,vector<string>& vs,unordered_map<string,int>& c2i){ int32_t N; f.read((char*)&N,4); vs.resize(N); c2i.clear();
    for(int i=0;i<N;++i){ int32_t L; f.read((char*)&L,4); vs[i].resize(L); f.read(&vs[i][0],L); c2i[vs[i]]=i; } }
static void wblk(ofstream& f,const Block& b){
    wv(f,b.ln1); wv(f,b.ln2);
    wW(f,b.at.Wqkv); wW(f,b.at.Wo);
    wv(f,b.at.bqkv); wv(f,b.at.bo); wv(f,b.at.qg); wv(f,b.at.kg);
    wW(f,b.ff.W1); wW(f,b.ff.W3); wW(f,b.ff.W2);
    wv(f,b.ff.b1); wv(f,b.ff.b3); wv(f,b.ff.b2);
}
static void rblk(ifstream& f,Block& b){
    rv(f,b.ln1); rv(f,b.ln2);
    rW(f,b.at.Wqkv); rW(f,b.at.Wo);
    rv(f,b.at.bqkv); rv(f,b.at.bo); rv(f,b.at.qg); rv(f,b.at.kg);
    rW(f,b.ff.W1); rW(f,b.ff.W3); rW(f,b.ff.W2);
    rv(f,b.ff.b1); rv(f,b.ff.b3); rv(f,b.ff.b2);
}
static bool savem(const string& p,const Model& m,const Vocab& voc){
    ofstream f(p,ios::binary); if(!f)return 0;
    f.write(MG,4); uint32_t v=VER; f.write((char*)&v,4);
    int32_t V=m.V,D=m.D,L=m.L,Hq=m.Hq,Hkv=m.Hkv,mt=m.maxT;
    f.write((char*)&V,4); f.write((char*)&D,4); f.write((char*)&L,4); f.write((char*)&Hq,4); f.write((char*)&Hkv,4); f.write((char*)&mt,4);
    wvstr(f,voc.chars);
    wv(f,m.E); wv(f,m.lnf);
    for(const auto& b:m.bl)wblk(f,b);
    return 1;
}
static bool loadm(const string& p,Model& m,Vocab& voc,const Config& c){
    ifstream f(p,ios::binary); if(!f)return 0;
    char mg[4]; f.read(mg,4); if(memcmp(mg,MG,4)){ cerr<<"magic\n"; return 0; }
    uint32_t v; f.read((char*)&v,4); if(v!=VER){ cerr<<"版本 "<<v<<"\n"; return 0; }
    int32_t V,D,L,Hq,Hkv,mt; f.read((char*)&V,4); f.read((char*)&D,4); f.read((char*)&L,4); f.read((char*)&Hq,4); f.read((char*)&Hkv,4); f.read((char*)&mt,4);
    rvstr(f,voc.chars,voc.c2i);
    mt19937 dg(0); m.init(V,D,L,Hq,Hkv,mt,dg,c);
    rv(f,m.E); rv(f,m.lnf);
    for(auto& b:m.bl)rblk(f,b);
    return 1;
}
static bool saveck(const string& p,const Model& m,int ep,long long st,float bl,mt19937& g){
    string tmp=p+".tmp"; ofstream f(tmp,ios::binary); if(!f)return 0;
    f.write(CM,4); uint32_t v=CV; f.write((char*)&v,4);
    int32_t V=m.V,D=m.D,L=m.L,Hq=m.Hq,Hkv=m.Hkv,mt=m.maxT;
    f.write((char*)&V,4); f.write((char*)&D,4); f.write((char*)&L,4); f.write((char*)&Hq,4); f.write((char*)&Hkv,4); f.write((char*)&mt,4);
    int32_t e=ep; f.write((char*)&e,4); f.write((char*)&st,8); f.write((char*)&bl,4);
    ostringstream oss; oss<<g; string rs=oss.str(); int32_t rn=(int32_t)rs.size(); f.write((char*)&rn,4); f.write(rs.data(),rn);
    wv(f,m.E); wv(f,m.aE.m); wv(f,m.aE.v);
    f.write((char*)&m.aE.t,8); f.write((char*)&m.aE.bc1,4); f.write((char*)&m.aE.bc2,4); f.write((char*)&m.aE.base_lr,4);
    wv(f,m.lnf);
    for(const auto& b:m.bl)wblk(f,b);
    f.flush(); if(!f.good()){ cerr<<"ckpt 写失败\n"; f.close(); std::remove(tmp.c_str()); return 0; }
    f.close();
    if(std::rename(tmp.c_str(),p.c_str())!=0){
        ifstream s(tmp,ios::binary); ofstream d(p,ios::binary);
        if(!s||!d){ std::remove(tmp.c_str()); return 0; }
        d<<s.rdbuf(); std::remove(tmp.c_str());
    }
    return 1;
}
static bool loadck(const string& p,Model& m,Config& c,int& ep,long long& st,float& bl,mt19937& g){
    ifstream f(p,ios::binary); if(!f)return 0;
    char mg[4]; f.read(mg,4); if(memcmp(mg,CM,4)){ cerr<<"ckpt magic\n"; return 0; }
    uint32_t v; f.read((char*)&v,4); if(v!=CV){ cerr<<"ckpt 版本\n"; return 0; }
    int32_t V,D,L,Hq,Hkv,mt; f.read((char*)&V,4); f.read((char*)&D,4); f.read((char*)&L,4); f.read((char*)&Hq,4); f.read((char*)&Hkv,4); f.read((char*)&mt,4);
    f.read((char*)&ep,4); f.read((char*)&st,8); f.read((char*)&bl,4);
    int32_t rn; f.read((char*)&rn,4); string rs(rn,0); f.read(&rs[0],rn); istringstream iss(rs); iss>>g;
    mt19937 dg(0); m.init(V,D,L,Hq,Hkv,mt,dg,c);
    rv(f,m.E); rv(f,m.aE.m); rv(f,m.aE.v);
    f.read((char*)&m.aE.t,8); f.read((char*)&m.aE.bc1,4); f.read((char*)&m.aE.bc2,4); f.read((char*)&m.aE.base_lr,4); m.aE.lr=m.aE.base_lr;
    rv(f,m.lnf);
    for(auto& b:m.bl)rblk(f,b);
    return 1;
}

static string lt(const string& p){ ifstream f(p,ios::binary); if(!f)return ""; return string((istreambuf_iterator<char>(f)),istreambuf_iterator<char>()); }
static bool iseos(const string& s){ return s.size()>=5&&s.compare(s.size()-5,5,"<eos>")==0; }

static void clip_grads(Model& m,const Config& c){
    double n2=0;
    auto acc=[&](const VF& g){ for(float x:g)n2+=(double)x*x; };
    for(auto& b:m.bl){
        acc(b.at.Wqkv.g); acc(b.at.Wo.g); acc(b.at.dbqkv); acc(b.at.dbo); acc(b.at.dqg); acc(b.at.dkg);
        acc(b.ff.W1.g); acc(b.ff.W3.g); acc(b.ff.W2.g); acc(b.ff.db1); acc(b.ff.db3); acc(b.ff.db2);
        acc(b.dln1); acc(b.dln2);
    }
    for(float x:m.dE)n2+=(double)x*x;
    for(float x:m.dlnf)n2+=(double)x*x;
    float gn=sqrtf((float)n2);
    if(gn<=c.grad_clip)return;
    float s=c.grad_clip/gn;
    auto mul=[&](VF& g){ for(float& x:g)x*=s; };
    for(auto& b:m.bl){
        mul(b.at.Wqkv.g); mul(b.at.Wo.g); mul(b.at.dbqkv); mul(b.at.dbo); mul(b.at.dqg); mul(b.at.dkg);
        mul(b.ff.W1.g); mul(b.ff.W3.g); mul(b.ff.W2.g); mul(b.ff.db1); mul(b.ff.db3); mul(b.ff.db2);
        mul(b.dln1); mul(b.dln2);
    }
    for(float& x:m.dE)x*=s;
    for(float& x:m.dlnf)x*=s;
}

static int run_train(Config& cfg){
    string text=lt(cfg.data);
    if(text.empty()){ cerr<<"打不开 "<<cfg.data<<"\n"; return 1; }
    Vocab voc; voc.build(text);
    int V=(int)voc.chars.size();
    vector<int> ids=voc.encode(text);
    cout<<"字节: "<<text.size()<<" 词表: "<<V<<" 序列: "<<ids.size()<<"\n";
    vector<vector<int>> dialogs;
    {
        vector<int> cur;
        for(int id:ids){ cur.push_back(id); if(voc.chars[id]=="\n"){ dialogs.push_back(std::move(cur)); cur.clear(); } }
        if(!cur.empty())dialogs.push_back(std::move(cur));
    }
    cout<<"对话数: "<<dialogs.size()<<"\n";
    mt19937 g(cfg.seed); Model m;
    int sep=0; long long stc=0; float bl=1e30f;
    int amT=cfg.infer_maxT>0?cfg.infer_maxT:(cfg.ctx+cfg.maxlen+16); if(amT<cfg.ctx)amT=cfg.ctx;
    if(cfg.resume){
        if(loadck(cfg.ckpt,m,cfg,sep,stc,bl,g)){
            V=m.V; cout<<"恢复 V="<<V<<" D="<<m.D<<" L="<<m.L<<" maxT="<<m.maxT<<" ep="<<sep<<" step="<<stc<<" best="<<bl<<"\n";
        } else { cout<<"从头\n"; m.init(V,cfg.D,cfg.L,cfg.nq,cfg.nkv,amT,g,cfg); }
    } else m.init(V,cfg.D,cfg.L,cfg.nq,cfg.nkv,amT,g,cfg);
    if(m.maxT<cfg.ctx){ cerr<<"[err] maxT<ctx\n"; return 1; }
    if(cfg.lr_override){ m.aE.base_lr=m.aE.lr=cfg.lr; m.sch.base_lr=cfg.lr; }
    cout<<"参数: "<<m.pc()<<" D="<<m.D<<" L="<<m.L<<" nq="<<m.Hq<<" nkv="<<m.Hkv<<" ctx="<<cfg.ctx<<"\n";
    int ndial=(int)dialogs.size();
    if(ndial<2){ cerr<<"对话太少\n"; return 1; }
    vector<int> ord(ndial); iota(ord.begin(),ord.end(),0); shuffle(ord.begin(),ord.end(),g);
    int nv=(int)(ndial*cfg.val_split); if(nv<1&&cfg.val_split>0)nv=1; if(nv>=ndial)nv=ndial-1;
    int nt=ndial-nv;
    vector<int> tid(ord.begin(),ord.begin()+nt), vid(ord.begin()+nt,ord.end());
    cout<<"训练对话 "<<nt<<" 验证 "<<nv<<"\n";
    long long total=(long long)cfg.epochs*nt/max(1,cfg.grad_accum);
    m.set_total(total);
    float sps=0.f;
    auto t0=chrono::steady_clock::now(); const int BW=24;
    bool es=0;
    for(int ep=sep;ep<cfg.epochs&&!es;++ep){
        shuffle(tid.begin(),tid.end(),g);
        m.zg(); float esum=0.f; int ecnt=0; int ac=0;
        for(int s=0;s<nt;++s){
            const auto& d=dialogs[tid[s]];
            int Tl=(int)d.size(); if(Tl<2)continue; if(Tl>cfg.ctx)Tl=cfg.ctx;
            vector<int> xb(d.begin(),d.begin()+Tl-1), yb(d.begin()+1,d.begin()+Tl);
            m.upd_lr(stc); m.fwd(xb,g);
            float loss=m.bwd(xb,yb);
            esum+=loss; ecnt++; ac++;
            if(ac>=cfg.grad_accum||s==nt-1){
                if(cfg.grad_clip>0.f)clip_grads(m,cfg);
                m.ag(); m.zg(); ac=0; stc++;
            }
            auto t1=chrono::steady_clock::now();
            double sec=chrono::duration<double>(t1-t0).count();
            double inst=stc/max(sec,1e-9);
            if(sps==0.f)sps=(float)inst; else sps=0.95f*sps+0.05f*(float)inst;
            double pct=100.0*stc/max((long long)1,total); if(pct>100)pct=100;
            int fl=(int)(BW*pct/100);
            double eta=(total-stc)/max((double)sps,1e-6);
            float al=esum/max(1,ecnt);
            cout<<"\r[";
            for(int i=0;i<BW;++i)cout<<(i<fl?'#':'.');
            cout<<"] "<<setprecision(1)<<setw(5)<<pct<<"%"
                <<" loss="<<setprecision(4)<<al
                <<" lr="<<setprecision(2)<<scientific<<m.aE.lr<<fixed
                <<" ETA "<<setprecision(0)<<setw(5)<<eta<<"s    "<<flush;
        }
        float vl=0.f;
        if(nv>0){
            float vs=0.f;
            for(int vi:vid){
                const auto& d=dialogs[vi];
                int Tl=(int)d.size(); if(Tl<2)continue; if(Tl>cfg.ctx)Tl=cfg.ctx;
                vector<int> xb(d.begin(),d.begin()+Tl-1), yb(d.begin()+1,d.begin()+Tl);
                for(auto& dd:m.df)dd.p=0.f;
                for(auto& dd:m.de)dd.p=0.f;
                m.fwd(xb,g);
                for(auto& dd:m.df)dd.p=cfg.dropout;
                for(auto& dd:m.de)dd.p=cfg.embed_dropout;
                const float* lp=m.lg.data();
                float sl=0.f;
                for(int t=0;t<Tl-1;++t){
                    const float* L=lp+(size_t)t*V;
                    float mx=-1e30f; for(int v=0;v<V;++v)mx=max(mx,L[v]);
                    float sm=0.f; for(int v=0;v<V;++v)sm+=expf(L[v]-mx);
                    sl+=-logf(max(expf(L[yb[t]]-mx)/sm,1e-9f));
                }
                vs+=sl/(Tl-1);
            }
            vl=vs/nv;
        }
        float tl=esum/max(1,ecnt);
        float mon=nv>0?vl:tl;
        if(std::isfinite(mon)&&mon<bl)bl=mon;
        cout<<"\r"<<string(120,' ')<<"\r"<<"ep "<<(ep+1)<<" train="<<setprecision(4)<<tl<<" val="<<vl<<" best="<<bl<<"\n";
        if(cfg.ckpt_interval>0&&(ep+1)%cfg.ckpt_interval==0)saveck(cfg.ckpt,m,ep+1,stc,bl,g);
        if(cfg.stop_loss>0.f&&std::isfinite(mon)&&mon<=cfg.stop_loss)es=1;
    }
    savem(cfg.model,m,voc); saveck(cfg.ckpt,m,cfg.epochs,stc,bl,g);
    cout<<"\r"<<string(120,' ')<<"\r";
    if(es)cout<<"早停\n";
    cout<<"完成 best="<<bl<<" 保存 "<<cfg.model<<" / "<<cfg.ckpt<<"\n";
    return 0;
}

static string gen_once(Model& m,const Vocab& voc,const vector<int>& ic,Config& cfg,mt19937& g,Sampler& sm,int forced){
    int V=m.V; vector<int> ct=ic; if(ct.empty())ct.push_back(0);
    int lim=min(cfg.ctx,m.maxT-1); if((int)ct.size()>lim)ct.erase(ct.begin(),ct.end()-lim);
    m.kv_reset();
    for(int i=0;i+1<(int)ct.size();++i)m.step(ct[i]);
    int last=ct.back();
    vector<int> hist=ct; string out; int mx=forced>0?forced:cfg.maxlen; int st=0;
    for(;st<mx&&m.kv<m.maxT;++st){
        const float* l=m.step(last);
        int c=sm.sample(l,V,cfg.temp,cfg.topk,cfg.top_p,hist,cfg.rep_window,cfg.rep_pen,g,cfg.greedy);
        out+=voc.chars[c]; hist.push_back(c); last=c;
        if(forced==0&&iseos(out))break;
    }
    if(m.kv>=m.maxT)cerr<<"[warn] maxT 截断\n";
    return out;
}
static int run_gen(Config& cfg){
    Model m; Vocab voc;
    if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; }
    cout<<"V="<<m.V<<" D="<<m.D<<" L="<<m.L<<"\n";
    mt19937 g(chrono::system_clock::now().time_since_epoch().count());
    Sampler sm(m.V); vector<int> ct=voc.encode(cfg.prompt);
    if(ct.empty())ct.push_back(0);
    cout<<cfg.prompt<<gen_once(m,voc,ct,cfg,g,sm,0)<<"\n";
    return 0;
}
static int run_next(Config& cfg){
    Model m; Vocab voc;
    if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; }
    mt19937 g(chrono::system_clock::now().time_since_epoch().count());
    Sampler sm(m.V); vector<int> ct=voc.encode(cfg.prompt);
    if(ct.empty())ct.push_back(0);
    cout<<cfg.prompt<<gen_once(m,voc,ct,cfg,g,sm,1)<<"\n";
    return 0;
}
static int run_chat(Config& cfg){
    Model m; Vocab voc;
    if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; }
    mt19937 g(chrono::system_clock::now().time_since_epoch().count()); Sampler sm(m.V);
    cout<<"chat /exit /reset /temp /topk /topp /greedy /one\n";
    string ln;
    auto sw=[](const string& s,const string& p){ return s.size()>=p.size()&&s.compare(0,p.size(),p)==0; };
    while(1){
        cout<<"\n> "<<flush; if(!getline(cin,ln))break; if(ln.empty())continue;
        if(ln=="/exit"||ln=="/quit")break;
        if(ln=="/reset"){ cout<<"ok\n"; continue; }
        if(sw(ln,"/temp ")){ cfg.temp=stof(trm(ln.substr(6))); cout<<"t="<<cfg.temp<<"\n"; continue; }
        if(sw(ln,"/topk ")){ cfg.topk=stoi(trm(ln.substr(6))); cout<<"k="<<cfg.topk<<"\n"; continue; }
        if(sw(ln,"/topp ")){ cfg.top_p=stof(trm(ln.substr(6))); cout<<"p="<<cfg.top_p<<"\n"; continue; }
        if(ln=="/greedy"){ cfg.greedy=!cfg.greedy; cout<<cfg.greedy<<"\n"; continue; }
        if(ln=="/one"){ cfg.one_token=!cfg.one_token; cout<<cfg.one_token<<"\n"; continue; }
        vector<int> ct=voc.encode(ln); if(ct.empty()){ cout<<"无有效\n"; continue; }
        cout<<gen_once(m,voc,ct,cfg,g,sm,cfg.one_token?1:0);
    }
    return 0;
}
static int run_batch(Config& cfg){
    Model m; Vocab voc;
    if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; }
    ifstream f(cfg.batch_file); if(!f){ cerr<<"打不开\n"; return 1; }
    mt19937 g(chrono::system_clock::now().time_since_epoch().count()); Sampler sm(m.V);
    string ln;
    while(getline(f,ln)){
        if(ln.empty())continue;
        vector<int> ct=voc.encode(ln);
        if(ct.empty()){ cout<<ln<<"\t[无效]\n"; continue; }
        cout<<ln<<"\t"<<gen_once(m,voc,ct,cfg,g,sm,cfg.one_token?1:0)<<"\n";
    }
    return 0;
}

int main(int argc,char** argv){
    Config cfg;
    for(int i=1;i<argc;++i){ if(string(argv[i])=="--config"&&i+1<argc){ cfg.config_file=argv[i+1]; ++i; } }
    load_cfg(cfg,cfg.config_file);
    if(!parse_args(cfg,argc,argv))return 0;
    cout<<fixed;
    if(cfg.mode=="train")return run_train(cfg);
    if(cfg.mode=="gen")return run_gen(cfg);
    if(cfg.mode=="next")return run_next(cfg);
    if(cfg.mode=="chat")return run_chat(cfg);
    if(cfg.mode=="batch")return run_batch(cfg);
    cerr<<"未知 mode\n"; return 1;
}