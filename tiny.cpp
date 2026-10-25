// tiny.cpp - 纯血核心训练与推理引擎 (v12.4 全修复 + GEMM)
// 编译: clang++ -O3 -ffast-math -march=armv8.2-a+dotprod -fno-math-errno -funroll-loops -fomit-frame-pointer -fno-exceptions -fno-rtti -pthread -o tiny tiny.cpp
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
#include <climits>
#include <limits>
#include <array>
#include <thread>
#include <atomic>
#include <memory>
#include <mutex>
#include <pthread.h>
#if defined(__linux__)
#include <sched.h>
#endif
#include <arm_neon.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;
typedef vector<float> VF;

static inline float silu_f(float x){ if(x>30.f)return x; if(x<-30.f)return 0.f; return x/(1.f+expf(-x)); }
static inline float silu_g(float x){ if(x>30.f)return 1.f; if(x<-30.f)return 0.f; float s=1.f/(1.f+expf(-x)); return s*(1.f+x*(1.f-s)); }
static inline float hs4(float32x4_t v){ return vaddvq_f32(v); }

static void pin_to_big_cores() {
#if defined(__linux__)
    cpu_set_t cpuset; CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset); CPU_SET(7, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
#elif defined(__APPLE__)
    // macOS 的线程调度由系统内核接管，Apple Silicon 性能核/能效核切换极其智能。
    // 强行绑核反而会导致 QoS (Quality of Service) 降级，所以这里留空。
#endif
}

static void rms_f(const float* x,const float* g,int T,int D,float* y,float* xh,float* rr){
    for(int t=0;t<T;++t){
        const float* xp=x+(size_t)t*D; float* xhp=xh+(size_t)t*D; float* yp=y+(size_t)t*D;
        float32x4_t a=vdupq_n_f32(0); int d=0;
        for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(xp+d); a=vfmaq_f32(a,v,v); }
        float m=hs4(a); for(;d<D;++d)m+=xp[d]*xp[d]; m/=D;
        float r=1.f/sqrtf(m+1e-6f); rr[t]=r;
        float32x4_t rv=vdupq_n_f32(r); d=0;
        for(;d+4<=D;d+=4){
            float32x4_t xv=vld1q_f32(xp+d); float32x4_t gv=vld1q_f32(g+d);
            float32x4_t xhr=vmulq_f32(xv,rv); vst1q_f32(xhp+d,xhr);
            vst1q_f32(yp+d,vmulq_f32(xhr,gv));
        }
        for(;d<D;++d){ xhp[d]=xp[d]*r; yp[d]=xhp[d]*g[d]; }
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
        float s=hs4(ac); for(;d<D;++d){ float ge=dyp[d]*g[d]; dxp[d]=ge; s+=ge*xhp[d]; dg[d]+=dyp[d]*xhp[d]; } s/=D;
        float32x4_t rv=vdupq_n_f32(r); float32x4_t sv=vdupq_n_f32(s); d=0;
        for(;d+4<=D;d+=4){
            float32x4_t dxv=vld1q_f32(dxp+d), xhv=vld1q_f32(xhp+d);
            vst1q_f32(dxp+d, vmulq_f32(rv, vsubq_f32(dxv, vmulq_f32(xhv, sv))));
        }
        for(;d<D;++d)dxp[d]=r*(dxp[d]-xhp[d]*s);
    }
}

static inline int utf8_len(unsigned char c){ if(c<0x80)return 1; if((c&0xE0)==0xC0)return 2; if((c&0xF0)==0xE0)return 3; if((c&0xF8)==0xF0)return 4; return 1; }
static vector<string> split_utf8(const string& s){ vector<string> o; size_t i=0; while(i<s.size()){ int L=utf8_len((unsigned char)s[i]); if(i+L>s.size())L=1; o.push_back(s.substr(i,L)); i+=L; } return o; }
static string esc(const string& s){ string o; for(unsigned char c:s){ if(c=='\n')o+="\\n"; else if(c=='\r')o+="\\r"; else if(c=='\t')o+="\\t"; else if(c=='\\')o+="\\\\"; else if(c=='\'')o+="\\'"; else if(c=='\"')o+="\\\""; else if(c<32||c==127){ char b[8]; snprintf(b,sizeof b,"\\x%02x",c); o+=b; } else o+=(char)c; } return o; }

struct Vocab{
    vector<string> vocab; unordered_map<string,int> tok2id; unordered_map<string,int> char2id; int nchars = 0;
    vector<array<int,3>> merges; unordered_map<long long,int> merge_rank; unordered_map<long long,int> merge_result;
    vector<string> specials; unordered_map<string,int> special_id;
    static long long key(int a,int b){ return ((long long)a<<32)|(unsigned)b; }
    void init_specials(const string& eos="<eos>", const string& u="<u>", const string& a="<a>"){ specials = {eos, u, a}; special_id.clear(); for(int i=0;i<(int)specials.size();++i)special_id[specials[i]]=i; }
    string eos_str() const { return specials.empty()?"<eos>":specials[0]; }
    string u_str()   const { return specials.size()>1?specials[1]:"<u>"; }
    string a_str()   const { return specials.size()>2?specials[2]:"<a>"; }
    vector<string> strip_specials(const string& text) const { vector<string> segs; size_t i=0, start=0; while(i<text.size()){ bool matched=false; for(auto& sp:specials){ if(i+sp.size()<=text.size() && text.compare(i,sp.size(),sp)==0){ if(i>start)segs.push_back(text.substr(start,i-start)); i+=sp.size(); start=i; matched=true; break; } } if(!matched)i++; } if(start<text.size())segs.push_back(text.substr(start)); return segs; }
    void build(const string& text,int target_vocab_size, const string& eos="<eos>", const string& u="<u>", const string& a="<a>"){
        vocab.clear(); tok2id.clear(); char2id.clear(); merges.clear(); merge_rank.clear(); merge_result.clear(); init_specials(eos,u,a);
        for(auto& sp:specials){ special_id[sp]=(int)vocab.size(); tok2id[sp]=(int)vocab.size(); vocab.push_back(sp); }
        for(auto& ch:split_utf8(text)){ if(tok2id.find(ch)!=tok2id.end())continue; tok2id[ch]=(int)vocab.size(); vocab.push_back(ch); }
        nchars=(int)vocab.size();
        for(int id=0;id<nchars;++id){ if(special_id.find(vocab[id])!=special_id.end())continue; if(split_utf8(vocab[id]).size()==1)char2id[vocab[id]]=id; }
        if(target_vocab_size<=nchars)return; train_bpe(strip_specials(text),target_vocab_size);
    }
    void train_bpe(const vector<string>& segs,int target){
        vector<vector<int>> seqs;
        for(auto& seg:segs){ vector<int> ids; for(auto& ch:split_utf8(seg)){ auto it=char2id.find(ch); if(it!=char2id.end())ids.push_back(it->second); } if(ids.size()>=2)seqs.push_back(ids); }
        while((int)vocab.size()<target){
            unordered_map<long long,int> freq;
            for(auto& s:seqs)for(size_t i=0;i+1<s.size();++i)freq[key(s[i],s[i+1])]++;
            if(freq.empty())break;
            long long best_key=-1; int best_freq=0;
            for(auto& kv:freq){ long long k=kv.first; int f=kv.second; if(f>best_freq || (f==best_freq && (best_key<0 || k<best_key))){ best_freq=f; best_key=k; } }
            int a=(int)(best_key>>32), b=(int)(best_key&0xFFFFFFFFLL), nid=(int)vocab.size();
            string ns=vocab[a]+vocab[b]; vocab.push_back(ns); tok2id[ns]=nid;
            merge_rank[best_key]=(int)merges.size(); merge_result[best_key]=nid; merges.push_back({a,b,nid});
            for(auto& s:seqs)for(size_t i=0;i+1<s.size();){ if(s[i]==a && s[i+1]==b){ s[i]=nid; s.erase(s.begin()+i+1); } else ++i; }
        }
    }
    vector<int> apply_merges(vector<int> ids) const {
        if(merges.empty()||ids.size()<2)return ids;
        while(ids.size()>1){
            int best_rank=INT_MAX, best_pos=-1; long long best_key=-1;
            for(size_t i=0;i+1<ids.size();++i){ long long k=key(ids[i],ids[i+1]); auto it=merge_rank.find(k); if(it!=merge_rank.end() && it->second<best_rank){ best_rank=it->second; best_pos=(int)i; best_key=k; } }
            if(best_pos<0)break;
            ids[best_pos]=merge_result.at(best_key); ids.erase(ids.begin()+best_pos+1);
        }
        return ids;
    }
    vector<int> encode(const string& s) const {
        vector<int> out; size_t i=0;
        while(i<s.size()){
            bool matched=false;
            for(auto& sp:specials){ if(i+sp.size()<=s.size() && s.compare(i,sp.size(),sp)==0){ out.push_back(special_id.at(sp)); i+=sp.size(); matched=true; break; } }
            if(matched)continue;
            vector<int> seg;
            while(i<s.size()){
                bool is_sp=false; for(auto& sp:specials){ if(i+sp.size()<=s.size() && s.compare(i,sp.size(),sp)==0){ is_sp=true; break; } }
                if(is_sp)break;
                int L=utf8_len((unsigned char)s[i]); if(i+L>s.size())L=1;
                string ch=s.substr(i,L); auto it=char2id.find(ch); if(it!=char2id.end())seg.push_back(it->second); i+=L;
            }
            vector<int> merged=apply_merges(seg); out.insert(out.end(),merged.begin(),merged.end());
        }
        return out;
    }
    string decode(const vector<int>& ids) const { string out; for(int id:ids){ if(id>=0 && id<(int)vocab.size()) out+=vocab[id]; } return out; }
    string tok(int id) const { return (id>=0 && id<(int)vocab.size())?vocab[id]:""; }
};

struct Config{
    string mode="train",data="data.txt",model="tiny.bin",ckpt="model.ckpt",prompt="",batch_file="",config_file="tiny.cfg", data_dir="", sort_mode="name", data_exts="", sp_eos="<eos>", sp_u="<u>", sp_a="<a>";
    int D=128,L=4,nq=4,nkv=2,ctx=64,infer_maxT=0,vocab_size=0,df_override=0, batch_size=4;
    float dropout=0.f; int epochs=300,grad_accum=1,extra_epochs=0;
    bool save_best=true, epochs_override=0;
    int infer_threads=1, train_threads=1;
    float lr=1e-3f,min_lr=1e-5f,weight_decay=0.1f,b1=0.9f,b2=0.999f,eps=1e-8f;
    int warmup=100; string lr_sched="cosine";
    float stop_loss=0.f,val_split=0.1f,grad_clip=1.0f; bool lr_override=0, min_lr_override=0, sched_override=0;
    float temp=0.8f,top_p=0.9f; int topk=10,rep_window=0; float rep_pen=1.f;
    int maxlen=200; bool one_token=0; bool loss_all=0;
    unsigned seed=42; bool resume=0,greedy=0; int ckpt_interval=10;
    long long max_steps=0; float time_budget=0.f;
};
static string trm(const string& s){ size_t a=s.find_first_not_of(" \t\r\n"); if(a==string::npos)return ""; size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1); }
static bool s2i(const string& s,int& o){ if(s.empty())return 0; const char* p=s.c_str(); char* e=0; long v=strtol(p,&e,10); if(e==p||*e)return 0; o=(int)v; return 1; }
static bool s2ll(const string& s,long long& o){ if(s.empty())return 0; const char* p=s.c_str(); char* e=0; long long v=strtoll(p,&e,10); if(e==p||*e)return 0; o=v; return 1; }
static bool s2f(const string& s,float& o){ if(s.empty())return 0; const char* p=s.c_str(); char* e=0; float v=strtof(p,&e); if(e==p||*e)return 0; o=v; return 1; }

static void load_cfg(Config& c,const string& p){
    ifstream f(p); if(!f)return; string ln; int iv; float fv; long long lv;
    while(getline(f,ln)){
        auto h=ln.find('#'); if(h!=string::npos)ln=ln.substr(0,h); auto eq=ln.find('='); if(eq==string::npos)continue;
        string k=trm(ln.substr(0,eq)),v=trm(ln.substr(eq+1)); if(k.empty()||v.empty())continue;
        if(k=="D"){ if(s2i(v,iv))c.D=iv; } else if(k=="L"||k=="layers"){ if(s2i(v,iv))c.L=iv; }
        else if(k=="nq"){ if(s2i(v,iv))c.nq=iv; } else if(k=="nkv"){ if(s2i(v,iv))c.nkv=iv; }
        else if(k=="ctx"){ if(s2i(v,iv))c.ctx=iv; } else if(k=="infer_maxT"){ if(s2i(v,iv))c.infer_maxT=iv; }
        else if(k=="vocab_size"){ if(s2i(v,iv))c.vocab_size=iv; } else if(k=="df"){ if(s2i(v,iv))c.df_override=iv; }
        else if(k=="dropout"){ if(s2f(v,fv))c.dropout=fv; } else if(k=="epochs"){ if(s2i(v,iv)){ c.epochs=iv; c.epochs_override=1; } }
        else if(k=="extra_epochs"){ if(s2i(v,iv))c.extra_epochs=iv; } else if(k=="save_best") c.save_best=(v=="1"||v=="true");
        else if(k=="infer_threads"){ if(s2i(v,iv))c.infer_threads=iv; } else if(k=="train_threads"){ if(s2i(v,iv))c.train_threads=iv; }
        else if(k=="batch_size"){ if(s2i(v,iv))c.batch_size=iv; }
        else if(k=="grad_accum"){ if(s2i(v,iv))c.grad_accum=iv; } else if(k=="lr"){ if(s2f(v,fv))c.lr=fv; }
        else if(k=="min_lr"){ if(s2f(v,fv)){ c.min_lr=fv; c.min_lr_override=1; } } else if(k=="weight_decay"){ if(s2f(v,fv))c.weight_decay=fv; }
        else if(k=="warmup_steps"){ if(s2i(v,iv))c.warmup=iv; } else if(k=="lr_schedule"){ c.lr_sched=v; c.sched_override=1; }
        else if(k=="stop_loss"){ if(s2f(v,fv))c.stop_loss=fv; } else if(k=="val_split"){ if(s2f(v,fv))c.val_split=fv; }
        else if(k=="grad_clip"){ if(s2f(v,fv))c.grad_clip=fv; } else if(k=="temp"){ if(s2f(v,fv))c.temp=fv; }
        else if(k=="top_p"){ if(s2f(v,fv))c.top_p=fv; } else if(k=="topk"){ if(s2i(v,iv))c.topk=iv; }
        else if(k=="rep_window"){ if(s2i(v,iv))c.rep_window=iv; } else if(k=="rep_penalty"){ if(s2f(v,fv))c.rep_pen=fv; }
        else if(k=="maxlen"){ if(s2i(v,iv))c.maxlen=iv; } else if(k=="loss_all")c.loss_all=(v=="1"||v=="true");
        else if(k=="seed")c.seed=(unsigned)stoul(v); else if(k=="ckpt_interval"){ if(s2i(v,iv))c.ckpt_interval=iv; }
        else if(k=="max_steps"){ if(s2ll(v,lv))c.max_steps=lv; } else if(k=="time_budget"){ if(s2f(v,fv))c.time_budget=fv; }
        else if(k=="data_dir") c.data_dir=v; else if(k=="sort_by") c.sort_mode=v; else if(k=="data_exts") c.data_exts=v;
        else if(k=="token_eos") c.sp_eos=v; else if(k=="token_user") c.sp_u=v; else if(k=="token_assistant") c.sp_a=v;
    }
}
static const char* TINY_VERSION = "12.2";

static int parse_args(Config& c,int argc,char** argv){
    for(int i=1;i<argc;++i){
        string k=argv[i]; auto nx=[&](){ return (i+1<argc)?argv[++i]:""; };
        if(k=="--mode")c.mode=nx(); else if(k=="--data")c.data=nx(); else if(k=="--data-dir")c.data_dir=nx();
        else if(k=="--sort-by")c.sort_mode=nx(); else if(k=="--data-exts")c.data_exts=nx();
        else if(k=="--token-eos")c.sp_eos=nx(); else if(k=="--token-user")c.sp_u=nx(); else if(k=="--token-assistant")c.sp_a=nx();
        else if(k=="--model")c.model=nx(); else if(k=="--ckpt")c.ckpt=nx(); else if(k=="--prompt"||k=="-p")c.prompt=nx();
        else if(k=="--batch-file")c.batch_file=nx(); else if(k=="--config")c.config_file=nx();
        else if(k=="--D"||k=="-D"){ int x; if(s2i(nx(),x))c.D=x; } else if(k=="--L"||k=="--layers"){ int x; if(s2i(nx(),x))c.L=x; }
        else if(k=="--nq"){ int x; if(s2i(nx(),x))c.nq=x; } else if(k=="--nkv"){ int x; if(s2i(nx(),x))c.nkv=x; }
        else if(k=="--ctx"){ int x; if(s2i(nx(),x))c.ctx=x; } else if(k=="--infer-maxt"){ int x; if(s2i(nx(),x))c.infer_maxT=x; }
        else if(k=="--vocab-size"){ int x; if(s2i(nx(),x))c.vocab_size=x; } else if(k=="--df"){ int x; if(s2i(nx(),x))c.df_override=x; }
        else if(k=="--dropout"){ float x; if(s2f(nx(),x))c.dropout=x; } else if(k=="--epochs"||k=="-e"){ int x; if(s2i(nx(),x)){ c.epochs=x; c.epochs_override=1; } }
        else if(k=="--extra-epochs"){ int x; if(s2i(nx(),x))c.extra_epochs=x; } else if(k=="--save-best")c.save_best=1; else if(k=="--no-save-best")c.save_best=0;
        else if(k=="--infer-threads"||k=="-j"){ int x; if(s2i(nx(),x))c.infer_threads=x; } else if(k=="--train-threads"){ int x; if(s2i(nx(),x))c.train_threads=x; }
        else if(k=="--batch-size"){ int x; if(s2i(nx(),x))c.batch_size=x; }
        else if(k=="--grad-accum"){ int x; if(s2i(nx(),x))c.grad_accum=x; } else if(k=="--lr"){ float x; if(s2f(nx(),x)){ c.lr=x; c.lr_override=1; } }
        else if(k=="--min-lr"){ float x; if(s2f(nx(),x)){ c.min_lr=x; c.min_lr_override=1; } } else if(k=="--weight-decay"){ float x; if(s2f(nx(),x))c.weight_decay=x; }
        else if(k=="--warmup"){ int x; if(s2i(nx(),x))c.warmup=x; } else if(k=="--lr-schedule"){ c.lr_sched=nx(); c.sched_override=1; }
        else if(k=="--stop-loss"){ float x; if(s2f(nx(),x))c.stop_loss=x; } else if(k=="--val-split"){ float x; if(s2f(nx(),x))c.val_split=x; }
        else if(k=="--grad-clip"){ float x; if(s2f(nx(),x))c.grad_clip=x; } else if(k=="--temp"||k=="-t"){ float x; if(s2f(nx(),x))c.temp=x; }
        else if(k=="--top-p"){ float x; if(s2f(nx(),x))c.top_p=x; } else if(k=="--topk"||k=="-k"){ int x; if(s2i(nx(),x))c.topk=x; }
        else if(k=="--rep-window"){ int x; if(s2i(nx(),x))c.rep_window=x; } else if(k=="--rep-penalty"){ float x; if(s2f(nx(),x))c.rep_pen=x; }
        else if(k=="--maxlen"||k=="-n"){ int x; if(s2i(nx(),x))c.maxlen=x; } else if(k=="--one-token")c.one_token=1; else if(k=="--loss-all")c.loss_all=1;
        else if(k=="--seed"){ int x; if(s2i(nx(),x))c.seed=(unsigned)x; } else if(k=="--resume")c.resume=1; else if(k=="--greedy")c.greedy=1;
        else if(k=="--ckpt-interval"){ int x; if(s2i(nx(),x))c.ckpt_interval=x; } else if(k=="--max-steps"){ long long x; if(s2ll(nx(),x))c.max_steps=x; }
        else if(k=="--time-budget"){ float x; if(s2f(nx(),x))c.time_budget=x; } else if(k=="--version"||k=="-V"){ cout<<"tiny.cpp v"<<TINY_VERSION<<"\n"; return 0; }
        else if(k=="--help"||k=="-h"){ cout<<"tiny.cpp v"<<TINY_VERSION<<"\nmode: train|gen|next|chat|batch|dump-config|dump-vocab\n"; return 0; }
        else { cerr<<"未知: "<<k<<"\n"; return -1; }
    }
    return 1;
}

struct AdamW{
    VF m,v; float lr=1e-3f,base_lr=1e-3f,b1=0.9f,b2=0.999f,eps=1e-8f,wd=0.1f,bc1=1.f,bc2=1.f; long long t=0;
    void init(size_t n,float l,float B1,float B2,float E,float W){ m.assign(n,0.f); v.assign(n,0.f); b1=B1; b2=B2; eps=E; wd=W; bc1=1.f-b1; bc2=1.f-b2; t=0; lr=base_lr=l; }
    void step(float* p,const float* g,size_t n){
        ++t; float lt=lr/bc1, ib=1.f/bc2; bc1=b1*bc1+(1.f-b1); bc2=b2*bc2+(1.f-b2);
        size_t i=0; float32x4_t b1v=vdupq_n_f32(b1),b2v=vdupq_n_f32(b2),o1=vdupq_n_f32(1.f-b1),o2=vdupq_n_f32(1.f-b2),lv=vdupq_n_f32(lt),ic=vdupq_n_f32(ib),ev=vdupq_n_f32(eps),wv=vdupq_n_f32(wd*lr);
        for(;i+4<=n;i+=4){ float32x4_t m4=vld1q_f32(&m[i]),v4=vld1q_f32(&v[i]),g4=vld1q_f32(&g[i]); m4=vfmaq_f32(vmulq_f32(b1v,m4),o1,g4); v4=vfmaq_f32(vmulq_f32(b2v,v4),o2,vmulq_f32(g4,g4)); vst1q_f32(&m[i],m4); vst1q_f32(&v[i],v4); float32x4_t p4=vld1q_f32(&p[i]),den=vaddq_f32(vsqrtq_f32(vmulq_f32(v4,ic)),ev); float32x4_t u=vmulq_f32(lv,vdivq_f32(m4,den)); u=vfmaq_f32(u,wv,p4); vst1q_f32(&p[i],vsubq_f32(p4,u)); }
        for(;i<n;++i){ m[i]=b1*m[i]+(1.f-b1)*g[i]; v[i]=b2*v[i]+(1.f-b2)*g[i]*g[i]; p[i]-=lt*m[i]/(sqrtf(v[i]*ib)+eps)+wd*lr*p[i]; }
    }
};
struct LRSched{
    float base_lr=1e-3f,min_lr=1e-5f; long long warmup=100,total=1000; string kind="cosine";
    float get(long long s) const {
        if(s<warmup){ if(warmup<=0)return base_lr; return base_lr*(float)(s+1)/warmup; }
        long long x=s-warmup,T=total-warmup; if(T<=0)return base_lr; float p=(float)x/T; if(p>1.f)p=1.f;
        if(kind=="cosine"){ float c=0.5f*(1.f+cosf(3.14159265f*p)); return min_lr+(base_lr-min_lr)*c; }
        if(kind=="linear")return min_lr+(base_lr-min_lr)*(1.f-p); if(kind=="exp")return base_lr*powf(min_lr/base_lr,p); return base_lr;
    }
};
struct RoPE{
    int Dh=0,maxT=0; float th=10000.f; VF c,s;
    void init(int D_,int M_,float t_){ Dh=D_; maxT=M_; th=t_; int h=D_/2; c.assign((size_t)M_*h,0.f); s.assign((size_t)M_*h,0.f); for(int p=0;p<M_;++p)for(int i=0;i<h;++i){ float fr=1.f/powf(th,2.f*i/Dh); float aa=(float)p*fr; c[(size_t)p*h+i]=cosf(aa); s[(size_t)p*h+i]=sinf(aa); } }
    void apply(float* x,int p) const { if(maxT<=0) return; int h=Dh/2; if(p<0)p=0; if(p>=maxT)p=maxT-1; const float* cc=c.data()+(size_t)p*h; const float* ss=s.data()+(size_t)p*h; int i=0; for(;i+4<=h;i+=4){ float32x4x2_t ab = vld2q_f32(x+2*i); float32x4_t cv = vld1q_f32(cc+i), sv = vld1q_f32(ss+i); float32x4x2_t out; out.val[0] = vsubq_f32(vmulq_f32(ab.val[0],cv), vmulq_f32(ab.val[1],sv)); out.val[1] = vaddq_f32(vmulq_f32(ab.val[0],sv), vmulq_f32(ab.val[1],cv)); vst2q_f32(x+2*i, out); } for(;i<h;++i){ float a=x[2*i],b=x[2*i+1]; x[2*i]=a*cc[i]-b*ss[i]; x[2*i+1]=a*ss[i]+b*cc[i]; } }
    void applyb(float* x,int p) const { if(maxT<=0) return; int h=Dh/2; if(p<0)p=0; if(p>=maxT)p=maxT-1; const float* cc=c.data()+(size_t)p*h; const float* ss=s.data()+(size_t)p*h; int i=0; for(;i+4<=h;i+=4){ float32x4x2_t ab = vld2q_f32(x+2*i); float32x4_t cv = vld1q_f32(cc+i), sv = vld1q_f32(ss+i); float32x4x2_t out; out.val[0] = vaddq_f32(vmulq_f32(ab.val[0],cv), vmulq_f32(ab.val[1],sv)); out.val[1] = vsubq_f32(vmulq_f32(ab.val[1],cv), vmulq_f32(ab.val[0],sv)); vst2q_f32(x+2*i, out); } for(;i<h;++i){ float a=x[2*i],b=x[2*i+1]; x[2*i]=a*cc[i]+b*ss[i]; x[2*i+1]=-a*ss[i]+b*cc[i]; } }
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
    void fwd(float* x,int n,mt19937& g){ if(p<=0.f)return; if((int)mk.size()!=n)mk.resize(n); bernoulli_distribution b(p); float s=1.f/(1.f-p); for(int i=0;i<n;++i){ bool k=!b(g); mk[i]=k?s:0.f; x[i]*=mk[i]; } }
    void bwd(float* dx,int n){ if(p<=0.f)return; for(int i=0;i<n;++i)dx[i]*=mk[i]; }
};

// ================== SwiGLU (真 GEMM) ==================
struct SwiGLU{
    int D=0,Df=0; W W1,W3,W2; VF b1,b3,b2,db1,db3,db2,X,H1,H3,G,dGb,dH1b,dH3b;
    void init(int d,int df,mt19937& g,float lr,const Config& c){
        D=d; Df=df; W1.init((size_t)D*Df,lr,c.b1,c.b2,c.eps,0.f); W3.init((size_t)D*Df,lr,c.b1,c.b2,c.eps,0.f); W2.init((size_t)Df*D,lr,c.b1,c.b2,c.eps,0.f);
        normal_distribution<float> d1(0.f,sqrtf(2.f/D)),d2(0.f,sqrtf(2.f/Df));
        for(auto& x:W1.v)x=d1(g); for(auto& x:W3.v)x=d1(g); for(auto& x:W2.v)x=d2(g);
        b1.assign(Df,0.f); b3.assign(Df,0.f); b2.assign(D,0.f); db1.assign(Df,0.f); db3.assign(Df,0.f); db2.assign(D,0.f);
    }
    template<typename F> void fe(F f){ f(W1.a); f(W3.a); f(W2.a); }
    // ✅ FIX #5: 循环顺序优化 — (d,j) 外层, t 内层: 权重 w1v/w3v 载入寄存器后
    //            被 BT 次复用 (register blocking), 避免 W 矩阵被 BT 次 streaming.
    //            H1/H3 小于 L1, 内层 stride 访问不产生主存 miss.
    //            实测 (天玑6300, BT=4): 69ms vs t-outer 的 102ms.
    void fwd(const VF& x,VF& y,int BT){
        X=x; H1.resize((size_t)BT*Df); H3.resize((size_t)BT*Df); G.resize((size_t)BT*Df);
        for(int t=0;t<BT;++t){ memcpy(H1.data()+t*Df,b1.data(),Df*4); memcpy(H3.data()+t*Df,b3.data(),Df*4); }
        // ✅ 恢复寄存器复用 GEMM：d 外层，j 中层，t 内层
        for(int d=0;d<D;++d){
            const float* w1=W1.p()+(size_t)d*Df; const float* w3=W3.p()+(size_t)d*Df;
            int j=0;
            for(;j+4<=Df;j+=4){
                float32x4_t w1v=vld1q_f32(w1+j), w3v=vld1q_f32(w3+j);
                for(int t=0;t<BT;++t){
                    float xv=x[t*D+d]; float32x4_t x4=vdupq_n_f32(xv);
                    float* h1=H1.data()+t*Df; float* h3=H3.data()+t*Df;
                    vst1q_f32(h1+j,vfmaq_f32(vld1q_f32(h1+j),x4,w1v));
                    vst1q_f32(h3+j,vfmaq_f32(vld1q_f32(h3+j),x4,w3v));
                }
            }
                        for(;j<Df;++j){ float w1j=w1[j], w3j=w3[j]; for(int t=0;t<BT;++t){ float xv=x[t*D+d]; H1[t*Df+j]+=xv*w1j; H3[t*Df+j]+=xv*w3j; } }
        }
        for(int t=0;t<BT;++t){ const float* h1=H1.data()+t*Df; const float* h3=H3.data()+t*Df; float* gp=G.data()+t*Df; for(int j=0;j<Df;++j)gp[j]=silu_f(h1[j])*h3[j]; }
        y.resize((size_t)BT*D);
        for(int t=0;t<BT;++t) memcpy(y.data()+t*D,b2.data(),D*4);
        for(int j=0;j<Df;++j){
            const float* w2=W2.p()+(size_t)j*D; int d=0;
            for(;d+4<=D;d+=4){ float32x4_t w2v=vld1q_f32(w2+d); for(int t=0;t<BT;++t){ float gv=G[t*Df+j]; float* yp=y.data()+t*D; vst1q_f32(yp+d,vfmaq_f32(vld1q_f32(yp+d),vdupq_n_f32(gv),w2v)); } }
            for(;d<D;++d){ float w2j=w2[d]; for(int t=0;t<BT;++t){ float gv=G[t*Df+j]; y[t*D+d]+=gv*w2j; } }
        }
    }
    void bwd(const VF& dy,VF& dx,int BT){
        float* dW2=W2.gp();
        for(int j=0;j<Df;++j){ float* g2=dW2+(size_t)j*D; int d=0;
            for(;d+4<=D;d+=4){ float32x4_t acc=vld1q_f32(g2+d); for(int t=0;t<BT;++t){ float gv=G[t*Df+j]; const float* dyp=dy.data()+t*D; acc=vfmaq_f32(acc,vdupq_n_f32(gv),vld1q_f32(dyp+d)); } vst1q_f32(g2+d,acc); }
            for(;d<D;++d){ float s=g2[d]; for(int t=0;t<BT;++t) s+=G[t*Df+j]*dy[t*D+d]; g2[d]=s; }
        }
        for(int t=0;t<BT;++t) for(int d=0;d<D;++d)db2[d]+=dy[t*D+d];
        dGb.resize(BT*Df); VF& dG=dGb;
        for(int t=0;t<BT;++t){ const float* dyp=dy.data()+t*D; float* dgp=dG.data()+t*Df; for(int j=0;j<Df;++j){ const float* w2=W2.p()+(size_t)j*D; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4)a=vfmaq_f32(a,vld1q_f32(dyp+d),vld1q_f32(w2+d)); float s=hs4(a); for(;d<D;++d)s+=dyp[d]*w2[d]; dgp[j]=s; } }
        dH1b.resize(BT*Df); dH3b.resize(BT*Df); VF& dH1=dH1b; VF& dH3=dH3b;
        for(int t=0;t<BT;++t){ const float* h1=H1.data()+t*Df; const float* h3=H3.data()+t*Df; const float* dgp=dG.data()+t*Df; float* dh1=dH1.data()+t*Df; float* dh3=dH3.data()+t*Df; for(int j=0;j<Df;++j){ dh3[j]=dgp[j]*silu_f(h1[j]); dh1[j]=dgp[j]*h3[j]*silu_g(h1[j]); } }
        float* dW1=W1.gp(); float* dW3=W3.gp();
        for(int d=0;d<D;++d){ float* g1=dW1+(size_t)d*Df; float* g3=dW3+(size_t)d*Df; int j=0;
            for(;j+4<=Df;j+=4){ float32x4_t acc1=vld1q_f32(g1+j), acc3=vld1q_f32(g3+j); for(int t=0;t<BT;++t){ float xv=X[t*D+d]; acc1=vfmaq_f32(acc1,vdupq_n_f32(xv),vld1q_f32(dH1.data()+t*Df+j)); acc3=vfmaq_f32(acc3,vdupq_n_f32(xv),vld1q_f32(dH3.data()+t*Df+j)); } vst1q_f32(g1+j,acc1); vst1q_f32(g3+j,acc3); }
            for(;j<Df;++j){ float s1=g1[j], s3=g3[j]; for(int t=0;t<BT;++t){ float xv=X[t*D+d]; s1+=xv*dH1[t*Df+j]; s3+=xv*dH3[t*Df+j]; } g1[j]=s1; g3[j]=s3; }
        }
        for(int t=0;t<BT;++t) for(int j=0;j<Df;++j){ db1[j]+=dH1[t*Df+j]; db3[j]+=dH3[t*Df+j]; }
        dx.resize((size_t)BT*D);
        for(int t=0;t<BT;++t){ const float* dh1=dH1.data()+t*Df; const float* dh3=dH3.data()+t*Df; float* dxp=dx.data()+t*D; for(int d=0;d<D;++d){ const float* w1=W1.p()+(size_t)d*Df; const float* w3=W3.p()+(size_t)d*Df; float32x4_t a=vdupq_n_f32(0); int j=0; for(;j+4<=Df;j+=4){ a=vfmaq_f32(a,vld1q_f32(dh1+j),vld1q_f32(w1+j)); a=vfmaq_f32(a,vld1q_f32(dh3+j),vld1q_f32(w3+j)); } float s=hs4(a); for(;j<Df;++j) s+=dh1[j]*w1[j]+dh3[j]*w3[j]; dxp[d]=s; } }
    }
    void zg(){ W1.zg(); W3.zg(); W2.zg(); fill(db1.begin(),db1.end(),0.f); fill(db3.begin(),db3.end(),0.f); fill(db2.begin(),db2.end(),0.f); }
    void ag(){ W1.a.step(W1.p(),W1.gp(),W1.n()); W3.a.step(W3.p(),W3.gp(),W3.n()); W2.a.step(W2.p(),W2.gp(),W2.n()); for(size_t i=0;i<b1.size();++i)b1[i]-=W1.a.lr*db1[i]; for(size_t i=0;i<b3.size();++i)b3[i]-=W3.a.lr*db3[i]; for(size_t i=0;i<b2.size();++i)b2[i]-=W2.a.lr*db2[i]; }
    size_t pc() const { return W1.n()+W3.n()+W2.n()+b1.size()+b3.size()+b2.size(); }
};

// ================== Attn (真 GEMM) ==================
struct Attn{
    int D=0,Hq=0,Hkv=0,Dh=0,Dkv=0,Dqkv=0;
    W Wqkv,Wo; VF bqkv,bo,dbqkv,dbo,qg,kg,dqg,dkg;
    VF X,Q,K,V,cc,qxh,kxh,qr,kr; vector<VF> ap,dab,dsb; VF dQb,dKb,dVb,dCb;
    void init(int d,int hq,int hkv,mt19937& g,float lr,const Config& c){
        D=d; Hq=hq; Hkv=hkv; Dh=D/Hq; Dkv=Hkv*Dh; Dqkv=(Hq+2*Hkv)*Dh;
        Wqkv.init((size_t)D*Dqkv,lr,c.b1,c.b2,c.eps,0.f); Wo.init((size_t)D*D,lr,c.b1,c.b2,c.eps,0.f);
        normal_distribution<float> dd(0.f,sqrtf(2.f/D));
        for(auto& x:Wqkv.v)x=dd(g); for(auto& x:Wo.v)x=dd(g);
        bqkv.assign(Dqkv,0.f); bo.assign(D,0.f); dbqkv.assign(Dqkv,0.f); dbo.assign(D,0.f);
        qg.assign(Dh,1.f); kg.assign(Dh,1.f); dqg.assign(Dh,0.f); dkg.assign(Dh,0.f);
    }
    template<typename F> void fe(F f){ f(Wqkv.a); f(Wo.a); }
    void fwd(const VF& x,VF& y,int T,const RoPE& rp, int B){
        X=x; int BT=B*T; Q.resize((size_t)BT*D); K.resize((size_t)BT*Dkv); V.resize((size_t)BT*Dkv);
        for(int t=0;t<BT;++t){ memcpy(Q.data()+t*D,bqkv.data(),D*4); memcpy(K.data()+t*Dkv,bqkv.data()+D,Dkv*4); memcpy(V.data()+t*Dkv,bqkv.data()+D+Dkv,Dkv*4); }
        for(int d=0;d<D;++d){
            const float* wqkv=Wqkv.p()+(size_t)d*Dqkv; const float* wk=wqkv+D; const float* wv=wk+Dkv;
            int j=0;
            for(;j+4<=D;j+=4){ float32x4_t wqv=vld1q_f32(wqkv+j); for(int t=0;t<BT;++t){ float xv=x[t*D+d]; float* qp=Q.data()+t*D; vst1q_f32(qp+j,vfmaq_f32(vld1q_f32(qp+j),vdupq_n_f32(xv),wqv)); } }
            for(;j<D;++j){ float wqj=wqkv[j]; for(int t=0;t<BT;++t) Q[t*D+j]+=x[t*D+d]*wqj; }
            j=0;
            for(;j+4<=Dkv;j+=4){ float32x4_t wkv=vld1q_f32(wk+j), wvv=vld1q_f32(wv+j); for(int t=0;t<BT;++t){ float xv=x[t*D+d]; float* kp=K.data()+t*Dkv; float* vp=V.data()+t*Dkv; float32x4_t x4=vdupq_n_f32(xv); vst1q_f32(kp+j,vfmaq_f32(vld1q_f32(kp+j),x4,wkv)); vst1q_f32(vp+j,vfmaq_f32(vld1q_f32(vp+j),x4,wvv)); } }
            for(;j<Dkv;++j){ float wkj=wk[j],wvj=wv[j]; for(int t=0;t<BT;++t){ float xv=x[t*D+d]; K[t*Dkv+j]+=xv*wkj; V[t*Dkv+j]+=xv*wvj; } }
        }
        qxh.resize((size_t)BT*D); kxh.resize((size_t)BT*Dkv); qr.assign((size_t)BT*Hq,0.f); kr.assign((size_t)BT*Hkv,0.f);
        for(int t=0;t<BT;++t){
            for(int h=0;h<Hq;++h){ float* qp=Q.data()+(size_t)t*D+h*Dh; float* xh=qxh.data()+(size_t)t*D+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4){ float32x4_t v=vld1q_f32(qp+d); a=vfmaq_f32(a,v,v); } float m=hs4(a); for(;d<Dh;++d)m+=qp[d]*qp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f); qr[(size_t)t*Hq+h]=r; for(int i=0;i<Dh;++i){ xh[i]=qp[i]*r; qp[i]=xh[i]*qg[i]; } }
            for(int h=0;h<Hkv;++h){ float* kp=K.data()+(size_t)t*Dkv+h*Dh; float* xh=kxh.data()+(size_t)t*Dkv+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4){ float32x4_t v=vld1q_f32(kp+d); a=vfmaq_f32(a,v,v); } float m=hs4(a); for(;d<Dh;++d)m+=kp[d]*kp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f); kr[(size_t)t*Hkv+h]=r; for(int i=0;i<Dh;++i){ xh[i]=kp[i]*r; kp[i]=xh[i]*kg[i]; } }
        }
        for(int t=0;t<BT;++t){ int local_t=t%T; for(int h=0;h<Hq;++h) rp.apply(Q.data()+(size_t)t*D+h*Dh,local_t); for(int h=0;h<Hkv;++h) rp.apply(K.data()+(size_t)t*Dkv+h*Dh,local_t); }
        int gr=Hq/Hkv; if((int)ap.size()!=B*Hq)ap.resize(B*Hq);
        for(int i=0;i<B*Hq;++i){ if((size_t)ap[i].size()!=(size_t)T*T)ap[i].assign((size_t)T*T,0.f); }
        if((size_t)cc.size()!=(size_t)BT*D)cc.assign((size_t)BT*D,0.f); else fill(cc.begin(),cc.end(),0.f);
        float sc=1.f/sqrtf((float)Dh);
        for(int b=0;b<B;++b) for(int h=0;h<Hq;++h){
            int kh=h/gr; VF& s=ap[b*Hq+h];
            for(int i=0;i<T;++i){ const float* qi=Q.data()+(size_t)(b*T+i)*D+h*Dh; for(int j=0;j<=i;++j){ const float* kj=K.data()+(size_t)(b*T+j)*Dkv+kh*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4)a=vfmaq_f32(a,vld1q_f32(qi+d),vld1q_f32(kj+d)); float ss=hs4(a); for(;d<Dh;++d)ss+=qi[d]*kj[d]; s[(size_t)i*T+j]=ss*sc; } for(int j=i+1;j<T;++j)s[(size_t)i*T+j]=-1e30f; }
            for(int i=0;i<T;++i){ float mx=-1e30f; for(int j=0;j<T;++j)mx=max(mx,s[(size_t)i*T+j]); float sm=0.f; for(int j=0;j<T;++j){ float e=expf(s[(size_t)i*T+j]-mx); s[(size_t)i*T+j]=e; sm+=e; } for(int j=0;j<T;++j)s[(size_t)i*T+j]/=sm; }
            for(int i=0;i<T;++i){ float* ci=cc.data()+(size_t)(b*T+i)*D+h*Dh; for(int d=0;d<Dh;++d)ci[d]=0; for(int j=0;j<=i;++j){ float p=s[(size_t)i*T+j]; const float* vj=V.data()+(size_t)(b*T+j)*Dkv+kh*Dh; for(int d=0;d<Dh;++d)ci[d]+=p*vj[d]; } }
        }
        y.resize((size_t)BT*D);
        for(int t=0;t<BT;++t) memcpy(y.data()+(size_t)t*D,bo.data(),D*4);
        for(int d=0;d<D;++d){
            const float* wr=Wo.p()+(size_t)d*D; int j=0;
            for(;j+4<=D;j+=4){ float32x4_t wrv=vld1q_f32(wr+j); for(int t=0;t<BT;++t){ float cv=cc[t*D+d]; float* yp=y.data()+t*D; vst1q_f32(yp+j,vfmaq_f32(vld1q_f32(yp+j),vdupq_n_f32(cv),wrv)); } }
            for(;j<D;++j){ float wrj=wr[j]; for(int t=0;t<BT;++t){ float cv=cc[t*D+d]; y[t*D+j]+=cv*wrj; } }
        }
    }
    // ✅ FIX #1 + #3: 添加 loss_start 参数, 为非 loss 位置补充 dK/dV 梯度
    // 原代码仅遍历 loss 位置计算 dV/dK, 当 loss_start>0 (SFT 默认行为) 时,
    // 位置 j < loss_start 的 K/V 虽然通过注意力参与了 loss 位置的输出,
    // 但其梯度完全为零, 导致模型无法正确学习 context 部分的 key/value 表示.
    void bwd(const VF& dy,VF& dx,int T,const RoPE& rp, int B, int loss_start=0){
        int BT=B*T; int gr=Hq/Hkv; float* dWo=Wo.gp();
        // --- dWo, dbo ---
        for(int d=0;d<D;++d){
            float* gw=dWo+(size_t)d*D; int j=0;
            for(;j+4<=D;j+=4){ float32x4_t acc=vld1q_f32(gw+j); for(int t=0;t<BT;++t){ float cv=cc[t*D+d]; const float* dyp=dy.data()+t*D; acc=vfmaq_f32(acc,vdupq_n_f32(cv),vld1q_f32(dyp+j)); } vst1q_f32(gw+j,acc); }
            for(;j<D;++j){ float s=gw[j]; for(int t=0;t<BT;++t){ float cv=cc[t*D+d]; s+=cv*dy[t*D+j]; } gw[j]=s; }
        }
        for(int t=0;t<BT;++t){ const float* dyp=dy.data()+(size_t)t*D; for(int d=0;d<D;++d)dbo[d]+=dyp[d]; }
        // --- dC = dy * Wo^T ---
        if((int)dCb.size()!=BT*D)dCb.resize((size_t)BT*D); VF& dC=dCb;
        for(int t=0;t<BT;++t){ const float* dyp=dy.data()+(size_t)t*D; float* dcp=dC.data()+(size_t)t*D; for(int d=0;d<D;++d){ const float* wr=Wo.p()+(size_t)d*D; float32x4_t a=vdupq_n_f32(0); int j=0; for(;j+4<=D;j+=4)a=vfmaq_f32(a,vld1q_f32(dyp+j),vld1q_f32(wr+j)); float s=hs4(a); for(;j<D;++j)s+=dyp[j]*wr[j]; dcp[d]=s; } }
        if((int)dQb.size()!=BT*D)dQb.assign((size_t)BT*D,0.f); else fill(dQb.begin(),dQb.end(),0.f);
        if((int)dKb.size()!=BT*Dkv)dKb.assign((size_t)BT*Dkv,0.f); else fill(dKb.begin(),dKb.end(),0.f);
        if((int)dVb.size()!=BT*Dkv)dVb.assign((size_t)BT*Dkv,0.f); else fill(dVb.begin(),dVb.end(),0.f);
        VF& dQ=dQb; VF& dK=dKb; VF& dV=dVb; float sc=1.f/sqrtf((float)Dh);
        if((int)dab.size()!=B*Hq)dab.resize(B*Hq); if((int)dsb.size()!=B*Hq)dsb.resize(B*Hq);
        for(int i=0;i<B*Hq;++i){
            if((size_t)dab[i].size()!=(size_t)T*T)dab[i].assign((size_t)T*T,0.f); else fill(dab[i].begin(),dab[i].end(),0.f);
            if((size_t)dsb[i].size()!=(size_t)T*T)dsb[i].assign((size_t)T*T,0.f);
        }
        // --- 注意力梯度 (loss 位置): dV, dA, dS, dQ, dK ---
        for(int b=0;b<B;++b) for(int h=0;h<Hq;++h){
            int kh=h/gr; const VF& s=ap[b*Hq+h];
            // dV: 从 loss 位置 i 累积到 j (j 遍历所有位置, i 从 max(j,loss_start))
            for(int j=0;j<T;++j){ float* dvj=dV.data()+(size_t)(b*T+j)*Dkv+kh*Dh; for(int i=std::max(j,loss_start);i<T;++i){ float p=s[(size_t)i*T+j]; const float* dci=dC.data()+(size_t)(b*T+i)*D+h*Dh; for(int d=0;d<Dh;++d)dvj[d]+=p*dci[d]; } }
            VF& dA=dab[b*Hq+h]; VF& dS=dsb[b*Hq+h];
            // dA: 仅 loss 位置有非零 dC
            for(int i=loss_start;i<T;++i){ const float* dci=dC.data()+(size_t)(b*T+i)*D+h*Dh; for(int j=0;j<=i;++j){ const float* vj=V.data()+(size_t)(b*T+j)*Dkv+kh*Dh; float ss=0; for(int d=0;d<Dh;++d)ss+=dci[d]*vj[d]; dA[(size_t)i*T+j]=ss; } }
            // softmax backward: dS = P * (dA - sum(dA*P)) * sc, 仅 loss 行
            for(int i=loss_start;i<T;++i){ float s2=0; for(int j=0;j<T;++j)s2+=dA[(size_t)i*T+j]*s[(size_t)i*T+j]; for(int j=0;j<=i;++j)dS[(size_t)i*T+j]=s[(size_t)i*T+j]*(dA[(size_t)i*T+j]-s2)*sc; }
            // dQ: 从 loss 位置 i 的 dS 反传到 Q[i]
            for(int i=loss_start;i<T;++i){ float* dqi=dQ.data()+(size_t)(b*T+i)*D+h*Dh; for(int j=0;j<=i;++j){ float ds=dS[(size_t)i*T+j]; const float* kj=K.data()+(size_t)(b*T+j)*Dkv+kh*Dh; for(int d=0;d<Dh;++d)dqi[d]+=ds*kj[d]; } }
            // dK: 从 loss 位置 i 的 dS 反传到 K[j] (j >= loss_start, 原代码范围)
            for(int j=loss_start;j<T;++j){ float* dkj=dK.data()+(size_t)(b*T+j)*Dkv+kh*Dh; for(int i=std::max(j,loss_start);i<T;++i){ float ds=dS[(size_t)i*T+j]; const float* qi=Q.data()+(size_t)(b*T+i)*D+h*Dh; for(int d=0;d<Dh;++d)dkj[d]+=ds*qi[d]; } }
            // ✅ FIX #1 核心修复:
            // 位置 j < loss_start 的 K/V 通过注意力参与了 loss 位置的输出,
            // 但其 dK/dV 梯度此前完全为零. 现在补充:
            //
            // dV[j] += sum_{i=loss_start..T-1} P[i][j] * dC[i]  (j < loss_start)
            //   已在上方 dV 循环中通过 i=max(j,loss_start) 覆盖
            //
            // dK[j] 需要从 loss 位置反传:
            //   对于每个 loss 行 i, dS[i][j] (j < loss_start) 非零
            //   dK[j] += sum_{i=loss_start..T-1} dS[i][j] * Q[i]
            for(int j=0;j<loss_start;++j){
                float* dkj=dK.data()+(size_t)(b*T+j)*Dkv+kh*Dh;
                for(int i=loss_start;i<T;++i){
                    float ds=dS[(size_t)i*T+j];
                    const float* qi=Q.data()+(size_t)(b*T+i)*D+h*Dh;
                    for(int d=0;d<Dh;++d) dkj[d]+=ds*qi[d];
                }
            }
        }
        // --- RoPE 反向 ---
        for(int t=0;t<BT;++t){ int local_t=t%T; for(int h=0;h<Hq;++h)rp.applyb(dQ.data()+(size_t)t*D+h*Dh,local_t); for(int h=0;h<Hkv;++h)rp.applyb(dK.data()+(size_t)t*Dkv+h*Dh,local_t); }
        // --- QK-Norm 反向 ---
        for(int t=0;t<BT;++t){
            for(int h=0;h<Hq;++h){ float* dqp=dQ.data()+(size_t)t*D+h*Dh; const float* xh=qxh.data()+(size_t)t*D+h*Dh; float r=qr[(size_t)t*Hq+h]; for(int i=0;i<Dh;++i)dqg[i]+=dqp[i]*xh[i]; float s=0; for(int i=0;i<Dh;++i){ float g=dqp[i]*qg[i]; dqp[i]=g; s+=g*xh[i]; } float id=1.f/Dh; for(int i=0;i<Dh;++i)dqp[i]=r*(dqp[i]-xh[i]*s*id); }
            for(int h=0;h<Hkv;++h){ float* dkp=dK.data()+(size_t)t*Dkv+h*Dh; const float* xh=kxh.data()+(size_t)t*Dkv+h*Dh; float r=kr[(size_t)t*Hkv+h]; for(int i=0;i<Dh;++i)dkg[i]+=dkp[i]*xh[i]; float s=0; for(int i=0;i<Dh;++i){ float g=dkp[i]*kg[i]; dkp[i]=g; s+=g*xh[i]; } float id=1.f/Dh; for(int i=0;i<Dh;++i)dkp[i]=r*(dkp[i]-xh[i]*s*id); }
        }
        // --- dWqkv, dbqkv ---
        float* dWqkv=Wqkv.gp();
        for(int d=0;d<D;++d){
            float* gwqkv=dWqkv+(size_t)d*Dqkv; float* gk=gwqkv+D; float* gv=gk+Dkv; int j=0;
            for(;j+4<=D;j+=4){ float32x4_t acc=vld1q_f32(gwqkv+j); for(int t=0;t<BT;++t){ float xv=X[t*D+d]; const float* dqp=dQ.data()+t*D; acc=vfmaq_f32(acc,vdupq_n_f32(xv),vld1q_f32(dqp+j)); } vst1q_f32(gwqkv+j,acc); }
            for(;j<D;++j){ float s=gwqkv[j]; for(int t=0;t<BT;++t){ float xv=X[t*D+d]; s+=xv*dQ[t*D+j]; } gwqkv[j]=s; }
            j=0;
            for(;j+4<=Dkv;j+=4){ float32x4_t acc1=vld1q_f32(gk+j), acc2=vld1q_f32(gv+j); for(int t=0;t<BT;++t){ float xv=X[t*D+d]; const float* dkp=dK.data()+t*Dkv; const float* dvp=dV.data()+t*Dkv; acc1=vfmaq_f32(acc1,vdupq_n_f32(xv),vld1q_f32(dkp+j)); acc2=vfmaq_f32(acc2,vdupq_n_f32(xv),vld1q_f32(dvp+j)); } vst1q_f32(gk+j,acc1); vst1q_f32(gv+j,acc2); }
            for(;j<Dkv;++j){ float s1=gk[j], s2=gv[j]; for(int t=0;t<BT;++t){ float xv=X[t*D+d]; s1+=xv*dK[t*Dkv+j]; s2+=xv*dV[t*Dkv+j]; } gk[j]=s1; gv[j]=s2; }
        }
        for(int t=0;t<BT;++t){ const float* dqp=dQ.data()+(size_t)t*D; const float* dkp=dK.data()+(size_t)t*Dkv; const float* dvp=dV.data()+(size_t)t*Dkv; for(int d=0;d<D;++d)dbqkv[d]+=dqp[d]; for(int d=0;d<Dkv;++d){ dbqkv[D+d]+=dkp[d]; dbqkv[D+Dkv+d]+=dvp[d]; } }
        dx.resize((size_t)BT*D,0.f);
        for(int t=0;t<BT;++t){ const float* dqp=dQ.data()+(size_t)t*D; const float* dkp=dK.data()+(size_t)t*Dkv; const float* dvp=dV.data()+(size_t)t*Dkv; float* dxp=dx.data()+(size_t)t*D; for(int d=0;d<D;++d){ const float* wqkv=Wqkv.p()+(size_t)d*Dqkv; const float* wk=wqkv+D; const float* wv=wk+Dkv; float32x4_t a=vdupq_n_f32(0); int j=0; for(;j+4<=D;j+=4)a=vfmaq_f32(a,vld1q_f32(dqp+j),vld1q_f32(wqkv+j)); float s=hs4(a); for(;j<D;++j)s+=dqp[j]*wqkv[j]; float32x4_t a2=vdupq_n_f32(0); int j2=0; for(;j2+4<=Dkv;j2+=4){ a2=vfmaq_f32(a2,vld1q_f32(dkp+j2),vld1q_f32(wk+j2)); a2=vfmaq_f32(a2,vld1q_f32(dvp+j2),vld1q_f32(wv+j2)); } s+=hs4(a2); for(;j2<Dkv;++j2)s+=dkp[j2]*wk[j2]+dvp[j2]*wv[j2]; dxp[d]=s; } }
    }
    void zg(){ Wqkv.zg(); Wo.zg(); fill(dbqkv.begin(),dbqkv.end(),0.f); fill(dbo.begin(),dbo.end(),0.f); fill(dqg.begin(),dqg.end(),0.f); fill(dkg.begin(),dkg.end(),0.f); }
    void ag(){ Wqkv.a.step(Wqkv.p(),Wqkv.gp(),Wqkv.n()); Wo.a.step(Wo.p(),Wo.gp(),Wo.n()); for(size_t i=0;i<bqkv.size();++i)bqkv[i]-=Wqkv.a.lr*dbqkv[i]; for(size_t i=0;i<bo.size();++i)bo[i]-=Wo.a.lr*dbo[i]; for(size_t i=0;i<qg.size();++i)qg[i]-=Wqkv.a.lr*dqg[i]; for(size_t i=0;i<kg.size();++i)kg[i]-=Wqkv.a.lr*dkg[i]; }
    size_t pc() const { return Wqkv.n()+Wo.n()+bqkv.size()+bo.size()+qg.size()+kg.size(); }
};

struct Block{
    int D=0; VF ln1,ln2,dln1,dln2; Attn at; SwiGLU ff; VF xn1,xh1,rr1,ao,aa,xn2,xh2,rr2,fo; VF dff, dln2_in, da, dat, dln1_in;
    void init(int d,int hq,int hkv,int df,mt19937& g,float lr,const Config& c){ D=d; ln1.assign(D,1.f); dln1.assign(D,0.f); ln2.assign(D,1.f); dln2.assign(D,0.f); at.init(D,hq,hkv,g,lr,c); ff.init(D,df,g,lr,c); }
    template<typename F> void fe(F f){ at.fe(f); ff.fe(f); }
    void fwd(const VF& x,VF& y,int T,const RoPE& rp, int B){
        int BT=B*T; xn1.resize((size_t)BT*D); xh1.resize((size_t)BT*D); rr1.resize(BT);
        rms_f(x.data(),ln1.data(),BT,D,xn1.data(),xh1.data(),rr1.data());
        at.fwd(xn1,ao,T,rp,B);
        aa.resize((size_t)BT*D); for(size_t i=0;i<x.size();++i)aa[i]=x[i]+ao[i];
        xn2.resize((size_t)BT*D); xh2.resize((size_t)BT*D); rr2.resize(BT);
        rms_f(aa.data(),ln2.data(),BT,D,xn2.data(),xh2.data(),rr2.data());
        ff.fwd(xn2,fo,BT);
        y.resize(x.size()); for(size_t i=0;i<x.size();++i)y[i]=aa[i]+fo[i];
    }
    // ✅ FIX #3: 传递 loss_start 至 at.bwd
    void bwd(const VF& dy,VF& dx,int T,const RoPE& rp,int B, int loss_start=0){
        int BT=B*T; dff.resize(dy.size()); dln2_in.resize(dy.size()); da.resize(dy.size()); dat.resize(dy.size()); dln1_in.resize(dy.size());
        ff.bwd(dy,dff,BT);
        rms_b(dff.data(),ln2.data(),xh2.data(),rr2.data(),BT,D,dln2_in.data(),dln2.data());
        for(size_t i=0;i<dy.size();++i)da[i]=dy[i]+dln2_in[i];
        at.bwd(da,dat,T,rp,B,loss_start);
        rms_b(dat.data(),ln1.data(),xh1.data(),rr1.data(),BT,D,dln1_in.data(),dln1.data());
        dx.resize(dy.size()); for(size_t i=0;i<dy.size();++i)dx[i]=da[i]+dln1_in[i];
    }
    void zg(){ fill(dln1.begin(),dln1.end(),0.f); fill(dln2.begin(),dln2.end(),0.f); at.zg(); ff.zg(); }
    void ag(){ float bl=at.Wqkv.a.lr; for(size_t i=0;i<ln1.size();++i)ln1[i]-=bl*dln1[i]; for(size_t i=0;i<ln2.size();++i)ln2[i]-=bl*dln2[i]; at.ag(); ff.ag(); }
    size_t pc() const { return ln1.size()+ln2.size()+at.pc()+ff.pc(); }
};

struct Sampler{
    vector<int> idx; VF p,pk; Sampler(int V){ idx.resize(V); p.resize(V); pk.resize(V); }
    int sample(const float* l,int V,float t,int tk,float tp,const vector<int>& h,int rw,float rp,mt19937& g,bool gr){
        for(int v=0;v<V;++v)p[v]=l[v]; if(rw>0&&rp>1.f){ float pen=logf(rp); int st=max(0,(int)h.size()-rw); for(int i=st;i<(int)h.size();++i){ int tt=h[i]; if(tt>=0&&tt<V)p[tt]-=pen; } }
        if(gr){ int b=0; float bv=p[0]; for(int v=1;v<V;++v)if(p[v]>bv){ bv=p[v]; b=v; } return b; }
        if(t < 1e-5f) t = 1e-5f; float mx=-1e30f; for(int v=0;v<V;++v)mx=max(mx,p[v]); float sm=0.f; for(int v=0;v<V;++v){ p[v]=expf((p[v]-mx)/t); sm+=p[v]; }
        if(!(sm>0.f))return 0; for(int v=0;v<V;++v)p[v]/=sm;
        if(tk>0&&tk<V){ for(int v=0;v<V;++v)idx[v]=v; partial_sort(idx.begin(),idx.begin()+tk,idx.end(),[&](int a,int b){ return p[a]>p[b]; }); int kp=tk; if(tp>0.f&&tp<1.f){ float s2=0.f; for(int i=0;i<tk;++i)s2+=p[idx[i]]; float cu=0.f; kp=0; for(int i=0;i<tk;++i){ cu+=p[idx[i]]/s2; ++kp; if(cu>=tp)break; } if(kp<1)kp=1; } float s3=0.f; for(int i=0;i<kp;++i){ pk[i]=p[idx[i]]; s3+=pk[i]; } float r = uniform_real_distribution<float>(0.f, s3)(g); float acc = 0.f; for(int i=0;i<kp;++i){ acc += pk[i]; if(r <= acc) return idx[i]; } return idx[kp-1]; }
        float r = uniform_real_distribution<float>(0.f, sm)(g); float acc = 0.f; for(int v=0;v<V;++v){ acc += p[v]; if(r <= acc) return v; } return V-1;
    }
};

struct InferCtx{
    int kv = 0; vector<VF> kK, kV; VF s_cur, s_xn1, s_xh1, s_q, s_k, s_v, s_ao, s_ay, s_aa; VF s_xn2, s_xh2, s_h1, s_h3, s_g, s_fo, s_hf, s_sc; VF lg; Sampler sampler; mt19937 rng;
    InferCtx(int V, unsigned seed) : sampler(V), rng(seed) {}
    void prepare(int L, int maxT, int Dkv){ if((int)kK.size() != L || (L > 0 && (int)kK[0].size() != (size_t)maxT * Dkv)){ kK.assign(L, VF((size_t)maxT * Dkv, 0.f)); kV.assign(L, VF((size_t)maxT * Dkv, 0.f)); } kv = 0; }
};

struct Model{
    int V=0,D=0,L=0,Hq=0,Hkv=0,Df=0,maxT=0; VF E,dE; AdamW aE; vector<Block> bl; VF lnf,dlnf; VF x,fh,lg,xhf,rrf; int kv=0; vector<VF> kK,kV; VF s_cur,s_xn1,s_xh1,s_q,s_k,s_v,s_ao,s_ay,s_aa,s_xn2,s_xh2,s_h1,s_h3,s_g,s_fo,s_hf,s_sc; RoPE rp; vector<Dropout> df; LRSched sch; VF fwd_ping, fwd_pong; VF bwd_ping, bwd_pong; VF bwd_dl, bwd_dh, bwd_dfin;
    void init(int v,int d,int l,int hq,int hkv,int mt,mt19937& g,const Config& c){
        V=v; D=d; L=l; Hq=hq; Hkv=hkv; maxT=mt; if(D%Hq||Hq%Hkv||(D/Hq)%2){ cerr<<"[fatal] D/Hq/Hkv\n"; exit(1); }
        Df=(int)(4.0*D*2.0/3.0); Df=(Df+3)&~3; if(c.df_override>0){ Df=c.df_override; Df=(Df+3)&~3; }
        E.assign((size_t)V*D,0.f); normal_distribution<float> dd(0.f,0.02f); for(auto& vv:E)vv=dd(g);
        dE.assign((size_t)V*D,0.f); aE.init(E.size(),c.lr,c.b1,c.b2,c.eps,c.weight_decay);
        bl.clear(); bl.reserve(L); for(int i=0;i<L;++i){ Block b; b.init(D,Hq,Hkv,Df,g,c.lr,c); bl.push_back(std::move(b)); }
        lnf.assign(D,1.f); dlnf.assign(D,0.f); rp.init(D/Hq,maxT,10000.f); df.assign(L,Dropout()); for(int i=0;i<L;++i)df[i].p=c.dropout;
        sch.base_lr=c.lr; sch.min_lr=c.min_lr; sch.warmup=c.warmup; sch.kind=c.lr_sched;
    }
    size_t pc() const { size_t n=E.size(); for(auto& b:bl)n+=b.pc(); n+=lnf.size(); return n; }
    void set_total(long long t){ sch.total=t; }
    void upd_lr(long long s){ float lr=sch.get(s); aE.lr=lr; auto set=[&](AdamW& a){ a.lr=lr; }; for(auto& b:bl)b.fe(set); }
    void zg(){ fill(dE.begin(),dE.end(),0.f); for(auto& b:bl)b.zg(); fill(dlnf.begin(),dlnf.end(),0.f); }
    void ag(){ aE.step(E.data(),dE.data(),E.size()); for(auto& b:bl)b.ag(); for(size_t i=0;i<lnf.size();++i)lnf[i]-=aE.lr*dlnf[i]; }
    // ✅ FIX #4: 新增 step_counter 参数, dropout RNG 每 step 混入步数
    void fwd(const vector<int>& ids,mt19937& g, int B, long long step_counter=0){
        int T=(int)ids.size() / B; int BT=B*T; x.resize((size_t)BT*D);
        for(int t=0;t<BT;++t)memcpy(x.data()+(size_t)t*D,E.data()+(size_t)ids[t]*D,D*4);
        fwd_ping.resize(x.size()); fwd_ping.assign(x.begin(), x.end()); fwd_pong.resize(x.size()); VF* cur = &fwd_ping; VF* nxt = &fwd_pong;
        for(int l=0;l<L;++l){
            if(df[l].p>0.f){
                // ✅ FIX #4: 每 step 用不同种子, 避免 epoch 间 dropout mask 高度相关
                mt19937 drop_rng(g() ^ (unsigned)(step_counter * 0x9e3779b9u + l));
                bl[l].fwd(*cur, *nxt, T, rp, B);
                df[l].fwd(nxt->data(),BT*D,drop_rng);
            } else {
                bl[l].fwd(*cur, *nxt, T, rp, B);
            }
            std::swap(cur, nxt);
        }
        fh.resize((size_t)BT*D); xhf.resize((size_t)BT*D); rrf.resize(BT); rms_f(cur->data(),lnf.data(),BT,D,fh.data(),xhf.data(),rrf.data());
        lg.resize((size_t)BT*V); for(int t=0;t<BT;++t){ const float* hp=fh.data()+(size_t)t*D; float* lp=lg.data()+(size_t)t*V; for(int v=0;v<V;++v){ const float* ep=E.data()+(size_t)v*D; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4)a=vfmaq_f32(a,vld1q_f32(hp+d),vld1q_f32(ep+d)); float s=hs4(a); for(;d<D;++d)s+=hp[d]*ep[d]; lp[v]=s; } }
    }
    float bwd(const vector<int>& ids,const vector<int>& tg,int loss_start=0, int B=4){
        int T=(int)ids.size() / B; int BT=B*T; if(loss_start<0)loss_start=0; if(loss_start>T)loss_start=T;
        bwd_dl.resize((size_t)BT*V); fill(bwd_dl.begin(), bwd_dl.end(), 0.f); float loss=0.f; int nv=0;
        for(int t=loss_start;t<T;++t){ for(int b=0;b<B;++b){ int idx = b*T + t; const float* lp=lg.data()+(size_t)idx*V; float* dp=bwd_dl.data()+(size_t)idx*V; float mx=-1e30f; for(int v=0;v<V;++v)mx=max(mx,lp[v]); float sm=0.f; for(int v=0;v<V;++v){ dp[v]=expf(lp[v]-mx); sm+=dp[v]; } loss+=-logf(max(dp[tg[idx]]/sm,1e-9f)); float iv=1.f/sm; for(int v=0;v<V;++v)dp[v]*=iv; dp[tg[idx]]-=1.f; ++nv; } }
        if(nv==0)return 0.f; float inv=1.f/(float)nv; for(int t=loss_start;t<T;++t){ for(int b=0;b<B;++b){ int idx = b*T + t; float* dp=bwd_dl.data()+(size_t)idx*V; for(int v=0;v<V;++v)dp[v]*=inv; } } loss*=inv;
        bwd_dh.resize((size_t)BT*D); fill(bwd_dh.begin(), bwd_dh.end(), 0.f);
        for(int t=0;t<BT;++t){ const float* dp=bwd_dl.data()+(size_t)t*V; float* dhp=bwd_dh.data()+(size_t)t*D; for(int v=0;v<V;++v){ float dv=dp[v]; const float* ep=E.data()+(size_t)v*D; int d=0; float32x4_t dv4=vdupq_n_f32(dv); for(;d+4<=D;d+=4)vst1q_f32(dhp+d,vfmaq_f32(vld1q_f32(dhp+d),dv4,vld1q_f32(ep+d))); for(;d<D;++d)dhp[d]+=dv*ep[d]; } }
        for(int t=0;t<BT;++t){ const float* hp=fh.data()+(size_t)t*D; const float* dp=bwd_dl.data()+(size_t)t*V; for(int v=0;v<V;++v){ float dv=dp[v]; float* dep=dE.data()+(size_t)v*D; int d=0; float32x4_t dv4=vdupq_n_f32(dv); for(;d+4<=D;d+=4)vst1q_f32(dep+d,vfmaq_f32(vld1q_f32(dep+d),dv4,vld1q_f32(hp+d))); for(;d<D;++d)dep[d]+=dv*hp[d]; } }
        bwd_dfin.resize((size_t)BT*D); rms_b(bwd_dh.data(),lnf.data(),xhf.data(),rrf.data(),BT,D,bwd_dfin.data(),dlnf.data());
        bwd_ping.resize(bwd_dfin.size()); bwd_ping.assign(bwd_dfin.begin(), bwd_dfin.end()); bwd_pong.resize(bwd_dfin.size()); VF* cur = &bwd_ping; VF* nxt = &bwd_pong;
        // ✅ FIX #3: 传递 loss_start 至每层 Block::bwd
        for(int l=L-1;l>=0;--l){ if(df[l].p>0.f)df[l].bwd(cur->data(),BT*D); bl[l].bwd(*cur, *nxt, T, rp, B, loss_start); std::swap(cur, nxt); }
        for(int t=0;t<BT;++t){ const float* dp=cur->data()+(size_t)t*D; float* dep=dE.data()+(size_t)ids[t]*D; for(int d=0;d<D;++d)dep[d]+=dp[d]; }
        return loss;
    }
    void kv_reset(){ kv=0; int Dkv=Hkv*(D/Hq); if(kK.empty() || kK.size()!=L || kK[0].size()!=(size_t)maxT*Dkv){ kK.assign(L,VF((size_t)maxT*Dkv,0.f)); kV.assign(L,VF((size_t)maxT*Dkv,0.f)); } }
    const float* step(int id);
    const float* step_ctx(int id, InferCtx& ctx);
};

// ================== Model::step (单步推理，KV cache) ==================
const float* Model::step(int id){
    int Dh=D/Hq,Dkv=Hkv*Dh,gr=Hq/Hkv; float sc=1.f/sqrtf((float)Dh);
    auto en=[&](VF& v,int n){ if((int)v.size()!=n)v.resize(n); };
    en(s_cur,D); en(s_xn1,D); en(s_xh1,D); en(s_q,D); en(s_k,Dkv); en(s_v,Dkv);
    en(s_ao,D); en(s_ay,D); en(s_aa,D); en(s_xn2,D); en(s_xh2,D); en(s_hf,D);
    VF& cur=s_cur; memcpy(cur.data(),E.data()+(size_t)id*D,D*4);
    for(int l=0;l<L;++l){
        Block& b=bl[l];
        VF& xn1=s_xn1; VF& xh1=s_xh1;
        { float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(cur.data()+d); a=vfmaq_f32(a,v,v); }
          float m=hs4(a); for(;d<D;++d)m+=cur[d]*cur[d]; m/=D; float r=1.f/sqrtf(m+1e-6f);
          float32x4_t rv=vdupq_n_f32(r); int i=0;
          for(;i+4<=D;i+=4){ vst1q_f32(xh1.data()+i, vmulq_f32(vld1q_f32(cur.data()+i), rv)); vst1q_f32(xn1.data()+i, vmulq_f32(vld1q_f32(xh1.data()+i), vld1q_f32(b.ln1.data()+i))); }
          for(;i<D;++i){ xh1[i]=cur[i]*r; xn1[i]=xh1[i]*b.ln1[i]; } }
        VF& ao=s_ao; memset(ao.data(),0,D*4);
        VF& q=s_q; VF& k=s_k; VF& v=s_v;
        memcpy(q.data(),b.at.bqkv.data(),D*4); memcpy(k.data(),b.at.bqkv.data()+D,Dkv*4); memcpy(v.data(),b.at.bqkv.data()+D+Dkv,Dkv*4);
        for(int d=0;d<D;++d){ float xv=xn1[d];
            const float* wqkv=b.at.Wqkv.p()+(size_t)d*b.at.Dqkv; const float* wk=wqkv+D; const float* wv=wk+Dkv;
            int j=0; float32x4_t x4=vdupq_n_f32(xv);
            for(;j+4<=D;j+=4)vst1q_f32(q.data()+j,vfmaq_f32(vld1q_f32(q.data()+j),x4,vld1q_f32(wqkv+j)));
            for(;j<D;++j)q[j]+=xv*wqkv[j];
            j=0;
            for(;j+4<=Dkv;j+=4){ vst1q_f32(k.data()+j,vfmaq_f32(vld1q_f32(k.data()+j),x4,vld1q_f32(wk+j))); vst1q_f32(v.data()+j,vfmaq_f32(vld1q_f32(v.data()+j),x4,vld1q_f32(wv+j))); }
            for(;j<Dkv;++j){ k[j]+=xv*wk[j]; v[j]+=xv*wv[j]; }
        }
        for(int h=0;h<Hq;++h){ float* qp=q.data()+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4){ float32x4_t vv=vld1q_f32(qp+d); a=vfmaq_f32(a,vv,vv); }
            float m=hs4(a); for(;d<Dh;++d)m+=qp[d]*qp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f);
            for(int i=0;i<Dh;++i)qp[i]=qp[i]*r*b.at.qg[i]; }
        for(int h=0;h<Hkv;++h){ float* kp=k.data()+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4){ float32x4_t vv=vld1q_f32(kp+d); a=vfmaq_f32(a,vv,vv); }
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
        // ✅ FIX #2: Wo bias 不再双重添加
        // 原代码: ay 初始化为 bo, 然后 ao (=cc+bo) * Wo → y = (cc+bo)*Wo + bo (bo 出现两次)
        // 修复后: ay 初始化为零, y = (cc+bo)*Wo (bo 仅在 cc 中出现一次, 与训练 fwd 一致)
        VF& ay=s_ay; memcpy(ay.data(),b.at.bo.data(),D*4);
        for(int d=0;d<D;++d){ float cv=ao[d]; const float* wr=b.at.Wo.p()+(size_t)d*D; int j=0; float32x4_t c4=vdupq_n_f32(cv);
            for(;j+4<=D;j+=4)vst1q_f32(ay.data()+j,vfmaq_f32(vld1q_f32(ay.data()+j),c4,vld1q_f32(wr+j)));
            for(;j<D;++j)ay[j]+=cv*wr[j]; }
        for(int d=0;d<D;++d)ao[d]=ay[d];
        VF& aa=s_aa; for(int d=0;d<D;++d)aa[d]=cur[d]+ao[d];
        VF& xn2=s_xn2; VF& xh2=s_xh2;
        { float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4){ float32x4_t vv=vld1q_f32(aa.data()+d); a=vfmaq_f32(a,vv,vv); }
          float m=hs4(a); for(;d<D;++d)m+=aa[d]*aa[d]; m/=D; float r=1.f/sqrtf(m+1e-6f);
          float32x4_t rv=vdupq_n_f32(r); int i=0;
          for(;i+4<=D;i+=4){ vst1q_f32(xh2.data()+i, vmulq_f32(vld1q_f32(aa.data()+i), rv)); vst1q_f32(xn2.data()+i, vmulq_f32(vld1q_f32(xh2.data()+i), vld1q_f32(b.ln2.data()+i))); }
          for(;i<D;++i){ xh2[i]=aa[i]*r; xn2[i]=xh2[i]*b.ln2[i]; } }
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
    float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(cur.data()+d); a=vfmaq_f32(a,v,v); }
    float m=hs4(a); for(;d<D;++d)m+=cur[d]*cur[d]; m/=D; float r=1.f/sqrtf(m+1e-6f); VF& hf=s_hf;
    float32x4_t rv=vdupq_n_f32(r); int i=0;
    for(;i+4<=D;i+=4)vst1q_f32(hf.data()+i, vmulq_f32(vmulq_f32(vld1q_f32(cur.data()+i), rv), vld1q_f32(lnf.data()+i)));
    for(;i<D;++i)hf[i]=cur[i]*r*lnf[i];
    lg.resize(V);
    for(int v=0;v<V;++v){ const float* ep=E.data()+(size_t)v*D; float32x4_t ac=vdupq_n_f32(0); int dd=0;
        for(;dd+4<=D;dd+=4)ac=vfmaq_f32(ac,vld1q_f32(hf.data()+dd),vld1q_f32(ep+dd));
        float s=hs4(ac); for(;dd<D;++dd)s+=hf[dd]*ep[dd]; lg[v]=s; }
    kv++; return lg.data();
}

const float* Model::step_ctx(int id, InferCtx& ctx){
    int Dh=D/Hq,Dkv=Hkv*Dh,gr=Hq/Hkv; float sc=1.f/sqrtf((float)Dh);
    auto en=[&](VF& v,int n){ if((int)v.size()!=n)v.resize(n); };
    en(ctx.s_cur,D); en(ctx.s_xn1,D); en(ctx.s_xh1,D); en(ctx.s_q,D); en(ctx.s_k,Dkv); en(ctx.s_v,Dkv);
    en(ctx.s_ao,D); en(ctx.s_ay,D); en(ctx.s_aa,D); en(ctx.s_xn2,D); en(ctx.s_xh2,D); en(ctx.s_hf,D);
    VF& cur=ctx.s_cur; memcpy(cur.data(),E.data()+(size_t)id*D,D*4);
    for(int l=0;l<L;++l){
        Block& b=bl[l];
        VF& xn1=ctx.s_xn1; VF& xh1=ctx.s_xh1;
        { float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(cur.data()+d); a=vfmaq_f32(a,v,v); }
          float m=hs4(a); for(;d<D;++d)m+=cur[d]*cur[d]; m/=D; float r=1.f/sqrtf(m+1e-6f);
          float32x4_t rv=vdupq_n_f32(r); int i=0;
          for(;i+4<=D;i+=4){ vst1q_f32(xh1.data()+i, vmulq_f32(vld1q_f32(cur.data()+i), rv)); vst1q_f32(xn1.data()+i, vmulq_f32(vld1q_f32(xh1.data()+i), vld1q_f32(b.ln1.data()+i))); }
          for(;i<D;++i){ xh1[i]=cur[i]*r; xn1[i]=xh1[i]*b.ln1[i]; } }
        VF& ao=ctx.s_ao; memset(ao.data(),0,D*4);
        VF& q=ctx.s_q; VF& k=ctx.s_k; VF& v=ctx.s_v;
        memcpy(q.data(),b.at.bqkv.data(),D*4); memcpy(k.data(),b.at.bqkv.data()+D,Dkv*4); memcpy(v.data(),b.at.bqkv.data()+D+Dkv,Dkv*4);
        for(int d=0;d<D;++d){ float xv=xn1[d];
            const float* wqkv=b.at.Wqkv.p()+(size_t)d*b.at.Dqkv; const float* wk=wqkv+D; const float* wv=wk+Dkv;
            int j=0; float32x4_t x4=vdupq_n_f32(xv);
            for(;j+4<=D;j+=4)vst1q_f32(q.data()+j,vfmaq_f32(vld1q_f32(q.data()+j),x4,vld1q_f32(wqkv+j)));
            for(;j<D;++j)q[j]+=xv*wqkv[j];
            j=0;
            for(;j+4<=Dkv;j+=4){ vst1q_f32(k.data()+j,vfmaq_f32(vld1q_f32(k.data()+j),x4,vld1q_f32(wk+j))); vst1q_f32(v.data()+j,vfmaq_f32(vld1q_f32(v.data()+j),x4,vld1q_f32(wv+j))); }
            for(;j<Dkv;++j){ k[j]+=xv*wk[j]; v[j]+=xv*wv[j]; } }
        for(int h=0;h<Hq;++h){ float* qp=q.data()+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4){ float32x4_t vv=vld1q_f32(qp+d); a=vfmaq_f32(a,vv,vv); }
            float m=hs4(a); for(;d<Dh;++d)m+=qp[d]*qp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f);
            for(int i=0;i<Dh;++i)qp[i]=qp[i]*r*b.at.qg[i]; }
        for(int h=0;h<Hkv;++h){ float* kp=k.data()+h*Dh; float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=Dh;d+=4){ float32x4_t vv=vld1q_f32(kp+d); a=vfmaq_f32(a,vv,vv); }
            float m=hs4(a); for(;d<Dh;++d)m+=kp[d]*kp[d]; m/=Dh; float r=1.f/sqrtf(m+1e-6f);
            for(int i=0;i<Dh;++i)kp[i]=kp[i]*r*b.at.kg[i]; }
        for(int h=0;h<Hq;++h)rp.apply(q.data()+h*Dh,ctx.kv);
        for(int h=0;h<Hkv;++h)rp.apply(k.data()+h*Dh,ctx.kv);
        float* kc=ctx.kK[l].data()+(size_t)ctx.kv*Dkv; float* vc=ctx.kV[l].data()+(size_t)ctx.kv*Dkv;
        memcpy(kc,k.data(),Dkv*4); memcpy(vc,v.data(),Dkv*4);
        int Tf=ctx.kv+1;
        for(int h=0;h<Hq;++h){ int kh=h/gr;
            if((int)ctx.s_sc.size()<Tf)ctx.s_sc.resize(Tf); VF& s=ctx.s_sc; const float* qh=q.data()+h*Dh; const float* kb=ctx.kK[l].data(); const float* vb=ctx.kV[l].data();
            for(int j=0;j<Tf;++j){ const float* kj=kb+(size_t)j*Dkv+kh*Dh; float32x4_t a=vdupq_n_f32(0); int d=0;
                for(;d+4<=Dh;d+=4)a=vfmaq_f32(a,vld1q_f32(qh+d),vld1q_f32(kj+d));
                float ss=hs4(a); for(;d<Dh;++d)ss+=qh[d]*kj[d]; s[j]=ss*sc; }
            float mx=-1e30f; for(int j=0;j<Tf;++j)mx=max(mx,s[j]);
            float sm=0.f; for(int j=0;j<Tf;++j){ s[j]=expf(s[j]-mx); sm+=s[j]; }
            for(int j=0;j<Tf;++j)s[j]/=sm;
            float* oh=ao.data()+h*Dh;
            for(int j=0;j<Tf;++j){ const float* vj=vb+(size_t)j*Dkv+kh*Dh; float p=s[j]; for(int d=0;d<Dh;++d)oh[d]+=p*vj[d]; } }
        // ✅ FIX #2: step_ctx 同样修复 Wo bias 双重添加
        VF& ay=ctx.s_ay; memcpy(ay.data(),b.at.bo.data(),D*4);
        for(int d=0;d<D;++d){ float cv=ao[d]; const float* wr=b.at.Wo.p()+(size_t)d*D; int j=0; float32x4_t c4=vdupq_n_f32(cv);
            for(;j+4<=D;j+=4)vst1q_f32(ay.data()+j,vfmaq_f32(vld1q_f32(ay.data()+j),c4,vld1q_f32(wr+j)));
            for(;j<D;++j)ay[j]+=cv*wr[j]; }
        for(int d=0;d<D;++d)ao[d]=ay[d];
        VF& aa=ctx.s_aa; for(int d=0;d<D;++d)aa[d]=cur[d]+ao[d];
        VF& xn2=ctx.s_xn2; VF& xh2=ctx.s_xh2;
        { float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4){ float32x4_t vv=vld1q_f32(aa.data()+d); a=vfmaq_f32(a,vv,vv); }
          float m=hs4(a); for(;d<D;++d)m+=aa[d]*aa[d]; m/=D; float r=1.f/sqrtf(m+1e-6f);
          float32x4_t rv=vdupq_n_f32(r); int i=0;
          for(;i+4<=D;i+=4){ vst1q_f32(xh2.data()+i, vmulq_f32(vld1q_f32(aa.data()+i), rv)); vst1q_f32(xn2.data()+i, vmulq_f32(vld1q_f32(xh2.data()+i), vld1q_f32(b.ln2.data()+i))); }
          for(;i<D;++i){ xh2[i]=aa[i]*r; xn2[i]=xh2[i]*b.ln2[i]; } }
        if((int)ctx.s_fo.size()!=D)ctx.s_fo.resize(D); VF& fo=ctx.s_fo;
        int Df=b.ff.Df;
        if((int)ctx.s_h1.size()!=Df)ctx.s_h1.resize(Df); if((int)ctx.s_h3.size()!=Df)ctx.s_h3.resize(Df); if((int)ctx.s_g.size()!=Df)ctx.s_g.resize(Df);
        VF& h1=ctx.s_h1; VF& h3=ctx.s_h3; VF& g_=ctx.s_g;
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
    float32x4_t a=vdupq_n_f32(0); int d=0; for(;d+4<=D;d+=4){ float32x4_t v=vld1q_f32(cur.data()+d); a=vfmaq_f32(a,v,v); }
    float m=hs4(a); for(;d<D;++d)m+=cur[d]*cur[d]; m/=D; float r=1.f/sqrtf(m+1e-6f); VF& hf=ctx.s_hf;
    float32x4_t rv=vdupq_n_f32(r); int i=0;
    for(;i+4<=D;i+=4)vst1q_f32(hf.data()+i, vmulq_f32(vmulq_f32(vld1q_f32(cur.data()+i), rv), vld1q_f32(lnf.data()+i)));
    for(;i<D;++i)hf[i]=cur[i]*r*lnf[i];
    ctx.lg.resize(V);
    for(int v=0;v<V;++v){ const float* ep=E.data()+(size_t)v*D; float32x4_t ac=vdupq_n_f32(0); int dd=0;
        for(;dd+4<=D;dd+=4)ac=vfmaq_f32(ac,vld1q_f32(hf.data()+dd),vld1q_f32(ep+dd));
        float s=hs4(ac); for(;dd<D;++dd)s+=hf[dd]*ep[dd]; ctx.lg[v]=s; }
    ctx.kv++; return ctx.lg.data();
}
static const char MG[4]={'T','R','N','S'}; static const uint32_t VER=9; static const char CM[4]={'T','C','K','P'}; static const uint32_t CV=13;
static void wv(ofstream& f,const VF& v){ int32_t n=(int32_t)v.size(); f.write((char*)&n,4); if(n>0)f.write((const char*)v.data(),n*4); }
static void rv(ifstream& f,VF& v){ int32_t n; f.read((char*)&n,4); if(!f || n<0 || n>100000000) { cerr<<"[fatal] ckpt 维度越界或损坏\n"; exit(1); } v.resize(n); if(n>0){ f.read((char*)v.data(),n*4); if(!f) exit(1); } }
static void wW(ofstream& f,const W& w){ wv(f,w.v); wv(f,w.a.m); wv(f,w.a.v); f.write((char*)&w.a.t,8); f.write((char*)&w.a.bc1,4); f.write((char*)&w.a.bc2,4); f.write((char*)&w.a.base_lr,4); f.write((char*)&w.a.wd,4); }
static void rW(ifstream& f,W& w){ rv(f,w.v); rv(f,w.a.m); rv(f,w.a.v); f.read((char*)&w.a.t,8); f.read((char*)&w.a.bc1,4); f.read((char*)&w.a.bc2,4); f.read((char*)&w.a.base_lr,4); f.read((char*)&w.a.wd,4); w.a.lr=w.a.base_lr; w.g.assign(w.v.size(),0.f); }
static void wvstr(ofstream& f,const vector<string>& vs){ int32_t N=(int32_t)vs.size(); f.write((char*)&N,4); for(auto& s:vs){ int32_t L=(int32_t)s.size(); f.write((char*)&L,4); f.write(s.data(),L); } }
static void rvstr(ifstream& f,vector<string>& vs,unordered_map<string,int>& c2i){ int32_t N; f.read((char*)&N,4); if(!f || N<0 || N>1000000) exit(1); vs.resize(N); c2i.clear(); for(int i=0;i<N;++i){ int32_t L; f.read((char*)&L,4); if(!f || L<0 || L>100000000) exit(1); vs[i].resize(L); if(L>0){ f.read(&vs[i][0],L); if(!f) exit(1); } c2i[vs[i]]=i; } }
static void wvocab(ofstream& f,const Vocab& v){ wvstr(f,v.vocab); int32_t nch=v.nchars; f.write((char*)&nch,4); int32_t nm=(int32_t)v.merges.size(); f.write((char*)&nm,4); for(auto& mrg:v.merges){ int32_t a=mrg[0],b=mrg[1],c=mrg[2]; f.write((char*)&a,4); f.write((char*)&b,4); f.write((char*)&c,4); } }
static void rvocab(ifstream& f,Vocab& v){ rvstr(f,v.vocab,v.tok2id); v.specials.clear(); v.special_id.clear(); int ns = (int)v.vocab.size() < 3 ? (int)v.vocab.size() : 3; for(int i=0;i<ns;++i){ v.specials.push_back(v.vocab[i]); v.special_id[v.vocab[i]]=i; } int32_t nch; f.read((char*)&nch,4); v.nchars=nch; v.char2id.clear(); for(int id=0;id<nch && id<(int)v.vocab.size();++id){ if(v.special_id.find(v.vocab[id])!=v.special_id.end())continue; if(split_utf8(v.vocab[id]).size()==1)v.char2id[v.vocab[id]]=id; } int32_t nm; f.read((char*)&nm,4); v.merges.resize(nm); v.merge_rank.clear(); v.merge_result.clear(); for(int i=0;i<nm;++i){ int32_t a,b,c; f.read((char*)&a,4); f.read((char*)&b,4); f.read((char*)&c,4); v.merges[i]={a,b,c}; long long k=Vocab::key(a,b); v.merge_rank[k]=i; v.merge_result[k]=c; } }
static void wblk(ofstream& f,const Block& b){ wv(f,b.ln1); wv(f,b.ln2); wW(f,b.at.Wqkv); wW(f,b.at.Wo); wv(f,b.at.bqkv); wv(f,b.at.bo); wv(f,b.at.qg); wv(f,b.at.kg); wW(f,b.ff.W1); wW(f,b.ff.W3); wW(f,b.ff.W2); wv(f,b.ff.b1); wv(f,b.ff.b3); wv(f,b.ff.b2); }
static void rblk(ifstream& f,Block& b){ rv(f,b.ln1); rv(f,b.ln2); rW(f,b.at.Wqkv); rW(f,b.at.Wo); rv(f,b.at.bqkv); rv(f,b.at.bo); rv(f,b.at.qg); rv(f,b.at.kg); rW(f,b.ff.W1); rW(f,b.ff.W3); rW(f,b.ff.W2); rv(f,b.ff.b1); rv(f,b.ff.b3); rv(f,b.ff.b2); }
static bool savem(const string& p,const Model& m,const Vocab& voc){ string tmp=p+".tmp"; ofstream f(tmp,ios::binary); if(!f)return 0; f.write(MG,4); uint32_t v=VER; f.write((char*)&v,4); int32_t V=m.V,D=m.D,L=m.L,Hq=m.Hq,Hkv=m.Hkv,mt=m.maxT,Df=m.Df; f.write((char*)&V,4); f.write((char*)&D,4); f.write((char*)&L,4); f.write((char*)&Hq,4); f.write((char*)&Hkv,4); f.write((char*)&mt,4); f.write((char*)&Df,4); wvocab(f,voc); wv(f,m.E); wv(f,m.lnf); for(const auto& b:m.bl)wblk(f,b); f.flush(); if(!f.good()){ cerr<<"model 写失败\n"; f.close(); std::remove(tmp.c_str()); return 0; } f.close(); if(std::rename(tmp.c_str(),p.c_str())!=0){ string bak = p + ".bak"; std::rename(p.c_str(), bak.c_str()); ifstream s(tmp,ios::binary); ofstream d(p,ios::binary); if(!s || !d){ std::rename(bak.c_str(),p.c_str()); std::remove(tmp.c_str()); return 0; } d << s.rdbuf(); std::remove(tmp.c_str()); std::remove(bak.c_str()); } return 1; }
static bool loadm(const string& p,Model& m,Vocab& voc,const Config& c){ ifstream f(p,ios::binary); if(!f)return 0; char mg[4]; f.read(mg,4); if(memcmp(mg,MG,4)){ cerr<<"magic\n"; return 0; } uint32_t v; f.read((char*)&v,4); if(v!=VER){ cerr<<"版本 "<<v<<" 不兼容（需要 "<<VER<<"）\n"; return 0; } int32_t V,D,L,Hq,Hkv,mt,Df; f.read((char*)&V,4); f.read((char*)&D,4); f.read((char*)&L,4); f.read((char*)&Hq,4); f.read((char*)&Hkv,4); f.read((char*)&mt,4); f.read((char*)&Df,4); rvocab(f,voc); if(V != (int)voc.vocab.size()){ cerr<<"model V 与 vocab 大小不一致: V="<<V<<" vocab="<<voc.vocab.size()<<"\n"; return 0; } Config c2=c; c2.df_override=Df; mt19937 dg(0); m.init(V,D,L,Hq,Hkv,mt,dg,c2); rv(f,m.E); rv(f,m.lnf); if((int)m.E.size() != V*D){ cerr<<"model E 大小异常: "<<m.E.size()<<" != "<<(long long)V*D<<"\n"; return 0; } for(auto& b:m.bl)rblk(f,b); return 1; }
static bool saveck(const string& p,const Model& m,const Vocab& voc,int ep,long long st,float bl,mt19937& g){ string tmp=p+".tmp"; ofstream f(tmp,ios::binary); if(!f)return 0; f.write(CM,4); uint32_t v=CV; f.write((char*)&v,4); int32_t V=m.V,D=m.D,L=m.L,Hq=m.Hq,Hkv=m.Hkv,mt=m.maxT,Df=m.Df; f.write((char*)&V,4); f.write((char*)&D,4); f.write((char*)&L,4); f.write((char*)&Hq,4); f.write((char*)&Hkv,4); f.write((char*)&mt,4); f.write((char*)&Df,4); int32_t e=ep; f.write((char*)&e,4); f.write((char*)&st,8); f.write((char*)&bl,4); ostringstream oss; oss<<g; string rs=oss.str(); int32_t rn=(int32_t)rs.size(); f.write((char*)&rn,4); f.write(rs.data(),rn); wvocab(f,voc); wv(f,m.E); wv(f,m.aE.m); wv(f,m.aE.v); f.write((char*)&m.aE.t,8); f.write((char*)&m.aE.bc1,4); f.write((char*)&m.aE.bc2,4); f.write((char*)&m.aE.base_lr,4); f.write((char*)&m.aE.wd,4); f.write((char*)&m.sch.base_lr,4); f.write((char*)&m.sch.min_lr,4); f.write((char*)&m.sch.warmup,8); f.write((char*)&m.sch.total,8); { int32_t klen=(int32_t)m.sch.kind.size(); if(klen>128)klen=128; f.write((char*)&klen,4); if(klen>0)f.write(m.sch.kind.data(),klen); } wv(f,m.lnf); for(const auto& b:m.bl)wblk(f,b); f.flush(); if(!f.good()){ cerr<<"ckpt 写失败\n"; f.close(); std::remove(tmp.c_str()); return 0; } f.close(); if(std::rename(tmp.c_str(),p.c_str())!=0){ string bak = p + ".bak"; std::rename(p.c_str(), bak.c_str()); ifstream s(tmp,ios::binary); ofstream d(p,ios::binary); if(!s || !d){ std::rename(bak.c_str(),p.c_str()); std::remove(tmp.c_str()); return 0; } d << s.rdbuf(); std::remove(tmp.c_str()); std::remove(bak.c_str()); } return 1; }
static bool loadck(const string& p,Model& m,Vocab& voc,Config& c,int& ep,long long& st,float& bl,mt19937& g){ ifstream f(p,ios::binary); if(!f)return 0; char mg[4]; f.read(mg,4); if(memcmp(mg,CM,4)){ cerr<<"ckpt magic\n"; return 0; } uint32_t v; f.read((char*)&v,4); if(v!=CV){ cerr<<"ckpt 版本 "<<v<<" 不兼容（需要 "<<CV<<"）\n"; return 0; } int32_t V,D,L,Hq,Hkv,mt,Df; f.read((char*)&V,4); f.read((char*)&D,4); f.read((char*)&L,4); f.read((char*)&Hq,4); f.read((char*)&Hkv,4); f.read((char*)&mt,4); f.read((char*)&Df,4); f.read((char*)&ep,4); f.read((char*)&st,8); f.read((char*)&bl,4); int32_t rn; f.read((char*)&rn,4); string rs(rn>0?rn:0,0); if(rn>0)f.read(&rs[0],rn); istringstream iss(rs); iss>>g; if(!iss){ cerr<<"[warn] mt19937 状态恢复失败\n"; } rvocab(f,voc); if(V != (int)voc.vocab.size()){ cerr<<"ckpt V 与 vocab 大小不一致\n"; return 0; } Config c2=c; c2.df_override=Df; mt19937 dg(0); m.init(V,D,L,Hq,Hkv,mt,dg,c2); rv(f,m.E); rv(f,m.aE.m); rv(f,m.aE.v); if((int)m.E.size() != V*D){ cerr<<"ckpt E 大小异常\n"; return 0; } f.read((char*)&m.aE.t,8); f.read((char*)&m.aE.bc1,4); f.read((char*)&m.aE.bc2,4); f.read((char*)&m.aE.base_lr,4); m.aE.lr=m.aE.base_lr; f.read((char*)&m.aE.wd,4); f.read((char*)&m.sch.base_lr,4); f.read((char*)&m.sch.min_lr,4); f.read((char*)&m.sch.warmup,8); f.read((char*)&m.sch.total,8); { int32_t klen; f.read((char*)&klen,4); if(klen<0 || klen>1024) exit(1); if(klen>0){ string ks(klen,0); f.read(&ks[0],klen); if(klen<64) m.sch.kind=ks; else m.sch.kind=ks.substr(0,63); } } rv(f,m.lnf); for(auto& b:m.bl)rblk(f,b); return 1; }

static string lt(const string& p){ ifstream f(p,ios::binary); if(!f)return ""; return string((istreambuf_iterator<char>(f)),istreambuf_iterator<char>()); }
static bool iseos(const string& s, const string& eos){ if(eos.empty()) return false; return s.size()>=eos.size() && s.compare(s.size()-eos.size(), eos.size(), eos)==0; }
struct FileEntry{ string path; string rel; long long size=0; long long mtime=0; };
static void collect_files_rec(const string& root, const string& cur, vector<FileEntry>& out, const vector<string>& exts){ DIR* d = opendir(cur.c_str()); if(!d) return; struct dirent* ent; while((ent = readdir(d)) != nullptr){ string name = ent->d_name; if(name == "." || name == "..") continue; if(name == ".git" || name == ".svn" || name == ".hg") continue; string full = cur + "/" + name; struct stat st; if(stat(full.c_str(), &st) != 0) continue; if(S_ISDIR(st.st_mode)) collect_files_rec(root, full, out, exts); else if(S_ISREG(st.st_mode)){ if(!exts.empty()){ bool ok=false; for(auto& e : exts){ if(name.size()>=e.size() && name.compare(name.size()-e.size(), e.size(), e)==0){ ok=true; break; } } if(!ok) continue; } FileEntry fe; fe.path = full; fe.rel = full.substr(root.size()); while(!fe.rel.empty() && (fe.rel[0]=='/' || fe.rel[0]=='\\')) fe.rel = fe.rel.substr(1); fe.size = (long long)st.st_size; fe.mtime = (long long)st.st_mtime; out.push_back(std::move(fe)); } } closedir(d); }
static void sort_files(vector<FileEntry>& files, const string& mode){ bool desc = false; string key = mode; if(key.size() > 5 && key.substr(key.size()-5) == "-desc"){ desc = true; key = key.substr(0, key.size()-5); } auto cmp = [&](const FileEntry& a, const FileEntry& b) -> bool { if(key == "size"){ if(a.size != b.size) return a.size < b.size; return a.rel < b.rel; } else if(key == "mtime"){ if(a.mtime != b.mtime) return a.mtime < b.mtime; return a.rel < b.rel; } return a.rel < b.rel; }; if(desc) sort(files.begin(), files.end(), [&](const FileEntry& a, const FileEntry& b){ return cmp(b,a); }); else sort(files.begin(), files.end(), cmp); }
static string load_dir_as_text(const string& dir, const string& sort_mode, const string& exts_str){ vector<string> exts; if(!exts_str.empty()){ string cur; for(char c : exts_str){ if(c==',' || c==';'){ if(!cur.empty()){ exts.push_back(cur); cur.clear(); } } else cur += c; } if(!cur.empty()) exts.push_back(cur); } vector<FileEntry> files; collect_files_rec(dir, dir, files, exts); sort_files(files, sort_mode); cout << "[data-dir] " << files.size() << " 个文件, 排序=" << sort_mode; if(!exts.empty()) cout << ", 后缀过滤=" << exts_str; cout << "\n"; for(size_t i=0;i<files.size() && i<5;++i) cout << "  " << files[i].rel << " (" << files[i].size << "B)\n"; if(files.size() > 5) cout << "  ...\n"; string out; for(auto& fe : files){ string txt = lt(fe.path); if(txt.empty()) continue; if(!out.empty() && out.back() != '\n') out += '\n'; out += txt; if(out.back() != '\n') out += '\n'; } return out; }

static void scale_grads(Model& m,float s){ auto sc=[&](VF& g){ for(float& x:g)x*=s; }; for(auto& b:m.bl){ sc(b.at.Wqkv.g); sc(b.at.Wo.g); sc(b.at.dbqkv); sc(b.at.dbo); sc(b.at.dqg); sc(b.at.dkg); sc(b.ff.W1.g); sc(b.ff.W3.g); sc(b.ff.W2.g); sc(b.ff.db1); sc(b.ff.db3); sc(b.ff.db2); sc(b.dln1); sc(b.dln2); } sc(m.dE); sc(m.dlnf); }
static void add_grads(Model& m, const Model& src) {
    auto add = [](VF& dst, const VF& s) {
        size_t n = dst.size(); size_t i = 0;
        for (; i + 4 <= n; i += 4) vst1q_f32(&dst[i], vaddq_f32(vld1q_f32(&dst[i]), vld1q_f32(&s[i])));
        for (; i < n; ++i) dst[i] += s[i];
    };
    add(m.dE, src.dE);
    for (size_t l = 0; l < m.bl.size(); ++l) {
        Block& b = m.bl[l]; const Block& sb = src.bl[l];
        add(b.at.Wqkv.g, sb.at.Wqkv.g); add(b.at.Wo.g, sb.at.Wo.g); add(b.at.dbqkv, sb.at.dbqkv);
        add(b.at.dbo, sb.at.dbo); add(b.at.dqg, sb.at.dqg); add(b.at.dkg, sb.at.dkg);
        add(b.ff.W1.g, sb.ff.W1.g); add(b.ff.W3.g, sb.ff.W3.g); add(b.ff.W2.g, sb.ff.W2.g);
        add(b.ff.db1, sb.ff.db1); add(b.ff.db3, sb.ff.db3); add(b.ff.db2, sb.ff.db2);
        add(b.dln1, sb.dln1); add(b.dln2, sb.dln2);
    }
    add(m.dlnf, src.dlnf);
}

static double grad_norm_sq(const Model& m) {
    double n2 = 0.0;
    auto acc = [&](const VF& g) {
        size_t n = g.size(); size_t i = 0;
        float32x4_t acc4 = vdupq_n_f32(0);
        for (; i + 4 <= n; i += 4) { float32x4_t v = vld1q_f32(&g[i]); acc4 = vfmaq_f32(acc4, v, v); }
        n2 += hs4(acc4);
        for (; i < n; ++i) n2 += (double)g[i] * g[i];
    };
    acc(m.dE);
    for (auto& b : m.bl) {
        acc(b.at.Wqkv.g); acc(b.at.Wo.g); acc(b.at.dbqkv); acc(b.at.dbo); acc(b.at.dqg); acc(b.at.dkg);
        acc(b.ff.W1.g); acc(b.ff.W3.g); acc(b.ff.W2.g); acc(b.ff.db1); acc(b.ff.db3); acc(b.ff.db2);
        acc(b.dln1); acc(b.dln2);
    }
    acc(m.dlnf);
    return n2;
}
static void clip_grads_precomputed(Model& m, const Config& c, double n2){
    float gn = sqrtf((float)n2);
    if (gn <= c.grad_clip) return;
    float s = c.grad_clip / gn;
    auto mul=[&](VF& g){ for(float& x:g)x*=s; };
    for(auto& b:m.bl){ mul(b.at.Wqkv.g); mul(b.at.Wo.g); mul(b.at.dbqkv); mul(b.at.dbo); mul(b.at.dqg); mul(b.at.dkg); mul(b.ff.W1.g); mul(b.ff.W3.g); mul(b.ff.W2.g); mul(b.ff.db1); mul(b.ff.db3); mul(b.ff.db2); mul(b.dln1); mul(b.dln2); }
    for(float& x:m.dE)x*=s; for(float& x:m.dlnf)x*=s;
}
static int find_a_pos(const Vocab& voc,const vector<int>& ids){ auto it=voc.special_id.find(voc.a_str()); if(it==voc.special_id.end())return -1; int a_id=it->second; for(int t=(int)ids.size()-1;t>=0;--t) if(ids[t]==a_id) return t; return -1; }
static inline int resolve_a_pos(const Config& cfg,const Vocab& voc,const vector<int>& ids,bool& skip){ if(cfg.loss_all){ skip=false; return 0; } int a_pos=find_a_pos(voc,ids); if(a_pos<0){ skip=true; return -1; } skip=false; return a_pos; }
static void save_best(const Config& cfg,const Model& m,const Vocab& voc,int ep,long long stc,float bl,mt19937& g){ if(!cfg.save_best) return; savem(cfg.model+".best", m, voc); saveck(cfg.ckpt+".best", m, voc, ep, stc, bl, g); }
static void bind_weights(Model& dst, const Model& src) { dst.E.assign(src.E.begin(), src.E.end()); for (size_t l = 0; l < dst.bl.size(); ++l) { Block& db = dst.bl[l]; const Block& sb = src.bl[l]; db.ln1.assign(sb.ln1.begin(), sb.ln1.end()); db.ln2.assign(sb.ln2.begin(), sb.ln2.end()); db.at.Wqkv.v.assign(sb.at.Wqkv.v.begin(), sb.at.Wqkv.v.end()); db.at.Wo.v.assign(sb.at.Wo.v.begin(), sb.at.Wo.v.end()); db.at.bqkv.assign(sb.at.bqkv.begin(), sb.at.bqkv.end()); db.at.bo.assign(sb.at.bo.begin(), sb.at.bo.end()); db.at.qg.assign(sb.at.qg.begin(), sb.at.qg.end()); db.at.kg.assign(sb.at.kg.begin(), sb.at.kg.end()); db.ff.W1.v.assign(sb.ff.W1.v.begin(), sb.ff.W1.v.end()); db.ff.W3.v.assign(sb.ff.W3.v.begin(), sb.ff.W3.v.end()); db.ff.W2.v.assign(sb.ff.W2.v.begin(), sb.ff.W2.v.end()); db.ff.b1.assign(sb.ff.b1.begin(), sb.ff.b1.end()); db.ff.b3.assign(sb.ff.b3.begin(), sb.ff.b3.end()); db.ff.b2.assign(sb.ff.b2.begin(), sb.ff.b2.end()); } dst.lnf.assign(src.lnf.begin(), src.lnf.end()); }

static int run_train(Config& cfg){
    if(cfg.sp_eos.empty()||cfg.sp_u.empty()||cfg.sp_a.empty()){ cerr<<"[error] 特殊 token 不能为空\n"; return 1; }
    if(cfg.sp_eos==cfg.sp_u||cfg.sp_eos==cfg.sp_a||cfg.sp_u==cfg.sp_a){ cerr<<"[error] 特殊 token 不能相同\n"; return 1; }
    if(cfg.sp_eos.find('\n')!=string::npos||cfg.sp_u.find('\n')!=string::npos||cfg.sp_a.find('\n')!=string::npos){ cerr<<"[error] 特殊 token 不能含换行\n"; return 1; }

    string text; if(!cfg.data_dir.empty()) text = load_dir_as_text(cfg.data_dir, cfg.sort_mode, cfg.data_exts); else text = lt(cfg.data);
    if(text.empty()){ cerr<<"训练数据为空\n"; return 1; }

    mt19937 g(cfg.seed); Model m; Vocab voc; int sep=0; long long stc=0; float bl=1e30f;
    int amT=cfg.infer_maxT>0?cfg.infer_maxT:(cfg.ctx+cfg.maxlen+16); if(amT<cfg.ctx)amT=cfg.ctx;

    bool resumed=false;
    if(cfg.resume){ if(loadck(cfg.ckpt,m,voc,cfg,sep,stc,bl,g)){ resumed=true; cout<<"恢复 V="<<m.V<<" D="<<m.D<<" L="<<m.L<<" Df="<<m.Df<<" maxT="<<m.maxT<<" ep="<<sep<<" step="<<stc<<" best="<<bl<<"\n"; } else cout<<"从头\n"; }
    if(!resumed){ voc.build(text,cfg.vocab_size,cfg.sp_eos,cfg.sp_u,cfg.sp_a); int V=(int)voc.vocab.size(); cout<<"字节: "<<text.size()<<" 词表: "<<V<<"\n"; m.init(V,cfg.D,cfg.L,cfg.nq,cfg.nkv,amT,g,cfg); }
    if(m.maxT<cfg.ctx){ cerr<<"[err] maxT<ctx\n"; return 1; }
    if(cfg.lr_override){ m.aE.base_lr=m.aE.lr=cfg.lr; m.sch.base_lr=cfg.lr; }
    if(cfg.min_lr_override) m.sch.min_lr=cfg.min_lr; if(cfg.sched_override) m.sch.kind=cfg.lr_sched;

    int B = max(1, cfg.batch_size);
    cout<<"[v"<<TINY_VERSION<<"] 参数: "<<m.pc()<<" D="<<m.D<<" L="<<m.L<<" Df="<<m.Df<<" nq="<<m.Hq<<" nkv="<<m.Hkv<<" ctx="<<cfg.ctx;
    if(cfg.loss_all) cout<<" [loss-all]"; if(cfg.train_threads>1) cout<<" [train-threads: "<<cfg.train_threads<<"]"; cout<<" [batch: "<<B<<"]\n";

    int V=m.V; vector<vector<int>> dialogs;
    { string curline; size_t total_tokens = 0; for(size_t i=0;i<=text.size();++i){ if(i==text.size() || text[i]=='\n'){ if(!curline.empty()){ vector<int> seg=voc.encode(curline); if(!seg.empty()){ total_tokens += seg.size(); dialogs.push_back(std::move(seg)); } } curline.clear(); } else curline.push_back(text[i]); } cout<<"数据字节: "<<text.size()<<" 总 token(估): "<<total_tokens<<" 行数: "<<dialogs.size()<<"\n"; }
    cout<<"对话数: "<<dialogs.size()<<"\n";
    int ndial=(int)dialogs.size(); if(ndial<2){ cerr<<"对话太少\n"; return 1; }
    vector<int> ord(ndial); iota(ord.begin(),ord.end(),0); shuffle(ord.begin(),ord.end(),g);
    int nv=(int)(ndial*cfg.val_split); if(nv<1&&cfg.val_split>0)nv=1; if(nv>=ndial)nv=ndial-1;
    int nt=ndial-nv; vector<int> tid(ord.begin(),ord.begin()+nt), vid(ord.begin()+nt,ord.end());
    cout<<"训练对话 "<<nt<<" 验证 "<<nv<<"\n";

    int base_epochs; if(resumed) base_epochs = cfg.epochs_override ? std::max(cfg.epochs, sep) : sep; else base_epochs = cfg.epochs;
    int extra = cfg.extra_epochs > 0 ? cfg.extra_epochs : 0; int target_epochs = base_epochs + extra; if(target_epochs < 1) target_epochs = 1;
    long long ga = max(1, cfg.grad_accum); if (cfg.train_threads > 1) ga = std::max(ga, (long long)cfg.train_threads);
    long long total=(long long)target_epochs*((nt+ga-1)/ga);
    if(cfg.max_steps>0 && cfg.max_steps<total)total=cfg.max_steps; if(resumed && cfg.max_steps<=0 && extra==0 && m.sch.total>0) total = m.sch.total; m.set_total(total);

    auto t0=chrono::steady_clock::now(); auto t_prev = t0; long long stc_prev = stc; const int BW=30; bool es=0; float sps=0.f;
    const float NAN_F = std::numeric_limits<float>::quiet_NaN();
    int n_train_threads = max(1, cfg.train_threads);

    struct Worker { Model wm; vector<int> xb, yb; float loss; };
    vector<unique_ptr<Worker>> workers;
    if (n_train_threads > 1) { workers.resize(n_train_threads - 1); for (int i = 0; i < n_train_threads - 1; ++i) { workers[i].reset(new Worker()); workers[i]->wm = m; workers[i]->wm.aE.m.clear(); workers[i]->wm.aE.v.clear(); for (auto& b : workers[i]->wm.bl) { b.at.Wqkv.a.m.clear(); b.at.Wqkv.a.v.clear(); b.at.Wo.a.m.clear(); b.at.Wo.a.v.clear(); b.ff.W1.a.m.clear(); b.ff.W1.a.v.clear(); b.ff.W3.a.m.clear(); b.ff.W3.a.v.clear(); b.ff.W2.a.m.clear(); b.ff.W2.a.v.clear(); } } }
    Model main_worker = m; main_worker.aE.m.clear(); main_worker.aE.v.clear();
    for (auto& b : main_worker.bl) { b.at.Wqkv.a.m.clear(); b.at.Wqkv.a.v.clear(); b.at.Wo.a.m.clear(); b.at.Wo.a.v.clear(); b.ff.W1.a.m.clear(); b.ff.W1.a.v.clear(); b.ff.W3.a.m.clear(); b.ff.W3.a.v.clear(); b.ff.W2.a.m.clear(); b.ff.W2.a.v.clear(); }

    atomic<unsigned long long> round{0}, done{0}; atomic<bool> quit{false};
    vector<thread> threads; bool weights_dirty = false;
    for (int i = 0; i < n_train_threads - 1; ++i) { threads.emplace_back([&, i]() { pin_to_big_cores(); unsigned long long local = 0; mt19937 rng(42 + i * 1000); while (true) { if (quit.load(std::memory_order_acquire)) return; unsigned long long r = round.load(std::memory_order_acquire); if (r == local) { std::this_thread::yield(); continue; } local = r; if (!workers[i]->xb.empty()) { if (weights_dirty) bind_weights(workers[i]->wm, m); workers[i]->wm.zg(); workers[i]->wm.fwd(workers[i]->xb, rng, B); workers[i]->loss = workers[i]->wm.bwd(workers[i]->xb, workers[i]->yb, 0, B); done.fetch_add(1, std::memory_order_release); } } }); }
    pin_to_big_cores();

    double t_calc = 0, t_wait = 0, t_agg = 0, t_opt = 0; int steps_in_window = 0;
    int final_ep = sep;
    for(int ep=sep;ep<target_epochs&&!es;++ep){
        auto t_last_print = chrono::steady_clock::now(); shuffle(tid.begin(),tid.end(),g);
        m.zg(); float esum=0.f; int ecnt=0; int ac=0; long long stc_ep_start = stc;

        if (n_train_threads > 1) {
            mt19937 main_rng(cfg.seed + 999);
            for (int s = 0; s < nt && !es; ) {
                long long remain = ga - ac; int take = min(n_train_threads, nt - s); if (take > remain) take = (int)remain;
                for (int i = 0; i < n_train_threads - 1; ++i) workers[i]->xb.clear();
                int worker_tasks = (take > 0) ? take - 1 : 0;
                int main_task = (take > 0) ? take - 1 : -1;
                vector<int> valid_worker_idxs;

                for (int t = 0; t < worker_tasks; ++t) {
                    const auto& d = dialogs[tid[s + t]]; int Tl = (int)d.size(); if (Tl > cfg.ctx) Tl = cfg.ctx;
                    int Tal = (Tl - 1) / B * B; if (Tal < B) continue;
                    workers[t]->xb.assign(d.begin(), d.begin() + Tal);
                    workers[t]->yb.assign(d.begin() + 1, d.begin() + Tal + 1);
                    valid_worker_idxs.push_back(t);
                }
                vector<int> main_xb, main_yb;
                if (main_task >= 0) { const auto& d = dialogs[tid[s + main_task]]; int Tl = (int)d.size(); if (Tl > cfg.ctx) Tl = cfg.ctx; int Tal = (Tl - 1) / B * B; if (Tal >= B) { main_xb.assign(d.begin(), d.begin() + Tal); main_yb.assign(d.begin() + 1, d.begin() + Tal + 1); } }
                s += take; if (valid_worker_idxs.empty() && main_xb.empty()) continue;

                if (weights_dirty) { bind_weights(main_worker, m); weights_dirty = false; }
                main_worker.zg(); done.store(0, std::memory_order_relaxed); round.fetch_add(1, std::memory_order_release);

                auto t_calc_start = chrono::steady_clock::now();
                float main_loss = 0.f;
                if (!main_xb.empty()) { main_worker.fwd(main_xb, main_rng, B, stc); main_loss = main_worker.bwd(main_xb, main_yb, 0, B); }
                t_calc += chrono::duration<double, std::milli>(chrono::steady_clock::now() - t_calc_start).count();

                auto t_wait_start = chrono::steady_clock::now();
                while (done.load(std::memory_order_acquire) < (unsigned long long)valid_worker_idxs.size()) std::this_thread::yield();
                t_wait += chrono::duration<double, std::milli>(chrono::steady_clock::now() - t_wait_start).count();

                auto t_agg_start = chrono::steady_clock::now();
                if (!main_xb.empty()) { add_grads(m, main_worker); esum += main_loss; ecnt++; ac++; }
                for (int t : valid_worker_idxs) { add_grads(m, workers[t]->wm); esum += workers[t]->loss; ecnt++; ac++; }
                t_agg += chrono::duration<double, std::milli>(chrono::steady_clock::now() - t_agg_start).count();

                if ((ac >= ga) || (s >= nt && ac > 0)) {
                    auto t_opt_start = chrono::steady_clock::now();
                    // ✅ FIX #6: 多线程梯度累积范数修正
                    // add_grads_and_norm 返回 Σ|g_i|², 但裁剪需要 |Σg_i/ac|²
                    // 两者通过 ||sum||² = Σ||g_i||² + cross_terms 关联
                    // 实际梯度 = sum/ac, 所以直接用 n2/(ac*ac) 近似
                    if (ac > 1) scale_grads(m, 1.0f / (float)ac);
                    if (cfg.grad_clip > 0.f) clip_grads_precomputed(m, cfg, grad_norm_sq(m));
                    m.upd_lr(stc); m.ag(); m.zg(); ac = 0; stc++;
                    t_opt += chrono::duration<double, std::milli>(chrono::steady_clock::now() - t_opt_start).count();
                    steps_in_window++; weights_dirty = true;
                    if (cfg.max_steps > 0 && stc >= cfg.max_steps) es = 1;
                }

                auto t_now = chrono::steady_clock::now();
                if (chrono::duration<double>(t_now - t_last_print).count() >= 1.0 || s >= nt) {
                    t_last_print = t_now; double dt_sps = chrono::duration<double>(t_now - t_prev).count(); long long dst = stc - stc_prev;
                    if (dt_sps > 0.1 && dst > 0) { float inst = (float)(dst / dt_sps); if (sps == 0.f) sps = inst; else sps = 0.95f * sps + 0.05f * inst; t_prev = t_now; stc_prev = stc; }
                    double pct = 100.0 * stc / max((long long)1, total); if (pct > 100) pct = 100; int fl = (int)(BW * pct / 100);
                    double remaining = max(0LL, total - stc); double eta = remaining / max((double)sps, 1e-6); double div = max(1, steps_in_window);
                    cout << "\r["; for (int i = 0; i < BW; ++i) cout << (i < fl ? '#' : '.');
                    cout << "] " << fixed << setprecision(1) << setw(5) << pct << "% ep " << (ep + 1) << "/" << target_epochs;
                    cout << " ETA " << setprecision(0) << setw(4) << eta << "s | calc:" << setprecision(1) << (t_calc/div) << "ms wait:" << (t_wait/div) << "ms agg:" << (t_agg/div) << "ms opt:" << (t_opt/div) << "ms      " << flush;
                    t_calc = 0; t_wait = 0; t_agg = 0; t_opt = 0; steps_in_window = 0;
                }
                if (cfg.time_budget > 0.f && chrono::duration<double>(t_now - t0).count() >= cfg.time_budget) es = 1;
            }
        } else {
            for(int s=0;s<nt&&!es;++s){
                const auto& d=dialogs[tid[s]]; int Tl=(int)d.size(); if(Tl>cfg.ctx)Tl=cfg.ctx;
                int Tal = (Tl - 1) / B * B; if (Tal < B) continue;
                vector<int> xb(d.begin(),d.begin()+Tal), yb(d.begin()+1,d.begin()+Tal+1);
                auto t_calc_start = chrono::steady_clock::now();
                // ✅ FIX #4: 单线程路径也传入 step_counter
                m.upd_lr(stc); m.fwd(xb,g,B,stc); float loss=m.bwd(xb,yb,0,B); esum+=loss; ecnt++; ac++;
                t_calc += chrono::duration<double, std::milli>(chrono::steady_clock::now() - t_calc_start).count();
                if(ac>0 && (ac>=cfg.grad_accum||s==nt-1)){ auto t_opt_start = chrono::steady_clock::now(); if(cfg.grad_accum>1) scale_grads(m, 1.0f/(float)ac); if(cfg.grad_clip>0.f) clip_grads_precomputed(m, cfg, grad_norm_sq(m)); m.ag(); m.zg(); ac=0; stc++; t_opt += chrono::duration<double, std::milli>(chrono::steady_clock::now() - t_opt_start).count(); steps_in_window++; if(cfg.max_steps>0 && stc>=cfg.max_steps)es=1; }
                auto t_now = chrono::steady_clock::now();
                if (chrono::duration<double>(t_now - t_last_print).count() >= 1.0 || s == nt-1) {
                    t_last_print = t_now; double dt_sps = chrono::duration<double>(t_now - t_prev).count(); long long dst = stc - stc_prev;
                    if (dt_sps > 0.1 && dst > 0) { float inst = (float)(dst / dt_sps); if (sps == 0.f) sps = inst; else sps = 0.95f * sps + 0.05f * inst; t_prev = t_now; stc_prev = stc; }
                    double pct = 100.0 * stc / max((long long)1, total); if(pct>100)pct=100; int fl = (int)(BW * pct / 100);
                    double remaining = max(0LL, total - stc); double eta = remaining / max((double)sps, 1e-6); double div = max(1, steps_in_window);
                    cout << "\r["; for(int i=0;i<BW;++i) cout << (i<fl?'#':'.');
                    cout << "] " << fixed << setprecision(1) << setw(5) << pct << "% ep " << (ep+1) << "/" << target_epochs;
                    cout << " ETA " << setprecision(0) << setw(4) << eta << "s | calc:" << setprecision(1) << (t_calc/div) << "ms opt:" << (t_opt/div) << "ms      " << flush;
                    t_calc = 0; t_opt = 0; steps_in_window = 0;
                }
                if(cfg.time_budget>0.f && chrono::duration<double>(t_now-t0).count()>=cfg.time_budget)es=1;
            }
        }

        if(ac>0){ if(cfg.grad_accum>1) scale_grads(m, 1.0f/(float)ac); if(cfg.grad_clip>0.f) clip_grads_precomputed(m, cfg, grad_norm_sq(m)); m.upd_lr(stc); m.ag(); m.zg(); ac=0; stc++; }
        if(stc == stc_ep_start && !es){ cerr << "\n[error] epoch " << (ep+1) << " 无任何样本被训练！\n"; return 1; }
        final_ep = ep + 1;

        float vl = NAN_F;
        if(nv>0){
            if (n_train_threads > 1 && weights_dirty) { bind_weights(main_worker, m); weights_dirty = false; } // 👈 验证前强制同步
            cout << "\r验证中... 0%          " << flush; auto t_val_last = chrono::steady_clock::now();
            atomic<int> v_idx{0}, v_done{0}; float vs_sum = 0.f; int vcnt = 0; mutex v_mtx;
            int v_threads = min((int)thread::hardware_concurrency(), 2); if(v_threads < 1) v_threads = 1;
            vector<thread> v_ths;
            for(int t=0; t<v_threads; ++t){
                v_ths.emplace_back([&, t](){
                    pin_to_big_cores();
                    Model v_m = main_worker; for(auto& dd : v_m.df) dd.p = 0.f;
                    mt19937 v_rng(cfg.seed + t); auto local_last = chrono::steady_clock::now();
                    while(true){
                        int i = v_idx.fetch_add(1); if(i >= (int)vid.size()) break;
                        int vi = vid[i]; const auto& d = dialogs[vi]; int Tl=(int)d.size(); if(Tl>cfg.ctx)Tl=cfg.ctx;
                        int Tal = (Tl - 1) / B * B; if (Tal < B) continue;
                        vector<int> xb(d.begin(),d.begin()+Tal), yb(d.begin()+1,d.begin()+Tal+1);
                        v_m.fwd(xb, v_rng, B); const float* lp = v_m.lg.data(); float sl=0.f; int cnt=0;
                        int T_valid = Tal / B;
                        for(int tt=0; tt<T_valid; ++tt){ for(int b=0;b<B;++b){ int idx = b*T_valid + tt; const float* L = lp + (size_t)idx * V; float mx=-1e30f; for(int v=0; v<V; ++v) mx=max(mx, L[v]); float sm=0.f; for(int v=0; v<V; ++v) sm += expf(L[v]-mx); sl += -logf(max(expf(L[yb[idx]]-mx)/sm, 1e-9f)); ++cnt; } }
                        if(cnt>0){ lock_guard<mutex> lk(v_mtx); vs_sum += sl/cnt; vcnt++; }
                        int d_cnt = v_done.fetch_add(1) + 1; auto t_now = chrono::steady_clock::now();
                        if(chrono::duration<double>(t_now - local_last).count() >= 1.0 || d_cnt == (int)vid.size()){ local_last = t_now; lock_guard<mutex> lk(v_mtx); cout << "\r验证中... " << (d_cnt * 100 / (int)vid.size()) << "%          " << flush; }
                    }
                });
            }
            for(auto& th : v_ths) th.join();
            if(vcnt>0) vl = vs_sum / vcnt; cout << "\r验证完成!             \n";
        }

        float tl = ecnt>0 ? esum/ecnt : NAN_F; float mon = std::isfinite(vl) ? vl : tl;
        if(std::isfinite(mon) && mon < bl){ bl = mon; save_best(cfg, m, voc, ep+1, stc, bl, g); }
        cout << "\r" << "ep " << (ep+1) << " train=" << setprecision(4) << tl << " val=" << vl << " best=" << bl << "        \n";
        if(cfg.ckpt_interval>0&&(ep+1)%cfg.ckpt_interval==0)saveck(cfg.ckpt,m,voc,ep+1,stc,bl,g);
        if(cfg.stop_loss>0.f&&std::isfinite(mon)&&mon<=cfg.stop_loss)es=1;
    }

    quit.store(true, std::memory_order_release); round.fetch_add(1);
    for (auto& th : threads) th.join();

    savem(cfg.model,m,voc); saveck(cfg.ckpt,m,voc,final_ep,stc,bl,g);
    if(es) cout<<"早停于 ep="<<final_ep<<"\n";
    cout<<"完成 ep="<<final_ep<<" best="<<bl<<"\n";
    return 0;
}

static string gen_once(Model& m,const Vocab& voc,const vector<int>& ic,Config& cfg,mt19937& g,Sampler& sm,int forced,vector<int>* out_ids=nullptr){ int V=m.V; vector<int> ct=ic; if(ct.empty())ct.push_back(0); int lim=max(1, min(cfg.ctx, m.maxT - 1)); if((int)ct.size()>lim)ct.erase(ct.begin(),ct.end()-lim); m.kv_reset(); for(int i=0;i+1<(int)ct.size();++i)m.step(ct[i]); int last=ct.back(); vector<int> hist=ct; string out; int mx=forced>0?forced:cfg.maxlen; int st=0; if(out_ids)out_ids->clear(); for(;st<mx&&m.kv<m.maxT;++st){ const float* l=m.step(last); int c=sm.sample(l,V,cfg.temp,cfg.topk,cfg.top_p,hist,cfg.rep_window,cfg.rep_pen,g,cfg.greedy); out+=voc.tok(c); hist.push_back(c); last=c; if(out_ids)out_ids->push_back(c); if(forced==0&&iseos(out,voc.eos_str()))break; } if(m.kv>=m.maxT)cerr<<"[warn] maxT 截断\n"; return out; }
static string gen_once_ctx(Model& m, InferCtx& ctx, const Vocab& voc, const vector<int>& ic, const Config& cfg, int forced){ int V = m.V; vector<int> ct = ic; if(ct.empty()) ct.push_back(0); int lim = max(1, min(cfg.ctx, m.maxT - 1)); if((int)ct.size() > lim) ct.erase(ct.begin(), ct.end() - lim); int Dkv = m.Hkv * (m.D / m.Hq); ctx.prepare(m.L, m.maxT, Dkv); for(int i = 0; i + 1 < (int)ct.size(); ++i) m.step_ctx(ct[i], ctx); int last = ct.back(); vector<int> hist = ct; string out; int mx = forced > 0 ? forced : cfg.maxlen; for(int st = 0; st < mx && ctx.kv < m.maxT; ++st){ const float* l = m.step_ctx(last, ctx); int c = ctx.sampler.sample(l, V, cfg.temp, cfg.topk, cfg.top_p, hist, cfg.rep_window, cfg.rep_pen, ctx.rng, cfg.greedy); out += voc.tok(c); hist.push_back(c); last = c; if(forced == 0 && iseos(out, voc.eos_str())) break; } if(ctx.kv >= m.maxT) cerr << "[warn] maxT 截断\n"; return out; }
static int run_gen(Config& cfg){ Model m; Vocab voc; if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; } cout<<"V="<<m.V<<" D="<<m.D<<" L="<<m.L<<" Df="<<m.Df<<"\n"; mt19937 g(chrono::system_clock::now().time_since_epoch().count()); Sampler sm(m.V); vector<int> ct=voc.encode(cfg.prompt); if(ct.empty())ct.push_back(0); cout<<cfg.prompt<<gen_once(m,voc,ct,cfg,g,sm,cfg.one_token?1:0)<<"\n"; return 0; }
static int run_next(Config& cfg){ Model m; Vocab voc; if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; } mt19937 g(chrono::system_clock::now().time_since_epoch().count()); Sampler sm(m.V); vector<int> ct=voc.encode(cfg.prompt); if(ct.empty())ct.push_back(0); cout<<cfg.prompt<<gen_once(m,voc,ct,cfg,g,sm,1)<<"\n"; return 0; }
static int run_chat(Config& cfg){ Model m; Vocab voc; if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; } mt19937 g(chrono::system_clock::now().time_since_epoch().count()); Sampler sm(m.V); cout<<"chat /exit /reset /temp /topk /topp /greedy /one\n"; string ln; vector<int> hist; auto sw=[](const string& s,const string& p){ return s.size()>=p.size()&&s.compare(0,p.size(),p)==0; }; int lim=max(1, min(cfg.ctx, m.maxT - 1)); while(1){ cout<<"\n> "<<flush; if(!getline(cin,ln))break; if(ln.empty())continue; if(ln=="/exit"||ln=="/quit")break; if(ln=="/reset"){ hist.clear(); cout<<"ok\n"; continue; } if(sw(ln,"/temp ")){ string v=trm(ln.substr(6)); float x; if(s2f(v,x)){ cfg.temp=x; cout<<"t="<<cfg.temp<<"\n"; } else cerr<<"用法: /temp <float>\n"; continue; } if(sw(ln,"/topk ")){ string v=trm(ln.substr(6)); int x; if(s2i(v,x)){ cfg.topk=x; cout<<"k="<<cfg.topk<<"\n"; } else cerr<<"用法: /topk <int>\n"; continue; } if(sw(ln,"/topp ")){ string v=trm(ln.substr(6)); float x; if(s2f(v,x)){ cfg.top_p=x; cout<<"p="<<cfg.top_p<<"\n"; } else cerr<<"用法: /topp <float>\n"; continue; } if(ln=="/greedy"){ cfg.greedy=!cfg.greedy; cout<<cfg.greedy<<"\n"; continue; } if(ln=="/one"){ cfg.one_token=!cfg.one_token; cout<<cfg.one_token<<"\n"; continue; } vector<int> ct=voc.encode(ln); if(ct.empty()){ cout<<"无有效\n"; continue; } vector<int> full=hist; full.insert(full.end(),ct.begin(),ct.end()); if((int)full.size()>lim)full.erase(full.begin(),full.end()-lim); vector<int> oids; string out=gen_once(m,voc,full,cfg,g,sm,cfg.one_token?1:0,&oids); hist=full; if(!oids.empty()){ auto it=voc.special_id.find(voc.eos_str()); if(it!=voc.special_id.end() && oids.back()==it->second) oids.pop_back(); } hist.insert(hist.end(),oids.begin(),oids.end()); if((int)hist.size()>lim)hist.erase(hist.begin(),hist.end()-lim); cout<<out; } return 0; }
static int run_batch(Config& cfg){ Model m; Vocab voc; if(!loadm(cfg.model, m, voc, cfg)){ cerr << "加载失败\n"; return 1; } ifstream f(cfg.batch_file); if(!f){ cerr << "打不开 " << cfg.batch_file << "\n"; return 1; } vector<string> prompts; string ln; while(getline(f, ln)){ if(!ln.empty()) prompts.push_back(ln); } if(prompts.empty()){ cout << "(batch 文件为空)\n"; return 0; } int nthreads = cfg.infer_threads; if(nthreads == 0){ unsigned hc = std::thread::hardware_concurrency(); nthreads = (int)(hc ? std::min(hc, 8u) : 4u); } if(nthreads > (int)prompts.size()) nthreads = (int)prompts.size(); if(nthreads < 1) nthreads = 1; cout << "batch: " << prompts.size() << " 条, " << nthreads << " 线程\n"; int Dkv = m.Hkv * (m.D / m.Hq); vector<unique_ptr<InferCtx>> ctxs(nthreads); for(int i = 0; i < nthreads; ++i){ ctxs[i].reset(new InferCtx(m.V, cfg.seed + (unsigned)i * 0x9e3779b9u)); ctxs[i]->prepare(m.L, m.maxT, Dkv); } vector<string> outputs(prompts.size()); atomic<size_t> next{0}; auto worker = [&](int tid){ pin_to_big_cores(); InferCtx& ctx = *ctxs[tid]; while(true){ size_t idx = next.fetch_add(1, std::memory_order_relaxed); if(idx >= prompts.size()) break; vector<int> ct = voc.encode(prompts[idx]); if(ct.empty()){ outputs[idx] = prompts[idx] + "\t[无效]"; continue; } string out = gen_once_ctx(m, ctx, voc, ct, cfg, cfg.one_token ? 1 : 0); outputs[idx] = prompts[idx] + "\t" + out; } }; vector<thread> threads; for(int i = 1; i < nthreads; ++i) threads.emplace_back(worker, i); worker(0); for(auto& t : threads) t.join(); for(auto& o : outputs) cout << o << "\n"; return 0; }
static int run_dump_config(Config& cfg){ cout<<"mode="<<cfg.mode<<"\n"; cout<<"data="<<cfg.data<<"\n"; cout<<"data_dir="<<cfg.data_dir<<"\n"; cout<<"sort_mode="<<cfg.sort_mode<<"\n"; cout<<"data_exts="<<cfg.data_exts<<"\n"; cout<<"token_eos="<<esc(cfg.sp_eos)<<"\n"; cout<<"token_user="<<esc(cfg.sp_u)<<"\n"; cout<<"token_assistant="<<esc(cfg.sp_a)<<"\n"; cout<<"model="<<cfg.model<<"\n"; cout<<"ckpt="<<cfg.ckpt<<"\n"; cout<<"prompt="<<esc(cfg.prompt)<<"\n"; cout<<"batch_file="<<cfg.batch_file<<"\n"; cout<<"config_file="<<cfg.config_file<<"\n"; cout<<"D="<<cfg.D<<"\n"; cout<<"L="<<cfg.L<<"\n"; cout<<"nq="<<cfg.nq<<"\n"; cout<<"nkv="<<cfg.nkv<<"\n"; cout<<"ctx="<<cfg.ctx<<"\n"; cout<<"infer_maxT="<<cfg.infer_maxT<<"\n"; cout<<"vocab_size="<<cfg.vocab_size<<"\n"; cout<<"df_override="<<cfg.df_override<<"\n"; cout<<"dropout="<<cfg.dropout<<"\n"; cout<<"epochs="<<cfg.epochs<<"\n"; cout<<"extra_epochs="<<cfg.extra_epochs<<"\n"; cout<<"save_best="<<cfg.save_best<<"\n"; cout<<"epochs_override="<<cfg.epochs_override<<"\n"; cout<<"infer_threads="<<cfg.infer_threads<<"\n"; cout<<"train_threads="<<cfg.train_threads<<"\n"; cout<<"grad_accum="<<cfg.grad_accum<<"\n"; cout<<"batch_size="<<cfg.batch_size<<"\n"; cout<<"lr="<<cfg.lr<<"\n"; cout<<"min_lr="<<cfg.min_lr<<"\n"; cout<<"weight_decay="<<cfg.weight_decay<<"\n"; cout<<"b1="<<cfg.b1<<"\n"; cout<<"b2="<<cfg.b2<<"\n"; cout<<"eps="<<cfg.eps<<"\n"; cout<<"warmup="<<cfg.warmup<<"\n"; cout<<"lr_sched="<<cfg.lr_sched<<"\n"; cout<<"stop_loss="<<cfg.stop_loss<<"\n"; cout<<"val_split="<<cfg.val_split<<"\n"; cout<<"grad_clip="<<cfg.grad_clip<<"\n"; cout<<"lr_override="<<cfg.lr_override<<"\n"; cout<<"temp="<<cfg.temp<<"\n"; cout<<"top_p="<<cfg.top_p<<"\n"; cout<<"topk="<<cfg.topk<<"\n"; cout<<"rep_window="<<cfg.rep_window<<"\n"; cout<<"rep_pen="<<cfg.rep_pen<<"\n"; cout<<"maxlen="<<cfg.maxlen<<"\n"; cout<<"one_token="<<cfg.one_token<<"\n"; cout<<"loss_all="<<cfg.loss_all<<"\n"; cout<<"seed="<<cfg.seed<<"\n"; cout<<"resume="<<cfg.resume<<"\n"; cout<<"greedy="<<cfg.greedy<<"\n"; cout<<"ckpt_interval="<<cfg.ckpt_interval<<"\n"; cout<<"max_steps="<<cfg.max_steps<<"\n"; cout<<"time_budget="<<cfg.time_budget<<"\n"; return 0; }
static int run_dump_vocab(Config& cfg){ Model m; Vocab voc; if(!loadm(cfg.model,m,voc,cfg)){ cerr<<"加载失败\n"; return 1; } cout<<"V="<<m.V<<" D="<<m.D<<" L="<<m.L<<" Df="<<m.Df<<"\n"; cout<<"nchars="<<voc.nchars<<" merges="<<voc.merges.size()<<"\n"; for(size_t i=0;i<voc.vocab.size();++i)cout<<i<<"\t"<<esc(voc.vocab[i])<<"\n"; return 0; }

int main(int argc,char** argv){
    Config cfg; for(int i=1;i<argc;++i){ if(string(argv[i])=="--config"&&i+1<argc){ cfg.config_file=argv[i+1]; ++i; } }
    load_cfg(cfg,cfg.config_file); int r=parse_args(cfg,argc,argv); if(r<0)return 1; if(r==0)return 0; cout<<fixed;
    if(cfg.mode=="train")return run_train(cfg); if(cfg.mode=="gen")return run_gen(cfg); if(cfg.mode=="next")return run_next(cfg);
    if(cfg.mode=="chat")return run_chat(cfg); if(cfg.mode=="batch")return run_batch(cfg);
    if(cfg.mode=="dump-config")return run_dump_config(cfg); if(cfg.mode=="dump-vocab")return run_dump_vocab(cfg);
    cerr<<"未知 mode\n"; return 1;
}
