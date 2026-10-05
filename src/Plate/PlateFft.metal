// Arbitrary-length DCT/DST via a reordered real DFT. Non-power-of-two lengths
// use Bluestein convolution; the physical grid and modal cutoff do not change.
constant uint FftOperation [[function_constant(0)]];
constant bool FftShared [[function_constant(1)]];
using FftComplex = Real2;

inline FftComplex FftAdd(FftComplex a,FftComplex b) { return {a.x+b.x,a.y+b.y}; }
inline FftComplex FftSub(FftComplex a,FftComplex b) { return {a.x-b.x,a.y-b.y}; }
inline FftComplex FftMul(FftComplex a,FftComplex b) { return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x}; }
inline FftComplex FftConj(FftComplex a) { return {a.x,-a.y}; }
inline void FftBarrier() { threadgroup_barrier(FftShared ? mem_flags::mem_threadgroup : mem_flags::mem_device); }
inline uint FftReverse(uint i,uint bits) { return reverse_bits(i)>>(32-bits); }
inline bool FftSynthesis() { return FftOperation<2 || FftOperation==FftPhiX || FftOperation==FftPhiY; }
inline uint FftChannels() { return FftSynthesis() ? 3 : 1; }
inline bool FftSine(uint component) {
    if(FftOperation<2) return component<2;
    if(FftOperation==FftPhiX || FftOperation==FftPhiY) return component==2;
    return FftOperation==FftForceX || FftOperation==FftForceY;
}

inline Real FftSpectrum(uint k,uint component,uint group,device const Real *input,
    device const FftComplex *table,constant PlateFftParams &p) {
    const bool xAxis=(FftOperation%2)==0;
    uint modes=xAxis?p.Plate.Mx:p.Plate.My;
    if(k>modes || (FftOperation<2 && !k)) return 0.f;
    FftComplex wave=table[p.Waves+k]; // wavenumber and its square
    Real value{};
    if(FftOperation==FftDerivativeX) {
        uint y=group%p.Plate.My,midpoint=group/p.Plate.My;
        Real4 s=((device const Real4 *)input)[(k-1)*p.Plate.My+y];
        value=s.x-(midpoint?.5f*s.y:Real(0.f));
    } else if(FftOperation==FftDerivativeY) {
        uint x=group%p.Plate.Nx,midpoint=group/p.Plate.Nx;
        Real4 s=((device const Real4 *)input)[(midpoint*p.Plate.Nx+x)*p.Plate.My+k-1];
        value=component==0?s.x:(component==1?s.y:s.z);
    } else if(FftOperation==FftPhiX) {
        value=input[k*(p.Plate.My+1)+group];
    } else {
        Real4 s=((device const Real4 *)input)[group*(p.Plate.My+1)+k];
        value=component==0?s.x:(component==1?s.y:s.z);
    }
    if(component==2) return value*wave.x;
    if(component==(xAxis?0u:1u)) return -value*wave.y;
    return value;
}

inline Real FftDctCoefficient(uint k,uint component,uint group,device const Real *input,
    device const FftComplex *table,constant PlateFftParams &p) {
    Real value=FftSpectrum(FftSine(component)?p.N-k:k,component,group,input,table,p);
    return (k?value:value*p.Sqrt2)*p.SynthesisNorm;
}

inline FftComplex FftInput(uint k,uint component,uint group,device const Real *input,
    device const FftComplex *table,constant PlateFftParams &p) {
    if(k>=p.N) return {};
    if(FftSynthesis()) {
        Real a=FftDctCoefficient(k,component,group,input,table,p);
        Real b=k?FftDctCoefficient(p.N-k,component,group,input,table,p):Real(0.f);
        return FftMul({a,-b},table[p.Phase+k]);
    }
    // Even samples followed by reversed odd samples reduce a length-N DCT to
    // one length-N DFT, including odd N.
    uint n=k<(p.N+1)/2?2*k:2*(p.N-k)-1;
    Real value{};
    if(FftOperation==FftProjectX) {
        uint midpoint=group/p.Plate.Ny,y=group%p.Plate.Ny;
        value=input[midpoint*p.Plate.Grid+n*p.Plate.Ny+y];
    } else if(FftOperation==FftAiryY) {
        value=input[group*p.Plate.Ny+n];
    } else if(FftOperation==FftForceX) {
        value=input[n*p.Plate.Ny+group];
    } else {
        value=input[group*p.Plate.Ny+n];
    }
    if(FftSine(0) && (n&1)) value=-value;
    return {value,0.f};
}

template<typename Scratch>
inline void FftRadix2(Scratch data,device const FftComplex *table,
    constant PlateFftParams &p,bool inverse,uint tid) {
    uint channels=FftChannels();
    for(uint width=2;width<=p.Length;width*=2) {
        for(uint i=tid;i<(p.Length/2)*channels;i+=256) {
            uint c=i%channels,b=i/channels,j=b%(width/2),base=(b/(width/2))*width;
            uint a=(base+j)*channels+c,z=a+(width/2)*channels;
            FftComplex w=table[j*(p.Length/width)];
            if(inverse) w=FftConj(w);
            FftComplex u=data[a],v=FftMul(data[z],w);
            data[a]=FftAdd(u,v); data[z]=FftSub(u,v);
        }
        FftBarrier();
    }
}

template<typename Scratch>
inline FftComplex FftValue(Scratch data,uint k,uint component,
    device const FftComplex *table,constant PlateFftParams &p) {
    FftComplex v=data[k*FftChannels()+component];
    if(p.Length!=p.N) {
        FftComplex chirp=table[p.Chirp+k];
        if(FftSynthesis()) chirp=FftConj(chirp);
        v=FftMul(v,chirp);
        v.x*=p.InvLength; v.y*=p.InvLength;
    }
    return v;
}

template<typename Scratch>
inline Real FftOutput(Scratch data,uint n,uint component,
    device const FftComplex *table,constant PlateFftParams &p) {
    if(FftSynthesis()) {
        uint k=(n&1)?p.N-1-n/2:n/2;
        Real value=FftValue(data,k,component,table,p).x;
        return FftSine(component)&&(n&1)?-value:value;
    }
    uint k=FftSine(0)?p.N-n:n;
    Real value=FftMul(FftValue(data,k,0,table,p),FftConj(table[p.Phase+k])).x;
    return value*(k?p.AnalysisNorm:p.AnalysisNorm/p.Sqrt2);
}

template<typename Scratch>
inline void FftRun(Scratch data,device const Real *input,device Real *output,
    device Real *derivatives,device const Real2 *coefficients,device const uint *active,
    device const FftComplex *table,constant PlateFftParams &p,uint tid,uint group) {
    uint channels=FftChannels();
    for(uint i=tid;i<p.Length*channels;i+=256) {
        uint k=i/channels,c=i%channels;
        FftComplex value=FftInput(k,c,group,input,table,p);
        if(p.Length!=p.N && k<p.N) {
            FftComplex chirp=table[p.Chirp+k];
            if(FftSynthesis()) chirp=FftConj(chirp);
            value=FftMul(value,chirp);
        }
        data[FftReverse(k,p.Bits)*channels+c]=value;
    }
    FftBarrier();
    FftRadix2(data,table,p,p.Length==p.N && FftSynthesis(),tid);
    if(p.Length!=p.N) {
        for(uint i=tid;i<p.Length*channels;i+=256) {
            uint k=i/channels;
            FftComplex spectrum=FftSynthesis()?FftConj(table[p.Kernel+(p.Length-k)%p.Length]):table[p.Kernel+k];
            data[i]=FftMul(data[i],spectrum);
        }
        FftBarrier();
        for(uint i=tid;i<p.Length*channels;i+=256) {
            uint k=i/channels,c=i%channels,r=FftReverse(k,p.Bits);
            if(k<r) {
                FftComplex value=data[i];
                data[i]=data[r*channels+c]; data[r*channels+c]=value;
            }
        }
        FftBarrier();
        FftRadix2(data,table,p,true,tid);
    }
    if(FftSynthesis()) {
        for(uint n=tid;n<p.N;n+=256) {
            Real4 v{FftOutput(data,n,0,table,p),FftOutput(data,n,1,table,p),FftOutput(data,n,2,table,p),0.f};
            if(FftOperation==FftDerivativeX) {
                uint y=group%p.Plate.My,midpoint=group/p.Plate.My;
                ((device Real4 *)output)[(midpoint*p.Plate.Nx+n)*p.Plate.My+y]=v;
            } else if(FftOperation==FftPhiX) {
                ((device Real4 *)output)[n*(p.Plate.My+1)+group]=v;
            } else {
                v.x*=p.Plate.Scale; v.y*=p.Plate.Scale; v.z*=p.Plate.Scale;
                if(FftOperation==FftDerivativeY) {
                    uint x=group%p.Plate.Nx,midpoint=group/p.Plate.Nx;
                    output[midpoint*p.Plate.Grid+x*p.Plate.Ny+n]=2.f*(v.x*v.y-v.z*v.z);
                    if(!midpoint) ((device Real4 *)derivatives)[x*p.Plate.Ny+n]=v;
                } else {
                    Real4 u=((device const Real4 *)derivatives)[group*p.Plate.Ny+n];
                    output[group*p.Plate.Ny+n]=u.x*v.y+u.y*v.x-2.f*u.z*v.z;
                }
            }
        }
    } else {
        uint modes=(FftOperation%2)?p.Plate.My:p.Plate.Mx;
        for(uint index=tid;index<modes+(FftSine(0)?0u:1u);index+=256) {
            uint k=index+(FftSine(0)?1u:0u);
            Real value=FftOutput(data,k,0,table,p);
            if(FftOperation==FftProjectX) {
                uint y=group%p.Plate.Ny,midpoint=group/p.Plate.Ny;
                output[(midpoint*(p.Plate.Mx+1)+k)*p.Plate.Ny+y]=value;
            } else if(FftOperation==FftAiryY) {
                uint x=group%(p.Plate.Mx+1),midpoint=group/(p.Plate.Mx+1),a=x*(p.Plate.My+1)+k;
                output[midpoint*p.Plate.AiryModes+a]=active[a]?value*coefficients[a].y:Real(0.f);
            } else if(FftOperation==FftForceX) {
                output[index*p.Plate.Ny+group]=value;
            } else {
                uint a=group*p.Plate.My+index;
                output[a]=active[a]?-value*p.Plate.InvScale:Real(0.f);
            }
        }
    }
}

kernel void PlateFftTransform(device const Real *input [[buffer(0)]],device Real *output [[buffer(1)]],
    device Real *derivatives [[buffer(2)]],device const Real2 *coefficients [[buffer(3)]],
    device const uint *active [[buffer(4)]],device const FftComplex *table [[buffer(5)]],
    device FftComplex *scratch [[buffer(6)]],constant PlateFftParams &p [[buffer(7)]],
    threadgroup FftComplex *shared [[threadgroup(0)]],uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    if(FftOperation==FftForceY && !p.Plate.Nonlinear) {
        for(uint i=tid;i<p.Plate.My;i+=256) output[group*p.Plate.My+i]=0.f;
        return;
    }
    if(FftShared) FftRun(shared,input,output,derivatives,coefficients,active,table,p,tid,group);
    else FftRun(scratch+group*p.Length*FftChannels(),input,output,derivatives,coefficients,active,table,p,tid,group);
}
