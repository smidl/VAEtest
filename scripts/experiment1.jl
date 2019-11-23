using DataFrames
using DrWatson
quickactivate(@__DIR__)

using Distributions;
using LinearAlgebra;
using Flux
using GenerativeModels
using EvalCurves

include(scriptsdir() * "/vae2d.jl")
include(scriptsdir() * "/visualize_vae.jl")

(x,xa,pxx,pxa) = toydata2d(0.1)

as = 1. .-[pxx[:];pxa[:]];
labe = [zeros(size(pxx));ones(size(pxa))];
fpr, tpr = EvalCurves.roccurve(as, labe)
auc_gt = EvalCurves.auc(fpr, tpr)


V1 = buildVAE(x,1,true,true,1.0,1.0) # ELBO
Vb = buildVAE(x,1,false,true,1.0,1.0) # ELBO
W1 = buildVAE(x,1,false,false,1.0,1.0)

function testbeta(beta)
    VA = buildVAE(x,1,false,true,beta,1.0) # ELBO
    res= evalmodel(VA,x,xa)
    sname =savename((@dict beta), "bson")
    mkpath(datadir("results", "VAEbeta"))
    
    wsave(datadir("results", "VAEbeta", sname), struct2dict(res))
    wsave(datadir("results", "model", sname), model)
    
end

function testepsilon(ε,ndat)
    (x,xa,pxx,pxa) = toydata2d(ε,ndat)

    for nz=1:2
        Dparam = (@dict ε nz ndat)

        VA = buildVAE(x,nz,true,true,0.01,0.1) # ELBO
        res= evalmodel(VA,x,xa)
    
        sname =savename(Dparam, "bson")

        D= struct2dict(res);
        wsave(datadir("results", "Vae_noise", sname),merge(D,Dparam))
        wsave(datadir("model", "Vae_noise", sname),struct2dict(VA))
    end
end


function testepsilonbeta(ε,beta)
    (x,xa,pxx,pxa) = toydata2d(ε)

    for nz=1:2
        Dparam = (@dict ε nz beta)

        VA = buildVAE(x,nz,false,true,beta,0.1) # ELBO
        res= evalmodel(VA,x,xa)
    
        sname =savename(Dparam, "bson")

        D= struct2dict(res);
        wsave(datadir("results", "Vae_noisebeta", sname),merge(D,Dparam))
        wsave(datadir("model", "Vae_noisebeta", sname),struct2dict(VA))
    end
end


function testWasser(beta,gamma)
    VA = buildVAE(x,1,false,false,beta,gamma) # ELBO
    res= evalmodel(VA,x,xa)
    Dbg = (@dict beta gamma)
    sname =savename(Dbg, "bson")
    mkpath(datadir("results", "WAEbg"))
    D= struct2dict(res);
    wsave(datadir("results", "WAEbg", sname),merge(D,Dbg))
    wsave(datadir("model", "WAEbg", sname),)
end


beta = exp.(-8:2);
gamma = exp.(-8:0);
vecb=[beta[i] for i=1:length(beta), j=1:length(gamma)]
vecg=[gamma[j] for i=1:length(beta), j=1:length(gamma)]
# map(testbeta,beta)
map(testWasser,vecb[:],vecg[:])


beta = exp.(-8:2);
eps = exp.(-8:0);
vecb=[beta[i] for i=1:length(beta), j=1:length(eps)]
vece=[eps[j] for i=1:length(beta), j=1:length(eps)]
# map(testbeta,beta)
map(testepsilonbeta,vece[:],vecb[:])

eps = exp.(-8:0);
nda = 20:50:500
vecnd=[nda[i] for i=1:length(nda), j=1:length(eps)]
vece=[eps[j] for i=1:length(nda), j=1:length(eps)]
# map(testbeta,beta)
map(testepsilon,vece[:],vecnd[:])
