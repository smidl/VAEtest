using Plots; 
using Distributions;
using LinearAlgebra;
## Generate data
ndat = 500
nz = 1

function toydata2d(ϵ)
    pz = MvNormal(0.5ones(nz),0.15*Matrix(Diagonal(ones(nz))));
    z = rand(pz,ndat);
    ftrue(z) = [z.^2; z] 
    x= ftrue(z) .+ ϵ*randn(2,ndat)    

    pxtrue(x) = mean(exp.(-0.5*(sum(((x.-ftrue(z))./ϵ).^2,dims=1))))

    xa = [-1;-1.5].+[4;3.5].*rand(2,ndat);

    pxx = [pxtrue(x[:,i]) for i = 1:ndat]
    pxa = [pxtrue(xa[:,i]) for i = 1:ndat]

    iii = pxa.<0.001;
    xa = xa[:,iii]

    (x,xa,pxx,pxa[iii])
end
nz = 2 # estimate with different dimension

## normalize data!!!
# p_hx=Plots.histogram(x,nbins=40)#,xlabel='x',ylabel='count')
using Flux
using GenerativeModels
import GenerativeModels: loglikelihood
import GenerativeModels: AbstractVAE, AbstractPDF, AbstractCPDF

## NN == inicializace??
nx = 2
nh = 5;
niter = 5000
mbsize = 500

T=Float32
f = Chain(Dense(nz,nh,swish),Dense(nh,nx)) 
g = Chain(Dense(nx,nh,swish),Dense(nh,nz+1)) 

dec_dist = SharedVarCGaussian{T}(nx, nz, f, param(ones(T,nx)))
enc_dist = CGaussian{T,ScalarVar}(nz, nx, g)
prior = Gaussian(zeros(T,nz),ones(T,nz))

struct WAE{T} <: AbstractVAE{T}
    prior
    encoder
    decoder
end
Flux.@treelike WAE

model = WAE{T}(prior,enc_dist, dec_dist)

params_init = Flux.params(model)
opt = ADAM()
L=zeros(niter);
cb(model, data, loss, opt) = nothing;# (@show(loss(x)); push(L)=loss(x))
data = [x for i in 1:10000];
k(x,y) = GenerativeModels.imq(x,y,0.001);
# function loglikelihood(p::SharedVarCGaussian, x::AbstractArray, z::AbstractArray)
#     (xp,xvar) = mean_var(p,z);
#     d = x-xp;
#     -sum(d .* d./xvar .+ log.(xvar), dims=1) / 2
# end
function elbo(m::WAE{T}, x::AbstractArray; β=1)
    z = rand(m.encoder, x)
    llh = mean(-loglikelihood(m.decoder, x, z))
    kl  = mean(kld(m.encoder, m.prior, x))
    llh + β*kl
end
# lossf(x) = mse(x, mean(model.decoder, mean(model.encoder,x))) + 1*mmd(k,rand(model.encoder,x), rand(model.prior,size(x,2)))
lossf(x) = sum(-loglikelihood(model.decoder,x,mean(model.encoder,x))) ./ size(x,2) + 1e-5*mmd(k,rand(model.encoder,x), rand(model.prior,size(x,2)))
# train!(model, data, lossf, opt, cb)
losse(x)=elbo(model,x;β =1)
train!(model, data, losse, opt, cb)

# include("visualize_vae.jl")
visualize2d(model,x)