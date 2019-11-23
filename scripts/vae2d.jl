## normalize data!!!
# p_hx=Plots.histogram(x,nbins=40)#,xlabel='x',ylabel='count')
using Flux
using GenerativeModels
import GenerativeModels: loglikelihood
import GenerativeModels: AbstractVAE, AbstractPDF, AbstractCPDF

T=Float32


function toydata2d(ϵ,ndat=500)
    nz = 1;
    pz = MvNormal(0.5ones(nz),0.15*Matrix(Diagonal(ones(nz))));
    z = rand(pz,ndat);
    ftrue(z) = [z.^2; z] 
    x= ftrue(z) .+ (ϵ.*randn(2,ndat))

    pxtrue(x) = mean(exp.(-0.5*(sum(((x.-ftrue(z))./ϵ).^2,dims=1))))

    xa = [-1;-1.5].+[4;3.5].*rand(2,ndat);

    pxx = [pxtrue(x[:,i]) for i = 1:ndat]
    pxa = [pxtrue(xa[:,i]) for i = 1:ndat]

    iii = pxa.<0.001;
    xa = xa[:,iii]

    (x,xa,pxx,pxa[iii])
end


function buildVAE(x::AbstractArray,nz::Int,estimσ::Bool,useelbo::Bool,β,lengthscale)
    ## NN == inicializace??
    nx = 2;
    nh = 40;

    # decoder
    f = Chain(Dense(nz,nh,swish),Dense(nh,nx)) 
    if estimσ
        σ = [1e-5];
        dec_dist = CMeanGaussian{T,ScalarVar}(f, σ, nx)
    else
        dec_dist = CMeanGaussian{T,ScalarVar}(f, 1.0, nx)
    end
    # encoder
    g = Chain(Dense(nx,nh,swish),Dense(nh,2*nz)) 
    enc_dist = CMeanVarGaussian{T,DiagVar}(g)

    prior = Gaussian(zeros(T,nz),ones(T,nz))

    model = VAE{T}(prior,enc_dist, dec_dist)

    params_init = Flux.params(model)
    opt = ADAM()
    mydata = Iterators.repeated((x,),10000);
    k(x,y) = GenerativeModels.imq(x,y,lengthscale);
    lossf(x) = sum(-loglikelihood(model.decoder,x,mean(model.encoder,x))) ./ size(x,2) + β*mmd(k,rand(model.encoder,x), rand(model.prior,size(x,2)))
    # train!(model, data, lossf, opt, cb)
    losse(x)=elbo(model,x;β=β)
    if useelbo
        Flux.train!(losse, params_init, mydata, opt)
    else
        Flux.train!(lossf, params_init, mydata, opt)
    end

    model
end
    # include("visualize_vae.jl")