using Plots; gr()
using Distributions;
using LinearAlgebra;
## Generate data
ndat = 500
nz = 1

pz = MvNormal(0.5ones(nz),0.15*Matrix(Diagonal(ones(nz))));
z = rand(pz,ndat);
x = [z.^2; z] .+ 0.1randn(2,ndat)

## normalize data!!!
# p_hx=Plots.histogram(x,nbins=40)#,xlabel='x',ylabel='count')

using Flux
using IPMeasures
using MLDataPattern

## NN == inicializace??
nx = 2
nh = 5;
niter = 5000
mbsize = 500
#g = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
# g  = Dense(nx, nz) 
if false
    f = Chain(Dense(nz,nh,swish),Dense(nh,nh,swish),Dense(nh,nx)) 
    g = Chain(Dense(nx,nh,swish),Dense(nh,nh,swish),Dense(nh,nz)) 
else
    f = Chain(Dense(nz,nh,swish),Dense(nh,nx)) 
    g = Chain(Dense(nx,nh,swish),Dense(nh,nz)) 
end
# f = Dense(nz,nx)

s = 100.0;#param(rand(2,1))
k = IPMeasures.IMQKernel(.001);
# k = IPMeasures.RQKernel(0.05)

#loss(x,z) = Flux.mse(f(mu(x)+exp.(0.5*lsig(x)).*z),x) + KL(mu(x),lsig(x))
function loss(x)
    #s+exp.(s).*0.5*sum((x.-f(z_given_x)).^2) + KL(mu(x),lsig(x))
    zsample = g(x);
    zprior=rand(pz,size(x,2));
    0.5*sum((x.-f(zsample)).^2 .+ s*IPMeasures.mmd(k,zsample,zprior));
end
ps = Flux.params(g, f)

opt = Flux.ADAM(1e-3)

LL = zeros(1,niter);


# Q: how to repeat? N -times?
#opt = runall(opt)
#z = randn(nz,1000)
#  Flux.train!(x->loss(x),ps,RandomBatches((x,),size=mbsize, count=niter),opt, cb=)
for i=1:niter
    # ind1 = convert(Array{Int},floor.(rand(mbsize)*ndat.+1))
    # xmb = x[:,ind1];

    l = loss(x);
    Flux.Tracker.back!(l);
    for p in ps
        D = Flux.Optimise.apply!(opt,p.data,p.grad)
        p.data .-= D;
        p.grad .= 0.0;
    end
    LL[i]=l.data;
    #cb() == :stop && break

end


xg1 = -0.5+minimum(x[1,:]):0.1:maximum(x[1,:])+0.5; # grid of x
xg2 = -0.5+minimum(x[2,:]):0.1:maximum(x[2,:])+0.5;
Xg = [[xgi,xgj] for xgi in xg1, xgj in xg2] #-0.5:0.01:1.5; # grid of x
Xg=hcat(Xg...)

# compare probability scores
var_px = var(x-f(g(x))).data

Px_vita(x1,x2) = exp.(-0.5*sum([x1,x2]-Flux.data(f(g([x1,x2])))).^2/var_px);
function Px_vita(X,var_px)
    px = zeros(1,size(X,2))
    for i=1:size(X,2)
        px[i] = exp.(-0.5*sum(X[:,i]-Flux.data(f(g(X[:,i])))).^2/var_px);
    end
    px
end
function Px_vitaZ(X,Z,var_px)
    px = zeros(1,size(X,2))
    for i=1:size(X,2)
        px[i] = exp.(-0.5*sum(X[:,i]-Flux.data(f(Z[:,i]))).^2/var_px);
    end
    px
end
function Jaco(x1,x2) 
    Sv= svd(Flux.Tracker.jacobian(g,[x1,x2]));
    abs(Sv.S[1].^2)
end
function Jaco(X) 
    Ja = zeros(1,size(X,2))
    for i=1:size(X,2)
        J = Flux.Tracker.jacobian(g,X[:,i]);
        Ja[i] = (J*J')[1]; # only for 1D !!!!!
    end
    Ja
end
function JacoD(x1,x2) 
    Sv= svd(Flux.Tracker.jacobian(f,[x1,x2]));
    1.0/abs(prod(Sv.S.^2))
end
function Pz_Jaco(x1,x2)
    z = g([x1,x2]).data;
    pdf(pz,z) .* Jaco(x1,x2)
end
function Pz_Jaco(X)
    z = g(X).data;
    pdf(pz,z)' .* Jaco(X)
end
function Pz_JacoD(x1,x2)
    z = g([x1,x2]).data;
    pdf(pz,z) .* JacoD(z[1],z[2])
end

function Pz(x1,x2)
    z = g([x1,x2]).data[1];
    pdf(pz,z)
end

function manifoldz(f, g, x, steps = 1000)
    z_x = Flux.data(g(x))
    z = param(z_x)
    ps = Flux.Tracker.Params([z])
    opt = ADAM()
    _info() = println("likelihood = ", lkl(model, x, z))
    li = Flux.data(-mean((x.-f(z)).^2))
    # Flux.train!((i) -> -mean(lkl(model, x, z)), ps, 1:steps, opt, cb = () -> Flux.throttle(_info(),5))
    Flux.train!((i) -> mean((x.-f(z)).^2), ps, 1:steps, opt)
    le = Flux.data(-mean((x.-f(z)).^2))
    # println("initial = ",li, " final = ",le)
    Flux.data(z), z_x, Flux.data(z)-z_x
end
Mz = manifoldz(f,g,Xg)[1]
Mzx = manifoldz(f,g,x)[1]

Px_Pz(x1,x2) = Px_vita(x1,x2).*Pz_Jaco(x1,x2)
Px_Pz(X) = Px_vita(X,var_px).*Pz_Jaco(X)
Px_PzP(X,Z) = Px_vita(X,var_px).*pdf(pz,Z)'.*Jaco(X)


KL(f,X)=mean(log.(f(X)))
Q(f,x)=quantile([log(f(x[1,i],x[2,i])) for i in 1:size(x,2)],0.05)

ppx=contour(xg1,xg2,Px_vita(Xg,var_px),title="Px_vita")
ppx=scatter!(x[1,:],x[2,:],mark='.')
ppz=contour(xg1,xg2,Pz_Jaco(Xg),title = "Pz with Jacobian")
ppz=scatter!(x[1,:],x[2,:])
ppxpz=contour(xg1,xg2,Px_Pz(Xg),title = "Px Vita * Pz with Jacobian")
ppxpz=scatter!(x[1,:],x[2,:])
ppxpzP=contour(xg1,xg2,Px_PzP(Xg),title = "Px Vita * Pz with Jacobian Proj.")
ppxpzP=scatter!(x[1,:],x[2,:])
    
kl_px=KL(Px_vita,x)
kl_pxpz=KL(Px_Pz,x)
kl_pxpz=KL(Px_PzP,x)


