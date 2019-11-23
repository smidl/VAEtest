using Plots

function jacobian(f, x)
    y = f(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    j = Array{T, 2}(undef, n, m)
    for i in 1:n
        j[i, :] .= gradient(x -> f(x)[i], x)[1]
    end
    return j
end

function lJaco(m,Xg)
    JJ = zeros(1,size(Xg,2));
    for i=1:size(Xg,2)
        jj,J = jacobian(x->mean(m.encoder,x)[:],Xg[:,i])
        (U,S,V) = svd(J)
        JJ[i]= sum(2*log.(S))
    end
    JJ
end
function lJacoD(m,Xg)
    JJ = zeros(1,size(Xg,2));
    Zg = mean(m.encoder,Xg);
    for i=1:size(Xg,2)
        jj,J = jacobian(x->mean(m.decoder,x)[:],Zg[:,i]);
        (U,S,V) = svd(J);
        JJ[i]= sum(2*log.(S));
    end
    JJ
end

lpx(m,Xg)= GenerativeModels.loglikelihood(m.decoder,Xg,mean(m.encoder,Xg)) # p(x| g(x))
lpz(m,Xg)= GenerativeModels.loglikelihood(m.prior,mean(m.encoder,Xg)) # p(x| g(x))

lp_orth(m,Xg) = (lpx(m,Xg) .+ lpz(m,Xg) .+ lJaco(m,Xg));
lp_orthD(m,Xg) = (lpx(m,Xg) .+ lpz(m,Xg) .- lJacoD(m,Xg));

function visualize2d(m,x,sname=nothing)
    z0=rand(m.prior,size(x,2));
    zp = rand(m.encoder,x);
    xp = rand(m.decoder,z0);
    pdata=scatter(x[1,:],x[2,:],label="data",marker = (:dot, 2, 0.3))
    pdata=scatter!(xp[1,:],xp[2,:],label="reconstruction",marker = (:dot, 2, 0.3))


    if size(z0,1)==1
        pzp=histogram(z0',nbins=20,label="prior",alpha=0.5);
        pzp=histogram!(zp',nbins=20,label="sampled",alpha=0.5);
    end
    if size(z0,1)==2
        pzp=scatter(z0[1,:],z0[2,:],mark='.',label="prior")
        pzp=scatter!(zp[1,:],zp[2,:],mark='.',label="sampled")
    end

    xg1 = -0.5+minimum(x[1,:]):0.1:maximum(x[1,:])+0.5; # grid of x
    xg2 = -0.5+minimum(x[2,:]):0.1:maximum(x[2,:])+0.5;
    Xg = [[xgi,xgj] for xgi in xg1, xgj in xg2] #-0.5:0.01:1.5; # grid of x
    Xg=hcat(Xg...)

    Zfromx= mean(m.encoder,Xg);



    # rs(Xg)=reshape(Xg,length(xg1),length(xg2));

    rs(lx)=reshape((lx.-maximum(lx)),length(xg1),length(xg2))';

    ppx=contour(xg1,xg2,rs(lpx(m,Xg)),title="Px_vita")
    ppx=scatter!(x[1,:],x[2,:],marker = (:dot, 2, 0.6))
    # ppx=plot!(Xfromz[1,:],Xfromz[2,:])

    ppz=contour(xg1,xg2,rs(lpz(m,Xg)),title = "Pz with Jacobian")
    ppz=scatter!(x[1,:],x[2,:],marker = (:dot, 2, 0.6))

    ppxo=contour(xg1,xg2,rs(lp_orth(m,Xg)),title = "Px Vita * Pz with Jacobian")
    ppxo=scatter!(x[1,:],x[2,:],marker = (:dot, 2, 0.6))

    ppxoD=contour(xg1,xg2,rs(lp_orthD(m,Xg)),title = "Px Vita * Pz with Jaco Deco")
    ppxoD=scatter!(x[1,:],x[2,:],marker = (:dot, 2, 0.6))

    plt=plot(pdata,pzp,ppx,ppz, ppxo,ppxoD, layout=(3,2), size=(800,800))
    if sname!=nothing
        savefig(sname)
    end
    plt
end

function evalmodel(m,x,xa)

    labe = [zeros(size(x,2));ones(size(xa,2))];
    xall = [x xa];

    Px = exp.(lpx(m,xall));
    as = 1. .- Px;
    fpr, tpr = EvalCurves.roccurve(as[:], labe)
    auc_x = EvalCurves.auc(fpr, tpr)

    Px = exp.(lpz(m,xall));
    as = 1. .- Px;
    fpr, tpr = EvalCurves.roccurve(as[:], labe)
    auc_z = EvalCurves.auc(fpr, tpr)

    Px = exp.(lp_orth(m,xall));
    as = 1. .- Px;
    fpr, tpr = EvalCurves.roccurve(as[:], labe)
    auc_j = EvalCurves.auc(fpr, tpr)

    Px = exp.(lp_orthD(m,xall));
    as = 1. .- Px;
    fpr, tpr = EvalCurves.roccurve(as[:], labe)
    auc_d = EvalCurves.auc(fpr, tpr)

    (auc_x = auc_x, auc_z = auc_z, auc_j = auc_j, auc_d = auc_d)
end