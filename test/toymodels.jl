function classicaltoys()

	ds=18   #6 days, 3 observed states 
	da=2
	dtheta=2    

	pi1=kron(hcat(zeros(6),eye(6,5)),rsm(3,3))
	pi2=kron(hcat(zeros(6),eye(6,5)),rsm(3,3))
	pi1[16,:]=[vec(RustModels.rsm(1,8));zeros(10)]
	pi1[17,:]=pi1[16,:]
	pi1[18,:]=pi1[16,:]
	pi2[16,:]=pi1[16,:]
	pi2[17,:]=pi1[16,:]
	pi2[18,:]=pi1[16,:]
	pis=[pi::Array{Float64,2} for pi=(pi1,pi2)]
	spis=[sparse(pi)::SparseMatrixCSC{Float64,Int} for pi=(pi1,pi2)]

	dk=ds

	beta=.95
	mu=fill(1/ds*da,ds,da)

	aa=rand(dk,da,dtheta)
	bb=zeros(dk,da)
	dirm=rustmodel(aa,bb,mu,beta,pis)
	sirm=rustmodel(aa,bb,mu,beta,spis)
	dfrm=rustmodel(aa,bb,mu,beta,pis,16)
	sfrm=rustmodel(aa,bb,mu,beta,spis,16)

	dirm,sirm,dfrm,sfrm
end

function hiddentoys()

	dx=2
	ds=18   #6 days, 3states 
	da=2

	pi1=kron(hcat(zeros(6),eye(6,5)),rsm(3,3))
	pi2=kron(hcat(zeros(6),eye(6,5)),rsm(3,3))
	pi1[16,:]=[vec(rsm(1,6));zeros(12)]
	pi1[17,:]=pi1[16,:]
	pi1[18,:]=pi1[16,:]
	pi2[16,:]=pi1[16,:]
	pi2[17,:]=pi1[16,:]
	pi2[18,:]=pi1[16,:]
	pis=[pi::Array{Float64,2} for pi=(pi1,pi2)]
	spis=[sparse(pi)::SparseMatrixCSC{Float64,Int} for pi=(pi1,pi2)]


	dy=ds*da
	dk=dx*ds

	beta=.95
	mu=rand(dx,dy)
	mu=mu./sum(mu)

	dlambda=2
	aa=rand(dk,da,dlambda)
	bb=zeros(dk,da)
	dirm=hiddenrustmodel(aa,bb,mu,beta,pis)
	sirm=hiddenrustmodel(aa,bb,mu,beta,spis)
	dfrm=hiddenrustmodel(aa,bb,mu,beta,pis,31)
	sfrm=hiddenrustmodel(aa,bb,mu,beta,spis,31)


	dirm,sirm,dfrm,sfrm
end


function hiddentoys(dx,days)

	ds=3*days
	da=2

	pi1=kron(hcat(zeros(days),eye(days,days-1)),rsm(3,3))
	pi2=kron(hcat(zeros(days),eye(days,days-1)),rsm(3,3))
	pi1[end-2,:]=[vec(rsm(1,6));zeros(days*3-6)]
	pi1[end-1,:]=pi1[end-2,:]
	pi1[end,:]=pi1[end-2,:]
	pi2[end-2,:]=pi1[end-2,:]
	pi2[end-1,:]=pi1[end-2,:]
	pi2[end,:]=pi1[end-2,:]
	pis=[pi::Array{Float64,2} for pi=(pi1,pi2)]
	spis=[sparse(pi)::SparseMatrixCSC{Float64,Int} for pi=(pi1,pi2)]
	horizonindex=dx*(3*(days-1))+1


	dy=ds*da
	dk=dx*ds

	beta=.95
	mu=rand(dx,dy)
	mu=mu./sum(mu)


	dlambda=2
	aa=rand(dk,da,dlambda)
	bb=zeros(dk,da)
	dirm=hiddenrustmodel(aa,bb,mu,beta,pis)
	sirm=hiddenrustmodel(aa,bb,mu,beta,spis)
	dfrm=hiddenrustmodel(aa,bb,mu,beta,pis,horizonindex)
	sfrm=hiddenrustmodel(aa,bb,mu,beta,spis,horizonindex)


	dirm,sirm,dfrm,sfrm
end


wage(ndays)=	500+50*max(0,ndays-10)

function teachermodel(dx,days)
	ds=3*days

	pi1=kron(hcat(zeros(days),eye(days,days-1)),rsm(3,3))
	pi2=kron(hcat(zeros(days),eye(days,days-1)),rsm(3,3))
	pi1[end-2,:]=[vec(rsm(1,6));zeros(days*3-6)]
	pi1[end-1,:]=pi1[end-2,:]
	pi1[end,:]=pi1[end-2,:]
	pi2[end-2,:]=pi1[end-2,:]
	pi2[end-1,:]=pi1[end-2,:]
	pi2[end,:]=pi1[end-2,:]
	pis=[pi::Array{Float64,2} for pi=(pi1,pi2)]
	spis=[sparse(pi)::SparseMatrixCSC{Float64,Int} for pi=(pi1,pi2)]
	horizonindex=dx*(3*(days-1))+1
	
	beta=0.9995

	da=2
	dk=ds*dx
	dy=ds*da
	dtheta=dx+1

	s2days=readcsv(Pkg.dir("HiddenRustModelsCode/data/states2statevariables.csv"),Int)
	aa=zeros(dx,ds,da,dtheta)
	for ix=1:dx
		aa[ix,:,1,ix]=1 		#leisure (ia=1) utility in a ix day => itheta =ix 
	end
	for is=352:ds
		aa[:,is,1,dtheta].+=wage(s2days[is,2])
		aa[:,is,2,dtheta].+=wage(s2days[is,2]+1)
	end
	aa=reshape(aa,(dk,da,dtheta))
	bb=zeros(dk,da)
	mu=fill(1/(dx*dy),dx,dy)

	hiddenrustmodel(aa,bb,mu,beta,spis,horizonindex)
end