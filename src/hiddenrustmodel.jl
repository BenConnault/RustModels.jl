### TYPES AND CONSTRUCTORS

immutable HiddenLayer <: DynamicDiscreteModel
	m::Array{Float64,4}			  	#the transition matrix given as m[x,y,x',y'] 
	mu::Array{Float64,2}  			#initial distribution (dx,dy)
	mjac::Array{Float64,5}			#jacobian
	
	#discrete filter variables
	rho::Array{Float64,1}				#container for the constant rho used in the discrete filter
	phi::Array{Float64,1}			#filter value today
	psi::Array{Float64,1}			#filter value tomorrow
	rhojac::Array{Float64,1}				
	phijac::Array{Float64,2}			
	psijac::Array{Float64,2}
end		

hiddenlayer(mu,dx,dy,dtheta)=HiddenLayer(
		#initialization to ZERO rather than NA is IMPORTANT here:
		zeros(dx,dy,dx,dy),							#m
		mu,											#mu
		Array(Float64,dx,dy,dx,dy,dtheta),			#mjac
		Array(Float64,1),							#rho
		Array(Float64,dx),							#phi
		Array(Float64,dx),							#psi
		Array(Float64,dtheta),						#rhojac
		Array(Float64,dx,dtheta),					#phijac
		Array(Float64,dx,dtheta)					#psijac
)

immutable HiddenRustModel{S} <: StatisticalModel
	#Linear Rust Model u[k,a]=aa[k,a,itheta]*theta + bb[k,a]
	aa::Array{Float64,3}		  	#aa[k,a,itheta]
	bb::Array{Float64,2}		  	#bb[k,a]

	#might need to "type" pis:T and HiddenRustModel{T} 
	pis::Array{S,1} 				#condition state transitions 
	sindices::Array{Array{Int,1},2}		
	kindices::Array{Array{Int,1},2}

	rustcore::RustModelCore
	hiddenlayer::HiddenLayer

end


function hiddenrustmodel(core::RustModelCore,aa,bb,mu,beta,pis)
	dx,dy=size(mu)
	dk,da,dlambda=size(aa)
	dtheta=dlambda+dx*(dx-1) 	
	if (dk%dx!=0) 
		error("ds/dx is not integer") 
	end
	ds=Int(dk/dx)

	sindices=Array(Array{Int64,1},ds,da)
	for ia=1:da
		temp=full(pis[ia])'
		for is=1:ds
			sindices[is,ia]=find(temp[:,is].!=0)
		end
	end

	q=fill(1/dx,dx,dx)

	kindices=Array(Array{Int64,1},dk,da)
	for ia=1:da
		temp=kron(pis[ia],q)'
		for ik=1:dk
			kindices[ik,ia]=find(temp[:,ik].!=0)
		end
	end

	layer=hiddenlayer(mu,dx,dy,dtheta)
	
	model=HiddenRustModel(aa,bb,pis,sindices,kindices,core,layer)

	#reasonable initial value for solving the first dp?
	u!(model,rand(size(model.aa)[3]))
	model.rustcore.V[:]=sum(model.rustcore.u,2)  
	
	#If needed: run a couple of fixed point iterations
	# model.rustcore.V[:]=model.rustcore.v[:,1]+log(sum(exp(model.rustcore.v.-modelrustcore..v[:,1]),2))
	# for l=1:5
	# 	for ia=1:da
	# 		model.rustcore.v[:,ia]=model.rustcore.u[:,ia]+model.rustcore.beta*model.rustcore.pi[ia]*model.rustcore.V
	# 	end
	# 	model.rustcore.V[:]=model.rustcore.v[:,1]+log(sum(exp(model.rustcore.v.-model.rustcore.v[:,1]),2))
	# 	for ia=1:da
	# 		model.rustcore.p[:,ia]=exp(model.rustcore.v[:,ia]-model.rustcore.V) 
	# 	end
	# 	# println(round(model.rustcore.p,3))
	# 	# checkdp(model.rustcore)
	# end

	model
end

pitemp(::Type{SparseMatrixCSC{Float64,Int64}},dk,da)=[spzeros(dk,dk)::SparseMatrixCSC{Float64,Int64} for ia=1:da]
pitemp(::Type{Array{Float64,2}},dk,da)=[zeros(dk,dk)::Array{Float64,2} for ia=1:da]


function hiddenrustmodel{S}(aa::Array,bb,mu,beta,pis::Array{S,1})
	dk,da=size(bb)
	core=rustmodelcore(beta,pitemp(S,dk,da))
	hiddenrustmodel(core,aa,bb,mu,beta,pis)
end

function hiddenrustmodel{S}(aa::Array,bb,mu,beta,pis::Array{S,1},horizonindex)
	dk,da=size(bb)
	core=rustmodelcore(beta,pitemp(S,dk,da),horizonindex)
	hiddenrustmodel(core,aa,bb,mu,beta,pis)
end

### StatisticalModel interface



function pi!{S}(model::HiddenRustModel{S},q::S)
	da=size(model.bb,2)
	for ia=1:da
		model.rustcore.pi[ia]=kron(model.pis[ia],q)
	end
end	

pi!(model::HiddenRustModel{SparseMatrixCSC{Float64,Int}},q::Array{Float64,2})=pi!(model,sparse(q))

function u!(model::HiddenRustModel,lambda::Array{Float64,1})
	dk,da=size(model.bb)
	dlambda=length(lambda)
	for ia=1:da
		for ik=1:dk
			model.rustcore.u[ik,ia]=model.bb[ik,ia]
			for ilambda=1:dlambda
				model.rustcore.u[ik,ia]+=model.aa[ik,ia,ilambda]*lambda[ilambda]
			end
		end
	end
end


function m!(model::HiddenRustModel)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices)[1]
	for iy=1:dy
		is,ia=ind2sub((ds,da),iy)
		for js in model.sindices[is,ia]    #only those indices which are nonzero #could store the matrix indexed by y, not sure if faster
			for ja=1:da
				jy=sub2ind((ds,da),js,ja)
				for jx=1:dx
					jk=sub2ind((dx,ds),jx,js)
					for ix=1:dx
						ik=sub2ind((dx,ds),ix,is)
						model.hiddenlayer.m[ix,iy,jx,jy]=model.rustcore.pi[ia][ik,jk]*model.rustcore.p[jk,ja]
					end
				end	
			end
		end
	end
	# println(round(sort(vec(unique(model.m))),2))
end



#for testing purposes
function naiveccpjac(model)
	dk,da=size(model.rustcore.p)
	mmm=Array(Array{Float64,2},da)
	mmu=Array(Array{Float64,2},da)
	for ia=1:da
		diagp=diagm(model.rustcore.p[:,ia])
		mmm[ia]=diagp*(I-model.rustcore.beta*model.rustcore.pi[ia])
		mmu[ia]=diagp*slice(model.aa,:,ia,:)
	end
	smm=sum(mmm)
	smu=sum(mmu)
	m=smm\smu
	jac=Array(Array{Float64,2},da)
	for ia=1:da
		jac[ia]=-mmm[ia]*m+mmu[ia]
	end
	jac
end


function ccpjac(model::HiddenRustModel)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices,1)
	dlambda=size(model.aa,3)
	mmm=Array(Array{Float64,2},da)
	mmu=Array(Array{Float64,2},da)
	mme=Array(Array{Float64,2},dx,dx,da)
	for ia=1:da
		mmm[ia]=zeros(dk,dk)
		mmu[ia]=zeros(dk,dlambda)
		for jk=1:dk
			for ik=1:dk
				mmm[ia][ik,jk]=model.rustcore.p[ik,ia]*((ik==jk)-model.rustcore.beta*model.rustcore.pi[ia][ik,jk])
			end
		end
		for jlambda=1:dlambda
			for ik=1:dk
				mmu[ia][ik,jlambda]=model.rustcore.p[ik,ia]*model.aa[ik,ia,jlambda]
			end
		end
		fpis=full(model.pis[ia])
		for jx=1:dx
			for ix=1:dx
				# mme[ix,jx,ia]=zeros(dk,dk)
				#ee=spzeros(dx,dx)  #for a later "sparse" version
				ee=zeros(dx,dx)
				ee[ix,jx]=1.0
				mme[ix,jx,ia]=kron(fpis,ee) #could easily take advantage of sparsity here
				scale!(model.rustcore.beta,mme[ix,jx,ia])
				scale!(model.rustcore.p[:,ia],mme[ix,jx,ia])
			end
		end
	end
	smm=sum(mmm)
	smmi=inv(smm)
	smu=sum(mmu)
	sme=sum(mme,3)
	m=smm\smu
	mlambda=smmi*smu
	jaclambda=zeros(dk,da,dlambda)
	jacq=zeros(dk,da,dx,dx)
	V=\(eye(dk)-model.rustcore.beta*model.rustcore.pi[1],model.u[:,1]-log(model.rustcore.p[:,1]))
	for ia=1:da
		for ilambda=1:dlambda
			for ik=1:dk
				for jk=1:dk
					jaclambda[ik,ia,ilambda]+=mmm[ia][ik,jk]*mlambda[jk,ilambda]
				end
				jaclambda[ik,ia,ilambda]=-jaclambda[ik,ia,ilambda]+mmu[ia][ik,ilambda]
			end
		end
		for ix=1:dx
			for jx=1:dx
				for ik=1:dk
					for lk=1:dk
						for jk=1:dk
							for kk=1:dk
								#in profiling, this appears to be the expensive line:
								jacq[ik,ia,ix,jx]+=-mmm[ia][ik,jk]*smmi[jk,kk]*sme[ix,jx,1][kk,lk]*V[lk]
							end
						end
						jacq[ik,ia,ix,jx]+=mme[ix,jx,ia][ik,lk]*V[lk]
					end
				end
			end
		end
	end	
	jaclambda,jacq
end

function mjac!(model::HiddenRustModel,theta)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices,1)
	dlambda=size(model.aa,3)
	dz=dx*(dx-1)
	z2qjacmat=z2qjac(theta[end-dz+1:end])
	z2qj=reshape(z2qjacmat,dx,dx,dz)
	ccpjaclambda,ccpjacq=ccpjac(model)
	# ccpjacz=reshape(reshape(ccpjacq,dk*da,dx*dx)*z2qjacmat,dk,da,dz) #can be devectorized for efficiency
	model.mjac[:]=0
	for iy=1:dy
		is,ia=ind2sub((ds,da),iy)
		for js in model.sindices[is,ia]    #only those indices which are nonzero #could store the matrix indexed by y, not sure if faster
			for ja=1:da
				jy=sub2ind((ds,da),js,ja)
				for jx=1:dx
					jk=sub2ind((dx,ds),jx,js)
					for ix=1:dx
						ik=sub2ind((dx,ds),ix,is)
						for ilambda=1:dlambda
							model.mjac[ix,iy,jx,jy,ilambda]=model.rustcore.pi[ia][ik,jk]*ccpjaclambda[jk,ja,ilambda]
						end
						for iz=1:dz
							# ixx=(jx-1)*dx+ix
							for lx=1:dx
								for kx=1:dx
									model.mjac[ix,iy,jx,jy,dlambda+iz]+=model.rustcore.pi[ia][ik,jk]*ccpjacq[jk,ja,lx,kx]*z2qj[lx,kx,iz]
								end
							end
							model.mjac[ix,iy,jx,jy,dlambda+iz]+=z2qj[ix,jx,iz]*model.pis[ia][is,js]*model.rustcore.p[jk,ja]
							# model.mjac[ix,iy,jx,jy,dlambda+iz]+=z2qjacmat[ixx,iz]*model.pis[ia][is,js]*model.rustcore.p[jk,ja]
						end
					end
				end	
			end
		end
	end
end



