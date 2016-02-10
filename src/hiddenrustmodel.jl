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


immutable HiddenRustModel{S,T} <: StatisticalModel
	#Linear Rust Model u[k,a]=aa[k,a,itheta]*theta + bb[k,a]
	aa::Array{Float64,3}		  	#aa[k,a,itheta]
	bb::Array{Float64,2}		  	#bb[k,a]

	pis::Array{S,1} 				#condition state transitions 
	sindices::Array{Array{Int,1},2}		

	rustcore::T
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

	layer=hiddenlayer(mu,dx,dy,dtheta)
	
	model=HiddenRustModel(aa,bb,pis,sindices,core,layer)

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

pitemp(::Type{SparseMatrixCSC{Float64,Int64}},pis,dx,da)=
	[kron(pis[ia],sparse(fill(1/dx,dx,dx)))::SparseMatrixCSC{Float64,Int64} for ia=1:da]
pitemp(::Type{Array{Float64,2}},pis,dx,da)=
	[kron(pis[ia],fill(1/dx,dx,dx))::Array{Float64,2} for ia=1:da]


function hiddenrustmodel{S}(aa::Array,bb,mu,beta,pis::Array{S,1})
	dx=size(mu,1)
	da=size(bb,2)
	core=rustmodelcore(beta,pitemp(S,pis,dx,da))
	hiddenrustmodel(core,aa,bb,mu,beta,pis)
end

function hiddenrustmodel{S}(aa::Array,bb,mu,beta,pis::Array{S,1},horizonindex)
	dx=size(mu,1)
	da=size(bb,2)
	core=rustmodelcore(beta,pitemp(S,pis,dx,da),horizonindex)
	hiddenrustmodel(core,aa,bb,mu,beta,pis)
end

### Model "evaluation" functions (intermediate steps between deep structural parameters and m)


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


function m!{S,T}(model::HiddenRustModel{S,T})
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
end




