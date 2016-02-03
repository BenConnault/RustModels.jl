"""
Check that the dynamic program is solved
"""
function checkdp(model::RustModelCore)
	V1=\(eye(model.pi[1])-model.beta*model.pi[1],model.u[:,1]-log(model.p[:,1]))
	V2=\(eye(model.pi[2])-model.beta*model.pi[2],model.u[:,2]-log(model.p[:,2]))
	norm(V1-V2)
end


#changing autoscale option might solve numerical issues sometimes
function solvedp!(model::DIRMC)
	da=size(model.u)[2]
	
	function ff!(z,fv)
		fv[:]=-1.0
		for ia=1:da
			fv[:]+=exp(model.u[:,ia]+model.beta*model.pi[ia]*z-z)	
		end
	end

	function fj!(z,fm)
		fm[:]=0.0							#inefficient reallocation of temp at each call
		for ia=1:da
			fm[:,:]+=diagm(exp(model.u[:,ia]+model.beta*model.pi[ia]*z-z))*(model.beta*model.pi[ia]-eye(size(fm,1)))
		end
	end

	df = NLsolve.DifferentiableMultivariateFunction(ff!,fj!)

	res=NLsolve.nlsolve(df,model.V,autoscale=false)	#will automatically start at previous value
	
	# println(res)
	model.V[:]=res.zero 		
	for ia=1:da
		model.p[:,ia]=exp(model.u[:,ia]+model.beta*model.pi[ia]*model.V-model.V) 
	end
	# println(round(model.p,2))
end


function solvedp!(model::SIRMC)
	da=size(model.u)[2]
	
	function ff!(z,fv)
		fv[:]=-1.0
		for ia=1:da
			# println(sum(model.u[:,ia]+model.beta*model.pi[ia]*z-z))
			fv[:]+=exp(model.u[:,ia]+model.beta*model.pi[ia]*z-z)	
		end
	end

	function fj!(z,fm)
		dpj=spzeros(size(fm)...)							#inefficient reallocation of temp at each call
		for ia=1:da
			dpj+=spdiagm(exp(model.u[:,ia]+model.beta*model.pi[ia]*z-z))*(model.beta*model.pi[ia]-speye(size(fm)[1]))
		end
		fm.nzval=dpj.nzval
		fm.rowval=dpj.rowval
		fm.colptr=dpj.colptr
	end

	df = NLsolve.DifferentiableSparseMultivariateFunction(ff!, fj!)
	res= NLsolve.nlsolve(df,model.V,autoscale=false)	#will automatically start at previous value
	# println(res)
	model.V[:]=res.zero 		
	for ia=1:da
		model.p[:,ia]=exp(model.u[:,ia]+model.beta*model.pi[ia]*model.V-model.V) 
	end
	# println(round(model.p,2))
end



### FINITE HORIZON METHODS

#Upon exiting any of the finite-horizon methods, model.v or model.V are NOT calibrated to any meaningful value.

#this works as long as observable states are ranked by chronological order.
function solvedp!(model::DFRMC)
	dk,da=size(model.p)

	model.V[:]=0		#could be slightly more efficient here, see the sparsefinitehorizon case
	for ik in dk:-1:1			#this works as long as observable states are ranked by chronological order. 
		for ia=1:da
			model.v[ik,ia]=model.u[ik,ia]+model.beta*dot(full(model.pi[ia])[ik,:],model.V)
			model.p[ik,ia]=exp(model.v[ik,ia]-model.v[ik,1])				#for numerical stability
		end
		model.p[ik,:]/=sum(model.p[ik,:])
		model.V[ik]=dot(model.p[ik,:],model.v[ik,:]-log(model.p[ik,:]))
	end
end

function solvedp!(model::SFRMC)
	dk,da=size(model.p)
	
	#I could do one loop instead of two
	#but speed matters
	# local pisix::Float64=0
	for ik in model.horizonindex:dk			
		model.V[ik]=0
		rho=0
		for ia=1:da
			model.p[ik,ia]=exp(model.u[ik,ia]-model.u[ik,1])				#for numerical stability
			rho+=model.p[ik,ia]
		end
		for ia=1:da
			model.p[ik,ia]/=rho
			model.V[ik]+=model.p[ik,ia]*(model.u[ik,ia]-log(model.p[ik,ia]))
		end
	end
	for ik in model.horizonindex-1:-1:1			#this works as long as observable states are ranked by chronological order. 
		model.V[ik]=0
		rho=0
		for ia=1:da
			model.v[ik,ia]=0
			# println(sum(full(model.pi[ia])[ik,model.kindices[ik,ia]]))
			for jk in model.kindices[ik,ia]
				model.v[ik,ia]+=model.pi[ia][ik,jk]*model.V[jk]
			end
			model.v[ik,ia]*=model.beta
			model.v[ik,ia]+=model.u[ik,ia]
			model.p[ik,ia]=exp(model.v[ik,ia]-model.v[ik,1])				#for numerical stability
			rho+=model.p[ik,ia]
		end
		for ia=1:da
			model.p[ik,ia]/=rho
			model.V[ik]+=model.p[ik,ia]*(model.v[ik,ia]-log(model.p[ik,ia]))
		end
	end
	# println(sort(vec(model.p)))
end


## DEPRECATED
## keep around for now for testing purposes

function solvedp_old!(model::RustModelCore)
	da=size(model.u)[2]
	
	function ff!(z,fv)
		fv[:]=-1
		for ia=1:da
			fv[:]=fv+ exp(model.u[:,ia]+model.beta*model.pi[ia]*z-z)	
		end
	end

	function fj!(z,fm)
		dpj=spzeros(size(fm)...)							#inefficient reallocation of temp at each call
		for ia=1:da
			dpj+=spdiagm(exp(model.u[:,ia]+model.beta*model.pi[ia]*z-z))*(model.beta*model.pi[ia]-speye(size(fm)[1]))
		end
		fm.nzval=dpj.nzval
		fm.rowval=dpj.rowval
		fm.colptr=dpj.colptr
	end

	df = NLsolve.DifferentiableSparseMultivariateFunction(ff!, fj!)
	res= NLsolve.nlsolve(df,model.V,autoscale=false)	#will automatically start at previous value
	# println(res)
	model.V[:]=res.zero 		
	for ia=1:da
		model.p[:,ia]=exp(model.u[:,ia]+model.beta*model.pi[ia]*model.V-model.V) 
	end
end