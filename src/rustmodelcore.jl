abstract RustModelCore

#Dense, Infinite-horizon RustModelCore
immutable DIRMC <: RustModelCore
	beta::Float64						# discount factor
	u::Array{Float64,2}					# u[k,a]: flow utilities						
	v::Array{Float64,2}					# v[k,a]: choice-specific discounted value
	V::Array{Float64,1}					# V[k]: expected value
	pi::Array{Array{Float64,2},1}						# pi[a][k,k']: condition state transitions 
	p::Array{Float64,2}      			# p[k,a]: conditional choice probabilities (CCP)
end

#Sparse, Infinite-horizon RustModelCore
immutable SIRMC <: RustModelCore
	beta::Float64						# discount factor
	u::Array{Float64,2}					# u[k,a]: flow utilities						
	v::Array{Float64,2}					# v[k,a]: choice-specific discounted value
	V::Array{Float64,1}					# V[k]: expected value
	pi::Array{SparseMatrixCSC{Float64,Int},1}						# pi[a][k,k']: condition state transitions 
	p::Array{Float64,2}      			# p[k,a]: conditional choice probabilities (CCP)
end


#Dense, Finite-horizon RustModelCore
immutable DFRMC <: RustModelCore
	beta::Float64						# discount factor
	u::Array{Float64,2}					# u[k,a]: flow utilities						
	v::Array{Float64,2}					# v[k,a]: choice-specific discounted value
	V::Array{Float64,1}					# V[k]: expected value
	pi::Array{Array{Float64,2},1}						# pi[a][k,k']: condition state transitions 
	p::Array{Float64,2}      			# p[k,a]: conditional choice probabilities (CCP)
	
	# for finite horizon models only
	horizonindex::Int 					# index of the FIRST state of the LAST day
end

#Sparse, Finite-horizon RustModelCore
immutable SFRMC <: RustModelCore
	beta::Float64						# discount factor
	u::Array{Float64,2}					# u[k,a]: flow utilities						
	v::Array{Float64,2}					# v[k,a]: choice-specific discounted value
	V::Array{Float64,1}					# V[k]: expected value
	pi::Array{SparseMatrixCSC{Float64,Int},1}						# pi[a][k,k']: condition state transitions 
	p::Array{Float64,2}      			# p[k,a]: conditional choice probabilities (CCP)
	
	# for finite horizon models only
	horizonindex::Int 					# index of the FIRST state of the LAST day

	# for :SparseFiniteHorizon only
	kindices::Array{Array{Int,1},2}		#holds the non-zero k' at a given [k,a]
end


function rustmodelcore(beta,pi::Array{Array{Float64,2},1})
	da=length(pi)
	dk=size(pi[1],1)
	DIRMC(
		beta,
		Array(Float64,(dk,da)),			#u
		zeros(dk,da),					#v  (needs to be zeros()?)
		Array(Float64,dk),				#V
		pi,
		Array(Float64,(dk,da))			#p
	)
end

function rustmodelcore(beta,pi::Array{SparseMatrixCSC{Float64,Int},1})
	da=length(pi)
	dk=size(pi[1],1)
	SIRMC(
		beta,
		Array(Float64,(dk,da)),			#u
		zeros(dk,da),					#v  (needs to be zeros()?)
		Array(Float64,dk),				#V
		pi,
		Array(Float64,(dk,da))			#p
	)
end

function rustmodelcore(beta,pi::Array{Array{Float64,2},1},horizonindex::Int)
	da=length(pi)
	dk=size(pi[1],1)
	DFRMC(
		beta,
		Array(Float64,(dk,da)),			#u
		zeros(dk,da),					#v  (needs to be zeros()?)
		Array(Float64,dk),				#V
		pi,
		Array(Float64,(dk,da)),			#p
		horizonindex
	)
end

function rustmodelcore(beta,pi::Array{SparseMatrixCSC{Float64,Int},1},horizonindex::Int)
	da=length(pi)
	dk=size(pi[1],1)
	kindices=Array(Array{Int64,1},dk,da)
	for ia=1:da
		temp=full(pi[ia]')
		for ik=1:dk
			kindices[ik,ia]=find(temp[:,ik].!=0)
		end
	end
	SFRMC(
		beta,
		Array(Float64,(dk,da)),			#u
		zeros(dk,da),					#v  (needs to be zeros()?)
		Array(Float64,dk),				#V
		pi,
		Array(Float64,(dk,da)),			#p
		horizonindex,
		kindices
	)
end
