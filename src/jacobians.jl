# As of now, the jacobian methods are tested with sparse pis models.
# It is unclear if mjac should be dense or sparse for a sparse model.

#jacobian from the canonical parameter theta=[lambda;z] where z parametrizes q.
function mjac!(model::HiddenRustModel,theta)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices,1)
	dlambda=size(model.aa,3)
	dz=dx*(dx-1)
	z2qjacmat=z2qjac(theta[end-dz+1:end])
	z2qj=reshape(z2qjacmat,dx,dx,dz)
	
	# @code_warntype(ccpjac(model))

	# filename=string(Int(Dates.datetime2unix(now())))*"-type-stability.log"
	# file=open(filename,"w")
	# code_warntype(file,ccpjac,(typeof(model),))
	# close(file)
	# error()

	ccpjaclambda,ccpjacq=ccpjac(model)
	# ccpjacz=reshape(reshape(ccpjacq,dk*da,dx*dx)*z2qjacmat,dk,da,dz) #can be devectorized for efficiency
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
							model.hiddenlayer.mjac[ix,iy,jx,jy,ilambda]=model.rustcore.pi[ia][ik,jk]*ccpjaclambda[jk,ja,ilambda]
						end
						for iz=1:dz
							# ixx=(jx-1)*dx+ix
							model.hiddenlayer.mjac[ix,iy,jx,jy,dlambda+iz]=0
							for lx=1:dx
								for kx=1:dx
									#derivatives of m by lambda through the CCPs p
									model.hiddenlayer.mjac[ix,iy,jx,jy,dlambda+iz]+=model.rustcore.pi[ia][ik,jk]*ccpjacq[jk,ja,lx,kx]*z2qj[lx,kx,iz]
								end
							end
							#derivatives of m by lambda through pi
							model.hiddenlayer.mjac[ix,iy,jx,jy,dlambda+iz]+=z2qj[ix,jx,iz]*model.pis[ia][is,js]*model.rustcore.p[jk,ja]
							# model.mjac[ix,iy,jx,jy,dlambda+iz]+=z2qjacmat[ixx,iz]*model.pis[ia][is,js]*model.rustcore.p[jk,ja]
						end
					end
				end	
			end
		end
	end
	reshape(reshape(ccpjacq,dk*da,dx*dx)*reshape(z2qj,dx*dx,dz),dk,da,dz),ccpjacq
end

#optimized ccpjac function
#the profiler should spend most of its time on "matrix inversion" operations:
#the three lines involving smmi
function ccpjac(model::HiddenRustModel)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices,1)
	dlambda=size(model.aa,3)
	mmm=Array(SparseMatrixCSC{Float64,Int64},da)          #(dk,dk) inside
	mmu=Array(Array{Float64,2},da)			#(dk,dlambda) inside
	mme=Array(SparseMatrixCSC{Float64,Int64},dx,dx,da)	#(dk,dk) inside (remember ds*dx=dk)
	for ia=1:da
		mmm[ia]=speye(dk)-model.rustcore.beta*model.rustcore.pi[ia]
		sparsescale!(model.rustcore.p[:,ia],mmm[ia]) #modify this to standard scale!() for julia 0.5
		mmu[ia]=copy(slice(model.aa,:,ia,:))
		scale!(model.rustcore.p[:,ia],mmu[ia])
		for jx=1:dx
			for ix=1:dx
				# mme[ix,jx,ia]=zeros(dk,dk)
				#ee=spzeros(dx,dx)  #for a later "sparse" version
				ee=zeros(dx,dx)
				ee[ix,jx]=1.0
				mme[ix,jx,ia]=kron(model.pis[ia],sparse(ee)) #could easily take advantage of sparsity here
				scale!(model.rustcore.beta,mme[ix,jx,ia])
				sparsescale!(model.rustcore.p[:,ia],mme[ix,jx,ia]) #modify this to standard scale!() for julia 0.5
			end
		end
	end
	smm=sum(mmm)      #sparse (dk,dk)
	# smmi=lufact(smm) 
	smu=sum(mmu)
	sme=sum(mme,3)  #one-liner but type-unstable
	# sme=Array(SparseMatrixCSC{Float64,Int64},dx,dx)   #(dk,dk) inside
	# for ix=1:dx
	# 	for jx=1:dx
	# 		sme[ix,jx]=spzeros(dk,dk)
	# 		for ia=1:da
	# 			# sme[ix,jx]+=mme[ix,jx,ia]
	# 			broadcast!(+,sme[ix,jx],sme[ix,jx],mme[ix,jx,ia])   #in place addition sme[ix,jx]+=mme[ix,jx,ia]
	# 		end
	# 	end
	# end
	
	# jacobian with respect to structural utility parameter ("lambda")
	jaclambda=zeros(dk,da,dlambda)
	# mlambda=smmi\smu     #mlambda is 100% dense
	println(size(smm))
	for ilambda=1:dlambda
		# ll,ll2=Krylov.lsqr(smm,smu[:,ilambda],atol=1e-14,itmax=2000)
		mlambdailambda=smm\smu[:,ilambda]
		println("jaclambda inversions: ", norm(smm*mlambdailambda-slice(smu,:,ilambda)))
		for ia=1:da
			A_mul_B!(slice(jaclambda,:,ia,ilambda),mmm[ia],mlambdailambda)
			scale!(-1,slice(jaclambda,:,ia,ilambda))
			broadcast!(+,slice(jaclambda,:,ia,ilambda),slice(jaclambda,:,ia,ilambda),slice(mmu[ia],:,ilambda))
		end
	end
	
	# jacobian with respect to q
	jacq=zeros(dk,da,dx,dx)
	# V,ll2=Krylov.lsqr(speye(dk)-model.rustcore.beta*model.rustcore.pi[1],slice(model.rustcore.u,:,1)-log(slice(model.rustcore.p,:,1)),atol=1e-14,itmax=2000)
	V=\(speye(dk)-model.rustcore.beta*model.rustcore.pi[1],slice(model.rustcore.u,:,1)-log(slice(model.rustcore.p,:,1)))
	println("V inversion: ", norm((speye(dk)-model.rustcore.beta*model.rustcore.pi[1])*V-(slice(model.rustcore.u,:,1)-log(slice(model.rustcore.p,:,1)))))
	for ix=1:dx
		for jx=1:dx
			invproduct=smm\(sme[ix,jx]*V)
			# invproduct,ll=Krylov.lsqr(smm,(sme[ix,jx]*V),atol=1e-14,itmax=2000)
			# ll=Krylov.crmr(smm,(sme[ix,jx]*V))
			println("jacq inversions: ", norm(smm*invproduct-(sme[ix,jx]*V)), " $(size(smm))")
			# println(ll[2])
			for ia=1:da
				broadcast!(+,slice(jacq,:,ia,ix,jx),-mmm[ia]*invproduct,sparse(mme[ix,jx,ia])*V)
			end
		end
	end	
	jaclambda,jacq
end


#### convenience functions

spindex(m)=sum(map(x-> x≈0,m))/sum(size(m))


#### sparsescale stopgap before the following makes its way to julia 0.5

function copyinds!(C::SparseMatrixCSC, A::SparseMatrixCSC)
    if C.colptr !== A.colptr
        resize!(C.colptr, length(A.colptr))
        copy!(C.colptr, A.colptr)
    end
    if C.rowval !== A.rowval
        resize!(C.rowval, length(A.rowval))
        copy!(C.rowval, A.rowval)
    end
end

function sparsescale!(C::SparseMatrixCSC, b::Vector, A::SparseMatrixCSC)
    m, n = size(A)
    (m==length(b) && size(A)==size(C)) || throw(DimensionMismatch())
    copyinds!(C, A)
    Cnzval = C.nzval
    Anzval = A.nzval
    Arowval = A.rowval
    resize!(Cnzval, length(Anzval))
    for col = 1:n, p = A.colptr[col]:(A.colptr[col+1]-1)
        @inbounds Cnzval[p] = Anzval[p] * b[Arowval[p]]
    end
    C
end

sparsescale!(b::Vector, A::SparseMatrixCSC)=sparsescale!(A,b,A)