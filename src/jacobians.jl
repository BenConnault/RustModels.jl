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
	model.hiddenlayer.mjac[:]=0
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
							for lx=1:dx
								for kx=1:dx
									model.hiddenlayer.mjac[ix,iy,jx,jy,dlambda+iz]+=model.rustcore.pi[ia][ik,jk]*ccpjacq[jk,ja,lx,kx]*z2qj[lx,kx,iz]
								end
							end
							model.hiddenlayer.mjac[ix,iy,jx,jy,dlambda+iz]+=z2qj[ix,jx,iz]*model.pis[ia][is,js]*model.rustcore.p[jk,ja]
							# model.mjac[ix,iy,jx,jy,dlambda+iz]+=z2qjacmat[ixx,iz]*model.pis[ia][is,js]*model.rustcore.p[jk,ja]
						end
					end
				end	
			end
		end
	end
end

#direct inv(smm) (a priori a bad idea but...)
function ccpjac_inv(model::HiddenRustModel)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices,1)
	dlambda=size(model.aa,3)
	mmm=Array(Array{Float64,2},da)          #(dk,dk) inside
	mmu=Array(Array{Float64,2},da)			#(dk,dlambda) inside
	mme=Array(Array{Float64,2},dx,dx,da)	#(dk,dk) inside (remember ds*dx=dk)
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
				scale!(slice(model.rustcore.p,:,ia),mme[ix,jx,ia])
			end
		end
	end
	smm=sum(mmm)
	smmi=inv(smm)  #need to make this efficient as I do several smm\ below
	smu=sum(mmu)
	# sme=sum(mme,3)  #one-liner but type-unstable
	sme=fill(zeros(dx*ds,dx*ds),dx,dx)
	for ix=1:dx
		for jx=1:dx
			for ia=1:da
				broadcast!(+,sme[ix,jx],sme[ix,jx],mme[ix,jx,ia])   #in place addition sme[ix,jx]+=mme[ix,jx,ia]
			end
		end
	end
	mlambda=smmi*sparse(smu)
	jaclambda=zeros(dk,da,dlambda)
	jacq=zeros(dk,da,dx,dx)
	V=\(eye(dk)-model.rustcore.beta*model.rustcore.pi[1],model.rustcore.u[:,1]-log(slice(model.rustcore.p,:,1)))
	for ia=1:da
		for ilambda=1:dlambda
			for ik=1:dk
				for jk=1:dk
					jaclambda[ik,ia,ilambda]+=mmm[ia][ik,jk]*mlambda[jk,ilambda]
				end
				jaclambda[ik,ia,ilambda]=-jaclambda[ik,ia,ilambda]+mmu[ia][ik,ilambda]
			end
		end
	end
	container1=zeros(dk,dk)
	container2=zeros(dk,dk)
	for ix=1:dx
		for jx=1:dx
			container1[:,:]=smmi*sparse(sme[ix,jx])   #container1=smmi*sparse(sme[ix,jx])
			for ia=1:da
				#jacq[:,ia,ix,jx]=(-mmm[ia]*invproduct+mme[ix,jx,ia])* V
				A_mul_B!(container2,mmm[ia],container1)
				scale!(-1,container2)
				broadcast!(+,container2,container2,mme[ix,jx,ia])
				A_mul_B!(slice(jacq,:,ia,ix,jx),container2, V)    
			end
		end
	end	
	jaclambda,jacq
end

#matrix inversion with qrfact (or lufact)
function ccpjac(model::HiddenRustModel)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	ds=size(model.sindices,1)
	dlambda=size(model.aa,3)
	mmm=Array(Array{Float64,2},da)          #(dk,dk) inside
	mmu=Array(Array{Float64,2},da)			#(dk,dlambda) inside
	mme=Array(SparseMatrixCSC{Float64,Int64},dx,dx,da)	#(dk,dk) inside (remember ds*dx=dk)
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
		# fpis=full(model.pis[ia])
		for jx=1:dx
			for ix=1:dx
				# mme[ix,jx,ia]=zeros(dk,dk)
				#ee=spzeros(dx,dx)  #for a later "sparse" version
				ee=zeros(dx,dx)
				ee[ix,jx]=1.0
				mme[ix,jx,ia]=kron(model.pis[ia],sparse(ee)) #could easily take advantage of sparsity here
				scale!(model.rustcore.beta,mme[ix,jx,ia])
				sparsescale!(model.rustcore.p[:,ia],mme[ix,jx,ia]) #modify to standard scale!() for julia 0.5
			end
		end
	end
	smm=sparse(sum(mmm))
	# smmi=lufact(smm)  #need to make this efficient as I do several smm\ below
	smmi=qrfact(smm)  #in tests qrfacts seemed faster than lufact
	smu=sum(mmu)
	# sme=sum(mme,3)  #one-liner but type-unstable
	sme=Array(SparseMatrixCSC{Float64,Int64},dx,dx)
	for ix=1:dx
		for jx=1:dx
			sme[ix,jx]=copy(mme[ix,jx,1])
			for ia=2:da
				# sme[ix,jx]+=mme[ix,jx,ia]
				broadcast!(+,sme[ix,jx],sme[ix,jx],mme[ix,jx,ia])   #in place addition sme[ix,jx]+=mme[ix,jx,ia]
			end
		end
	end
	mlambda=smmi\smu
	jaclambda=zeros(dk,da,dlambda)
	jacq=zeros(dk,da,dx,dx)
	V=\(eye(dk)-model.rustcore.beta*model.rustcore.pi[1],model.rustcore.u[:,1]-log(model.rustcore.p[:,1]))
	for ia=1:da
		for ilambda=1:dlambda
			for ik=1:dk
				for jk=1:dk
					jaclambda[ik,ia,ilambda]+=mmm[ia][ik,jk]*mlambda[jk,ilambda]
				end
				jaclambda[ik,ia,ilambda]=-jaclambda[ik,ia,ilambda]+mmu[ia][ik,ilambda]
			end
		end
	end
	smmm=[-sparse(mmm[ia])::SparseMatrixCSC{Float64,Int64} for ia=1:da]
	for ix=1:dx
		for jx=1:dx
			invproduct=smmi\(sme[ix,jx]*V)
			for ia=1:da
				jacq[:,ia,ix,jx]=smmm[ia]*invproduct+mme[ix,jx,ia]*V
				# broadcast!(+,slice(jacq,:,ia,ix,jx),smmm[ia]*invproduct,sparse(mme[ix,jx,ia])*V)
			end
		end
	end	
	jaclambda,jacq
end

#the final nested loop is too heavy 
function ccpjac_explicit(model::HiddenRustModel)
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
	# sme=sum(mme,3)  #one-liner but type-unstable
	sme=fill(zeros(dx*ds,dx*ds),dx,dx)
	for ix=1:dx
		for jx=1:dx
			for ia=1:da
				sme[ix,jx]+=mme[ix,jx,ia]
			end
		end
	end
	# m=smm\smu   #useless! old code remnant 
	mlambda=smmi*smu
	jaclambda=zeros(dk,da,dlambda)
	jacq=zeros(dk,da,dx,dx)
	V=\(eye(dk)-model.rustcore.beta*model.rustcore.pi[1],model.rustcore.u[:,1]-log(model.rustcore.p[:,1]))
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
								jacq[ik,ia,ix,jx]+=-mmm[ia][ik,jk]*smmi[jk,kk]*sme[ix,jx][kk,lk]*V[lk]
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


#### sparsescale stopgap before this makes its way to julia 0.5

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