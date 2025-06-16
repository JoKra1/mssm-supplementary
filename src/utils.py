from mssm.models import *
from mssmViz.sim import *
from mssmViz.plot import *
from mssm.src.python.compare import compare_CDL
from mssm.src.python.gamm_solvers import deriv_transform_mu_eta,deriv_transform_eta_beta
from mssm.src.python.utils import *
import copy
import numpy as np
import scipy as scp

############################# Define General Smooth Model ############################# 
class GAMLSSGENSMOOTHFamily(GENSMOOTHFamily):
    """Implementation of the ``GENSMOOTHFamily`` class that uses only information about the likelihood to estimate
    a GAMLSS model.

    References:

        - Wood, Pya, & Säfken (2016). Smoothing Parameter and Model Selection for General Smooth Models.
        - Nocedal & Wright (2006). Numerical Optimization. Springer New York.
    """

    def __init__(self, pars: int, links:[Link], llkfun:Callable, *llkargs) -> None:
        super().__init__(pars, links, *llkargs)
        self.llkfun = llkfun
    
    def llk(self, coef, coef_split_idx, y, Xs):
        return self.llkfun(coef, coef_split_idx, self.links, y, Xs,*self.llkargs)
    
    def gradient(self, coef, coef_split_idx, y, Xs):
        """
        Function to evaluate gradient for gsmm model.
        """
        coef = coef.reshape(-1,1)
        split_coef = np.split(coef,coef_split_idx)
        eta_mu = Xs[0]@split_coef[0]
        if len(Xs) > 1:
            eta_sd = Xs[1]@split_coef[1]
        
        # Get the Gamlss family
        gammlss_family = self.llkargs[0]
        
        if gammlss_family.d_eta == False:
         
            if len(Xs) > 1:
                d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,[self.links[0].fi(eta_mu),self.links[1].fi(eta_sd)],gammlss_family)
            else:
                d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,[self.links[0].fi(eta_mu)],gammlss_family)
        else:
            if len(Xs) > 1:
                d1eta = [fd1(y,*[self.links[0].fi(eta_mu),self.links[1].fi(eta_sd)]) for fd1 in gammlss_family.d1]
                d2eta = [fd2(y,*[self.links[0].fi(eta_mu),self.links[1].fi(eta_sd)]) for fd2 in gammlss_family.d2]
                d2meta = [fd2m(y,*[self.links[0].fi(eta_mu),self.links[1].fi(eta_sd)]) for fd2m in gammlss_family.d2m]
            else:
                d1eta = [fd1(y,*[self.links[0].fi(eta_mu)]) for fd1 in gammlss_family.d1]
                d2eta = [fd2(y,*[self.links[0].fi(eta_mu)]) for fd2 in gammlss_family.d2]
                d2meta = [fd2m(y,*[self.links[0].fi(eta_mu)]) for fd2m in gammlss_family.d2m]
            
                
        grad,_ = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=True)
        #print(pgrad.flatten())
        return grad.reshape(-1,1)
    
    def hessian(self, coef, coef_split_idx, y, Xs):
        return None
            

def llk_gamm_fun(coef,coef_split_idx,links,y,Xs,gammlss_family):
    """Likelihood for a GAM(LSS) - implemented so
    that the model can be estimated using the general smooth code.

    Note, gammlss_family is passed via llkargs so that this code works with
    Gaussian and Gamma models.
    """
    coef = coef.reshape(-1,1)
    split_coef = np.split(coef,coef_split_idx)
    eta_mu = Xs[0]@split_coef[0]
    if len(Xs) > 1:
        eta_sd = Xs[1]@split_coef[1]
    
    mu_mu = links[0].fi(eta_mu)
    if len(Xs) > 1:
        mu_sd = links[1].fi(eta_sd)
    
    if len(Xs) > 1:
        llk = gammlss_family.llk(y,mu_mu,mu_sd)
    else:
        llk = gammlss_family.llk(y,mu_mu)

    if np.isnan(llk):
        return -np.inf
    
    return llk

def init_lambda(formulas):
    n_lam = np.sum([len(fm.penalties) for fm in formulas])
    return [1.1 for _ in range(n_lam)]

######################################################################### Exported/Modified MSSM functions #########################################################################

def _compute_VB_corr_terms_MP(family,address_y,address_dat,address_ptr,address_idx,address_datXX,address_ptrXX,address_idxXX,shape_y,shape_dat,shape_ptr,shape_datXX,shape_ptrXX,rows,cols,rPen,offset,r):
   """
   Multi-processing code for Grevel & Scheipl correction for Gaussian additive model - see ``correct_VB`` for details. Taken from mssm.
   """
   dat_shared = shared_memory.SharedMemory(name=address_dat,create=False)
   ptr_shared = shared_memory.SharedMemory(name=address_ptr,create=False)
   idx_shared = shared_memory.SharedMemory(name=address_idx,create=False)
   dat_sharedXX = shared_memory.SharedMemory(name=address_datXX,create=False)
   ptr_sharedXX = shared_memory.SharedMemory(name=address_ptrXX,create=False)
   idx_sharedXX = shared_memory.SharedMemory(name=address_idxXX,create=False)
   y_shared = shared_memory.SharedMemory(name=address_y,create=False)

   data = np.ndarray(shape_dat,dtype=np.double,buffer=dat_shared.buf)
   indptr = np.ndarray(shape_ptr,dtype=np.int64,buffer=ptr_shared.buf)
   indices = np.ndarray(shape_dat,dtype=np.int64,buffer=idx_shared.buf)
   dataXX = np.ndarray(shape_datXX,dtype=np.double,buffer=dat_sharedXX.buf)
   indptrXX = np.ndarray(shape_ptrXX,dtype=np.int64,buffer=ptr_sharedXX.buf)
   indicesXX = np.ndarray(shape_datXX,dtype=np.int64,buffer=idx_sharedXX.buf)
   y = np.ndarray(shape_y,dtype=np.double,buffer=y_shared.buf)

   X = scp.sparse.csc_array((data,indices,indptr),shape=(rows,cols),copy=False)
   XX = scp.sparse.csc_array((dataXX,indicesXX,indptrXX),shape=(cols,cols),copy=False)

   # Prepare penalties with current lambda candidate r
   for ridx,rc in enumerate(r):
        rPen[ridx].lam = rc

   # Now compute REML - and all other terms needed for correction proposed by Greven & Scheipl (2017)
   S_emb,_,_,_ = compute_S_emb_pinv_det(X.shape[1],rPen,"svd")
   LP, Pr, coef, code = cpp_solve_coef(y-offset,X,S_emb)

   if code != 0:
       raise ValueError("Forming coefficients for specified penalties was not possible.")
   
   eta = (X @ coef).reshape(-1,1) + offset
   
   # Compute scale
   _,_,edf,_,_,scale = update_scale_edf(y,None,eta,None,X.shape[0],X.shape[1],LP,None,Pr,None,family,rPen,None,None,10)

   llk = family.llk(y,eta,scale)

   # Now compute REML for candidate
   reml = REML(llk,XX/scale,coef,scale,rPen)
   coef = coef.reshape(-1,1)

   # Form VB, first solve LP{^-1}
   LPinv = compute_Linv(LP,1)
   Linv = apply_eigen_perm(Pr,LPinv)

   # Now collect what we need for the remaining terms
   return Linv,coef,reml,scale,edf,llk


def correct_VB(model,nR = 250,grid_type = 'JJJ1',a=1e-7,b=1e7,df=40,n_c=10,form_t=True,form_t1=False,verbose=False,drop_NA=True,method="Chol",only_expected_edf=False,Vp_fidiff=False,use_importance_weights=True,prior=None,recompute_H=False,seed=None,**bfgs_options):
    """Estimate :math:`\\tilde{\mathbf{V}}`, the covariance matrix of the marginal posterior :math:`\\boldsymbol{\\beta} | y` to account for smoothness uncertainty. Taken from mssm.
    """
    np_gen = np.random.default_rng(seed)

    family = model.family

    if not grid_type in ["JJJ1","JJJ2","JJJ3"]:
        raise ValueError("'grid_type' has to be set to one of 'JJJ1', 'JJJ2', or 'JJJ3'.")

    if isinstance(family,GENSMOOTHFamily):
        if not bfgs_options:
            bfgs_options = {"ftol":1e-9,
                            "maxcor":30,
                            "maxls":100}

    if isinstance(family,Family):
        nPen = len(model.formula.penalties)
        rPen = copy.deepcopy(model.formula.penalties)
        S_emb,_,_,_ = compute_S_emb_pinv_det(model.hessian.shape[1],model.formula.penalties,"svd")

    else: # GAMMLSS and GSMM case
        nPen = len(model.overall_penalties)
        rPen = copy.deepcopy(model.overall_penalties)
        S_emb,_,_,_ = compute_S_emb_pinv_det(model.hessian.shape[1],model.overall_penalties,"svd")
    
    if isinstance(family,Family):
        y = model.formula.y_flat[model.formula.NOT_NA_flat]
        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _,orig_scale = model.get_pars()
    else:
        if drop_NA:
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]
        else:
            y = model.formulas[0].y_flat
        Xs = model.get_mmat(drop_NA=drop_NA)
        X = Xs[0]
        orig_scale = 1
        init_coef = copy.deepcopy(model.overall_coef)

    Vp = None
    Vpr = None
    if grid_type == "JJJ1" or grid_type == "JJJ2" or grid_type == "JJJ3":
        # Approximate Vp via finitie differencing
        if Vp_fidiff:
            Vp, Vpr, Vr, Vrr, _ = estimateVp(model,n_c=n_c,grid_type="JJJ1",Vp_fidiff=True)
        else:
            # Take PQL approximation instead
            Vp, Vpr, Vr, Vrr, _, dBetadRhos = compute_Vp_WPS(model.lvi if isinstance(family,Family) else model.overall_lvi,
                                                 model.hessian,
                                                 S_emb,
                                                 model.formula.penalties if isinstance(family,Family) else model.overall_penalties,
                                                 model.coef.reshape(-1,1) if isinstance(family,Family) else model.overall_coef.reshape(-1,1),
                                                 scale=orig_scale if isinstance(family,Family) else 1)

        # Compute approximate WPS (2016) correction
        if grid_type == "JJJ1" or (grid_type == "JJJ3" and only_expected_edf == False):
            if isinstance(family,Family):
                Vc,Vcc = compute_Vb_corr_WPS(model.lvi,Vpr,Vr,model.hessian,S_emb,model.formula.penalties,model.coef.reshape(-1,1),scale=orig_scale)
            else:
                Vc,Vcc = compute_Vb_corr_WPS(model.overall_lvi,Vpr,Vr,model.hessian,S_emb,model.overall_penalties,model.overall_coef.reshape(-1,1))
                
            if isinstance(family,Family):
                V = Vc + Vcc + ((model.lvi.T@model.lvi)*orig_scale)
            else:
                V = Vc + Vcc + model.overall_lvi.T@model.overall_lvi
            
            if grid_type == "JJJ3":
                # Can enforce lower bound of JJJ1 here
                Vlb = copy.deepcopy(V)

    rGrid = np.array([])
    remls = []
    Vs = []
    coefs = []
    edfs = []
    llks = []
    Linvs = []
    scales = []

    if grid_type != "JJJ1":
        # Generate \lambda values from Vpr for which to compute REML, and Vb
        if isinstance(family,Family):
            ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.formula.penalties]).reshape(-1,1))
        else:
            ep = np.log(np.array([min(b,max(a,pen.lam)) for pen in model.overall_penalties]).reshape(-1,1))
        #print(ep)

        n_est = nR
        
        if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1: # Parallelize grid search
            # Generate next \lambda values for which to compute REML, and Vb
            p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,size=n_est,random_state=seed)
            p_sample = np.exp(p_sample)

            if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
            
                if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                
            elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1,-1)

            minDiag = 0.1*min(np.sqrt(Vpr.diagonal()))
            # Make sure actual estimate is included once.
            if np.any(np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])),axis=1) < minDiag) == False:
                p_sample = np.concatenate((np.array([pen2.lam for pen2 in rPen]).reshape(1,-1),p_sample),axis=0)
            
            if isinstance(family,Gaussian) and isinstance(family.link,Identity) and method == "Chol": # Fast Strictly additive case
                with managers.SharedMemoryManager() as manager, mp.Pool(processes=n_c) as pool:
                    # Create shared memory copies of data, indptr, and indices for X, XX, and y

                    # X
                    rows, cols, _, data, indptr, indices = map_csc_to_eigen(X)
                    shape_dat = data.shape
                    shape_ptr = indptr.shape
                    shape_y = y.shape

                    dat_mem = manager.SharedMemory(data.nbytes)
                    dat_shared = np.ndarray(shape_dat, dtype=np.double, buffer=dat_mem.buf)
                    dat_shared[:] = data[:]

                    ptr_mem = manager.SharedMemory(indptr.nbytes)
                    ptr_shared = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_mem.buf)
                    ptr_shared[:] = indptr[:]

                    idx_mem = manager.SharedMemory(indices.nbytes)
                    idx_shared = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_mem.buf)
                    idx_shared[:] = indices[:]

                    #XX
                    _, _, _, dataXX, indptrXX, indicesXX = map_csc_to_eigen((X.T@X).tocsc())
                    shape_datXX = dataXX.shape
                    shape_ptrXX = indptrXX.shape

                    dat_memXX = manager.SharedMemory(dataXX.nbytes)
                    dat_sharedXX = np.ndarray(shape_datXX, dtype=np.double, buffer=dat_memXX.buf)
                    dat_sharedXX[:] = dataXX[:]

                    ptr_memXX = manager.SharedMemory(indptrXX.nbytes)
                    ptr_sharedXX = np.ndarray(shape_ptrXX, dtype=np.int64, buffer=ptr_memXX.buf)
                    ptr_sharedXX[:] = indptrXX[:]

                    idx_memXX = manager.SharedMemory(indicesXX.nbytes)
                    idx_sharedXX = np.ndarray(shape_datXX, dtype=np.int64, buffer=idx_memXX.buf)
                    idx_sharedXX[:] = indicesXX[:]

                    # y
                    y_mem = manager.SharedMemory(y.nbytes)
                    y_shared = np.ndarray(shape_y, dtype=np.double, buffer=y_mem.buf)
                    y_shared[:] = y[:]

                    # Now compute reml for new candidates in parallel
                    args = zip(repeat(family),repeat(y_mem.name),repeat(dat_mem.name),
                            repeat(ptr_mem.name),repeat(idx_mem.name),repeat(dat_memXX.name),
                            repeat(ptr_memXX.name),repeat(idx_memXX.name),repeat(shape_y),
                            repeat(shape_dat),repeat(shape_ptr),repeat(shape_datXX),repeat(shape_ptrXX),
                            repeat(rows),repeat(cols),repeat(rPen),repeat(model.offset),p_sample)
                    
                    sample_Linvs, sample_coefs, sample_remls, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(_compute_VB_corr_terms_MP,args))
                    
                    if only_expected_edf == False:
                        Linvs.extend(list(sample_Linvs))
                    scales.extend(list(sample_scales))
                    coefs.extend(list(sample_coefs))
                    remls.extend(list(sample_remls))
                    edfs.extend(list(sample_edfs))
                    llks.extend(list(sample_llks))
                    rGrid = p_sample
            else: # all other models

                rPens = []
                for ps in p_sample:
                    rcPen = copy.deepcopy(rPen)
                    for ridx,rc in enumerate(ps):
                        rcPen[ridx].lam = rc
                    rPens.append(rcPen)

                if isinstance(family,Family):
                    origNH = None
                    if only_expected_edf:
                        if isinstance(family,Gaussian) and isinstance(family.link,Identity):
                            origNH = orig_scale
                        else:
                            origNH = -1*model.hessian
                    args = zip(repeat(family),repeat(y),repeat(X),rPens,
                               repeat(1),repeat(model.offset),repeat(None),
                               repeat(method),repeat(True),repeat(origNH))
                    with mp.Pool(processes=n_c) as pool:
                        sample_remls, sample_Linvs, _, _,sample_coefs, sample_scales, sample_edfs, sample_llks = zip(*pool.starmap(compute_reml_candidate_GAMM,args))
                else:
                    origNH = None
                    if only_expected_edf:
                        origNH = -1*model.hessian
                    args = zip(repeat(family),repeat(y),repeat(Xs),rPens,
                               repeat(init_coef),repeat(len(init_coef)),
                               repeat(model.coef_split_idx),repeat(method),
                               repeat(1e-7),repeat(1),repeat(bfgs_options),
                               repeat(origNH))
                    with mp.Pool(processes=n_c) as pool:
                        sample_remls, _, sample_Linvs, sample_coefs, sample_edfs, sample_llks = zip(*pool.starmap(compute_REML_candidate_GSMM,args))
                    sample_scales = np.ones(p_sample.shape[0])

                if only_expected_edf == False:
                    Linvs.extend(list(sample_Linvs))
                scales.extend(list(sample_scales))
                coefs.extend(list(sample_coefs))
                remls.extend(list(sample_remls))
                edfs.extend(list(sample_edfs))
                llks.extend(list(sample_llks))
                rGrid = p_sample
                rPens = None
            
        else:
            # Generate \lambda values for which to compute REML, and Vb
            p_sample = scp.stats.multivariate_t.rvs(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,size=n_est,random_state=seed)
            p_sample = np.exp(p_sample)

            if len(np.ndarray.flatten(ep)) == 1: # Single lambda parameter in model
                
                if n_est == 1: # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(n_est,1) # p_sample needs to be shape (n_est,1)
                
            elif n_est == 1: # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1,-1)
            
            # Make sure actual estimate is included once.
            minDiag = 0.1*min(np.sqrt(Vpr.diagonal()))
            if np.any(np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])),axis=1) < minDiag) == False:
                p_sample = np.concatenate((np.array([pen2.lam for pen2 in rPen]).reshape(1,-1),p_sample),axis=0)

            for ps in p_sample:
                for ridx,rc in enumerate(ps):
                    rPen[ridx].lam = rc
                
                if isinstance(family,Family):
                    reml,Linv,LP,Pr,coef,scale,edf,llk = compute_reml_candidate_GAMM(family,y,X,rPen,n_c,model.offset,method=method,compute_inv=True)
                    #coef = coef.reshape(-1,1)

                    # Form VB, first solve LP{^-1}
                    #LPinv = compute_Linv(LP,n_c)
                    #Linv = apply_eigen_perm(Pr,LPinv)

                    # Collect conditional posterior covariance matrix for this set of coef
                    Vb = Linv.T@Linv*scale
                    #Vb += coef@coef.T
                else:
                    try:
                        reml,V,_,coef,edf,llk = compute_REML_candidate_GSMM(family,y,Xs,rPen,init_coef,len(init_coef),model.coef_split_idx,n_c=n_c,method=method,bfgs_options=bfgs_options)
                    except:
                        warnings.warn(f"Unable to compute REML score for sample {np.exp(ps)}. Skipping.")
                        continue

                    #coef = coef.reshape(-1,1)

                    # Collect conditional posterior covariance matrix for this set of coef
                    Vb = V #+ coef@coef.T

                # Collect all necessary objects for G&S correction.
                coefs.append(coef)
                remls.append(reml)
                edfs.append(edf)
                llks.append(llk)
                if only_expected_edf == False:
                    Vs.append(Vb)
                
                if len(rGrid) == 0:
                    rGrid = ps.reshape(1,-1)
                else:
                    rGrid = np.concatenate((rGrid,ps.reshape(1,-1)),axis=0)
    
    ###################################################### Prepare computation of tau ###################################################### 

    if grid_type != "JJJ1":
        # Compute weights proposed by Greven & Scheipl (2017) - still work under importance sampling case instead of grid case if we assume Vp
        # is prior for \rho|\mathbf{y}.
        if use_importance_weights and prior is None:
            ws = scp.special.softmax(remls)

        elif use_importance_weights and prior is not None: # Standard importance weights (e.g., Branchini & Elvira, 2024)
            logp = prior.logpdf(np.log(rGrid))
            q = scp.stats.multivariate_t(loc=np.ndarray.flatten(ep),shape=Vpr,df=df,allow_singular=True)
            logq = q.logpdf(np.log(rGrid))

            ws = scp.special.softmax(remls + logp - logq)

        else:
            # Normal weights result in cancellation, i.e., just average
            ws = scp.special.softmax(np.ones_like(remls))

        Vcr = Vr @ dBetadRhos.T

        # Optionally re-compute negative Hessian at posterior mean for coef.
        if recompute_H and (only_expected_edf == False):

            # Estimate mean of posterior beta|y
            mean_coef = ws[0]*coefs[0]
            for ri in range(1,len(rGrid)):
                mean_coef += ws[ri]*coefs[ri]
            
            # Recompute Hessian at mean
            if isinstance(family,Family):
                yb = y
                Xb = X

                S_emb,_,S_root,_ = compute_S_emb_pinv_det(X.shape[1],model.formula.penalties,"svd",method != 'Chol')

                if isinstance(family,Gaussian) and isinstance(family.link,Identity): # strictly additive case
                    nH = (-1*model.hessian)*orig_scale

                else: # Generalized case
                    eta = (X @ mean_coef).reshape(-1,1) + model.offset
                    mu = family.link.fi(eta)
                    
                    # Compute pseudo-dat and weights for mean coef
                    yb,Xb,z,Wr = update_PIRLS(y,yb,mu,eta-model.offset,X,Xb,family)

                    inval_check =  np.any(np.isnan(z))

                    if inval_check:
                        _, w, inval = PIRLS_pdat_weights(y,mu,eta-model.offset,family)
                        w[inval] = 0

                        # Re-compute weight matrix
                        Wr_fix = scp.sparse.spdiags([np.sqrt(np.ndarray.flatten(w))],[0])
                    else:
                        Wr_fix = Wr
                        
                    W = Wr_fix@Wr_fix
                    nH = (X.T@W@X).tocsc() 

                # Solve for coef to get Cholesky needed to re-compute scale
                _,_,_,Pr,_,LP,keep,drop = update_coef(yb,X,Xb,family,S_emb,S_root,n_c,None,model.offset)

                # Re-compute scale
                _,_,_,_,_,scale = update_scale_edf(y,z,eta,Wr,X.shape[0],X.shape[1],LP,None,Pr,None,family,model.formula.penalties,keep,drop,n_c)
                
                # And negative hessian
                nH /= scale

                if drop is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        nH[:,drop] = 0
                        nH[drop,:] = 0

            else: # GSMM/GAMLSS case
                if isinstance(Family,GAMLSSFamily): #GAMLSS case
                    split_coef = np.split(mean_coef,model.coef_split_idx)

                    # Update etas and mus
                    etas = [Xs[i]@split_coef[i] for i in range(family.n_par)]
                    mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

                    # Get derivatives with respect to eta
                    if family.d_eta == False:
                        d1eta,d2eta,d2meta = deriv_transform_mu_eta(y,mus,family)
                    else:
                        d1eta = [fd1(y,*mus) for fd1 in family.d1]
                        d2eta = [fd2(y,*mus) for fd2 in family.d2]
                        d2meta = [fd2m(y,*mus) for fd2m in family.d2m]

                    # Get derivatives with respect to coef
                    _,H = deriv_transform_eta_beta(d1eta,d2eta,d2meta,Xs,only_grad=False)

                else: # GSMM
                    H = family.hessian(mean_coef,model.coef_split_idx,y,Xs)

                nH = -1 * H

            if verbose:
                print(f"Recomputed negative Hessian. 2 Norm of coef. difference: {np.linalg.norm(mean_coef-model.coef.reshape(-1,1))}. F. Norm of n. Hessian difference: {scp.sparse.linalg.norm(nH + model.hessian)}")

        else:
            mean_coef = None
            nH = -1*model.hessian

        ###################################################### Compute tau ###################################################### 

        if only_expected_edf:
            # Compute correction of edf directly..
            expected_edf = max(model.edf,np.sum(ws*edfs))

            if grid_type == 'JJJ2':
                # Can now add remaining correction term
                
                # Now have Vc = Vcr.T @ Vcr, so:
                # tr(Vc@(-1*model.hessian)) = tr(Vcr.T @ Vcr@(-1*model.hessian)) = tr(Vcr@ (-1*model.hessian)@Vcr.T)
                #expected_edf += (Vc@(-1*model.hessian)).trace()
                expected_edf += (Vcr@ (nH)@Vcr.T).trace()

            elif grid_type == 'JJJ3':
                # Correct based on G&S expectations instead
                tr1 = ws[0]* (coefs[0].T@(nH)@coefs[0])[0,0]

                # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
                tr2 = ws[0]*coefs[0] 
                
                # Now sum over remaining r
                for ri in range(1,len(rGrid)):
                    tr1 += ws[ri]* (coefs[ri].T@(nH)@coefs[ri])[0,0]
                    tr2 += ws[ri]*coefs[ri]
                
                # Enforce lower bound of JJJ2
                if (Vcr@ (nH)@Vcr.T).trace() > (tr1 - (tr2.T@(nH)@tr2)[0,0]):
                    expected_edf += (Vcr@ (nH)@Vcr.T).trace()
                else:
                    expected_edf += tr1 - (tr2.T@(nH)@tr2)[0,0]
                    
            if verbose:
                print(f"Correction was based on {rGrid.shape[0]} samples in total.")

            return None,None,None,None,None,None,None,None,expected_edf,mean_coef
        else:
            expected_edf = None
        
        ###################################################### Compute tau and full covariance matrix ###################################################### 

        if grid_type == 'JJJ2':
            
            if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
                Vr1 = ws[0]* (Linvs[0].T@Linvs[0]*scales[0])

                # Now sum over remaining r
                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*((Linvs[ri].T@Linvs[ri]*scales[ri]))

            else:
                Vr1 = ws[0]*Vs[0]

                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*Vs[ri]
            
            if (Vr1@(nH)).trace() < model.edf:
                nH = -1 * model.hessian # Reset nH
                
                if isinstance(family,Family):
                    Vr1 = model.lvi.T@model.lvi*orig_scale
                else:
                    Vr1 = model.overall_lvi.T@model.overall_lvi

            V = Vr1 + Vcr.T @ Vcr

        elif grid_type == 'JJJ3':

            # Now compute \hat{cov(\boldsymbol{\beta}|y)}
            if X.shape[0] < 1e5 and X.shape[1] < 2000 and n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
                Vr1 = ws[0]* (Linvs[0].T@Linvs[0]*scales[0]) 
                Vr2 = ws[0]* (coefs[0]@coefs[0].T)
                # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
                Vr3 = ws[0]*coefs[0] 
                
                # Now sum over remaining r
                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*(Linvs[ri].T@Linvs[ri]*scales[ri])
                    Vr2 += ws[ri]* (coefs[ri]@coefs[ri].T)
                    Vr3 += ws[ri]*coefs[ri]

            else:
                Vr1 = ws[0]*Vs[0]
                Vr2 = ws[0]*(coefs[0]@coefs[0].T)
                Vr3 = ws[0]*coefs[0] 

                for ri in range(1,len(rGrid)):
                    Vr1 += ws[ri]*Vs[ri]
                    Vr2 += ws[ri]*(coefs[ri]@coefs[ri].T)
                    Vr3 += ws[ri]*coefs[ri]
            
            #if (Vr1@(nH)).trace() < model.edf:
            #    nH = -1 * model.hessian # Reset nH
            #
            #    if isinstance(family,Family):
            #        Vr1 = model.lvi.T@model.lvi*orig_scale
            #    else:
            #        Vr1 = model.overall_lvi.T@model.overall_lvi

            # Now, Greven & Scheipl provide final estimate =
            # E_{p|y}[V_\boldsymbol{\beta}(\lambda)] + E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] - E_{p|y}[\boldsymbol{\beta}] E_{p|y}[\boldsymbol{\beta}]^T
            # Enforce lower bound of JJJ2 again
            #if (Vcr @ (nH) @ Vcr.T).trace() > ((Vr2 - (Vr3@Vr3.T)) @ (nH)).trace():
            #    V = Vr1 + Vcr.T @ Vcr
            #else:
            V = Vr1 + Vr2 - (Vr3@Vr3.T)

            if (V@(-1 * model.hessian)).trace() > (V@nH).trace():
                nH = -1 * model.hessian # Reset nH
            
            # Enforce lower bound of JJJ1
            #if (Vlb@nH).trace() > (V@nH).trace():
            #    V = Vlb

            # Some lower bounds for paper simulation:
            if isinstance(family,Family):
                Vr1lb = model.lvi.T@model.lvi*orig_scale
            else:
                Vr1lb = model.overall_lvi.T@model.overall_lvi

            if (Vr1@(nH)).trace() < (Vr1lb@(nH)).trace():
                Vr1 = Vr1lb
                V = Vr1 + Vr2 - (Vr3@Vr3.T)
            
            if (Vcr @ (nH) @ Vcr.T).trace() > ((Vr2 - (Vr3@Vr3.T)) @ (nH)).trace():
                V = Vr1 + Vcr.T @ Vcr
            
    else:
        mean_coef = None
        expected_edf = None
        nH = -1 * model.hessian

    # Check V is full rank - can use LV for sampling as well..
    LV,code = cpp_chol(scp.sparse.csc_array(V))
    if code != 0:
        raise ValueError("Failed to estimate marginal covariance matrix for ecoefficients.")

    # Compute corrected edf (e.g., for AIC; Wood, Pya, & Saefken, 2016)
    total_edf = None
    if form_t or form_t1:
        F = V@(nH)

        edf = F.diagonal()
        total_edf = F.trace()

    # In mgcv, an upper limit is enforced on edf and total_edf when they are uncertainty corrected - based on t1 in section 6.1.2 of Wood (2017)
    # so the same is done here.
    total_edf2 = None
    if grid_type == "JJJ1" or grid_type == "JJJ2" or grid_type == "JJJ3":
        if isinstance(family,Family):
            if isinstance(family,Gaussian) and isinstance(family.link,Identity): # Strictly additive case
                ucF = (model.lvi.T@model.lvi)@((X.T@X))
            else: # Generalized case
                W = model.Wr@model.Wr
                ucF = (model.lvi.T@model.lvi)@((X.T@W@X))
        else: # GSMM/GAMLSS case
            ucF = (model.overall_lvi.T@model.overall_lvi)@(-1*model.hessian)

        total_edf2 = 2*model.edf - (ucF@ucF).trace()
        if total_edf > total_edf2:
            #print(edf)
            total_edf = total_edf2
            #edf = None

    # Compute uncertainty corrected smoothness bias corrected edf (t1 in section 6.1.2 of Wood, 2017)
    edf2 = None
    if form_t1:
        edf2 = 2*edf - (F@F).diagonal()
        if total_edf2 is None: # Otherwise respect upper bound.
            total_edf2 = 2*total_edf - (F@F).trace()

    if verbose and grid_type != "JJJ1":
        print(f"Correction was based on {rGrid.shape[0]} samples in total.")    

    return V,LV,Vp,Vpr,edf,total_edf,edf2,total_edf2,expected_edf,mean_coef