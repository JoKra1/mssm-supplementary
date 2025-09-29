from mssm.models import *
from mssmViz.sim import *
from mssmViz.plot import *
from mssm.src.python.compare import compare_CDL
from mssm.src.python.gamm_solvers import (
    deriv_transform_mu_eta,
    deriv_transform_eta_beta,
)
from mssm.src.python.utils import *
import copy
import numpy as np
import scipy as scp


def init_lambda(penalties):
    return [1.1 for _ in range(len(penalties))]


############## Exported/Modified MSSM functions ############## # noqa


def _compute_VB_corr_terms_MP(
    family: Gaussian,
    address_y: str,
    address_dat: str,
    address_ptr: str,
    address_idx: str,
    address_datXX: str,
    address_ptrXX: str,
    address_idxXX: str,
    shape_y: tuple,
    shape_dat: tuple,
    shape_ptr: tuple,
    shape_datXX: tuple,
    shape_ptrXX: tuple,
    rows: int,
    cols: int,
    rPen: list[LambdaTerm],
    offset: float | np.ndarray,
    r: np.ndarray,
) -> tuple[scp.sparse.csc_array, np.ndarray, float, float, float, float]:
    """Multi-processing code for Grevel & Scheipl correction for Gaussian additive
    model - see ``correct_VB`` for details.

    :param family: Family of model
    :type family: Gaussian
    :param address_y: Memory address for y vector
    :type address_y: str
    :param address_dat: Memory address for X data
    :type address_dat: str
    :param address_ptr: Memory address for X pointers
    :type address_ptr: str
    :param address_idx: Memory address for X indices
    :type address_idx: str
    :param address_datXX: Memory address for X.T@X data
    :type address_datXX: str
    :param address_ptrXX: Memory address for X.T@X pointers
    :type address_ptrXX: str
    :param address_idxXX: Memory address for X.T@X indices
    :type address_idxXX: str
    :param shape_y: shape of y data array
    :type shape_y: tuple
    :param shape_dat: shape of X data array
    :type shape_dat: tuple
    :param shape_ptr: shape of X pointers array
    :type shape_ptr: tuple
    :param shape_datXX: shape of X.T@X data array
    :type shape_datXX: tuple
    :param shape_ptrXX: shape of X data array
    :type shape_ptrXX: tuple
    :param rows: Rows of X
    :type rows: int
    :param cols: Columns of X
    :type cols: int
    :param rPen: List of penalties
    :type rPen: list[LambdaTerm]
    :param offset: Any fixed offset to add to the linear predictor/mean
    :type offset: float | np.ndarray
    :param r: List of log(lambda) values for which to evaluate the reml score.
    :type r: np.ndarray
    :return: Un-pivoted inverse of pivoted Cholesky of negative penalized hessian, coefficients,
        reml score, scale etimate, total edf, llk
    :rtype: tuple[scp.sparse.csc_array,np.ndarray,float,float,float,float]
    """
    dat_shared = shared_memory.SharedMemory(name=address_dat, create=False)
    ptr_shared = shared_memory.SharedMemory(name=address_ptr, create=False)
    idx_shared = shared_memory.SharedMemory(name=address_idx, create=False)
    dat_sharedXX = shared_memory.SharedMemory(name=address_datXX, create=False)
    ptr_sharedXX = shared_memory.SharedMemory(name=address_ptrXX, create=False)
    idx_sharedXX = shared_memory.SharedMemory(name=address_idxXX, create=False)
    y_shared = shared_memory.SharedMemory(name=address_y, create=False)

    data = np.ndarray(shape_dat, dtype=np.double, buffer=dat_shared.buf)
    indptr = np.ndarray(shape_ptr, dtype=np.int64, buffer=ptr_shared.buf)
    indices = np.ndarray(shape_dat, dtype=np.int64, buffer=idx_shared.buf)
    dataXX = np.ndarray(shape_datXX, dtype=np.double, buffer=dat_sharedXX.buf)
    indptrXX = np.ndarray(shape_ptrXX, dtype=np.int64, buffer=ptr_sharedXX.buf)
    indicesXX = np.ndarray(shape_datXX, dtype=np.int64, buffer=idx_sharedXX.buf)
    y = np.ndarray(shape_y, dtype=np.double, buffer=y_shared.buf)

    X = scp.sparse.csc_array((data, indices, indptr), shape=(rows, cols), copy=False)
    XX = scp.sparse.csc_array(
        (dataXX, indicesXX, indptrXX), shape=(cols, cols), copy=False
    )

    # Prepare penalties with current lambda candidate r
    for ridx, rc in enumerate(r):
        rPen[ridx].lam = rc

    # Now compute REML - and all other terms needed for correction proposed
    # by Greven & Scheipl (2017)
    S_emb, _, _, _ = compute_S_emb_pinv_det(X.shape[1], rPen, "svd")
    LP, Pr, coef, code = cpp_solve_coef(y - offset, X, S_emb)

    if code != 0:
        raise ValueError(
            "Forming coefficients for specified penalties was not possible."
        )

    eta = (X @ coef).reshape(-1, 1) + offset

    # Compute scale
    _, _, edf, _, _, scale = update_scale_edf(
        y,
        None,
        eta,
        None,
        X.shape[0],
        X.shape[1],
        LP,
        None,
        Pr,
        None,
        None,
        family,
        rPen,
        None,
        None,
        1,
    )

    llk = family.llk(y, eta, scale)

    # Now compute REML for candidate
    reml = REML(llk, XX / scale, coef, scale, rPen)
    coef = coef.reshape(-1, 1)

    # Form VB, first solve LP{^-1}
    LPinv = compute_Linv(LP, 1)
    Linv = apply_eigen_perm(Pr, LPinv)

    # Now collect what we need for the remaining terms
    return Linv, coef, reml, scale, edf, llk


def correct_VB(
    model,
    nR: int = 250,
    grid_type: str = "JJJ1",
    a: float = 1e-7,
    b: float = 1e7,
    df: int = 40,
    n_c: int = 10,
    form_t1: bool = False,
    verbose: bool = False,
    drop_NA: bool = True,
    method: str = "Chol",
    only_expected_edf: bool = False,
    Vp_fidiff: bool = False,
    use_importance_weights: bool = True,
    prior: Callable | None = None,
    recompute_H: bool = False,
    seed: int | None = None,
    compute_Vcc: bool = True,
    VP_grid_type: str = "JJJ1",
    **bfgs_options,
) -> tuple[
    scp.sparse.csc_array | None,
    scp.sparse.csc_array | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    float | None,
    np.ndarray | None,
    float | None,
    float,
    np.ndarray,
]:
    """Estimate :math:`\\tilde{\\mathbf{V}}`, the covariance matrix of the marginal posterior
    :math:`\\boldsymbol{\\beta} | y` to account for smoothness uncertainty. Taken from mssm.
    """
    # np_gen = np.random.default_rng(seed)

    family = model.family

    if grid_type not in ["JJJ1", "JJJ2", "JJJ3"]:
        raise ValueError(
            "'grid_type' has to be set to one of 'JJJ1', 'JJJ2', or 'JJJ3'."
        )

    if isinstance(family, Family) and model.rho is not None and grid_type != "JJJ1":
        raise ValueError(
            "For models with an ar1 model only grid_type='JJJ1' is supported."
        )

    if compute_Vcc is False and Vp_fidiff:
        warnings.warn(
            (
                "Ignoring request to rely on finite differencing to obtain covariance matrix of "
                "$\\rho$, because `compute_Vcc` was set to False."
            )
        )
        Vp_fidiff = False

    if isinstance(family, GSMMFamily):
        if not bfgs_options:
            bfgs_options = {"ftol": 1e-9, "maxcor": 30, "maxls": 100}

    # nPen = len(model.overall_penalties)
    rPen = copy.deepcopy(model.overall_penalties)
    S_emb, _, _, _ = compute_S_emb_pinv_det(
        model.hessian.shape[1], model.overall_penalties, "svd"
    )

    if isinstance(family, Family):
        y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

        if not model.formulas[0].get_lhs().f is None:
            # Optionally apply function to dep. var. before fitting.
            y = model.formulas[0].get_lhs().f(y)

        X = model.get_mmat()

        orig_scale = family.scale
        if family.twopar:
            _, orig_scale = model.get_pars()
    else:
        if isinstance(family, GAMLSSFamily):
            y = model.formulas[0].y_flat[model.formulas[0].NOT_NA_flat]

            if not model.formulas[0].get_lhs().f is None:
                # Optionally apply function to dep. var. before fitting. Not sure why that would be
                # desirable for this model class...
                y = model.formulas[0].get_lhs().f(y)

            Xs = model.get_mmat()

        else:  # Need all y vectors in y, i.e., y is actually ys
            ys = []
            for fi, form in enumerate(model.formulas):

                # Repeated y-variable - don't have to pass all of them
                if (
                    fi > 0
                    and form.get_lhs().variable == model.formulas[0].get_lhs().variable
                ):
                    ys.append(None)
                    continue

                # New y-variable
                if drop_NA:
                    y = form.y_flat[form.NOT_NA_flat]
                else:
                    y = form.y_flat

                # Optionally apply function to dep. var. before fitting. Not sure why that would be
                # desirable for this model class...
                if not form.get_lhs().f is None:
                    y = form.get_lhs().f(y)

                # And collect
                ys.append(y)

            y = ys
            Xs = model.get_mmat(drop_NA=drop_NA)

        keep_drop = None
        if model.info.dropped is not None:
            keep = np.array(
                [
                    cidx
                    for cidx in range(model.hessian.shape[1])
                    if cidx not in model.info.dropped
                ]
            )
            keep_drop = (keep, model.info.dropped)

        X = Xs[0]
        orig_scale = 1
        init_coef = copy.deepcopy(model.coef)

    Vp = None
    Vpr = None
    if grid_type == "JJJ1" or grid_type == "JJJ2" or grid_type == "JJJ3":
        # Approximate Vp via finitie differencing
        if Vp_fidiff:
            Vp, Vpr, Vr, Vrr, _, dBetadRhos = estimateVp(
                model, n_c=n_c, grid_type="JJJ1", Vp_fidiff=True
            )
        elif VP_grid_type != "JJJ1":
            Vp, Vpr, Vr, Vrr, _, dBetadRhos = estimateVp(
                model,
                n_c=n_c,
                grid_type=VP_grid_type,
                Vp_fidiff=Vp_fidiff,
                use_importance_weights=use_importance_weights,
                prior=prior,
                method=method,
                seed=seed,
            )
        else:
            # Take PQL approximation instead
            Vp, Vpr, Vr, Vrr, _, dBetadRhos = compute_Vp_WPS(
                model.lvi,
                model.hessian,
                S_emb,
                model.overall_penalties,
                model.coef,
                scale=orig_scale if isinstance(family, Family) else 1,
            )

        # Compute approximate WPS (2016) correction
        if grid_type == "JJJ1" or (grid_type == "JJJ3" and only_expected_edf is False):

            if compute_Vcc:
                if isinstance(family, Family):
                    Vc, Vcc = compute_Vb_corr_WPS(
                        model.lvi,
                        Vpr,
                        Vr,
                        model.hessian,
                        S_emb,
                        model.overall_penalties,
                        model.coef,
                        scale=orig_scale,
                        drop=model.info.dropped,
                    )
                else:
                    Vc, Vcc = compute_Vb_corr_WPS(
                        model.lvi,
                        Vpr,
                        Vr,
                        model.hessian,
                        S_emb,
                        model.overall_penalties,
                        model.coef,
                        drop=model.info.dropped,
                    )
            else:
                # Only compute first correction term
                Vc = Vr @ dBetadRhos.T
                Vc = Vc.T @ Vc
                Vcc = 0

            if isinstance(family, Family):
                V = Vc + Vcc + ((model.lvi.T @ model.lvi) * orig_scale)
            else:
                V = Vc + Vcc + model.lvi.T @ model.lvi

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
        # Generate \\lambda values from Vpr for which to compute REML, and Vb
        ep = np.log(
            np.array(
                [min(b, max(a, pen.lam)) for pen in model.overall_penalties]
            ).reshape(-1, 1)
        )
        # print(ep)

        n_est = nR

        if n_c > 1:  # Parallelize grid search
            # Generate next \\lambda values for which to compute REML, and Vb
            p_sample = scp.stats.multivariate_t.rvs(
                loc=np.ndarray.flatten(ep),
                shape=Vpr,
                df=df,
                size=n_est,
                random_state=seed,
            )
            p_sample = np.exp(p_sample)
            p_sample[np.isinf(p_sample)] = b

            if len(np.ndarray.flatten(ep)) == 1:  # Single lambda parameter in model

                if (
                    n_est == 1
                ):  # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(
                    n_est, 1
                )  # p_sample needs to be shape (n_est,1)

            elif (
                n_est == 1
            ):  # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1, -1)

            minDiag = 0.1 * min(np.sqrt(Vpr.diagonal()))
            # Make sure actual estimate is included once.
            if not np.any(
                np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])), axis=1)
                < minDiag
            ):
                p_sample = np.concatenate(
                    (np.array([pen2.lam for pen2 in rPen]).reshape(1, -1), p_sample),
                    axis=0,
                )

            if (
                isinstance(family, Gaussian)
                and isinstance(family.link, Identity)
                and method == "Chol"
            ):  # Fast Strictly additive case
                with (
                    managers.SharedMemoryManager() as manager,
                    mp.Pool(processes=n_c) as pool,
                ):
                    # Create shared memory copies of data, indptr, and indices for X, XX, and y

                    # X
                    rows, cols, _, data, indptr, indices = map_csc_to_eigen(X)
                    shape_dat = data.shape
                    shape_ptr = indptr.shape
                    shape_y = y.shape

                    dat_mem = manager.SharedMemory(data.nbytes)
                    dat_shared = np.ndarray(
                        shape_dat, dtype=np.double, buffer=dat_mem.buf
                    )
                    dat_shared[:] = data[:]

                    ptr_mem = manager.SharedMemory(indptr.nbytes)
                    ptr_shared = np.ndarray(
                        shape_ptr, dtype=np.int64, buffer=ptr_mem.buf
                    )
                    ptr_shared[:] = indptr[:]

                    idx_mem = manager.SharedMemory(indices.nbytes)
                    idx_shared = np.ndarray(
                        shape_dat, dtype=np.int64, buffer=idx_mem.buf
                    )
                    idx_shared[:] = indices[:]

                    # XX
                    _, _, _, dataXX, indptrXX, indicesXX = map_csc_to_eigen(
                        (X.T @ X).tocsc()
                    )
                    shape_datXX = dataXX.shape
                    shape_ptrXX = indptrXX.shape

                    dat_memXX = manager.SharedMemory(dataXX.nbytes)
                    dat_sharedXX = np.ndarray(
                        shape_datXX, dtype=np.double, buffer=dat_memXX.buf
                    )
                    dat_sharedXX[:] = dataXX[:]

                    ptr_memXX = manager.SharedMemory(indptrXX.nbytes)
                    ptr_sharedXX = np.ndarray(
                        shape_ptrXX, dtype=np.int64, buffer=ptr_memXX.buf
                    )
                    ptr_sharedXX[:] = indptrXX[:]

                    idx_memXX = manager.SharedMemory(indicesXX.nbytes)
                    idx_sharedXX = np.ndarray(
                        shape_datXX, dtype=np.int64, buffer=idx_memXX.buf
                    )
                    idx_sharedXX[:] = indicesXX[:]

                    # y
                    y_mem = manager.SharedMemory(y.nbytes)
                    y_shared = np.ndarray(shape_y, dtype=np.double, buffer=y_mem.buf)
                    y_shared[:] = y[:]

                    # Now compute reml for new candidates in parallel
                    args = zip(
                        repeat(family),
                        repeat(y_mem.name),
                        repeat(dat_mem.name),
                        repeat(ptr_mem.name),
                        repeat(idx_mem.name),
                        repeat(dat_memXX.name),
                        repeat(ptr_memXX.name),
                        repeat(idx_memXX.name),
                        repeat(shape_y),
                        repeat(shape_dat),
                        repeat(shape_ptr),
                        repeat(shape_datXX),
                        repeat(shape_ptrXX),
                        repeat(rows),
                        repeat(cols),
                        repeat(rPen),
                        repeat(model.offset),
                        p_sample,
                    )

                    (
                        sample_Linvs,
                        sample_coefs,
                        sample_remls,
                        sample_scales,
                        sample_edfs,
                        sample_llks,
                    ) = zip(*pool.starmap(_compute_VB_corr_terms_MP, args))

                    if only_expected_edf is False:
                        Linvs.extend(list(sample_Linvs))
                    scales.extend(list(sample_scales))
                    coefs.extend(list(sample_coefs))
                    remls.extend(list(sample_remls))
                    edfs.extend(list(sample_edfs))
                    llks.extend(list(sample_llks))
                    rGrid = p_sample
            else:  # all other models

                rPens = []
                for ps in p_sample:
                    rcPen = copy.deepcopy(rPen)
                    for ridx, rc in enumerate(ps):
                        rcPen[ridx].lam = rc
                    rPens.append(rcPen)

                if isinstance(family, Family):
                    origNH = None
                    if only_expected_edf:
                        if isinstance(family, Gaussian) and isinstance(
                            family.link, Identity
                        ):
                            origNH = orig_scale
                        else:
                            origNH = -1 * model.hessian
                    args = zip(
                        repeat(family),
                        repeat(y),
                        repeat(X),
                        rPens,
                        repeat(1),
                        repeat(model.offset),
                        repeat(None),
                        repeat(method),
                        repeat(True),
                        repeat(origNH),
                    )
                    with mp.Pool(processes=n_c) as pool:
                        (
                            sample_remls,
                            sample_Linvs,
                            _,
                            _,
                            sample_coefs,
                            sample_scales,
                            sample_edfs,
                            sample_llks,
                        ) = zip(*pool.starmap(compute_reml_candidate_GAMM, args))
                else:
                    origNH = None
                    if only_expected_edf:
                        origNH = -1 * model.hessian
                    args = zip(
                        repeat(family),
                        repeat(y),
                        repeat(Xs),
                        rPens,
                        repeat(init_coef),
                        repeat(len(init_coef)),
                        repeat(model.coef_split_idx),
                        repeat(method),
                        repeat(1e-7),
                        repeat(1),
                        repeat(bfgs_options),
                        repeat(origNH),
                        repeat(keep_drop),
                    )
                    with mp.Pool(processes=n_c) as pool:
                        (
                            sample_remls,
                            _,
                            sample_Linvs,
                            sample_coefs,
                            sample_edfs,
                            sample_llks,
                        ) = zip(*pool.starmap(compute_REML_candidate_GSMM, args))
                    sample_scales = np.ones(p_sample.shape[0])

                if only_expected_edf is False:
                    Linvs.extend(list(sample_Linvs))
                scales.extend(list(sample_scales))
                coefs.extend(list(sample_coefs))
                remls.extend(list(sample_remls))
                edfs.extend(list(sample_edfs))
                llks.extend(list(sample_llks))
                rGrid = p_sample
                rPens = None

        else:
            # Generate \\lambda values for which to compute REML, and Vb
            p_sample = scp.stats.multivariate_t.rvs(
                loc=np.ndarray.flatten(ep),
                shape=Vpr,
                df=df,
                size=n_est,
                random_state=seed,
            )
            p_sample = np.exp(p_sample)
            p_sample[np.isinf(p_sample)] = b

            if len(np.ndarray.flatten(ep)) == 1:  # Single lambda parameter in model

                if (
                    n_est == 1
                ):  # and single sample (n_est==1) - so p_sample needs to be shape (1,1)
                    p_sample = np.array([p_sample])

                p_sample = p_sample.reshape(
                    n_est, 1
                )  # p_sample needs to be shape (n_est,1)

            elif (
                n_est == 1
            ):  # multiple lambdas - so p_sample needs to be shape (1,n_lambda)
                p_sample = np.array([p_sample]).reshape(1, -1)

            # Make sure actual estimate is included once.
            minDiag = 0.1 * min(np.sqrt(Vpr.diagonal()))
            if not np.any(
                np.max(np.abs(p_sample - np.array([pen2.lam for pen2 in rPen])), axis=1)
                < minDiag
            ):
                p_sample = np.concatenate(
                    (np.array([pen2.lam for pen2 in rPen]).reshape(1, -1), p_sample),
                    axis=0,
                )

            for ps in p_sample:
                for ridx, rc in enumerate(ps):
                    rPen[ridx].lam = rc

                if isinstance(family, Family):
                    reml, Linv, LP, Pr, coef, scale, edf, llk = (
                        compute_reml_candidate_GAMM(
                            family,
                            y,
                            X,
                            rPen,
                            n_c,
                            model.offset,
                            method=method,
                            compute_inv=True,
                        )
                    )
                    # coef = coef.reshape(-1,1)

                    # Form VB, first solve LP{^-1}
                    # LPinv = compute_Linv(LP,n_c)
                    # Linv = apply_eigen_perm(Pr,LPinv)

                    # Collect conditional posterior covariance matrix for this set of coef
                    Vb = Linv.T @ Linv * scale
                    # Vb += coef@coef.T
                else:
                    try:
                        reml, V, _, coef, edf, llk = compute_REML_candidate_GSMM(
                            family,
                            y,
                            Xs,
                            rPen,
                            init_coef,
                            len(init_coef),
                            model.coef_split_idx,
                            n_c=n_c,
                            method=method,
                            bfgs_options=bfgs_options,
                            keep_drop=keep_drop,
                        )
                    except:  # noqa: E722
                        warnings.warn(
                            f"Unable to compute REML score for sample {ps}. Skipping."
                        )
                        continue

                    # coef = coef.reshape(-1,1)

                    # Collect conditional posterior covariance matrix for this set of coef
                    Vb = V  # + coef@coef.T

                # Collect all necessary objects for G&S correction.
                coefs.append(coef)
                remls.append(reml)
                edfs.append(edf)
                llks.append(llk)
                if only_expected_edf is False:
                    Vs.append(Vb)

                if len(rGrid) == 0:
                    rGrid = ps.reshape(1, -1)
                else:
                    rGrid = np.concatenate((rGrid, ps.reshape(1, -1)), axis=0)

    # Prepare computation of tau #

    if grid_type != "JJJ1":
        # Compute weights proposed by Greven & Scheipl (2017) - still work under importance
        # sampling case instead of grid case if we assume Vp
        # is prior for \rho|\mathbf{y}.
        if use_importance_weights and prior is None:
            ws = scp.special.softmax(remls)

        elif (
            use_importance_weights and prior is not None
        ):  # Standard importance weights (e.g., Branchini & Elvira, 2024)
            logp = prior.logpdf(np.log(rGrid))
            q = scp.stats.multivariate_t(
                loc=np.ndarray.flatten(ep), shape=Vpr, df=df, allow_singular=True
            )
            logq = q.logpdf(np.log(rGrid))

            ws = scp.special.softmax(remls + logp - logq)

        else:
            # Normal weights result in cancellation, i.e., just average
            ws = scp.special.softmax(np.ones_like(remls))

        Vcr = Vr @ dBetadRhos.T

        # Optionally re-compute negative Hessian at posterior mean for coef.
        if recompute_H and (only_expected_edf is False):

            # Estimate mean of posterior beta|y
            mean_coef = ws[0] * coefs[0]
            for ri in range(1, len(rGrid)):
                mean_coef += ws[ri] * coefs[ri]

            # Recompute Hessian at mean
            if isinstance(family, Family):
                yb = y
                Xb = X
                z = None
                Wr = None

                S_emb, _, S_root, _ = compute_S_emb_pinv_det(
                    X.shape[1], model.overall_penalties, "svd", method != "Chol"
                )

                eta = (X @ mean_coef).reshape(-1, 1) + model.offset

                if isinstance(family, Gaussian) and isinstance(
                    family.link, Identity
                ):  # strictly additive case
                    nH = (-1 * model.hessian) * orig_scale

                else:  # Generalized case

                    mu = family.link.fi(eta)

                    # Compute pseudo-dat and weights for mean coef
                    yb, Xb, z, Wr = update_PIRLS(
                        y, yb, mu, eta - model.offset, X, Xb, family, None
                    )

                    inval_check = np.any(np.isnan(z))

                    if inval_check:
                        _, w, inval = PIRLS_pdat_weights(
                            y, mu, eta - model.offset, family
                        )
                        w[inval] = 0

                        # Re-compute weight matrix
                        Wr_fix = scp.sparse.spdiags(
                            [np.sqrt(np.ndarray.flatten(w))], [0]
                        )
                    else:
                        Wr_fix = Wr

                    W = Wr_fix @ Wr_fix
                    nH = (X.T @ W @ X).tocsc()

                # Solve for coef to get Cholesky needed to re-compute scale
                _, _, _, Pr, _, LP, keep, drop = update_coef(
                    yb, X, Xb, family, S_emb, S_root, n_c, None, model.offset
                )

                # Re-compute scale
                _, _, _, _, _, scale = update_scale_edf(
                    y,
                    z,
                    eta,
                    Wr,
                    X.shape[0],
                    X.shape[1],
                    LP,
                    None,
                    Pr,
                    None,
                    None,
                    family,
                    model.overall_penalties,
                    keep,
                    drop,
                    n_c,
                )

                # And negative hessian
                nH /= scale

                if drop is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        nH[:, drop] = 0
                        nH[drop, :] = 0

            else:  # GSMM/GAMLSS case
                if isinstance(Family, GAMLSSFamily):  # GAMLSS case
                    split_coef = np.split(mean_coef, model.coef_split_idx)

                    # Update etas and mus
                    etas = [Xs[i] @ split_coef[i] for i in range(family.n_par)]
                    mus = [family.links[i].fi(etas[i]) for i in range(family.n_par)]

                    # Get derivatives with respect to eta
                    if family.d_eta is False:
                        d1eta, d2eta, d2meta = deriv_transform_mu_eta(y, mus, family)
                    else:
                        d1eta = [fd1(y, *mus) for fd1 in family.d1]
                        d2eta = [fd2(y, *mus) for fd2 in family.d2]
                        d2meta = [fd2m(y, *mus) for fd2m in family.d2m]

                    # Get derivatives with respect to coef
                    _, H = deriv_transform_eta_beta(
                        d1eta, d2eta, d2meta, Xs, only_grad=False
                    )

                else:  # GSMM
                    H = family.hessian(mean_coef, model.coef_split_idx, y, Xs)

                nH = -1 * H

            if verbose:
                print(
                    (
                        f"Recomputed negative Hessian. 2 Norm of coef. difference: "
                        f"{np.linalg.norm(mean_coef-model.coef)}. F. Norm of n. Hessian "
                        f"difference: {scp.sparse.linalg.norm(nH + model.hessian)}"
                    )
                )

        else:
            mean_coef = None
            nH = -1 * model.hessian

        # Compute tau #

        if only_expected_edf:
            # Compute correction of edf directly..
            expected_edf = max(model.edf, np.sum(ws * edfs))

            if grid_type == "JJJ2":
                # Can now add remaining correction term

                # Now have Vc = Vcr.T @ Vcr, so:
                # tr(Vc@(-1*model.hessian)) = tr(Vcr.T @ Vcr@(-1*model.hessian)) =
                # tr(Vcr@ (-1*model.hessian)@Vcr.T)
                # expected_edf += (Vc@(-1*model.hessian)).trace()
                expected_edf += (Vcr @ (nH) @ Vcr.T).trace()

            elif grid_type == "JJJ3":
                # Correct based on G&S expectations instead
                tr1 = ws[0] * (coefs[0].T @ (nH) @ coefs[0])[0, 0]

                # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
                tr2 = ws[0] * coefs[0]

                # Now sum over remaining r
                for ri in range(1, len(rGrid)):
                    tr1 += ws[ri] * (coefs[ri].T @ (nH) @ coefs[ri])[0, 0]
                    tr2 += ws[ri] * coefs[ri]

                # Enforce lower bound of JJJ2
                if (Vcr @ (nH) @ Vcr.T).trace() > (tr1 - (tr2.T @ (nH) @ tr2)[0, 0]):
                    expected_edf += (Vcr @ (nH) @ Vcr.T).trace()
                else:
                    expected_edf += tr1 - (tr2.T @ (nH) @ tr2)[0, 0]

            if verbose:
                print(f"Correction was based on {rGrid.shape[0]} samples in total.")

            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                expected_edf,
                mean_coef,
            )
        else:
            expected_edf = None

        # Compute tau and full covariance matrix #

        if grid_type == "JJJ2":

            if n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\\lambda)] +
                # E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
                Vr1 = ws[0] * (Linvs[0].T @ Linvs[0] * scales[0])

                # Now sum over remaining r
                for ri in range(1, len(rGrid)):
                    Vr1 += ws[ri] * ((Linvs[ri].T @ Linvs[ri] * scales[ri]))

            else:
                Vr1 = ws[0] * Vs[0]

                for ri in range(1, len(rGrid)):
                    Vr1 += ws[ri] * Vs[ri]

            if (Vr1 @ (nH)).trace() < model.edf:
                nH = -1 * model.hessian  # Reset nH

                if isinstance(family, Family):
                    Vr1 = model.lvi.T @ model.lvi * orig_scale
                else:
                    Vr1 = model.lvi.T @ model.lvi

            V = Vr1 + Vcr.T @ Vcr

        elif grid_type == "JJJ3":

            # Now compute \\hat{cov(\boldsymbol{\beta}|y)}
            if n_c > 1:
                # E_{p|y}[V_\boldsymbol{\beta}(\\lambda)] +
                # E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] in Greven & Scheipl (2017)
                Vr1 = ws[0] * (Linvs[0].T @ Linvs[0] * scales[0])
                Vr2 = ws[0] * (coefs[0] @ coefs[0].T)
                # E_{p|y}[\boldsymbol{\beta}] in Greven & Scheipl (2017)
                Vr3 = ws[0] * coefs[0]

                # Now sum over remaining r
                for ri in range(1, len(rGrid)):
                    Vr1 += ws[ri] * (Linvs[ri].T @ Linvs[ri] * scales[ri])
                    Vr2 += ws[ri] * (coefs[ri] @ coefs[ri].T)
                    Vr3 += ws[ri] * coefs[ri]

            else:
                Vr1 = ws[0] * Vs[0]
                Vr2 = ws[0] * (coefs[0] @ coefs[0].T)
                Vr3 = ws[0] * coefs[0]

                for ri in range(1, len(rGrid)):
                    Vr1 += ws[ri] * Vs[ri]
                    Vr2 += ws[ri] * (coefs[ri] @ coefs[ri].T)
                    Vr3 += ws[ri] * coefs[ri]

            # if (Vr1@(nH)).trace() < model.edf:
            #    nH = -1 * model.hessian # Reset nH
            #
            #    if isinstance(family,Family):
            #        Vr1 = model.lvi.T@model.lvi*orig_scale
            #    else:
            #        Vr1 = model.lvi.T@model.lvi

            # Now, Greven & Scheipl provide final estimate =
            # E_{p|y}[V_\boldsymbol{\beta}(\\lambda)] +
            # E_{p|y}[\boldsymbol{\beta}\boldsymbol{\beta}^T] -
            # E_{p|y}[\boldsymbol{\beta}] E_{p|y}[\boldsymbol{\beta}]^T
            # Enforce lower bound of JJJ2 again
            # if (Vcr @ (nH) @ Vcr.T).trace() > ((Vr2 - (Vr3@Vr3.T)) @ (nH)).trace():
            #    V = Vr1 + Vcr.T @ Vcr
            # else:
            V = Vr1 + Vr2 - (Vr3 @ Vr3.T)

            if (V @ (-1 * model.hessian)).trace() > (V @ nH).trace():
                nH = -1 * model.hessian  # Reset nH

            # Enforce lower bound of JJJ1
            # if (Vlb@(-1 * model.hessian)).trace() > (V@nH).trace():
            #    nH = -1 * model.hessian
            #    V = Vlb

            # Some lower bounds for paper simulation:
            if isinstance(family, Family):
                Vr1lb = model.lvi.T @ model.lvi * orig_scale
            else:
                Vr1lb = model.lvi.T @ model.lvi

            if (Vr1 @ (nH)).trace() < (Vr1lb @ (nH)).trace():
                Vr1 = Vr1lb
                V = Vr1 + Vr2 - (Vr3 @ Vr3.T)

            if (Vcr @ (nH) @ Vcr.T).trace() > ((Vr2 - (Vr3 @ Vr3.T)) @ (nH)).trace():
                V = Vr1 + Vcr.T @ Vcr

    else:
        mean_coef = None
        expected_edf = None
        nH = -1 * model.hessian

    # Check V is full rank - can use LV for sampling as well..
    LV, code = cpp_chol(scp.sparse.csc_array(V))
    if code != 0:

        # Check first whether Cholesky can be formed for set of identifiable coefficients.
        code2 = 1
        if model.info.dropped is not None:

            keep = np.array(
                [
                    cidx
                    for cidx in range(model.hessian.shape[1])
                    if cidx not in model.info.dropped
                ]
            )

            Vdrop = V[keep, :]
            Vdrop = Vdrop[:, keep]

            LVdrop, code2 = cpp_chol(scp.sparse.csc_array(Vdrop))

            if code2 == 0:

                warnings.warn(
                    (
                        "Marginal covariance matrix is not of full rank. "
                        "Possibly because coefficients were dropped. "
                        "Returned Cholesky ``LV`` was computed based on set of coefficients "
                        "deemed identifiable during estimation."
                    )
                )

                LVdat, LVrow, LVcol = translate_sparse(LVdrop)
                LVrow = keep[LVrow]
                LVcol = keep[LVcol]

                LV = scp.sparse.csc_array(
                    (LVdat, (LVrow, LVcol)), shape=model.hessian.shape
                )

        # V is definitely not PD..
        if code2 != 0:

            warnings.warn(
                (
                    "Marginal covariance matrix is not of full rank. "
                    "Possibly because coefficients were dropped. "
                    "Returned Cholesky ``LV`` was set to identity and is invalid!"
                )
            )

    # Compute corrected edf (e.g., for AIC; Wood, Pya, & Saefken, 2016)
    F = V @ (nH)

    edf = F.diagonal()
    total_edf = F.trace()

    # In mgcv, an upper limit is enforced on edf and total_edf when they are uncertainty corrected
    # - based on t1 in section 6.1.2 of Wood (2017)
    # so the same is done here.
    if isinstance(family, Family):
        ucF = model.lvi.T @ model.lvi @ ((-1 * model.hessian) * orig_scale)
    else:  # GSMM/GAMLSS case
        ucF = (model.lvi.T @ model.lvi) @ (-1 * model.hessian)

    # Compute upper bound
    ucFFd = ucF.multiply(ucF.T).sum(axis=0)
    total_edf2 = 2 * model.edf - np.sum(ucFFd)
    edf2 = 2 * ucF.diagonal() - ucFFd

    # Enforce upper bound
    if total_edf > total_edf2:
        total_edf = total_edf2
        edf = edf2

    # Compute uncertainty corrected smoothness bias corrected edf (t1 in section 6.1.2 of Wood,
    # 2017)
    if form_t1:
        total_edf2 += total_edf - model.edf
        edf2 += edf - ucF.diagonal()

    if verbose and grid_type != "JJJ1":
        print(f"Correction was based on {rGrid.shape[0]} samples in total.")

    return V, LV, Vp, Vpr, edf, total_edf, edf2, total_edf2, expected_edf, mean_coef
