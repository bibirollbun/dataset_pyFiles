# try:
#     import celer
#     print("Thư viện celer đã được cài đặt.")
# except ImportError:
#     print("Thư viện celer chưa được cài đặt. Tiến hành cài đặt...")
#     !pip install celer
#     print("Thư viện celer đã được cài đặt thành công. Hãy khởi động lại Kernel (Runtime) để sử dụng.")

try:
    import skglm
    print("Thư viện skglm đã được cài đặt.")
except ImportError:
    print("Thư viện skglm chưa được cài đặt. Tiến hành cài đặt...")
    !pip install skglm
    print("Thư viện skglm đã được cài đặt thành công. Hãy khởi động lại Kernel (Runtime) để sử dụng.")


from numpy.linalg import pinv
import numpy as np
from scipy.linalg import block_diag
from skglm import Lasso
from mpmath import mp
import random
import pickle
import warnings
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


def generate_synthetic_data(p, s, K, M, n_list, h_list, true_beta=0.5, gamma=0.1):
    beta = np.concatenate([np.full(s, true_beta), np.zeros(p - s)]) # (p,)
    coeffs = []
    for k in range(K):
        del_k = np.zeros(p)
        if h_list[k] > 0:
            idx = np.random.choice(p, h_list[k], replace=False)
            signs = np.random.choice([-1, 1], h_list[k])
            del_k[idx] = signs * gamma

        wk = beta - del_k
        coeffs.append(wk)

    coeffs.append(beta)

    X_list, Y_list, true_Y_list = [], [], []
    cov = np.eye(p)

    for k in range(K + 1):
        Xk = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n_list[k])
        
        true_Yk = Xk @ coeffs[k]
        noise = np.random.normal(0, 1, n_list[k])
        # noise = np.random.laplace(0, 1, n_list[k])
        # noise = skewnorm.rvs(a=10, loc=0, scale=1, size=n_list[k])
        # noise = np.random.standard_t(df=20, size=n_list[k])
        Yk = true_Yk + noise 
        X_list.append(Xk)
        Y_list.append(Yk)
        true_Y_list.append(true_Yk)

    XS_list, YS_list = X_list[:-1], Y_list[:-1]
    X0, Y0 = X_list[-1], Y_list[-1]
    true_Y = np.concatenate(true_Y_list)

    SigmaS_list = [np.eye(nk) for nk in n_list[:M]]
    Sigma0 = np.eye(n_list[-1])

    return XS_list, YS_list, X0, Y0, true_Y, beta, SigmaS_list, Sigma0

def gen_X_syn(X0, r, nk):
    # np.random.seed(42)
    indices = np.random.choice(X0.shape[0], size=r * nk, replace=True) 

    X_syn = X0[indices]

    return X_syn


# utils.py
mp.dps = 500
# CONSTRUCT ACTIVE SET
def construct_active_set (coef_hat, X): 
    coef_active, signs, active_set, inactive_set = [], [], [], []
    p = X.shape[1]
    for i, val in enumerate(coef_hat):
        if val == 0.0:
            inactive_set.append(i)
        else:
            active_set.append(i)
            coef_active.append(val)
            signs.append(np.sign(val))
    
    X_active = X[:, active_set] if active_set else np.zeros((X.shape[0], 0))
    X_inactive = X[:, inactive_set] if inactive_set else np.zeros((X.shape[0], 0))

    coef_active = np.array(coef_active).reshape(-1, 1)
    signs = np.array(signs).reshape(-1, 1)

    return {
        "coef_active": coef_active, 
        "signs": signs, 
        "active_set": active_set, 
        "X_active": X_active, 
        "inactive_set": inactive_set, 
        "X_inactive": X_inactive,
        "E": np.eye(p)[:, active_set] if len(active_set) > 0 else np.zeros((p, 0))
    }


def construct_Pk(k, n_list, n):
    start_col = sum(n_list[:k])
    nk = n_list[k]
    Pk = np.zeros((nk, n))
    Pk[:, start_col : start_col + nk] = np.eye(nk)
    return Pk


#_________________________________________________
def construct_test_statistic(j, X0M, Y, M_obs, n0, n):
    idx = M_obs.index(j)          
    ej = np.zeros((len(M_obs), 1))
    ej[idx, 0] = 1

    inv = pinv(X0M.T @ X0M)   
    eta_tail = X0M @ inv @ ej     # n_T × 1

    etaj = np.zeros((n, 1))
    etaj[-n0:, 0] = eta_tail.ravel()

    etajTy = (etaj.T @ Y.reshape(-1, 1))[0, 0] 
    return etaj, etajTy



def calculate_a_b(etaj, Y, Sigma, n):
    e1 = etaj.T @ Sigma @ etaj
    b = (Sigma @ etaj)/e1

    e2 = np.eye(n) - b @ etaj.T
    a = e2 @ Y

    return a.reshape(-1, 1), b.reshape(-1, 1)



def merge_intervals(intervals, tol=1e-4):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] - merged[-1][1] > tol:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged


def pivot(intervals, etajTy, etaj, Sigma, tn_mu=0):
    if len(intervals) == 0: return None 
    intervals = merge_intervals(intervals, tol=1e-2)

    etaj = etaj.ravel()
    stdev = np.sqrt(etaj @ (Sigma @ etaj))

    numerator = mp.mpf('0')
    denominator = mp.mpf('0')

    for (left, right) in intervals:
        cdf_left= mp.ncdf((left- tn_mu)/ stdev)
        cdf_right= mp.ncdf((right- tn_mu)/ stdev)
        piece = cdf_right- cdf_left
        denominator += piece

        if etajTy >= right:
            numerator += piece
        elif left <= etajTy < right:
            numerator += mp.ncdf((etajTy - tn_mu)/ stdev) - cdf_left

    if denominator == 0:
        return None
    return float(numerator/ denominator)


def calculate_TN_p_value(intervals, etaj, etajTY, Sigma, tn_mu=0.0):
    cdf = pivot(intervals, etajTY, etaj, Sigma, tn_mu)
    if cdf: 
        return 2.0 * min(cdf, 1.0 - cdf)
    else: 
        return None



# commute.py
def calculate_unshare_w_hat(unsh_XS_list, unsh_YS_list, unsh_lambdas_w):
    # Step 1: Calculate wk of COMMUTE (In unsharedsource sites)
    unsh_w_hat_list = []
    for Xk, Yk, lam in zip(unsh_XS_list, unsh_YS_list, unsh_lambdas_w):
        mdl = Lasso(alpha=lam, fit_intercept=False, tol=1e-10)
        mdl.fit(Xk, Yk)
        unsh_w_hat_list.append(mdl.coef_)
    
    return unsh_w_hat_list

def COMMUTE(share_XS_list, share_YS_list, X0, Y0, X_syn_list, unsh_w_hat_list, M, K, r, n_list, lambdas_w, lambdas_del, lambda_c):
    # Step 1: Calculate wk of COMMUTE (In unsharedsource sites)
    share_w_hat_list = []
    for Xk, Yk, lam in zip(share_XS_list, share_YS_list, lambdas_w):
        # mdl = Lasso(alpha=lam, fit_intercept=False, tol=1e-10)
        mdl = Lasso(alpha=lam, fit_intercept=False, tol=1e-10, max_iter=int(5e3), max_epochs=int(5e6))
        mdl.fit(Xk, Yk)
        share_w_hat_list.append(mdl.coef_)
        
    w_hat_list = share_w_hat_list + unsh_w_hat_list

    # In target site
    # Step 2: Calculate δk
    delta_hat_list = []
    for k, lam in enumerate(lambdas_del):
        # mdl = Lasso(alpha=lam, fit_intercept=False, tol=1e-10)
        mdl = Lasso(alpha=lam, fit_intercept=False, tol=1e-10, max_iter=int(5e3), max_epochs=int(5e6))
        mdl.fit(X0, Y0 - X0 @ w_hat_list[k])
        delta_hat_list.append(mdl.coef_)
    
    # Step 3: Calculate β_COMMUTE 
    n0 = n_list[-1]
    n_tilde = n0 + sum(n_list[:M]) + sum(nk * r for nk in n_list[M:K])
    X_tilde, Y_tilde = [], []

    # Target 
    weight0 = np.sqrt(n_tilde / n0)
    

    X_tilde.append(weight0 * X0)
    Y_tilde.append(weight0 * Y0)

    # M sharable sources
    for k in range(M):
        Xk, Yk = share_XS_list[k], share_YS_list[k]
        Y_adj = Yk + Xk @ delta_hat_list[k]
        
        sh_weight = np.sqrt(n_tilde / n_list[k])
        X_tilde.append(sh_weight * Xk)
        Y_tilde.append(sh_weight * Y_adj)


    # K−M unsharable sources
    for k in range(K - M):
        beta_hat_k = unsh_w_hat_list[k] + delta_hat_list[k + M]
        X_syn = X_syn_list[k]
        Y_syn = X_syn @ beta_hat_k

        
        unsh_weight = np.sqrt(n_tilde / (n_list[k+M] * r))
        X_tilde.append(unsh_weight * X_syn)
        Y_tilde.append(unsh_weight * Y_syn)

    
    X_tilde = np.vstack(X_tilde)
    Y_tilde = np.concatenate(Y_tilde)
    

    # beta_mdl = Lasso(alpha=lambda_c, fit_intercept=False, tol=1e-10)
    beta_mdl = Lasso(alpha=lambda_c, fit_intercept=False, tol=1e-10, max_iter=int(5e3), max_epochs=int(5e6))
    beta_mdl.fit(X_tilde, Y_tilde)
    beta_commute = beta_mdl.coef_

    return w_hat_list, delta_hat_list, beta_commute


# sub_prob.py
def compute_interval_from_inequalities(A, B, Z = None):
    # Az < B 
    l, r = -np.inf, np.inf

    for i in range(len(A)):
        if A[i] == 0:
            if B[i] < 0: 
                return np.inf, -np.inf
       
        elif A[i] > 0: 
            r = min(r, B[i] / A[i])
        
        else: 
            l = max(l, B[i] / A[i])

    if l > r:
      print(f"Lỗi l > r ở {Z}")
    
    return l, r


def compute_Zuk(wk_info, Pk, a, b, nk, lambda_wk, Yz):
    psi0 = gamma0 = psi1 = gamma1 = np.empty(0)

    O, Oc, XkO, XkOc, SO = wk_info["active_set"], wk_info["inactive_set"], wk_info["X_active"], wk_info["X_inactive"], wk_info["signs"]

    if len(O) > 0:
        inv = pinv(XkO.T @ XkO)
        XkO_plus = inv @ XkO.T    

        # Calculate psi0, gamma0
        psi0 = (-SO * (XkO_plus @ Pk @ b)).ravel()
        gamma0 = (SO * ((XkO_plus @ Pk @ a) - nk * lambda_wk * (inv @ SO))).ravel()

        # Check KKT
        # wO = inv @ (XkO.T @ Pk @ Yz.reshape(-1, 1) - nk * lambda_wk * SO)
        # wO_Lasso = wk_info["coef_active"]
        # for i in range(len(wO)):
        #     if not np.isclose(wO[i][0], wO_Lasso[i][0]):
        #         print (f"wO[{i}] = {wO[i][0]} - wO_Lasso[{i}] = {wO_Lasso[i][0]}")

    if len(Oc) > 0:
        if len(O) ==  0:
            proj = np.eye(nk)
            temp2 = np.zeros((len(Oc), 1))
        else:
            proj = np.eye(nk) - XkO @ XkO_plus
            temp2 = (XkOc.T @ XkO_plus.T) @ SO #

        temp1 = (XkOc.T @ proj) / (lambda_wk * nk)

        # Calculate psi1
        term_b = (temp1 @ (Pk @ b)).ravel()
        psi1 = np.concatenate([term_b, -term_b])

        # Calculate gamma1
        term_a  = (temp1 @ Pk @ a).ravel()
        gamma1 = np.concatenate([(1 - temp2.ravel() - term_a), (1 + temp2.ravel() + term_a) ])
    
        # Check KKT SOc
        # SOc = temp2 + temp1 @ Pk @ Yz.reshape(-1, 1)
        # for i in range (len(SOc)):
        #     if abs(SOc[i][0]) > 1:
        #         print(f"SOc[{i}] = {SOc[i][0]}")

    
    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))

    return compute_interval_from_inequalities(psi, gamma, "Zuk")


def calculate_phi_iota(wk_info, X0, P0, Pk, n0, nk, lambda_wk, sharable=True):    
    phi_u = P0.copy()
    iota_u = np.zeros((n0, 1))

    if sharable:
        if len(wk_info["active_set"]) > 0: 
            Ek, XkO, SO = wk_info["E"], wk_info["X_active"], wk_info["signs"]
            inv_XkOT_XkO = pinv(XkO.T @ XkO)
            phi_u -= X0 @ Ek @ (inv_XkOT_XkO @ XkO.T) @ Pk
            iota_u  = nk * lambda_wk * X0 @ Ek @ inv_XkOT_XkO @ SO

    else:
        iota_u = -X0 @ wk_info.reshape(-1, 1)
    
    return phi_u, iota_u


def compute_Zvk(delk_info, a, b, phi_u, iota_u, n0, lambda_delk, Yz):
    nu0 = kappa0 = nu1 = kappa1 = np.empty(0)
    L, Lc, X0L, X0Lc, SL = delk_info["active_set"], delk_info["inactive_set"], delk_info["X_active"], delk_info["X_inactive"], delk_info["signs"]

    if len(L) > 0:
        inv_X0LT_X0L = pinv(X0L.T @ X0L)
        X0L_plus = inv_X0LT_X0L @ X0L.T

        # Calculate nu0
        nu0 = (-SL * (X0L_plus @ phi_u @ b)).ravel()

        # Calculate kappa0
        temp = X0L_plus @ (phi_u @ a + iota_u)
        kappa0 = (SL * (temp - n0 * lambda_delk * (inv_X0LT_X0L @ SL))).ravel()

        # Check KKT
        # dL = inv_X0LT_X0L @ (X0L.T @ (phi_u @ Yz.reshape(-1, 1) + iota_u) - n0 * lambda_delk * SL)
        # dL_Lasso = delk_info["coef_active"]
        # for i in range(len(dL)):
        #     if not np.isclose(dL[i][0], dL_Lasso[i][0]):
        #         print (f"dL[{i}] = {dL[i][0]} - dL_Lasso[{i}] = {dL_Lasso[i][0]}")
    
    if len(Lc) > 0:
        if len(L) == 0:
            proj = np.eye(n0)
            temp2 = np.zeros((len(Lc), 1))

        else:
            proj = np.eye(n0) - X0L @ X0L_plus
            temp2 = (X0Lc.T @ X0L_plus.T) @ SL

        temp1 = (X0Lc.T @ proj) / (n0 * lambda_delk)

        # Calculate nu1
        term_b = (temp1 @ (phi_u @ b)).ravel()
        nu1 = np.concatenate([term_b, -term_b])

        # Calculate kappa1
        term_a = (temp1 @ (phi_u @ a + iota_u)).ravel()
        kappa1 = np.concatenate([(1 - temp2.ravel() - term_a), (1 + temp2.ravel() + term_a)])

        # Check KKT SLc
        # SLc = temp2 + temp1 @ (phi_u @ Yz.reshape(-1, 1) + iota_u)
        # for i in range (len(SLc)):
        #     if abs(SLc[i][0]) > 1:
        #         print(f"SLc[{i}] = {SLc[i][0]}")
    nu = np.concatenate((nu0, nu1))
    kappa = np.concatenate((kappa0, kappa1))
    
    return compute_interval_from_inequalities(nu, kappa, "Zvk")


def calculate_mk_Nk(delk_info, phi_u, iota_u, p, n0, n, lambda_delk):
    L, Lc, X0L, X0Lc, SL = delk_info["active_set"], delk_info["inactive_set"], delk_info["X_active"], delk_info["X_inactive"], delk_info["signs"]

    mk = np.zeros((p, 1))
    Nk = np.zeros((p, n))

    if len(L) > 0:
        Fk = delk_info["E"]
        inv_X0LT_X0L = pinv(X0L.T @ X0L)            
        mk = Fk @ inv_X0LT_X0L @ (X0L.T @ iota_u - n0 * lambda_delk * SL)
        Nk = Fk @ inv_X0LT_X0L @ X0L.T @ phi_u

    return mk, Nk


def compute_Zt(beta_info, a, b, share_XS_list, X_syn_list, n_list, P_list, m_list, N_list, p_list, Q_list, M, K, r, lambda_c, Yz):

    omega0 = rho0 = omega1 = rho1 = np.empty(0)
    M_set, Mc, X0M, X0Mc, SM = beta_info["active_set"], beta_info["inactive_set"], beta_info["X_active"], beta_info["X_inactive"], beta_info["signs"]
    
    n0, P0 = n_list[-1], P_list[-1]
    if len(M_set) > 0: 
        # Calculate ut, Vt
        ut = - lambda_c * SM
        Vt = (1/n0) * X0M.T @ P0
        temp = (1/n0) * X0M.T @ X0M 

        for k in range(M):
            Xk = share_XS_list[k]
            XkM = Xk[:, M_set]
            nk = n_list[k]
            ut += (1/nk) * XkM.T @ Xk @ m_list[k]
            Vt += (1/nk) * XkM.T @ (P_list[k] + Xk @ N_list[k])
            temp += (1/nk) * XkM.T @ XkM

        for k in range(K - M):
            nk = n_list[k + M]
            XkM_syn = X_syn_list[k][:, M_set]
            ut += (1/(nk * r)) * XkM_syn.T @ p_list[k]
            Vt += (1/(nk * r)) * XkM_syn.T @ Q_list[k]
            temp += (1/(nk * r)) * XkM_syn.T @ XkM_syn 

        temp = pinv(temp)
        ut = temp @ ut
        Vt = temp @ Vt

        # Calculate omega0, rho0
        omega0 = (-SM * Vt @ b).ravel()
        rho0 = (SM * Vt @ a + SM * ut).ravel()

        # Check KKT bM
        # bM = ut + Vt @ Yz.reshape(-1, 1)
        # bM_Lasso = beta_info["coef_active"]
        # for i in range(len(bM)):
        #     if not np.isclose(bM[i][0], bM_Lasso[i][0]):
        #         print (f"bM[{i}] = {bM[i][0]} - bM_Lasso[{i}] = {bM_Lasso[i][0]}")
    
    if len(Mc) > 0: 
        if len(M_set) == 0:
            ht = np.zeros((len(Mc), 1))
            Kt = (1/n0) * X0Mc.T @ P0

            for k in range(M):
                nk = n_list[k]
                Xk = share_XS_list[k]
                XkMc = Xk[:, Mc]
                ht += (1/nk) * XkMc.T @ (Xk @ m_list[k])
                Kt += (1/nk) * XkMc.T @ (P_list[k] + Xk @ N_list[k])

            for k in range(K - M):
                nk = n_list[k + M]
                XkMc_syn = X_syn_list[k][:, Mc]
                ht += (1/(nk * r)) * XkMc_syn.T @ p_list[k] 
                Kt += (1/(nk * r)) * XkMc_syn.T @ Q_list[k]   
        
        else:
            ht = (-1/n0) * X0Mc.T @ X0M @ ut
            Kt = (1/n0) * X0Mc.T @ (P0 - X0M @ Vt)

            for k in range(M):
                nk = n_list[k]
                Xk = share_XS_list[k]
                XkM = Xk[:, M_set]
                XkMc = Xk[:, Mc]
                ht += (1/nk) * XkMc.T @ (Xk @ m_list[k] - XkM @ ut)
                Kt += (1/nk) * XkMc.T @ (P_list[k] + Xk @ N_list[k] - XkM @ Vt)

            for k in range(K - M):
                nk = n_list[k + M]
                XkM_syn = X_syn_list[k][:, M_set]
                XkMc_syn = X_syn_list[k][:, Mc] 
                ht += (1/(nk * r)) * XkMc_syn.T @ (p_list[k] - XkM_syn @ ut)
                Kt += (1/(nk * r)) * XkMc_syn.T @ (Q_list[k] - XkM_syn @ Vt)
            
        
        ht = (1/lambda_c) * ht
        Kt = (1/lambda_c) * Kt

        # Calculate omega1
        term_b = (Kt @ b).ravel()
        omega1 = np.concatenate([term_b, -term_b])

        # Calculate rho1
        term_a = (Kt @ a).ravel()
        rho1 = np.concatenate([(1 - ht.ravel() - term_a), (1 + ht.ravel() + term_a)])

        # Check KKT SMc
        # SMc = ht + Kt @ Yz.reshape(-1, 1)
        # for i in range (len(SMc)):
        #     if abs(SMc[i][0]) > 1:
        #         print(f"SMc[{i}] = {SMc[i][0]}")

    omega = np.concatenate((omega0, omega1))
    rho = np.concatenate((rho0, rho1))  

    return compute_interval_from_inequalities(omega, rho, "Zt")    


import os, time, numpy as np
from joblib import Parallel, delayed

# -----------------------------------------------------------
# 1) merge intervals
# -----------------------------------------------------------
def _merge_intervals(ivals, eps=1e-4):
    if not ivals:
        return []
    ivals = sorted(ivals, key=lambda x: x[0])
    merged = [ivals[0]]
    for l, r in ivals[1:]:
        if l <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))
    return merged

# -----------------------------------------------------------
# 2) worker: trả về (intervals, oc_intervals, logs)
# -----------------------------------------------------------
def _segment_worker(
    seg_idx,
    share_XS_list, X_syn_list, X0, unsh_w_hat_list,
    a, b, Mobs, M, K, r, p, P_list, n_list, n,
    lambdas_w, lambdas_del, lambda_c,
    z_start, z_end
):
    # hạn chế BLAS đa luồng
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # log = []
    # pid = os.getpid()
    # t0  = time.time()
    # log.append(f"[PID {pid}] ▶️ seg {seg_idx} [{z_start:.2f},{z_end:.2f}] START")

    P0, n0 = P_list[-1], n_list[-1]
    intervals, oc_intervals = [], []
    global OBS_W_ACTIVE_SETS, OBS_D_ACTIVE_SETS

    z = z_start
    while z < z_end:
        Yz        = (a + b * z).ravel()
        share_YSz = [P @ Yz for P in P_list[:-1]]
        Y0z       = (P0 @ Yz).ravel()

        wz_list, dz_list, bz = COMMUTE(
            share_XS_list, share_YSz, X0, Y0z,
            X_syn_list, unsh_w_hat_list,
            M, K, r, n_list, lambdas_w, lambdas_del, lambda_c
        )

        wz_infos = [construct_active_set(wz_list[k], share_XS_list[k]) for k in range(M)]
        dz_infos = [construct_active_set(dz_list[k], X0) for k in range(K)]
        bz_info  = construct_active_set(bz, X0)

        l_list, r_list = [], []
        m_list, N_list, p_list, Q_list = [], [], [], []

        # --- w_k
        for k in range(M):
            lu, ru = compute_Zuk(wz_infos[k], P_list[k], a, b,
                                 n_list[k], lambdas_w[k], Yz)
            l_list.append(lu); r_list.append(ru)

            phi_u, iota_u = calculate_phi_iota(
                wz_infos[k], X0, P0, P_list[k],
                n0, n_list[k], lambdas_w[k], sharable=True
            )
            mk, Nk = calculate_mk_Nk(dz_infos[k], phi_u, iota_u,
                                     p, n0, n, lambdas_del[k])
            m_list.append(mk); N_list.append(Nk)

            lv, rv = compute_Zvk(dz_infos[k], a, b, phi_u, iota_u,
                                 n0, lambdas_del[k], Yz)
            l_list.append(lv); r_list.append(rv)

        # --- δ_k (unshare)
        for k in range(K - M):
            phi_u, iota_u = calculate_phi_iota(
                unsh_w_hat_list[k], X0, P0, None,
                n0, None, None, sharable=False
            )
            mk, Nk = calculate_mk_Nk(dz_infos[k+M], phi_u, iota_u,
                                     p, n0, n, lambdas_del[k+M])

            pk = X_syn_list[k] @ (unsh_w_hat_list[k].reshape(-1,1) + mk)
            Qk = X_syn_list[k] @ Nk
            p_list.append(pk); Q_list.append(Qk)

            lv, rv = compute_Zvk(dz_infos[k+M], a, b, phi_u, iota_u,
                                 n0, lambdas_del[k+M], Yz)
            l_list.append(lv); r_list.append(rv)

        # --- β
        lt, rt = compute_Zt(
            bz_info, a, b,
            share_XS_list, X_syn_list,
            n_list, P_list,
            m_list, N_list, p_list, Q_list,
            M, K, r, lambda_c, Yz
        )
        l_list.append(lt); r_list.append(rt)

        left, right = max(l_list), min(r_list)
        if right < left or right < z:
            print ("Error!!!")
            return [], []

        Mt = bz_info["active_set"]
        if np.array_equal(Mobs, Mt):
            intervals.append((left, right))

        OC_match = (np.array_equal(Mobs, Mt) and
                    all(np.array_equal(wz_infos[k]["active_set"], OBS_W_ACTIVE_SETS[k]) for k in range(M)) and
                    all(np.array_equal(dz_infos[k]["active_set"], OBS_D_ACTIVE_SETS[k]) for k in range(K)))
        if OC_match:
            oc_intervals.append((left, right))

        z = right + 1e-5

    # log.append(f"[PID {pid}] ⏹ seg {seg_idx} DONE "
    #            f"({len(intervals)} ivals) {time.time()-t0:.1f}s")

    return intervals, oc_intervals, # log

# -----------------------------------------------------------
# 3. Hàm chính
# -----------------------------------------------------------
def divide_and_conquer(
    share_XS_list, X_syn_list, X0, unsh_w_hat_list,
    a, b, Mobs, M, K, r, p, P_list, n_list, n,
    lambdas_w, lambdas_del, lambda_c,
    z_min=-20, z_max=20, num_segments=48
):
    global OBS_W_ACTIVE_SETS, OBS_D_ACTIVE_SETS

    seg_w  = (z_max - z_min) / num_segments
    segments = [(z_min+i*seg_w, z_min+(i+1)*seg_w) for i in range(num_segments)]

    n_jobs = min(num_segments, os.cpu_count())
    # print(f"[MAIN] cores={os.cpu_count()}, use n_jobs={n_jobs}")  #=====================================================================================================

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_segment_worker)(
            idx,
            share_XS_list, X_syn_list, X0, unsh_w_hat_list,
            a, b, Mobs, M, K, r, p, P_list, n_list, n,
            lambdas_w, lambdas_del, lambda_c,
            seg_start, seg_end
        )
        for idx, (seg_start, seg_end) in enumerate(segments)
    )

    intervals, oc_intervals = [], []
    for seg_intervals, seg_oc in results:
        intervals.extend(seg_intervals)
        oc_intervals.extend(seg_oc)
        # all_logs.extend(logs)

    # In log sau cùng (đảm bảo đúng thứ tự)
    # print("\n".join(all_logs)) #=====================================================================================================

    intervals    = _merge_intervals(intervals)
    oc_intervals = _merge_intervals(oc_intervals)
    # OC_INTERVALS_BUFFER = oc_intervals

    # print(f"[MAIN] merged_intervals={len(intervals)}, merged_OC={len(oc_intervals)}")  #=====================================================================================================
    return intervals, oc_intervals



import time

def fpr_experiment():
    global OBS_W_ACTIVE_SETS, OBS_D_ACTIVE_SETS
    p = 500
    s = 0
    true_beta = 0.5
    gamma = 0.5
    n0 = 100
    # n0_list = [200]
    M_list = [0]
    coeff = [20, 20, 20, 10]
    K = 3
    r = 10
    n_list = [300, 400, 500, 50]
    h_list = [10, 10, 10]
    thresold = 20
    alpha = 0.05

    num_trials = 500
    fpr_values = {}
    oc_fpr_values = {}
    num_err = 0

    for M in M_list:
        n_list[-1] = n0
        total_false_positives_detected = 0
        total_false_positives_rejected = 0
        oc_total_false_positives_detected = 0
        oc_total_false_positives_rejected = 0
        print (f'M: {M}')

        trial = 0
        while trial < num_trials:
            print (f"======== trial {trial+1} ========")
            start_time = time.perf_counter()
            XS_list, YS_list, X0, Y0, true_Y, beta, SigmaS_list, Sigma0 = generate_synthetic_data(p, s, K, M, n_list, h_list, true_beta, gamma)
            # XS_list, YS_list, X0, Y0, true_Y, beta, n_list, SigmaS_list, Sigma0 = generate_data(p, s, nS, n0, K, M, H, true_beta, gamma)
            n_tilde = (n0 + sum(n_list[k] for k in range(M)) + sum(n_list[k] * r for k in range(M, K)))
            share_XS_list, unsh_XS_list = XS_list[:M], XS_list[M:]
            share_YS_list, unsh_YS_list = YS_list[:M], YS_list[M:]

            lambdas_w = [np.sqrt(5 * np.log(p) / nk)  for nk in n_list[:K]]
            lambdas_del = [np.sqrt(1 * np.log(p) / n0) ] * K
            lambda_c = np.sqrt(np.log(p) / n_tilde) * coeff[M]

            # Unsharable sources
            unsh_w_hat_list = calculate_unshare_w_hat(unsh_XS_list, unsh_YS_list, lambdas_w[M:])
            n = sum(n_list[: M]) + n0
            Y = np.concatenate(share_YS_list + [Y0])
            Sigma = block_diag(*SigmaS_list, Sigma0)
            
            X_syn_list = []
            for k in range(K - M):
                X_syn = gen_X_syn(X0, r, n_list[k + M])
                X_syn_list.append(X_syn)
            
            w_hat_list, del_hat_list, beta_hat = COMMUTE(share_XS_list, share_YS_list, X0, Y0, X_syn_list, unsh_w_hat_list, M, K, r, n_list, lambdas_w, lambdas_del, lambda_c)

            print(f"beta: {np.count_nonzero(beta_hat)}/{p} non-zero")

            for k, w in enumerate(w_hat_list, 1):
                print(f"w{k}: {np.count_nonzero(w)}/{len(w)} non-zero")
                
            for k, d in enumerate(del_hat_list, 1):
                print(f"δ{k}: {np.count_nonzero(d)}/{len(d)} non-zero")

            M_obs = [i for i in range(p) if beta_hat[i] != 0.0]
            false_positives = [i for i in M_obs if beta[i] == 0.0]

            if len(false_positives) == 0:
                continue
            
            X0M = X0[:, M_obs]

            P_list = []
            for k in range(M):
                Pk = construct_Pk(k, n_list, n)
                P_list.append(Pk)
            P0 = construct_Pk(M, n_list[:M] + [n0], n)
            P_list.append(P0)
            OBS_W_ACTIVE_SETS = [construct_active_set(w_hat_list[k], share_XS_list[k])["active_set"] for k in range(M)]
            OBS_D_ACTIVE_SETS = [construct_active_set(del_hat_list[k], X0)["active_set"] for k in range(K)]
            
            j = random.choice(false_positives)
            etaj, etajTY = construct_test_statistic(j, X0M, Y, M_obs, n0, n)
            a, b = calculate_a_b(etaj, Y, Sigma, n)
            intervals, oc_intervals = divide_and_conquer(share_XS_list, X_syn_list, X0, unsh_w_hat_list, a, b, M_obs, M, K, r, p, P_list, n_list, n, lambdas_w, lambdas_del, lambda_c, -thresold, thresold)
            p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
            p_value_oc = calculate_TN_p_value(oc_intervals, etaj, etajTY, Sigma, 0)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"⏱️ Trial {trial+1} took {end_time - start_time:.2f} seconds")
            if p_value:
                total_false_positives_detected += 1                
                if p_value >= 0 and p_value <= alpha:
                    total_false_positives_rejected += 1
                trial += 1            
            else:
                print ("p value is None!!!")
                num_err += 1

            if p_value_oc:
                oc_total_false_positives_detected += 1                
                if p_value_oc >= 0 and p_value_oc <= alpha:
                    oc_total_false_positives_rejected += 1
            else:
                print ("p value OC is None!!!")
                
        if total_false_positives_detected > 0:
            fpr = total_false_positives_rejected / total_false_positives_detected
        else:
            fpr = 0

        if oc_total_false_positives_detected > 0:
            oc_fpr = oc_total_false_positives_rejected / oc_total_false_positives_detected
        else:
            oc_fpr = 0
            
        print(f'M = {M}, FPR = {fpr:.4f} - FPR OC = {oc_fpr:.4f}')
        print (f'num err = {num_err}')

        fpr_values[M] = fpr
        oc_fpr_values[M] = oc_fpr
        print ("________________________________________________________")
    return fpr_values, oc_fpr_values


# fpr_experiment()


# import time

# def tpr_experiment():
#     global OBS_W_ACTIVE_SETS, OBS_D_ACTIVE_SETS
#     p = 500
#     s = 10
#     true_beta = 0.5
#     gamma = 0.5
#     n0_list = [100]
#     M = 2
#     K = 3
#     r = 10
#     n_list = [300, 400, 500, 100]
#     h_list = [10, 10, 10]
#     thresold = 20
#     alpha = 0.05

#     num_trials = 100
#     tpr_values = {}
#     oc_tpr_values = {}
#     num_err = 0

#     for n0 in n0_list:
#         tpr_values[n0] = []
#         oc_tpr_values[n0] = []
#         n_list[-1] = n0
#         total_true_positives_detected = 0
#         total_true_positives_rejected = 0
#         oc_total_true_positives_detected = 0
#         oc_total_true_positives_rejected = 0
#         print (f'n0: {n0}')

    
#         for i in range(1):
#             trial = 0
#             while trial < num_trials:
#                 print (f"======== trial {trial+1} ========")
#                 start_time = time.perf_counter()
#                 XS_list, YS_list, X0, Y0, true_Y, beta, SigmaS_list, Sigma0 = generate_synthetic_data(p, s, K, M, n_list, h_list, true_beta, gamma)
#                 # XS_list, YS_list, X0, Y0, true_Y, beta, n_list, SigmaS_list, Sigma0 = generate_data(p, s, nS, n0, K, M, H, true_beta, gamma)
#                 n_tilde = (n0 + sum(n_list[k] for k in range(M)) + sum(n_list[k] * r for k in range(M, K)))
#                 share_XS_list, unsh_XS_list = XS_list[:M], XS_list[M:]
#                 share_YS_list, unsh_YS_list = YS_list[:M], YS_list[M:]
    
#                 lambdas_w = [np.sqrt(2 * np.log(p) / nk)  for nk in n_list[:K]]
#                 lambdas_del = [np.sqrt(2 * np.log(p) / n0) ] * K
#                 lambda_c = np.sqrt(np.log(p) / n_tilde) * 10
#                 # Unsharable sources
#                 unsh_w_hat_list = calculate_unshare_w_hat(unsh_XS_list, unsh_YS_list, lambdas_w[M:])
#                 n = sum(n_list[: M]) + n0
#                 Y = np.concatenate(share_YS_list + [Y0])
#                 Sigma = block_diag(*SigmaS_list, Sigma0)
                
#                 X_syn_list = []
#                 for k in range(K - M):
#                     X_syn = gen_X_syn(X0, r, n_list[k + M])
#                     X_syn_list.append(X_syn)
                
#                 w_hat_list, del_hat_list, beta_hat = COMMUTE(share_XS_list, share_YS_list, X0, Y0, X_syn_list, unsh_w_hat_list, M, K, r, n_list, lambdas_w, lambdas_del, lambda_c)
    
#                 print(f"beta: {np.count_nonzero(beta_hat)}/{p} non-zero")
    
#                 for k, w in enumerate(w_hat_list, 1):
#                     print(f"w{k}: {np.count_nonzero(w)}/{len(w)} non-zero")
                    
#                 for k, d in enumerate(del_hat_list, 1):
#                     print(f"δ{k}: {np.count_nonzero(d)}/{len(d)} non-zero")
    
#                 M_obs = [i for i in range(p) if beta_hat[i] != 0.0]
#                 true_positives = [i for i in M_obs if beta[i] != 0.0]
    
#                 if len(true_positives) == 0:
#                     continue
                
#                 X0M = X0[:, M_obs]
    
#                 P_list = []
#                 for k in range(M):
#                     Pk = construct_Pk(k, n_list, n)
#                     P_list.append(Pk)
#                 P0 = construct_Pk(M, n_list[:M] + [n0], n)
#                 P_list.append(P0)
#                 OBS_W_ACTIVE_SETS = [construct_active_set(w_hat_list[k], share_XS_list[k])["active_set"] for k in range(M)]
#                 OBS_D_ACTIVE_SETS = [construct_active_set(del_hat_list[k], X0)["active_set"] for k in range(K)]
    
#                 j = random.choice(true_positives)
#                 etaj, etajTY = construct_test_statistic(j, X0M, Y, M_obs, n0, n)
#                 a, b = calculate_a_b(etaj, Y, Sigma, n)
#                 intervals, oc_intervals = divide_and_conquer(share_XS_list, X_syn_list, X0, unsh_w_hat_list, a, b, M_obs, M, K, r, p, P_list, n_list, n, lambdas_w, lambdas_del, lambda_c, -thresold, thresold)
#                 p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
#                 p_value_oc = calculate_TN_p_value(oc_intervals, etaj, etajTY, Sigma, 0)
#                 end_time = time.perf_counter()
#                 total_time = end_time - start_time
#                 print(f"⏱️ Trial {trial+1} took {end_time - start_time:.2f} seconds")
#                 if p_value:
#                     total_true_positives_detected += 1                
#                     if p_value >= 0 and p_value <= alpha:
#                         total_true_positives_rejected += 1
#                     trial += 1            
#                 else:
#                     print ("p value is None!!!")
#                     num_err += 1
    
#                 if p_value_oc:
#                     oc_total_true_positives_detected += 1                
#                     if p_value_oc >= 0 and p_value_oc <= alpha:
#                         oc_total_true_positives_rejected += 1
#                 else:
#                     print ("p value OC is None!!!")
                    
#             if total_true_positives_detected > 0:
#                 tpr = total_true_positives_rejected / total_true_positives_detected
#             else:
#                 tpr = 0
    
#             if oc_total_true_positives_detected > 0:
#                 oc_tpr = oc_total_true_positives_rejected / oc_total_true_positives_detected
#             else:
#                 oc_tpr = 0
    
#             print(f'n0 = {n0}, TPR = {tpr:.4f} - TPR OC = {oc_tpr:.4f}')
#             print (f'num err = {num_err}')
    
#             tpr_values[n0].append(tpr)
#             oc_tpr_values[n0].append(oc_tpr)
#             print ("________________________________________________________")
#         print (tpr_values, oc_tpr_values)
#     return tpr_values, oc_tpr_values


# import time

# def tpr_experiment():
#     global OBS_W_ACTIVE_SETS, OBS_D_ACTIVE_SETS
#     p = 500
#     s = 10
#     true_beta = 0.5
#     gamma = 0.5
#     n0_list = [150]
#     coeff = [30, 25,]
#     M = 2
#     K = 3
#     r = 10
#     n_list = [300, 400, 500, 100]
#     h_list = [10, 10, 10]
#     thresold = 20
#     alpha = 0.05

#     num_trials = 100
#     tpr_values = {}
#     oc_tpr_values = {}
#     num_err = 0

#     for n0 in n0_list:
#         tpr_values[n0] = []
#         oc_tpr_values[n0] = []
#         n_list[-1] = n0
#         total_true_positives_detected = 0
#         total_true_positives_rejected = 0
#         oc_total_true_positives_detected = 0
#         oc_total_true_positives_rejected = 0
#         print (f'n0: {n0}')

    
#         for i in range(10):
#             trial = 0
#             while trial < num_trials:
#                 print (f"======== trial {trial+1} ========")
#                 start_time = time.perf_counter()
#                 XS_list, YS_list, X0, Y0, true_Y, beta, SigmaS_list, Sigma0 = generate_synthetic_data(p, s, K, M, n_list, h_list, true_beta, gamma)
#                 # XS_list, YS_list, X0, Y0, true_Y, beta, n_list, SigmaS_list, Sigma0 = generate_data(p, s, nS, n0, K, M, H, true_beta, gamma)
#                 n_tilde = (n0 + sum(n_list[k] for k in range(M)) + sum(n_list[k] * r for k in range(M, K)))
#                 share_XS_list, unsh_XS_list = XS_list[:M], XS_list[M:]
#                 share_YS_list, unsh_YS_list = YS_list[:M], YS_list[M:]
    
#                 lambdas_w = [np.sqrt(5 * np.log(p) / nk)  for nk in n_list[:K]]
#                 lambdas_del = [np.sqrt(1 * np.log(p) / n0) ] * K
#                 lambda_c = np.sqrt(np.log(p) / n_tilde) * 20
#                 # Unsharable sources
#                 unsh_w_hat_list = calculate_unshare_w_hat(unsh_XS_list, unsh_YS_list, lambdas_w[M:])
#                 n = sum(n_list[: M]) + n0
#                 Y = np.concatenate(share_YS_list + [Y0])
#                 Sigma = block_diag(*SigmaS_list, Sigma0)
                
#                 X_syn_list = []
#                 for k in range(K - M):
#                     X_syn = gen_X_syn(X0, r, n_list[k + M])
#                     X_syn_list.append(X_syn)
                
#                 w_hat_list, del_hat_list, beta_hat = COMMUTE(share_XS_list, share_YS_list, X0, Y0, X_syn_list, unsh_w_hat_list, M, K, r, n_list, lambdas_w, lambdas_del, lambda_c)
    
#                 print(f"beta: {np.count_nonzero(beta_hat)}/{p} non-zero")
    
#                 for k, w in enumerate(w_hat_list, 1):
#                     print(f"w{k}: {np.count_nonzero(w)}/{len(w)} non-zero")
                    
#                 for k, d in enumerate(del_hat_list, 1):
#                     print(f"δ{k}: {np.count_nonzero(d)}/{len(d)} non-zero")
    
#                 M_obs = [i for i in range(p) if beta_hat[i] != 0.0]
#                 true_positives = [i for i in M_obs if beta[i] != 0.0]
    
#                 if len(true_positives) == 0:
#                     continue
                
#                 X0M = X0[:, M_obs]
    
#                 P_list = []
#                 for k in range(M):
#                     Pk = construct_Pk(k, n_list, n)
#                     P_list.append(Pk)
#                 P0 = construct_Pk(M, n_list[:M] + [n0], n)
#                 P_list.append(P0)
#                 OBS_W_ACTIVE_SETS = [construct_active_set(w_hat_list[k], share_XS_list[k])["active_set"] for k in range(M)]
#                 OBS_D_ACTIVE_SETS = [construct_active_set(del_hat_list[k], X0)["active_set"] for k in range(K)]
    
#                 j = random.choice(true_positives)
#                 etaj, etajTY = construct_test_statistic(j, X0M, Y, M_obs, n0, n)
#                 a, b = calculate_a_b(etaj, Y, Sigma, n)
#                 intervals, oc_intervals = divide_and_conquer(share_XS_list, X_syn_list, X0, unsh_w_hat_list, a, b, M_obs, M, K, r, p, P_list, n_list, n, lambdas_w, lambdas_del, lambda_c, -thresold, thresold)
#                 p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
#                 p_value_oc = calculate_TN_p_value(oc_intervals, etaj, etajTY, Sigma, 0)
#                 end_time = time.perf_counter()
#                 total_time = end_time - start_time
#                 print(f"⏱️ Trial {trial+1} took {end_time - start_time:.2f} seconds")
#                 if p_value:
#                     total_true_positives_detected += 1                
#                     if p_value >= 0 and p_value <= alpha:
#                         total_true_positives_rejected += 1
#                     trial += 1            
#                 else:
#                     print ("p value is None!!!")
#                     num_err += 1
    
#                 if p_value_oc:
#                     oc_total_true_positives_detected += 1                
#                     if p_value_oc >= 0 and p_value_oc <= alpha:
#                         oc_total_true_positives_rejected += 1
#                 else:
#                     print ("p value OC is None!!!")
                    
#             if total_true_positives_detected > 0:
#                 tpr = total_true_positives_rejected / total_true_positives_detected
#             else:
#                 tpr = 0
    
#             if oc_total_true_positives_detected > 0:
#                 oc_tpr = oc_total_true_positives_rejected / oc_total_true_positives_detected
#             else:
#                 oc_tpr = 0
    
#             print(f'n0 = {n0}, TPR = {tpr:.4f} - TPR OC = {oc_tpr:.4f}')
#             print (f'num err = {num_err}')
    
#             tpr_values[n0].append(tpr)
#             oc_tpr_values[n0].append(oc_tpr)
#             print ("________________________________________________________")
#         print (tpr_values, oc_tpr_values)
#     return tpr_values, oc_tpr_values


import time

def tpr_experiment():
    global OBS_W_ACTIVE_SETS, OBS_D_ACTIVE_SETS
    p = 500
    s = 10
    gamma = 0.5
    n0 = 100
    true_beta_list = [0.25]
    coeff = {0.25: 15, 0.5: 15, 0.75: 15, 1: 15}
    M = 2
    K = 3
    r = 10
    n_list = [300, 400, 500, 100]
    h_list = [10, 10, 10]
    thresold = 20
    alpha = 0.05

    num_trials = 100
    tpr_values = {}
    oc_tpr_values = {}
    num_err = 0

    for true_beta in true_beta_list:
        tpr_values[true_beta] = []
        oc_tpr_values[true_beta] = []
        n_list[-1] = n0
        total_true_positives_detected = 0
        total_true_positives_rejected = 0
        oc_total_true_positives_detected = 0
        oc_total_true_positives_rejected = 0
        print (f'true_beta: {true_beta}')

    
        for i in range(10):
            trial = 0
            while trial < num_trials:
                print (f"======== trial {trial+1} ========")
                start_time = time.perf_counter()
                XS_list, YS_list, X0, Y0, true_Y, beta, SigmaS_list, Sigma0 = generate_synthetic_data(p, s, K, M, n_list, h_list, true_beta, gamma)
                # XS_list, YS_list, X0, Y0, true_Y, beta, n_list, SigmaS_list, Sigma0 = generate_data(p, s, nS, n0, K, M, H, true_beta, gamma)
                n_tilde = (n0 + sum(n_list[k] for k in range(M)) + sum(n_list[k] * r for k in range(M, K)))
                share_XS_list, unsh_XS_list = XS_list[:M], XS_list[M:]
                share_YS_list, unsh_YS_list = YS_list[:M], YS_list[M:]
    
                lambdas_w = [np.sqrt(5 * np.log(p) / nk)  for nk in n_list[:K]]
                lambdas_del = [np.sqrt(1 * np.log(p) / n0)] * K
                lambda_c = np.sqrt(2 * np.log(p) / n_tilde) * coeff[true_beta]
                # Unsharable sources
                unsh_w_hat_list = calculate_unshare_w_hat(unsh_XS_list, unsh_YS_list, lambdas_w[M:])
                n = sum(n_list[: M]) + n0
                Y = np.concatenate(share_YS_list + [Y0])
                Sigma = block_diag(*SigmaS_list, Sigma0)
                
                X_syn_list = []
                for k in range(K - M):
                    X_syn = gen_X_syn(X0, r, n_list[k + M])
                    X_syn_list.append(X_syn)
                
                w_hat_list, del_hat_list, beta_hat = COMMUTE(share_XS_list, share_YS_list, X0, Y0, X_syn_list, unsh_w_hat_list, M, K, r, n_list, lambdas_w, lambdas_del, lambda_c)
    
                print(f"beta: {np.count_nonzero(beta_hat)}/{p} non-zero")
    
                for k, w in enumerate(w_hat_list, 1):
                    print(f"w{k}: {np.count_nonzero(w)}/{len(w)} non-zero")
                    
                for k, d in enumerate(del_hat_list, 1):
                    print(f"δ{k}: {np.count_nonzero(d)}/{len(d)} non-zero")
    
                M_obs = [i for i in range(p) if beta_hat[i] != 0.0]
                true_positives = [i for i in M_obs if beta[i] != 0.0]
    
                if len(true_positives) == 0:
                    continue
                
                X0M = X0[:, M_obs]
    
                P_list = []
                for k in range(M):
                    Pk = construct_Pk(k, n_list, n)
                    P_list.append(Pk)
                P0 = construct_Pk(M, n_list[:M] + [n0], n)
                P_list.append(P0)
                OBS_W_ACTIVE_SETS = [construct_active_set(w_hat_list[k], share_XS_list[k])["active_set"] for k in range(M)]
                OBS_D_ACTIVE_SETS = [construct_active_set(del_hat_list[k], X0)["active_set"] for k in range(K)]
    
                j = random.choice(true_positives)
                etaj, etajTY = construct_test_statistic(j, X0M, Y, M_obs, n0, n)
                a, b = calculate_a_b(etaj, Y, Sigma, n)
                intervals, oc_intervals = divide_and_conquer(share_XS_list, X_syn_list, X0, unsh_w_hat_list, a, b, M_obs, M, K, r, p, P_list, n_list, n, lambdas_w, lambdas_del, lambda_c, -thresold, thresold)
                p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
                p_value_oc = calculate_TN_p_value(oc_intervals, etaj, etajTY, Sigma, 0)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                print(f"⏱️ Trial {trial+1} took {end_time - start_time:.2f} seconds")
                if p_value:
                    total_true_positives_detected += 1                
                    if p_value >= 0 and p_value <= alpha:
                        total_true_positives_rejected += 1
                    trial += 1            
                else:
                    print ("p value is None!!!")
                    num_err += 1
    
                if p_value_oc:
                    oc_total_true_positives_detected += 1                
                    if p_value_oc >= 0 and p_value_oc <= alpha:
                        oc_total_true_positives_rejected += 1
                else:
                    print ("p value OC is None!!!")
                    
            if total_true_positives_detected > 0:
                tpr = total_true_positives_rejected / total_true_positives_detected
            else:
                tpr = 0
    
            if oc_total_true_positives_detected > 0:
                oc_tpr = oc_total_true_positives_rejected / oc_total_true_positives_detected
            else:
                oc_tpr = 0
    
            print(f'M = {M}, TPR = {tpr:.4f} - TPR OC = {oc_tpr:.4f}')
            print (f'num err = {num_err}')
    
            tpr_values[true_beta].append(tpr)
            oc_tpr_values[true_beta].append(oc_tpr)
            print ("________________________________________________________")
        print (tpr_values, oc_tpr_values)
    return tpr_values, oc_tpr_values


tpr_experiment()







