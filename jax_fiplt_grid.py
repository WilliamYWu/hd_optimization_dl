import numpy as np
from numba import njit, float64, int64, prange, types, set_num_threads
import decimal
import time
import threading
import psutil
import concurrent.futures
set_num_threads(4)
decimal.getcontext().prec = 10


@njit(types.UniTuple(float64[:,:],2)(float64[:], float64[:]), fastmath=True, cache=True)
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size))
    yy = np.empty(shape=(x.size, y.size))
    for j in range(x.size):
        for k in range(y.size):
            xx[j,k] = x[j]
            yy[j,k] = y[k]
    return yy, xx

@njit(types.UniTuple(int64[:],2)(int64[:], int64[:], int64, int64),fastmath=True, cache=True)
def index_finder(left1, left2, nb, nz):
    left1 = np.where(left1 >= nb-2, nb-2, left1)
    left2 = np.where(left2 >= nz-2, nz-2, left2)
    return left1, left2

@njit((float64[:,:])(float64[:,:], int64[:,:]), fastmath=True, cache=True)
def pad_array(array, pad_shape):
    m, n = array.shape 
    pm, pn = m + 1, n + 1
    padded_array = np.zeros((pm, pn))

    pad_top, pad_bottom = pad_shape[1]
    pad_left, pad_right = pad_shape[0]

    start_m = pad_top if pad_top else 0
    end_m = m if not pad_bottom else m - pad_bottom

    start_n = pad_left if pad_left else 0
    end_n = n if not pad_right else n - pad_right

    padded_array[start_m:end_m + 1, start_n:end_n + 1] = array[:end_m - start_m + 1, :end_n - start_n + 1]

    return padded_array

@njit(types.Tuple((float64[:],float64[:,:]))(float64, float64, int64), fastmath=True, cache=True)
def rowenhorst(rho, sigma_eps, n):
    mu_eps = 0
    q = (rho+1) / 2
    nu = ((n-1)/(1-rho**2))**(1/2) * sigma_eps
    z_grid = np.linspace(mu_eps/(1-rho)-nu, mu_eps/(1-rho)+nu, n)

    # Looks like we are always using a 2 by 2
    P = np.array(((q, 1-q), (1-q, q)))

    for i in range (1,n-1):
        a1 = pad_array(P,np.array([[0,1],[0,1]]))
        a2 = pad_array(P,np.array([[1,0],[1,0]]))
        a3 = pad_array(P,np.array([[0,1],[1,0]]))
        a4 = pad_array(P,np.array([[1,0],[0,1]]))
        P = q * (a1 + a2) + (1-q) * (a3 + a4)
        P[1:i+1,:] /= 2
    return z_grid, P

@njit((float64[:])(float64[:,:], int64[:], int64[:]), fastmath=True, cache=True)
def get_value(array,x,y):
    index = x * array.shape[1] + y
    tmp = array.ravel()
    tmp = tmp[index]
    return tmp

@njit((float64[:,:])(float64[:],float64[:,:],float64[:],float64[:,:],float64[:,:],int64[:,:],float64, int64, int64, float64), 
      fastmath=True, 
      cache=True, 
      parallel=True
      )
def solveFiPit(bgrid, p, ygrid, b_dec_use, c_dec_use, combo, beta, gammac, m, r):
    nz,nb = b_dec_use.shape
    int_ct_prime = np.empty((1, nz), dtype=np.float64)
    c_dec_new = np.empty((nz, nb), dtype=np.float64)
    b_dec_new = np.empty((nz, nb), dtype=np.float64)
    w1 = np.empty((1,nz), dtype=np.float64)
    w2 = np.empty((1,nz), dtype=np.float64)
    pf = np.empty(nz*4, dtype=np.float64)
    for i in prange(len(combo)):
        iz = combo[i,0]
        ib = combo[i,1]
        b_use = bgrid[ib]
        z_use = ygrid[iz]
        b_prime = b_dec_use[iz, ib]

        if b_prime < bgrid[0]:
            b_prime = bgrid[0]

        x1i = b_dec_use[0:nz, ib]
        left1 = np.searchsorted(bgrid, x1i) - 1
        left2 = np.searchsorted(ygrid, ygrid)
        left1, left2 = index_finder(left1, left2, nb, nz)
        
        left1_values = bgrid[left1]
        right1_values = bgrid[left1+1]
        left2_values = ygrid[left2]
        right2_values = ygrid[left2+1]

        left_values = np.concatenate((left1_values, left2_values))
        right_values = np.concatenate((right1_values, right2_values))
        
        xi = np.concatenate((x1i, ygrid))
        w1 = (right_values - xi) / (right_values - left_values)
        w2 = 1 - w1

        w11 = np.concatenate((w1[0:nz], w2[0:nz], w1[0:nz], w2[0:nz]))
        w22 = np.concatenate((w1[nz:len(w1)], w1[nz:len(w1)], w2[nz:len(w2)], w2[nz:len(w2)]))
        p_x = np.concatenate((left2, left2, left2+1, left2+1))
        p_y = np.concatenate((left1, left1+1, left1, left1+1))
        pf = get_value(c_dec_use, p_x, p_y)
        w_new = w11 * w22 * pf
        w_new = w_new.reshape(4,11)
        int_ct_prime = np.sum(w_new, axis=0)
        int_ct_prime= np.maximum(int_ct_prime, 1e-20) ** (-gammac)

        sol_ct2 = np.ravel(p[iz, :]) @ int_ct_prime
        c_dec_new[iz, ib] = (beta * r * sol_ct2) ** (-1 / gammac)
        b_dec_new[iz, ib] = c_dec_new[iz, ib] - z_use + r * b_use
        if b_dec_new[iz, ib] > m * z_use:
                b_dec_new[iz, ib] = m * z_use
    return b_dec_new

@njit((int64)(float64[:], int64[:,:], int64, int64), fastmath=True, cache=True)
def parameter_grid_search(parameter_combo, combo, nz, nb):

    r,beta,rho,stdu,m,gammac = np.ravel(parameter_combo)

    # Rowenhorst Discretization
    log_y, p = rowenhorst(rho=rho, sigma_eps=stdu, n=nz)

    ygrid = np.exp(log_y)

    # Borrowing Grid
    b_min = 0.75 * m
    b_max = m * ygrid[-1]
    bgrid = np.linspace(b_min, b_max, nb)

    # Initial guess
    z_m, b_m = meshgrid(bgrid,ygrid)
    z_m, b_m = z_m.T, b_m.T
    c_dec_old = np.maximum(1e-100, -r * b_m + (1 + m) * z_m)
    b_dec_use = c_dec_old + r * b_m - z_m

    count = 1
    dist = 100
    
    while dist > 1e-10:
        count += 1
        c_dec = b_dec_use - r * b_m + z_m
        b_dec_new = solveFiPit(
            bgrid=bgrid, 
            p=p, 
            ygrid=ygrid, 
            b_dec_use=b_dec_use, 
            c_dec_use = c_dec,
            combo=combo,
            beta = beta,
            gammac = gammac,
            m = m,
            r = r,
            )
        b_dec_new2 = b_dec_new * 0.25 + b_dec_use * (1 - 0.25)
        dist = np.linalg.norm(np.abs(b_dec_new - b_dec_use))
        # if count % 10 == 0:
        #     print(count,dist)
        b_dec_use = b_dec_new2
    return count

def parallel_grid_search(parameter_combo, combo, nz, nb):
    start_time = time.time()
    count = parameter_grid_search(
        parameter_combo=parameter_combo,
        combo=combo,
        nz=nz,
        nb=nb,
        )
    end_time = time.time()
    time_passed = end_time - start_time
    print(f"Iterations: {count}, Total time passed: {time_passed} sec")

if __name__ == "__main__":
    # Model Parameters
    # r = 1.05
    # beta = 0.945
    # rho = 0.9
    # stdu = 0.010
    # m = 1   
    # gammac = 1
    
    r_range = np.arange(1,1.06,0.01)
    beta_range = np.arange(0.940,0.946,0.001)
    rho_range = np.arange(0.85,0.92,0.01)
    stdu_range = np.arange(0.01,0.02,0.01)
    m_range = np.arange(1,2,1)
    gammac_range = np.arange(1,2,1)

    parallel_grid = 1
    # r_range = np.arange(1.05,1.06,0.05)
    # beta_range = np.arange(0.945,0.946,0.005)
    # rho_range = np.arange(0.9,0.91,0.01)
    # stdu_range = np.arange(0.01,0.20,0.01)
    # m_range = np.arange(1,2,5)
    # gammac_range = np.arange(1,2,5)
    parameter_combos = np.array(np.meshgrid(r_range, beta_range, rho_range, stdu_range, m_range, gammac_range)).T.reshape(-1,6)
    
    # Grid for Decision Rules and Integration Nodes
    nz = 11  # Income grid in terms of z=log(Y)
    nb = 200  # Grid points for debt

    nb_range = range(0,nb,1)
    nz_range = range(0,nz,1)
    combo = np.array(np.meshgrid(nz_range, nb_range)).T.reshape(-1,2)

    total_start_time = time.time()
    if parallel_grid:
        # threads = []
        # for param_combo in parameter_combos:
        #     thread = threading.Thread(target=parallel_grid_search, args=(param_combo, combo, nz, nb))
        #     threads.append(thread)
        #     thread.start()

        # for thread in threads:
        #     thread.join()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Map the combinations to the worker threads, passing additional parameters
            results = list(executor.map(parallel_grid_search, parameter_combos, [combo]*len(parameter_combos), [nz]*len(parameter_combos), [nb]*len(parameter_combos)))

    else:
        for i in range(len(parameter_combos)):
            start_time = time.time()
            count = parameter_grid_search(
                parameter_combo=parameter_combos[i],
                combo=combo,
                nz=nz,
                nb=nb,
                )
            end_time = time.time()
            time_passed = end_time - start_time
            print(f"Combo: {i}, Iterations: {count}, Total time passed: {time_passed} sec")
    total_end_time = time.time()
    total_passed_time = total_end_time - total_start_time
    print(f"Total time passed: {total_passed_time}")
    