import numpy as np
import decimal
import time 
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

decimal.getcontext().prec = 10

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_numpy_dtype_promotion', 'standard')

def memory_function():
        # Declare the ranges we work with and then generate all combinations 
        r_range = jnp.linspace(start=1.05, stop=1.05, num=1, endpoint=True, dtype=jnp.float64)
        beta_range = jnp.linspace(start=0.945, stop=0.945, num=1, endpoint=True, dtype=jnp.float64)
        rho_range = jnp.linspace(start=0.9, stop=0.9, num=1, endpoint=True, dtype=jnp.float64)
        stdu_range = jnp.linspace(start=0.01, stop=0.01, num=1, endpoint=True, dtype=jnp.float64)
        m_range = jnp.linspace(start=1., stop=1., num=1, endpoint=True, dtype=jnp.float64)
        gammac_range = jnp.linspace(start=1., stop=1., num=1, endpoint=True, dtype=jnp.float64)

        parameter_combos = jnp.array(jnp.meshgrid(r_range, beta_range, rho_range, stdu_range, m_range, gammac_range), dtype=jnp.float64).T.reshape(-1,6)
        r,beta,rho,stdu,m,gammac = np.ravel(parameter_combos[0])

        nz = 11  # Income grid in terms of z=log(Y)
        nb = 200  # Grid points for debt
        nz_range = range(0,nz,1)
        nb_range = range(0,nb,1)

        # Rowenhorst Discretization
        mu_eps = 0
        q = (rho+1) / 2
        nu = ((nz-1)/(1-rho**2))**(1/2) * stdu
        log_y = np.linspace(mu_eps/(1-rho)-nu, mu_eps/(1-rho)+nu, nz)
    
        # Looks like we are always using a 2 by 2
        p = np.array(((q, 1-q), (1-q, q)))

        for i in range (1,nz-1):
            a1 = np.pad(array=p,pad_width=((0,1),(0,1)),mode='constant',constant_values=0)
            a2 = np.pad(array=p,pad_width=((1,0),(1,0)),mode='constant',constant_values=0)
            a3 = np.pad(array=p,pad_width=((0,1),(1,0)),mode='constant',constant_values=0)
            a4 = np.pad(array=p,pad_width=((1,0),(0,1)),mode='constant',constant_values=0)
            p = q * (a1 + a2) + (1-q) * (a3 + a4)
            p[1:i+1,:] /= 2
        ygrid = jnp.exp(log_y)

        # Borrowing Grid
        b_min = 0.75 * m
        b_max = m * ygrid[-1]
        bgrid = jnp.linspace(b_min, b_max, nb)

        # Initial guess
        z_m, b_m = jnp.meshgrid(ygrid,bgrid)
        z_m, b_m = z_m.T, b_m.T
        c_dec_old = jnp.maximum(1e-100, -r * b_m + (1 + m) * z_m)
        b_dec_use = c_dec_old + r * b_m - z_m

        combo = jnp.array(np.meshgrid(ygrid.T, bgrid), dtype=jnp.float64).T.reshape(-1,2).reshape(nz,nb,2)
        ygrid_search = jnp.tile(ygrid[:, jnp.newaxis, jnp.newaxis], (1, nz, nb))
        p = jnp.repeat(p.T[:, :, np.newaxis], 200, axis=2)

        count = 0
        dist = 100
        nz,nb = b_dec_use.shape

        while dist > 1e-10: 
                count += 1
                c_dec_use = b_dec_use - r * b_m + z_m

                x1i = jnp.repeat(b_dec_use.T,11,axis=1).T.reshape(nz,nz,nb)
                int_ct_prime = jnp.empty((1, nz), dtype=np.float64)
                c_dec_new = b_dec_new = jnp.empty((nz, nb), dtype=np.float64)
                w1 = w2 = jnp.empty((1,nz), dtype=np.float64)
                pf = jnp.empty(nz*4, dtype=np.float64)

                z_use = combo[:,:,0]
                b_use = combo[:,:,1]
                left1 = jnp.searchsorted(bgrid, x1i) - 1
                left2 = jnp.searchsorted(ygrid, ygrid_search)
                left1 = jnp.where(left1 >= nb-2, nb-2, left1)
                left2 = jnp.where(left2 >= nz-2, nz-2, left2)

                left_values = jnp.concatenate((bgrid[left1], ygrid[left2]))
                right_values = jnp.concatenate((bgrid[left1+1], ygrid[left2+1]))

                left_values2 = jnp.stack((bgrid[left1], ygrid[left2]), axis=0).reshape(nz*2,nz,nb)
                right_values2 = jnp.stack((bgrid[left1+1], ygrid[left2+1]), axis=0).reshape(nz*2,nz,nb)

                xi = jnp.concatenate((x1i, ygrid_search))
                w1 = (right_values - xi) / (right_values - left_values)
                w2 = 1 - w1

                w11 = jnp.concatenate((w1[0:nz,:,:], w2[0:nz,:,:], w1[0:nz,:,:], w2[0:nz,:,:]))
                w22 = jnp.concatenate((w1[nz:len(w1),:,:], w1[nz:len(w1),:,:], w2[nz:len(w2),:,:], w2[nz:len(w2),:,:]))
                p_x = jnp.concatenate((left2, left2, left2+1, left2+1))
                p_y = jnp.concatenate((left1, left1+1, left1, left1+1))
                w_new = w11 * w22 * c_dec_use[p_x, p_y]
                w_new = w_new.reshape(4,11,nz,nb)
                int_ct_prime = jnp.sum(w_new, axis=0)
                int_ct_prime= jnp.maximum(int_ct_prime, 1e-20) ** (-gammac)

                sol_ct2 = jnp.sum(p * int_ct_prime, axis=0)
                c_dec_new = (beta * r * sol_ct2) ** (-1 / gammac)
                b_dec_new = c_dec_new - z_use + r * b_use

                check_array = m * z_use
                b_dec_new = jnp.where(b_dec_new > check_array, check_array, b_dec_new)
                b_dec_new2 = b_dec_new * 0.25 + b_dec_use * (1 - 0.25)
                dist = jnp.linalg.norm(jnp.abs(b_dec_new - b_dec_use))
                if count % 10 == 0:
                        print(count,dist)
                b_dec_use = b_dec_new2