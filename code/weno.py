import numpy as np 
import matplotlib.pyplot as plt 

#set up the spatial and time grids
x_min = 0
x_max = 2
dx = .01
x_grid = np.arange(x_min, x_max, dx)

t_min = 0
t_max = 5.1 
dt = 0.01 
t_grid = np.arange(t_min, t_max, dt)

epsilon = 1e-6

k = 3
c_rj = [ # from Table 2.1 in Shu's lecture notes
    [1/3,5/6,-1/6],
    [-1/6,5/6,1/3],
    [1/3,-7/6,11/6],
]

c_rj_tilde = [ # from Table 2.1 in Shu's lecture notes
    [11/6,-7/6,1/3],
    [1/3,5/6,-1/6],
    [-1/6,5/6,1/3],
]

d_r = [3/10, 3/5, 1/10] # between eqs 2.54 and 2.55 
d_r_tilde = [d_r[2], d_r[1], d_r[0]] # d_tilde[r] = d[k-1-r]

def f(u):
    return 0.5 * u**2

def weno_flux(u):
    N = len(x_grid)
    flux = np.zeros(N)
    f_hat = np.zeros(N)
    for i in range(2,N-2):

        #computing Roe speed
        a_ip12 = (f(u[i+1])-f(u[i]))/(u[i+1]-u[i])

        # WENO procedure
        vr_i12_pos = np.zeros(k)
        vr_i12_neg = np.zeros(k)
        for r in range(k):
            for j in range(k):
                vr_i12_pos[r]+= c_rj[r][j]*u[i-r+j]
                vr_i12_neg[r]+= c_rj_tilde[r][j]*u[i-r+j]

        v_i12_pos = 0
        v_i12_neg = 0
        for r in range(k):
            v_i12_pos += d_r[r]*vr_i12_pos[r]
            v_i12_neg += d_r_tilde[r]*vr_i12_neg[r]

        beta0 = 13/12 * (u[i] - 2*u[i+1] + u[i+2])**2 + 1/4 * (3*u[i] - 4*u[i+1] + u[i+2])
        beta1 = 13/12 * (u[i-1] - 2*u[i] + u[i+1])**2 + 1/4 * (u[i-1] - u[i+1])
        beta2 = 13/12 * (u[i-2] - 2*u[i-1] + u[i])**2 + 1/4 * (u[i-2] - 4*u[i-1] + 3*u[i])
        beta_r = [beta0, beta1, beta2]

        alpha_r = np.zeros(k)
        alpha_r_tilde = np.zeros(k)
        for r in range(k):
            alpha_r[r] = d_r[r]/(epsilon+beta_r[r])**2
            alpha_r_tilde[r] = d_r_tilde[r]/(epsilon+beta_r[r])**2

        w_r = np.zeros(k)
        w_r_tilde = np.zeros(k)
        for r in range(k):
            w_r[r] = alpha_r[r]/np.sum(alpha_r)
            w_r_tilde[r] = alpha_r_tilde[r]/np.sum(alpha_r_tilde)

        v_neg_i12 = 0
        v_pos_i12 = 0

        for r in range(k):
            v_neg_i12 += w_r[r]*vr_i12_pos[r]
            v_pos_i12 += w_r_tilde[r]*vr_i12_neg[r]

        # assigning numerical flux based on sign of the Roe speed
        if a_ip12 >= 0:
            f_hat[i] = v_neg_i12
        else:
            f_hat[i] = v_pos_i12

        flux[i] = -1/dx * (f_hat[i]-f_hat[i-1])

    return flux

def rk4(ux_initial):
    u_grid = [ux_initial]
    for i in range(len(t_grid)-1):
        k1= weno_flux(u_grid[i])
        k2= weno_flux(u_grid[i]+dt*k1/2)
        k3= weno_flux(u_grid[i]+dt*k2/2)
        k4= weno_flux(u_grid[i]+dt*k3)

        u = u_grid[i]+(k1+2*k2+2*k3+k4)*dt/6   
        u_grid.append(u)
    return(u_grid)


def u_initial(x):
    return np.sin(np.pi*x)

u_grid =  rk4(u_initial(x_grid),)

for i in range(len(t_grid)):
    plt.clf()
    plt.plot(x_grid,u_grid[i],label='t={:.2f}'.format(t_grid[i]))
    plt.legend()
    plt.pause(0.00000000005)

