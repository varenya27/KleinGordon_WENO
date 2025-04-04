import numpy as np 
import matplotlib.pyplot as plt 

class circularArray(np.ndarray):
    def __new__(cls, *args) :
        return np.asarray(args[0]).view(cls)
    
    def __getitem__(self, index):
        return np.ndarray.__getitem__(self, index%(len(self)))

def flux_at_cell_boundaries(f:circularArray, split:bool, epsilon:float):
    """
    returns: np.array of length N+1 containing flux values for the cell boundaries from -1/2 to N-1/2
    f: array of flux values for each flux 
    split: True if positive flux
           False if negative flux
    """
    N = len(f)
    f_hat = np.zeros(N+1)
    if split:
        fp = circularArray(f)
        for j in range(N+1):
            beta_0 = 13/12 * (fp[j-3]-2*fp[j-2]+fp[j-1])**2 + 1/4 * (fp[j-3]-4*fp[j-2]+3*fp[j-1])**2
            beta_1 = 13/12 * (fp[j-2]-2*fp[j-1]+fp[j])**2 + 1/4 * (fp[j-2]-fp[j])**2
            beta_2 = 13/12 * (fp[j-1]-2*fp[j]+fp[j+1])**2 + 1/4 * (3*fp[j-1]-4*fp[j]+fp[j+1])**2

            alpha_0 = 1/10 * (1/(epsilon+beta_0))**2
            alpha_1 = 6/10 * (1/(epsilon+beta_1))**2
            alpha_2 = 3/10 * (1/(epsilon+beta_2))**2
            sum_alpha = alpha_0+alpha_1+alpha_2

            w_0 = alpha_0/sum_alpha
            w_1 = alpha_1/sum_alpha
            w_2 = alpha_2/sum_alpha
            
            f_hat[j] = (
                w_0*(2/6*fp[j-3] - 7/6*fp[j-2] + 11/6*fp[j-1]) 
                + w_1*(-1/6*fp[j-2] + 5/6*fp[j-1] + 2/6*fp[j]) 
                + w_2*(2/6*fp[j-1] + 5/6*fp[j] - 1/6*fp[j+1])
            )
        return f_hat
    fn = circularArray(f) 
    for j in range(N+1):
        beta_0 = 13/12 * (fn[j]-2*fn[j+1]+fn[j+2])**2 + 1/4 * (3*fn[j]-4*fn[j+1]+fn[j+2])**2
        beta_1 = 13/12 * (fn[j-1]-2*fn[j]+fn[j+1])**2 + 1/4 * (fn[j-1]+fn[j+1])**2
        beta_2 = 13/12 * (fn[j-2]-2*fn[j-1]+fn[j])**2 + 1/4 * (fn[j-2]-4*fn[j-1]+3*fn[j])**2

        alpha_0 = 1/10 * (1/(epsilon+beta_0))**2
        alpha_1 = 6/10 * (1/(epsilon+beta_1))**2
        alpha_2 = 3/10 * (1/(epsilon+beta_2))**2
        sum_alpha = alpha_0+alpha_1+alpha_2

        w_0 = alpha_0/sum_alpha
        w_1 = alpha_1/sum_alpha
        w_2 = alpha_2/sum_alpha
        
        f_hat[j] = (
            w_2*(-1/6*fn[j-2] + 5/6*fn[j-1] + 2/6*fn[j]) 
            + w_1*(2/6*fn[j-1] + 5/6*fn[j] - 1/6*fn[j+1]) 
            + w_0*(11/6*fn[j] - 7/6*fn[j+1] + 2/6*fn[j+2])
        )
    return f_hat

def flux_at_cell_boundaries_modulos(f, split:bool, epsilon:float):
    """
    returns: np.array of length N+1 containing flux values for the cell boundaries from -1/2 to N-1/2
    f: array of flux values for each flux 
    split: True if positive flux
           False if negative flux
    """
    N = len(f)
    f_hat = np.zeros(N+1)
    if split:
        fp = f
        for j in range(N+1):
            beta_0 = 13/12 * (fp[j-3] - 2*fp[j-2]   + fp[j-1])**2     + 1/4 * (  fp[j-3] - 4*fp[j-2] + 3*fp[j-1]    )**2
            beta_1 = 13/12 * (fp[j-2] - 2*fp[j-1]   + fp[(j)%N])**2   + 1/4 * (  fp[j-2] -   fp[(j)%N]              )**2
            beta_2 = 13/12 * (fp[j-1] - 2*fp[(j)%N] + fp[(j+1)%N])**2 + 1/4 * (3*fp[j-1] - 4*fp[j%N] +   fp[(j+1)%N])**2
            # print(beta_0, beta_1, beta_2)

            alpha_0 = 1/10 * (1/(epsilon+beta_0))**2
            alpha_1 = 6/10 * (1/(epsilon+beta_1))**2
            alpha_2 = 3/10 * (1/(epsilon+beta_2))**2
            sum_alpha = alpha_0+alpha_1+alpha_2

            w_0 = alpha_0/sum_alpha
            w_1 = alpha_1/sum_alpha
            w_2 = alpha_2/sum_alpha
            
            f_hat[j] = (
                w_0*(2/6*f[j-3] - 7/6*f[j-2] + 11/6*f[j-1]) +
                w_1*(-1/6*f[j-2] + 5/6*f[j-1] + 2/6*f[j%N]) +
                w_2*(2/6*f[j-1] + 5/6*f[j%N] - 1/6*f[(j+1)%N])
            )
        return f_hat
    fn = f 
    for j in range(N+1):
        beta_0 = 13/12 * (fn[(j)%N] - 2*fn[(j+1)%N] + fn[(j+2)%N])**2 + 1/4 * (3*fn[(j)%N] - 4*fn[(j+1)%N] +   fn[(j+2)%N])**2
        beta_1 = 13/12 * (fn[j-1]   - 2*fn[(j)%N]   + fn[(j+1)%N])**2 + 1/4 * (  fn[j-1]   -   fn[(j+1)%N]                )**2
        beta_2 = 13/12 * (fn[j-2]   - 2*fn[j-1]     + fn[(j)%N]  )**2 + 1/4 * (  fn[j-2]   - 4*fn[j-1]     + 3*fn[j%N]    )**2

        alpha_0 = 1/10 * (1/(epsilon+beta_0))**2
        alpha_1 = 6/10 * (1/(epsilon+beta_1))**2
        alpha_2 = 3/10 * (1/(epsilon+beta_2))**2
        sum_alpha = alpha_0+alpha_1+alpha_2

        w_0 = alpha_0/sum_alpha
        w_1 = alpha_1/sum_alpha
        w_2 = alpha_2/sum_alpha
        
        f_hat[j] = (
            w_2*(-1/6*f[j-2] + 5/6*f[j-1] + 2/6*f[(j)%N]) +
            w_1*(2/6*f[j-1] + 5/6*f[(j)%N] - 1/6*f[(j+1)%N]) +
            w_0*(11/6*f[(j)%N] - 7/6*f[(j+1)%N] + 2/6*f[(j+2)%N])
        )
    return f_hat

def flux(u):
    return 0.5*u**2

def rhs_f(u):
    # initialize flux for given u
    f = flux(u)

    # Lax-Friedrichs splitting
    fp= 0.5*(f + np.max(np.abs(u))*u)
    fn= 0.5*(f - np.max(np.abs(u))*u)
    
    # compute flux values at cell boundaries
    fhat_p = flux_at_cell_boundaries((fp), True, epsilon)
    fhat_n = flux_at_cell_boundaries((fn), False, epsilon)

    # spatial derivatives for each cell
    fx_p = 1/dx * (fhat_p[1:] - fhat_p[:-1])
    fx_n = 1/dx * (fhat_n[1:] - fhat_n[:-1])

    return -(fx_p+fx_n)

def rk4(t_grid, u_i):
    u = [u_i]
    dt = t_grid[1]-t_grid[0]
    n = len(t_grid)

    for j in range(n):
        k1 = rhs_f(u[j])
        k2 = rhs_f(u[j] + dt/2 * k1)
        k3 = rhs_f(u[j] + dt/2 * k2)
        k4 = rhs_f(u[j] + dt * k3)

        unext = u[j] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        u.append(unext)
    return u

#set up the spatial and time grids
x_min = 0
x_max = 2
dx = .01
x_grid = np.arange(x_min, x_max, dx)
N = len(x_grid)

t_min = 0
t_max = 1.1
dt = 0.001 
t_grid = np.arange(t_min, t_max, dt)

epsilon = 1e-6

u_i = 9+np.sin(np.pi*x_grid)

u_grid = rk4(t_grid, u_i)

for i in range(len(t_grid)):
    plt.clf()
    plt.plot(x_grid,u_grid[0],label='initial conditions')
    plt.plot(x_grid,u_grid[i],label='t={:.2f}'.format(t_grid[i]))
    plt.legend()
    plt.pause(0.00005)







