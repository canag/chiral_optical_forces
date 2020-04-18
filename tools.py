'''
Ensemble of tool functions to perform calculations on vector and scalar fields maps
and to derive adimensional quantities related to optical fields and related forces.
'''
import numpy as np
import matplotlib.pyplot as plt


# define a 3D grid

def define_3Dgrid(grid_range=[10, 10, 10], grid_step = 0.1):
    '''
    function that defines the grid for the 3D maps
    x, y and z all have shape (Nx, Ny, Nz) 
    '''

    x, y, z = np.meshgrid(np.arange(-grid_range[0]/2, grid_range[0]/2, grid_step),
                          np.arange(-grid_range[1]/2, grid_range[1]/2, grid_step),
                          np.arange(-grid_range[2]/2, grid_range[2]/2, grid_step),
                          indexing='ij')

    print("Maps of size (Nx, Ny, Nz) =", x.shape)

    return x, y, z


# define a planar wave going in the +/-z direction

def define_planar_wave(x, y, z, pol='X', dir=1):
    '''
    function that returns the adimensional E and H fields
    for a planar wave going in the dir*z direction 
    with a specific polarization 
    '''

    # term in the exponential
    t = 1j * dir * z

    if pol=='X':
        Ex = np.exp(t)
        Ey = 0 * x
        Ez = 0 * x
        Hx = 0 * x
        Hy = dir * np.exp(t)
        Hz = 0 * x
    elif pol=='Y':
        Ex = 0 * x
        Ey = np.exp(t)
        Ez = 0 * x
        Hx = - dir * np.exp(t)
        Hy = 0 * x
        Hz = 0 * x
    elif pol=='L':
        Ex = np.exp(t) / np.sqrt(2)
        Ey = - dir * 1j * np.exp(t) / np.sqrt(2)
        Ez = 0 * x
        Hx = 1j * np.exp(t) / np.sqrt(2)
        Hy = dir * np.exp(t) / np.sqrt(2)
        Hz = 0 * x
    elif pol=='R':
        Ex = np.exp(t) / np.sqrt(2)
        Ey = dir * 1j * np.exp(t) / np.sqrt(2)
        Ez = 0 * x
        Hx = -1j * np.exp(t) / np.sqrt(2)
        Hy = dir * np.exp(t) / np.sqrt(2)
        Hz = 0 * x
    else:
        print("Issue: polarization must be among X, Y, L, R.")

    E = np.stack([Ex, Ey, Ez])
    H = np.stack([Hx, Hy, Hz])
    print("E and H have shape", E.shape)
    return E, H 


def define_planar_counterwaves(x, y, z, pol='XX'):
    '''
    function that returns the adimensional E and H fields
    for a planar wave going in the +z direction 
    with a specific polarization 
    '''

    E1, H1 = define_planar_wave(x, y, z, pol=pol[0], dir=1)
    E2, H2 = define_planar_wave(x, y, z, pol=pol[1], dir=-1)

    E = E1 + E2
    H = H1 + H2
    print("E and H have shape", E.shape)
    return E, H


# define a Gaussian (paraxial) beam going in the +/-z direction

def define_gaussian_beam(x, y, z, NA, n, grid_step, pol='X', dir=1):
    '''
    function that returns the adimensional E and H fields
    for a Gaussian beam going in the dir*z direction 
    with a specific polarization 
    '''

    # distance to z-axis (first cylinder coordinate)
    rho = np.sqrt(x**2 + y**2)

    # amplitude of the E field
    w0 =  2 * np.pi * n 
    zR = w0**2 / n

    w = w0 * np.sqrt(1 + (z/zR)**2)
    R = z * (1 + (zR/z)**2)

    zeta = np.arctan(z/zR)
    phi = z + rho**2/(2*R) - zeta

    f = w0/w * np.exp(-(rho/w)**2) * np.exp(1j*phi)

    Ex = f
    Ey = 0 * x
    Ez = 0 * x
    E = np.stack([Ex, Ey, Ez])

    if pol=='X':
        Ex = f
        Ey = 0 * x
        Ez = 0 * x
    elif pol=='Y':
        Ex = 0 * x
        Ey = f
        Ez = 0 * x
    elif pol=='L':
        Ex = f / np.sqrt(2)
        Ey = - 1j * f / np.sqrt(2)
        Ez = 0 * x
    elif pol=='R':
        Ex = f / np.sqrt(2)
        Ey = 1j * f / np.sqrt(2)
        Ez = 0 * x
    else:
        print("Issue: polarization must be among X, Y, L, R.")

    E = np.stack([Ex, Ey, Ez])
    H = -1j * curl(E, grid_step)
    print("E and H have shape", E.shape, H.shape)
    return E, H 


# low-level tool functions


def crossprod(A, B):
    '''
    function that derives the map of cross products
    of two vector field maps
    '''

    if (A.ndim!=4)|(A.shape[0]!=3):
        print("Issue: first input must have shape (3, Nx, Ny, Nz).")
    if (B.ndim!=4)|(B.shape[0]!=3):
        print("Issue: second input must have shape (3, Nx, Ny, Nz).")
    return np.cross(A, B, axis=0)


def dotprod(A, B):
    '''
    function that derives the map of dot products
    of two vector field maps
    '''

    if (A.ndim!=4)|(A.shape[0]!=3):
        print("Issue: first input must have shape (3, Nx, Ny, Nz).")
    if (B.ndim!=4)|(B.shape[0]!=3):
        print("Issue: second input must have shape (3, Nx, Ny, Nz).")
    return np.sum(A*B, axis=0)


def grad(A, grid_step):
    '''
    function that computes the map of gradient of scalar field A
    A must have shape (Nx, Ny, Nz)
    Returns a vector field map of shape (3, Nx, Ny, Nz)
    '''

    # A must have shape (Nx, Ny, Nz)
    if A.ndim!=3:
        print("Issue: first input must have shape (Nx, Ny, Nz).")
    
    return np.stack(np.gradient(A)) / grid_step


def curl(A, grid_step):
    '''
    function that computes the map of curl of vector field A
    A must have shape (3, Nx, Ny, Nz)
    Returns a vector field map of shape (3, Nx, Ny, Nz)
    '''
    if (A.ndim!=4)|(A.shape[0]!=3):
        print("Issue: first input must have shape (3, Nx, Ny, Nz).")
    
    # Jabobian matrix (first two dimensions)
    J = np.stack(np.gradient(A, axis=(1,2,3))) / grid_step
    curl = 0*A
    curl[0,:,:,:] = J[1,2,:,:,:] - J[2,1,:,:,:]
    curl[1,:,:,:] = J[2,0,:,:,:] - J[0,2,:,:,:]
    curl[2,:,:,:] = J[0,1,:,:,:] - J[1,0,:,:,:]
    return curl
   

def plot_vectorfield_zx(A, x, y, z, q='real'):
    '''
    function that plots (z, x) maps for the real parts of 
    the 3 components of a vector field int the (y=0)-plane
    '''

    # A must have shape (3, Nx, Ny, Nz)
    if (A.ndim!=4)|(A.shape[0]!=3):
        print("Issue: first input must have shape (3, Nx, Ny, Nz).")

    # ind where y is zero
    ind_med = int(A.shape[2]/2)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
    mycmap = 'bwr'

    if q=='real':
        Mx = np.real(A[0, :, ind_med, :])
        My = np.real(A[1, :, ind_med, :])
        Mz = np.real(A[2, :, ind_med, :])
    elif q=='imag':
        Mx = np.imag(A[0, :, ind_med, :])
        My = np.imag(A[1, :, ind_med, :])
        Mz = np.imag(A[2, :, ind_med, :])
    elif q=='abs':
        Mx = np.abs(A[0, :, ind_med, :])
        My = np.abs(A[1, :, ind_med, :])
        Mz = np.abs(A[2, :, ind_med, :])
    else:
        print("Issue: q must be among ['real', 'imag', 'abs']")

    pcm = axs[0].imshow(Mx, cmap=mycmap, vmin=-np.max(abs(Mx)), vmax=np.max(abs(Mx)),
                        interpolation='spline16', extent=[z.min(), z.max(), x.min(), x.max()],
                        origin='lower')
    fig.colorbar(pcm, ax=axs[0])

    pcm = axs[1].imshow(My, cmap=mycmap, vmin=-np.max(abs(My)), vmax=np.max(abs(My)),
                        interpolation='spline16', extent=[z.min(), z.max(), x.min(), x.max()],
                        origin='lower')
    fig.colorbar(pcm, ax=axs[1])

    pcm = axs[2].imshow(Mz, cmap=mycmap, vmin=-np.max(abs(Mz)), vmax=np.max(abs(Mz)),
                        interpolation='spline16', extent=[z.min(), z.max(), x.min(), x.max()],
                        origin='lower')
    fig.colorbar(pcm, ax=axs[2])

    axs[0].set(xlabel='kz', ylabel='kx')
    axs[1].set(xlabel='kz', ylabel='kx')
    axs[2].set(xlabel='kz', ylabel='kx')
    plt.show()
    pass


def plot_scalarfield_zx(A, x, y, z, q='real'):
    '''
    function that plots (z, x) maps for the real parts of 
    a scalar field in the (y=0)-plane
    '''

    # A must have shape (Nx, Ny, Nz)
    if A.ndim!=3:
        print("Issue: first input must have shape (Nx, Ny, Nz).")
    
    # ind where y is zero
    ind_med = int(A.shape[1]/2)

    _, ax = plt.subplots(figsize=(5, 3))

    if q=='real':
        M = np.real(A[:, ind_med, :])
    elif q=='imag':
        M = np.imag(A[:, ind_med, :])
    elif q=='abs':
        M = np.abs(A[:, ind_med, :])
    else:
        print("Issue: q must be among ['real', 'imag', 'abs']")
    
    plt.imshow(M, cmap='bwr', vmin=-np.max(abs(M)), vmax=np.max(abs(M)),
               interpolation='spline16', extent=[z.min(), z.max(), x.min(), x.max()],
               origin='lower')
    plt.colorbar()
    ax.set(xlabel='kz', ylabel='kx')
    plt.show()
    pass
