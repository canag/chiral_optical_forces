'''
Ensemble of tool functions to perform calculations on vector and scalar fields maps
and to derive adimensional quantities related to optical fields and related forces.
'''
import numpy as np
import matplotlib.pyplot as plt


# define a 3D grid

def define_3Dgrid(grid_range=10, grid_step = 0.1):
    '''
    function that defines the grid for the 3D maps
    x, y and z all have shape (Nx, Ny, Nz) 
    '''

    x, y, z = np.meshgrid(np.arange(-grid_range/2, grid_range/2, grid_step),
                          np.arange(-grid_range/2, grid_range/2, grid_step),
                          np.arange(-grid_range/2, grid_range/2, grid_step),
                          indexing='ij')

    print("Maps of size (Nx, Ny, Nz) =", x.shape)

    return x, y, z


# define a planar wave going in the z direction

def define_planar_wave(x, y, z, pol='X'):
    '''
    function that returns the adimensional E and H fields
    for a planar wave going in the +z direction 
    with a specific polarization 
    '''

    if pol=='X':
        Ex = np.exp(1j * z)
        Ey = 0 * x
        Ez = 0 * x
        Hx = 0 * x
        Hy = np.exp(1j * z)
        Hz = 0 * x
    elif pol=='Y':
        Ex = 0 * x
        Ey = np.exp(1j * z)
        Ez = 0 * x
        Hx = - np.exp(1j * z)
        Hy = 0 * x
        Hz = 0 * x
    elif pol=='L':
        Ex = np.exp(1j * z) / np.sqrt(2)
        Ey = - 1j * np.exp(1j * z) / np.sqrt(2)
        Ez = 0 * x
        Hx = 1j * np.exp(1j * z) / np.sqrt(2)
        Hy = np.exp(1j * z) / np.sqrt(2)
        Hz = 0 * x
    elif pol=='R':
        Ex = np.exp(1j * z) / np.sqrt(2)
        Ey = 1j * np.exp(1j * z) / np.sqrt(2)
        Ez = 0 * x
        Hx = -1j * np.exp(1j * z) / np.sqrt(2)
        Hy = np.exp(1j * z) / np.sqrt(2)
        Hz = 0 * x
    else:
        print("Issue: polarization must be among X, Y, L, R.")

    E = np.stack([Ex, Ey, Ez])
    H = np.stack([Hx, Hy, Hz])
    print("E and H have shape", E.shape)
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
   

def plot_vectorfield_zx(A, x, y, z):
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

    M = np.real(A[0, :, ind_med, :])
    pcm = axs[0].imshow(M, cmap=mycmap, vmin=-np.max(abs(M)), vmax=np.max(abs(M)),
                        interpolation='spline16', extent=[x.min(), x.max(), z.min(), z.max()])
    fig.colorbar(pcm, ax=axs[0])

    M = np.real(A[1, :, ind_med, :])
    pcm = axs[1].imshow(M, cmap=mycmap, vmin=-np.max(abs(M)), vmax=np.max(abs(M)),
                        interpolation='spline16', extent=[x.min(), x.max(), z.min(), z.max()])
    fig.colorbar(pcm, ax=axs[1])

    M = np.real(A[2, :, ind_med, :])
    pcm = axs[2].imshow(M, cmap=mycmap, vmin=-np.max(abs(M)), vmax=np.max(abs(M)),
                        interpolation='spline16', extent=[x.min(), x.max(), z.min(), z.max()])
    fig.colorbar(pcm, ax=axs[2])

    axs[0].set(xlabel='kz', ylabel='kx')
    axs[1].set(xlabel='kz', ylabel='kx')
    axs[2].set(xlabel='kz', ylabel='kx')
    plt.show()
    pass


def plot_scalarfield_zx(A, x, y, z):
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

    M = np.real(A[:, ind_med, :])
    plt.imshow(M, cmap='bwr', vmin=-np.max(abs(M)), vmax=np.max(abs(M)),
               interpolation='spline16', extent=[x.min(), x.max(), z.min(), z.max()])
    plt.colorbar()
    ax.set(xlabel='kz', ylabel='kx')
    plt.show()
    pass
