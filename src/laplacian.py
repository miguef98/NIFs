import torch
import numpy as np
from scipy import sparse

# 2D

def gridLaplacianOld( N ):
    h = 2 / (N-1)
    ones = np.ones( N**2 )
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)

    L = sparse.csr_matrix((N**2,N**2))
    L.setdiag( ones * - 4, k=0)
    L.setdiag( decoupled_ones, k=1 )
    L.setdiag( decoupled_ones, k=-1)

    L.setdiag( ones, k=-N)
    L.setdiag( ones, k=N)

    L *= 1 / (h ** 2)

    return L

def gridDx(N):
    h = 2 / (N-1)
    ones = np.ones(N**2)

    # Handle decoupling across rows
    decoupled_ones = ones - (np.arange(1, N**2 + 1) % N == 0)

    # Initialize the gradient matrix
    Gx = sparse.lil_matrix((N**2, N**2))

    # Set the diagonals for the central difference in x-direction
    Gx.setdiag(decoupled_ones[:-1] * 0.5, k=1)   # Forward difference (next point)
    Gx.setdiag(decoupled_ones[:-1] * -0.5, k=-1) # Backward difference (previous point)

    Gx = Gx.tocsr()  # Convert to CSR format for efficient operations
    Gx *= (1 / h)    # Scale by grid spacing

    return Gx

def gridDy(N):
    h = 2 / (N-1)
    ones = np.ones(N**2)

    # Initialize the gradient matrix
    Gy = sparse.lil_matrix((N**2, N**2))

    # Set the diagonals for the central difference in y-direction
    Gy.setdiag(ones * 0.5, k=N)   # Forward difference (next row)
    Gy.setdiag(ones * -0.5, k=-N) # Backward difference (previous row)

    Gy = Gy.tocsr()  # Convert to CSR format for efficient operations
    Gy *= (1 / h)    # Scale by grid spacing

    return Gy

def sdfLaplacian( N, X, psiX, dpsiX, LpsiX ):
    L = gridLaplacianOld( N )
    
    Gx = gridDx( N )
    Gy = gridDy( N )

    Dpsix = sparse.diags([dpsiX[:,0]], [0], shape=(N**2, N**2), format='csr')
    Dpsiy = sparse.diags([dpsiX[:,1]], [0], shape=(N**2, N**2), format='csr')
    DLpsi = sparse.diags([LpsiX.flatten()], [0], shape=(N**2, N**2), format='csr')

    return L - DLpsi @ (Dpsix @ Gx + Dpsiy @ Gy)

def sdfGradient( N, X, psiX, dpsiX ):
    Gx = gridDx( N )
    Gy = gridDy( N )

    Dpsix = sparse.diags([dpsiX[:,0]], [0], shape=(N**2, N**2), format='csr')
    Dpsiy = sparse.diags([dpsiX[:,1]], [0], shape=(N**2, N**2), format='csr')

    return Gx - Dpsix @ (Dpsix @ Gx + Dpsiy @ Gy), Gy - Dpsiy @ (Dpsix @ Gx + Dpsiy @ Gy)


def inverse( v ):
    return np.where(
        v < 1e-5,
        np.ones_like(v) * np.sign(v) * 1e1 ,
        np.divide( np.ones_like(v), v )
    ).reshape( v.shape )

def fLaplacian( N, X, psiX, dpsiX, HpsiX, LpsiX ):
    k1 = inverse( np.sum( dpsiX ** 2, axis= 1) )
    k2 = (
        dpsiX[:,0] ** 2 * HpsiX[:,0,0] + dpsiX[:,1] ** 2 * HpsiX[:,1,1] + 2 * dpsiX[:,0] * dpsiX[:,1] * HpsiX[:,0,1]
    ) * k1 ** 2 + LpsiX * k1
    
    v1 = np.einsum('ijk,ik->ij', HpsiX, dpsiX)
    
    diag = lambda v : sparse.diags([v.flatten()], [0], shape=(N**2, N**2), format='csr')

    Gx = gridDx( N )
    Gy = gridDy( N )

    L = gridLaplacian( N )

    return (
        diag(k1) @ ( diag(v1[:,0]) @ Gx + diag(v1[:,1]) @ Gy ) -
        diag(k2) @ ( diag(dpsiX[:,0]) @ Gx + diag(dpsiX[:,1] @ Gy) ) +
        L
    )

def gridLaplacian( N, mask, boundary='dirichlet' ):
    L = sparse.csr_matrix((N**2, N**2))

    ones = np.ones(N**2)
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)

    L.setdiag( decoupled_ones * np.roll( mask, 1 ), k=1 )
    L.setdiag( decoupled_ones * np.roll( mask, -1 ), k=-1)
    L.setdiag( ones * np.roll( mask, -N ), k=-N)
    L.setdiag( ones * np.roll( mask, N ), k=N)

    if boundary=='neumann':
        L.setdiag( -1 * ( 
            np.pad( L.diagonal(1), (0,1), 'constant', constant_values=(0,0)) +
            np.pad( L.diagonal(-1), (1,0), 'constant', constant_values=(0,0)) + 
            np.pad( L.diagonal(N), (0,N), 'constant', constant_values=(0,0)) + 
            np.pad( L.diagonal(-N), (N,0), 'constant', constant_values=(0,0))
        ))
    elif boundary=='dirichlet':
        L.setdiag( -4 * ones * mask )
    else:
        raise ValueError('Uknown boundary condition')

    h = 2 / (N-1)
    return L * 1 / (h ** 2)

def gridDxMasked( N, mask, boundary='dirichlet' ):
    h = 2 / (N-1)
    invert = lambda x : np.where( x != 0, np.zeros_like(x), np.ones_like(x) )

    ones = np.ones(N**2)
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)    
    
    Gx = sparse.csr_matrix( (N**2, N**2) )
    Gx.setdiag( decoupled_ones * np.roll( mask, 1 ), k=1 )
    Gx.setdiag( -1 * decoupled_ones * np.roll( mask, -1 ), k=-1)
    
    if boundary=='neumann':
        Gx.setdiag( -1 * ( 
            invert( np.pad( Gx.diagonal(1), (0,1), 'constant', constant_values=(0,0)) )+
            invert( np.pad( Gx.diagonal(-1), (1,0), 'constant', constant_values=(0,0)))
        ))
    elif boundary=='dirichlet':
        pass
    else:
        raise ValueError('Uknown boundary condition')
    
    Gx *= 1 / (2*h)

    return Gx

def gridDyMasked( N, mask, boundary='dirichlet'  ):
    h = 2 / (N-1)
    invert = lambda x : np.where( x != 0, np.zeros_like(x), np.ones_like(x) )

    ones = np.ones(N**2)
    
    Gy = sparse.csr_matrix( (N**2, N**2) )
    Gy.setdiag( ones * np.roll( mask, N ), k=N )
    Gy.setdiag( -1 * ones * np.roll( mask, -N ), k=-N )

    if boundary=='neumann':
        Gy.setdiag( -1 * ( 
            invert( np.pad( Gy.diagonal(N), (0,N), 'constant', constant_values=(0,0)) )+
            invert( np.pad( Gy.diagonal(-N), (N,0), 'constant', constant_values=(0,0)) )
        ))
    elif boundary=='dirichlet':
        pass
    else:
        raise ValueError('Uknown boundary condition')
    
    Gy *= 1 / (2*h)

    return Gy

def gridLaplaceBeltrami( N, dfX, LfX, mask, boundary='dirichlet' ):
    L = gridLaplacian( N, mask, boundary=boundary)

    Gx = gridDxMasked( N, mask, boundary=boundary )

    Dpsix = sparse.csr_matrix( (N**2, N**2) )
    Dpsix.setdiag( dfX[:,0], k=0)

    Gy = gridDyMasked( N, mask, boundary=boundary )

    Dpsiy = sparse.csr_matrix( (N**2, N**2) )
    Dpsiy.setdiag( dfX[:,1], k=0)

    DLpsi = sparse.csr_matrix( (N**2, N**2) )
    DLpsi.setdiag( LfX.flatten(), k=0)

    return L - DLpsi @ (Dpsix @ Gx + Dpsiy @ Gy)

def gridIntrinsicGradient(  N, dfX, mask, boundary='dirichlet' ):
    Gx = gridDxMasked( N, mask, boundary=boundary )
    Gy = gridDyMasked( N, mask, boundary=boundary )

    Dpsix = sparse.diags([dfX[:,0]], [0], shape=(N**2, N**2), format='csr')
    Dpsiy = sparse.diags([dfX[:,1]], [0], shape=(N**2, N**2), format='csr')

    return Gx - Dpsix @ (Dpsix @ Gx + Dpsiy @ Gy), Gy - Dpsiy @ (Dpsix @ Gx + Dpsiy @ Gy)