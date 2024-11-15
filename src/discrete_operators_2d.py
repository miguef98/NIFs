import numpy as np
from scipy import sparse

def inverse( v ):
    return np.divide( np.ones_like(v), v )

def gridLaplacian( N, boundary='dirichlet' ):
    L = sparse.csr_matrix((N**2, N**2))

    ones = np.ones(N**2)
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)

    L.setdiag( decoupled_ones, k=1 )
    L.setdiag( decoupled_ones, k=-1)
    L.setdiag( ones, k=-N)
    L.setdiag( ones, k=N)

    if boundary=='neumann':
        L.setdiag( -1 * ( 
            np.pad( L.diagonal(1), (0,1), 'constant', constant_values=(0,0)) +
            np.pad( L.diagonal(-1), (1,0), 'constant', constant_values=(0,0)) + 
            np.pad( L.diagonal(N), (0,N), 'constant', constant_values=(0,0)) + 
            np.pad( L.diagonal(-N), (N,0), 'constant', constant_values=(0,0))
        ))
    elif boundary=='dirichlet':
        L.setdiag( -4 * ones )
    else:
        raise ValueError('Uknown boundary condition')

    h = 2 / (N-1)
    return L * 1 / (h ** 2)

def gridLaplacianDirichlet( N, mask ):
    L = sparse.csr_matrix((N**2, N**2))

    ones = np.ones(N**2) 
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)

    L.setdiag( decoupled_ones * mask.astype(np.float64), k=1 )
    L.setdiag( decoupled_ones * np.roll(mask.astype(np.float64), -1), k=-1)
    L.setdiag( ones * np.roll(mask.astype(np.float64), -N), k=-N)
    L.setdiag( ones * mask.astype(np.float64), k=N)
    
    h = 2 / (N-1)
    L *= 1 / (h ** 2)
    
    L.setdiag( (-4 / h**2) * ones * mask.astype(np.float64) + ones * np.logical_not(mask).astype(np.float64))

    return L

def gridDx( N, boundary='dirichlet' ):
    h = 2 / (N-1)
    invert = lambda x : np.where( x != 0, np.zeros_like(x), np.ones_like(x) )

    ones = np.ones(N**2)
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)    
    
    Gx = sparse.csr_matrix( (N**2, N**2) )
    Gx.setdiag( decoupled_ones, k=1 )
    Gx.setdiag( -1 * decoupled_ones, k=-1)
    
    if boundary=='neumann':
        Gx.setdiag( ( 
            invert( np.pad( Gx.diagonal(1), (0,1), 'constant', constant_values=(0,0)) ) +
            -1 * invert( np.pad( Gx.diagonal(-1), (1,0), 'constant', constant_values=(0,0)))
        ))
    elif boundary=='dirichlet':
        pass
    else:
        raise ValueError('Uknown boundary condition')
    
    Gx *= 1 / (2*h)

    return Gx

def gridDxx(N, boundary='dirichlet'):
    h = 2 / (N-1)

    ones = np.ones(N**2)
    decoupled_ones = ones - (np.arange(1,N**2 + 1) % N == 0)    
    
    Gx = sparse.csr_matrix( (N**2, N**2) )
    Gx.setdiag( decoupled_ones, k=1 )
    Gx.setdiag( decoupled_ones, k=-1)
    
    if boundary=='neumann':
        Gx.setdiag( -1 * ( 
            np.pad( Gx.diagonal(1), (0,1), 'constant', constant_values=(0,0))  +
            np.pad( Gx.diagonal(-1), (1,0), 'constant', constant_values=(0,0))
        ))
    elif boundary=='dirichlet':
        Gx.setdiag( -2 * ones )
    else:
        raise ValueError('Uknown boundary condition')
    
    Gx *= 1 / (h ** 2)

    return Gx

def gridDyy(N, boundary='dirichlet'):
    h = 2 / (N-1)

    ones = np.ones(N**2)
    
    Gy = sparse.csr_matrix( (N**2, N**2) )
    Gy.setdiag( ones, k=N )
    Gy.setdiag( ones, k=-N )

    if boundary=='neumann':
        Gy.setdiag( -1 * ( 
            np.pad( Gy.diagonal(N), (0,N), 'constant', constant_values=(0,0))  +
            np.pad( Gy.diagonal(-N), (N,0), 'constant', constant_values=(0,0))
        ))
    elif boundary=='dirichlet':
        Gy.setdiag( -2 * ones)
    else:
        raise ValueError('Uknown boundary condition')
    
    Gy *= 1 / (h ** 2)

    return Gy

def gridDxy(N, boundary='dirichlet'):
    h = 2 / (N - 1)
    invert = lambda x : np.where( x != 0, np.zeros_like(x), np.ones_like(x) )
    
    ones = np.ones(N**2)
    decoupled_ones_fx = ones - (np.arange(1,N**2 + 1) % N == 0)    
    decoupled_ones_bx = np.roll( decoupled_ones_fx, 1 )
    
    Gxy = sparse.csr_matrix((N**2, N**2))
    Gxy.setdiag( np.roll( decoupled_ones_fx, N+1), k=N + 1)  # Forward in both x and y
    Gxy.setdiag(- np.roll( decoupled_ones_bx, N-1 ), k=N - 1) # Backward in x, forward in y
    Gxy.setdiag(- np.roll( decoupled_ones_fx, -N+1), k=-N + 1) # Forward in x, backward in y
    Gxy.setdiag( np.roll( decoupled_ones_bx, -N-1 ), k=-N - 1) # Backward in both x and y
    
    if boundary == 'dirichlet':
        pass
    elif boundary == 'neumann':
        Gxy.setdiag( ( 
            invert( np.pad( Gxy.diagonal(N + 1), (0,N+1), 'constant', constant_values=(0,0)) ) +
            -1 * invert( np.pad( Gxy.diagonal(N - 1), (N-1,0), 'constant', constant_values=(0,0)) ) + 
            -1 * invert( np.pad( Gxy.diagonal(-N + 1), (0,N-1), 'constant', constant_values=(0,0)) ) + 
            invert( np.pad( Gxy.diagonal(-N - 1), (N+1,0), 'constant', constant_values=(0,0)) )
        ) )
    else:
        raise ValueError("Unknown boundary condition")
    
    Gxy *= 1 / (4 * h**2)
    
    return Gxy


def gridDy( N, boundary='dirichlet'  ):
    h = 2 / (N-1)
    invert = lambda x : np.where( x != 0, np.zeros_like(x), np.ones_like(x) )

    ones = np.ones(N**2)
    
    Gy = sparse.csr_matrix( (N**2, N**2) )
    Gy.setdiag( ones, k=N )
    Gy.setdiag( -1 * ones, k=-N )

    if boundary=='neumann':
        Gy.setdiag(  ( 
            invert( np.pad( Gy.diagonal(N), (0,N), 'constant', constant_values=(0,0)) ) +
            -1 * invert( np.pad( Gy.diagonal(-N), (N,0), 'constant', constant_values=(0,0)) )
        ))
    elif boundary=='dirichlet':
        pass
    else:
        raise ValueError('Uknown boundary condition')
    
    Gy *= 1 / (2*h)

    return Gy

def gridIntrinsicGradient(  N, dfX, boundary='dirichlet' ):
    Gx = gridDx( N, boundary=boundary )
    Gy = gridDy( N, boundary=boundary )

    sqnorm_dfX = np.sum( dfX ** 2, axis=1 )

    D_inv_sqnorm_dfX = sparse.diags([ inverse(sqnorm_dfX) ], [0], shape=(N**2, N**2), format='csr')
    D_dfX_x = sparse.diags([dfX[:,0]], [0], shape=(N**2, N**2), format='csr')
    D_dfX_y = sparse.diags([dfX[:,1]], [0], shape=(N**2, N**2), format='csr')

    projected_gradient = D_inv_sqnorm_dfX @ (D_dfX_x @ Gx + D_dfX_y @ Gy)

    return (
        Gx - D_dfX_x @ projected_gradient, 
        Gy - D_dfX_y @ projected_gradient
    )

def gridLaplaceBeltrami( N, dfX, boundary='dirichlet' ):
    norm_dfX = np.sqrt( np.sum( dfX ** 2, axis=1 ) )

    Gx = gridDx( N, boundary=boundary )
    Gy = gridDy( N, boundary=boundary )

    D_norm_dfX = sparse.diags([ norm_dfX ], [0], shape=(N**2, N**2), format='csr')
    D_inv_norm_dfX = sparse.diags([ inverse(norm_dfX) ], [0], shape=(N**2, N**2), format='csr')

    PGx, PGy = gridIntrinsicGradient( N, dfX, boundary )

    return D_inv_norm_dfX @ ( Gx @ D_norm_dfX @ PGx + Gy @ D_norm_dfX @ PGy )

def gridLaplaceBeltramiTV( N, dfX, LfX, tv, boundary='dirichlet' ):
    L = gridLaplacian( N, boundary=boundary)

    Gx = gridDx( N, boundary=boundary )
    Gy = gridDy( N, boundary=boundary )

    Gxx = gridDxx( N, boundary=boundary )
    Gyy = gridDyy( N, boundary=boundary )
    Gxy = gridDxy( N, boundary=boundary )

    Dfx = sparse.csr_matrix( (N**2, N**2) )
    Dfx.setdiag( dfX[:,0], k=0)
    Dfy = sparse.csr_matrix( (N**2, N**2) )
    Dfy.setdiag( dfX[:,1], k=0)
    DLf = sparse.csr_matrix( (N**2, N**2) )
    DLf.setdiag( LfX.flatten(), k=0)

    norm_dfX = np.sqrt( np.sum( dfX ** 2, axis=1 ) )
    D_inv_norm_dfX = sparse.diags([ inverse(norm_dfX) ], [0], shape=(N**2, N**2), format='csr')

    Dtvx = sparse.diags([ tv[:,0] ], [0], shape=(N**2, N**2), format='csr')
    Dtvy = sparse.diags([ tv[:,1] ], [0], shape=(N**2, N**2), format='csr')

    return (
        L -
        DLf @ (D_inv_norm_dfX ** 2 ) @ ( Dfx @ Gx + Dfy @ Gy ) -
        (D_inv_norm_dfX ** 2 ) @ ( (Dfx ** 2) @ Gxx + 2 * Dfx * Dfy @ Gxy + (Dfy ** 2) @ Gyy ) +
        (Dtvx * Dfx + Dtvy * Dfy) @ ( D_inv_norm_dfX ** 3 ) @ ( Dfx @ Gx + Dfy @ Gy )
    )

def gridInstrinsicGradientBertalmio( N, fX, boundary='dirichlet' ):
    Gx = gridDx( N, boundary=boundary )
    Gy = gridDy( N, boundary=boundary )

    dfXx = (Gx @ fX).flatten()
    dfXy = (Gy @ fX).flatten()

    sqnorm_dfX = dfXx ** 2 + dfXy ** 2
    D_inv_sqnorm_dfX = sparse.diags([ inverse(sqnorm_dfX) ], [0], shape=(N**2, N**2), format='csr')

    D_dfX_x = sparse.diags([dfXx], [0], shape=(N**2, N**2), format='csr')
    D_dfX_y = sparse.diags([dfXy], [0], shape=(N**2, N**2), format='csr')

    projected_gradient = D_inv_sqnorm_dfX @ (D_dfX_x @ Gx + D_dfX_y @ Gy)

    PGx, PGy = (
        Gx - D_dfX_x @ projected_gradient, 
        Gy - D_dfX_y @ projected_gradient
    )

    return PGx, PGy

def gridLaplaceBeltramiBertalmio( N, fX, boundary='dirichlet' ):
    Gx = gridDx( N, boundary=boundary )
    Gy = gridDy( N, boundary=boundary )

    dfXx = (Gx @ fX).flatten()
    dfXy = (Gy @ fX).flatten()

    sqnorm_dfX = dfXx ** 2 + dfXy ** 2

    norm_dfX = np.sqrt( sqnorm_dfX )
    D_norm_dfX = sparse.diags([ norm_dfX ], [0], shape=(N**2, N**2), format='csr')
    D_inv_norm_dfX = sparse.diags([ inverse(norm_dfX) ], [0], shape=(N**2, N**2), format='csr')
    
    PGx, PGy = gridInstrinsicGradientBertalmio( N, fX, boundary=boundary )

    return D_inv_norm_dfX @ ( Gx @ D_norm_dfX @ PGx + Gy @ D_norm_dfX @ PGy )