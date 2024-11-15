import numpy as np
from scipy import sparse

def inverse( v ):
    return np.divide( np.ones_like(v), v )

def gridDx( N ):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    decoupled_ones = ones - (np.arange(1,N**3 + 1) % N == 0)    
    
    Gx = sparse.csr_matrix( (N**3, N**3) )
    Gx.setdiag( decoupled_ones, k=1 )
    Gx.setdiag( -1 * decoupled_ones, k=-1)
    
    Gx *= 1 / (2*h)

    return Gx

def gridDy( N ):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    decoupled_ones_f = ones - ((np.arange(N**3) // N) % N == 0)
    decoupled_ones_b = ones - ((np.arange(N**3) // N) % N == 1)
    
    Gy = sparse.csr_matrix( (N**3, N**3) )
    Gy.setdiag( decoupled_ones_f, k=N )
    Gy.setdiag( -1 * np.roll(decoupled_ones_b, -N), k=-N )
    
    Gy *= 1 / (2*h)

    return Gy

def gridDz( N ):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    
    Gy = sparse.csr_matrix( (N**3, N**3) )
    Gy.setdiag( ones, k=N**2 )
    Gy.setdiag( -1 * ones, k=-(N**2) )
    
    Gy *= 1 / (2*h)

    return Gy

def gridDxx(N):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    decoupled_ones = ones - (np.arange(1,N**3 + 1) % N == 0)    
    
    Gx = sparse.csr_matrix( (N**3, N**3) )
    Gx.setdiag( decoupled_ones, k=1 )
    Gx.setdiag( decoupled_ones, k=-1)
    
    Gx.setdiag( -2 * ones )
    
    Gx *= 1 / (h ** 2)

    return Gx

def gridDyy(N):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    decoupled_ones = ones - ((1 + np.arange(1,N**3 + 1) // N) % N == 0)
    
    Gy = sparse.csr_matrix( (N**3, N**3) )
    Gy.setdiag( decoupled_ones, k=N )
    Gy.setdiag( decoupled_ones, k=-N )

    Gy.setdiag( -2 * ones)
    
    Gy *= 1 / (h ** 2)

    return Gy

def gridDzz(N):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    
    Gz = sparse.csr_matrix( (N**3, N**3) )
    Gz.setdiag( ones, k=N ** 2 )
    Gz.setdiag( ones, k=-(N ** 2) )

    Gz.setdiag( -2 * ones)
    
    Gz *= 1 / (h ** 2)

    return Gz

def gridDxy(N):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    decoupled_ones_fx = ones - (np.arange(1,N**3 + 1) % (N) == 0)
    decoupled_ones_fy = ones - (np.arange(1,N**3 + 1) % (N ** 2) == 0)

    decoupled_ones_bx = np.roll(decoupled_ones_bx, 1)
    decoupled_ones_by = ones - (np.arange(1,N**3 + 1) % (N ** 2) == 0)
    
    Gxy = sparse.csr_matrix( (N**3, N**3) )
    Gxy.setdiag( decoupled_ones_bx * decoupled_ones_bx, k = N + 1 )
    Gxy.setdiag( -decoupled_ones_bx * decoupled_ones_bx, k = -N + 1 )
    Gxy.setdiag( -decoupled_ones_bx * decoupled_ones_bx, k = N - 1 )
    Gxy.setdiag( decoupled_ones_bx * decoupled_ones_bx, k = -N - 1 )

    Gxy.setdiag( ones, k=-(N ** 2) )

    Gxy.setdiag( -2 * ones)
    
    Gxy *= 1 / (h ** 2)

    return Gxy

def gridLaplacian( N ):
    h = 2 / (N-1)

    ones = np.ones(N**3)
    decoupled_ones_x = ones - (np.arange(1,N**3 + 1) % N == 0)    
    decoupled_ones_y = ones - (np.arange(1,N**3 + 1) % (N ** 2) == 0)
    
    L = sparse.csr_matrix( (N**3, N**3) )
    L.setdiag( decoupled_ones_x, k=1 )
    L.setdiag( decoupled_ones_x, k=-1 )
    L.setdiag( decoupled_ones_y, k=N )
    L.setdiag( decoupled_ones_y, k=-N )
    L.setdiag( ones, k=N **2 )
    L.setdiag( ones, k=-(N **2) )

    L.setdiag( -6 * ones)
    
    L *= 1 / (h ** 2)

    return L

def gridIntrinsicGradient( N, dfX ):
    Gx = gridDx( N )
    Gy = gridDy( N )
    Gz = gridDz( N )

    sqnorm_dfX = np.sum( dfX ** 2, axis=1 )

    D_inv_sqnorm_dfX = sparse.diags([ inverse(sqnorm_dfX) ], [0], shape=(N**3, N**3), format='csr')
    D_dfX_x = sparse.diags([dfX[:,0]], [0], shape=(N**3, N**3), format='csr')
    D_dfX_y = sparse.diags([dfX[:,1]], [0], shape=(N**3, N**3), format='csr')
    D_dfX_z = sparse.diags([dfX[:,2]], [0], shape=(N**3, N**3), format='csr')

    projected_gradient = D_inv_sqnorm_dfX @ (D_dfX_x @ Gx + D_dfX_y @ Gy + D_dfX_z @ Gz)

    return (
        Gx - D_dfX_x @ projected_gradient, 
        Gy - D_dfX_y @ projected_gradient,
        Gz - D_dfX_z @ projected_gradient
    )

def gridLaplaceBeltrami( N, dfX ):
    norm_dfX = np.sqrt( np.sum( dfX ** 2, axis=1 ) )

    Gx = gridDx( N )
    Gy = gridDy( N )
    Gz = gridDz( N )

    D_norm_dfX = sparse.diags([ norm_dfX ], [0], shape=(N**3, N**3), format='csr')
    D_inv_norm_dfX = sparse.diags([ inverse(norm_dfX) ], [0], shape=(N**3, N**3), format='csr')

    PGx, PGy, PGz = gridIntrinsicGradient( N, dfX )

    return D_inv_norm_dfX @ ( Gx @ D_norm_dfX @ PGx + Gy @ D_norm_dfX @ PGy + Gz @ D_norm_dfX @ PGz )


def gridLaplaceBeltramiTV( N, dfX, LfX, tv ):
    L = gridLaplacian( N )

    Gx = gridDx( N )
    Gy = gridDy( N )
    Gz = gridDz( N )

    Gxx = gridDxx( N )
    Gyy = gridDyy( N )
    Gzz = gridDzz( N )

    Dfx = sparse.csr_matrix( (N**3, N**3) )
    Dfx.setdiag( dfX[:,0], k=0)
    Dfy = sparse.csr_matrix( (N**3, N**3) )
    Dfy.setdiag( dfX[:,1], k=0)
    Dfz = sparse.csr_matrix( (N**3, N**3) )
    Dfz.setdiag( dfX[:,2], k=0)

    DLf = sparse.csr_matrix( (N**3, N**3) )
    DLf.setdiag( LfX.flatten(), k=0)

    norm_dfX = np.sqrt( np.sum( dfX ** 2, axis=1 ) )
    D_inv_norm_dfX = sparse.diags([ inverse(norm_dfX) ], [0], shape=(N**3, N**3), format='csr')

    Dtvx = sparse.diags([ tv[:,0] ], [0], shape=(N**2, N**2), format='csr')
    Dtvy = sparse.diags([ tv[:,1] ], [0], shape=(N**2, N**2), format='csr')

    return (
        L -
        DLf @ (D_inv_norm_dfX ** 2 ) @ ( Dfx @ Gx + Dfy @ Gy ) -
        (D_inv_norm_dfX ** 2 ) @ ( (Dfx ** 2) @ Gxx + 2 * Dfx * Dfy @ Gxy + (Dfy ** 2) @ Gyy ) +
        (Dtvx * Dfx + Dtvy * Dfy) @ ( D_inv_norm_dfX ** 3 ) @ ( Dfx @ Gx + Dfy @ Gy )
    )


def gridIntrinsicGradientBertalmio( N, fX ):
    Gx = gridDx( N )
    Gy = gridDy( N )
    Gz = gridDz( N )

    dfXx = (Gx @ fX).flatten()
    dfXy = (Gy @ fX).flatten()
    dfXz = (Gz @ fX).flatten()

    sqnorm_dfX = dfXx ** 2 + dfXy ** 2 + dfXz ** 2
    D_inv_sqnorm_dfX = sparse.diags([ inverse(sqnorm_dfX) ], [0], shape=(N**3, N**3), format='csr')

    D_dfX_x = sparse.diags([dfXx], [0], shape=(N**3, N**3), format='csr')
    D_dfX_y = sparse.diags([dfXy], [0], shape=(N**3, N**3), format='csr')
    D_dfX_z = sparse.diags([dfXz], [0], shape=(N**3, N**3), format='csr')

    projected_gradient = D_inv_sqnorm_dfX @ (D_dfX_x @ Gx + D_dfX_y @ Gy + D_dfX_z @ Gz)

    PGx, PGy, PGz = (
        Gx - D_dfX_x @ projected_gradient, 
        Gy - D_dfX_y @ projected_gradient,
        Gz - D_dfX_z @ projected_gradient
    )

    return PGx, PGy, PGz

def gridLaplaceBeltramiBertalmio( N, fX ):
    Gx = gridDx( N )
    Gy = gridDy( N )
    Gz = gridDy( N )

    dfXx = (Gx @ fX).flatten()
    dfXy = (Gy @ fX).flatten()
    dfXz = (Gz @ fX).flatten()

    sqnorm_dfX = dfXx ** 2 + dfXy ** 2 + dfXz ** 2

    norm_dfX = np.sqrt( sqnorm_dfX )
    D_norm_dfX = sparse.diags([ norm_dfX ], [0], shape=(N**3, N**3), format='csr')
    D_inv_norm_dfX = sparse.diags([ inverse(norm_dfX) ], [0], shape=(N**3, N**3), format='csr')
    
    PGx, PGy, PGz = gridIntrinsicGradientBertalmio( N, fX )

    return D_inv_norm_dfX @ ( Gx @ D_norm_dfX @ PGx + Gy @ D_norm_dfX @ PGy + Gz @ D_norm_dfX @ PGz )