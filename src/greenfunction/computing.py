import numpy as np

def grid_points(corner = -2-2j, width = 4., n_grid = 100):
    '''
    Inputs:
        - corner: the bottom left corner of the square grid
        - width: the desired width of the grid
        - n_grid: the number of points on each grid axis, for a total number of n_grid**2 points

    Returns: the grid where we want to evaluate the Green's function.
    '''
    min_re = corner.real
    max_re = corner.real + width
    min_im = corner.imag
    max_im = corner.imag + width

    xx = np.linspace(min_re, max_re, n_grid + 1)
    yy = np.linspace(min_im, max_im, n_grid + 1)
    re, im = np.meshgrid( xx, yy )
    grid = re + im*1j

    return re, im, grid

def region_points(grid, width, condition, n_grid = 100):
    '''
    Inputs:
        - grid: the grid used for compuation
        - width: the width of said grid
        - condition: function that describes the region, condition(z)==true if z is in the region
        - n_grid: the number of points on the axis of each grid

    Returns: A 2d boolean array of the same size as the grid, where true means the point is inside the region,
             along with the area of each grid square
    '''
    raw_cond = np.asarray(condition(grid), dtype=bool)
    mask = np.ones_like(grid, dtype=bool) & raw_cond
    points = grid[mask]

    weight = (width / n_grid)**2

    return points, weight


def boundary_points(gamma=lambda t: 0j + 0.9 * np.exp(1j * t), boundary_n=10000):
    '''
    Inputs:
        - gamma: a closed simple curve [0,1]->C
        - boundary_n: the number of points on the curve

    Returns: An array of evenly spaced points on the curve along with the weight of their segments.
    '''
    t = np.linspace(0,2*np.pi,boundary_n,endpoint=False)
    dt = 1/boundary_n

    points = gamma(t)

    gamma_der = (gamma(t+dt/2)-gamma(t-dt/2))/dt
    weight = np.abs(gamma_der) * dt

    return points, weight

def orthogonal_basis(grid, region, n, q_func):
    '''
    Generates orthogonal polynomials of degree <n on region and simultaneously evaluates
    them on the grid using Arnoldi iteration.

    Input:
        - grid: The grid where we want to evaluate
        - region: The region we want to find the orthogonal polynomials over, can either be a real region or a curve
        - n: The number of polynomials desired
        - q_func: weight function

    Returns: An array consisting of the values the n polynomials take on the grid.

    '''
    points, weight = region
    W = weight * np.exp(-2. * n * q_func(points))

    def innerProduct(p_eval, q_eval):
        return np.sum(p_eval * np.conjugate(q_eval) * W)

    polys_region = np.zeros((n, len(points)), dtype=complex)

    polys_grid = np.zeros((n,) + np.shape(grid), dtype=complex)

    polys_region[0] = np.ones_like(points)
    polys_grid[0] = np.ones_like(grid)

    norm = np.sqrt(innerProduct(polys_region[0], polys_region[0]).real)
    polys_region[0] /= norm
    polys_grid[0] /= norm

    for d in range(1, n):
        pol_d_region = points * polys_region[d-1]
        pol_d_grid = grid * polys_grid[d-1]

        for j in range(d):
            h = innerProduct(pol_d_region, polys_region[j])

            pol_d_region -= h * polys_region[j]
            pol_d_grid -= h * polys_grid[j]

        norm = np.sqrt(innerProduct(pol_d_region, pol_d_region).real)

        if norm > 1e-14:
            polys_region[d] = pol_d_region / norm
            polys_grid[d] = pol_d_grid / norm
        else:
            print(f"Warning: Norm collapsed at degree {d}.")
            break

    return polys_grid

def bergman(basis):

    '''
    Example of usage:

        S = bergman(B)

    Input: An orthogonal basis evaluated on the grid points
    Returns: sum_j  |p_j(z)|^2
    '''

    S = np.sum(basis*np.conjugate(basis), axis=0)

    # S has no imaginary part. Cast to real.
    return S.real

def ingmar(basis):
    '''
    Example of usage:

        S = ingmar( z, B )

    Input: An orthogonal basis evaluated on the grid points
    Returns:

        S = |sum_j  a_j*p_j(z)|
    where
        a_j 	normal distributed complex numbers with mean 0
    '''

    S = 0

    realParts = np.random.normal(0, 1, len(basis))
    imagParts = np.random.normal(0, 1, len(basis))
    a = realParts + imagParts*1j

    S = np.tensordot(a, basis, axes=1)
    L = np.abs(S)

    # L has no imaginary part. Cast to real.
    return L.real

def green_approx(z, n, region, q_func = lambda z: 0*z):

    '''
    Usage: g = Green( z, n, Q, K )

    Inputs:
        -z:		complex array (or number)
        -n:		integer
        -region = (Points, weight):
                Points:		complex array
                weight:		real
        -q_func:
            weight function	(default: constant function 0)


    Returns: The n-th approximation of the weighted Green function
            G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K }
            evaluated at z.
    '''


    basis = orthogonal_basis(z, region, n, q_func)

    return np.log(bergman(basis))/(2.*n)


def george_approx(z, n, region, q_func = lambda z: 0*z):
    '''
    Usage: 	g = George( z, n, Q, K )

    Revives:

        z 		complex array (or number)
        n 		integer
        Q 		weight function				(default: constant function 0)
        K = ( Points, weight )
                Points 		complex array
                weight 		real

    Returns: The n-th approximation of the weighted Green function
    G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K } evaluated at z.
    '''

    basis = orthogonal_basis(z, region, n, q_func)

    return np.log(ingmar(basis))/(1.*n)

def dirichlet_szego(grid, gamma, boundary_func, n, boundary_n, q_func = lambda z: 0*z):

    boundary = boundary_points(gamma, boundary_n)
    boundary_vals = boundary[1]*boundary_func(boundary[0])

    grid_flat = grid.ravel()
    eval_points = np.concatenate([grid_flat,boundary[0]])

    basis=orthogonal_basis(eval_points, boundary, n, q_func)

    basis_grid = basis[:, :len(grid_flat)]
    basis_boundary = basis[:, len(grid_flat):]

    S_zz=bergman(basis_grid)

    M = (np.conjugate(basis_boundary) * boundary_vals) @ basis_boundary.T

    u = np.sum(basis_grid * np.tensordot(M, np.conjugate(basis_grid), axes=(1, 0)), axis=0).real
    u = u / (S_zz+1e-15)

    return u.reshape(grid.shape)

