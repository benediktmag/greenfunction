

# Python 2.7.6
#
#	TG
#	11/08/2015
#
#	MBH
#	22/06/2018
#


import numpy as np
import InnerProductSpace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def regionPoints( corner = 0j, width = 1., condition = lambda z: 0*z + 1, N = 100 ):

	'''
	Example of usage:

		Points, weight = regionPoints( -5.0 - 5.0*1j, 10.0, lambda z: (z.real)**2 + (z.imag)**2 < 1/4., 1000 )

	Returns:

		Points:		A complex 1d-array of the numbers which fulfill the condition.
		weight:		A real 1d-array corresponding to the area each point represents.

	Recieves:

		corner		complex number 		(default: 0)
		width		real number 		(default: 1)
		condition	lambda function		(default: constant function 1)
		N 			integer 			(default 100)

	We let S be the square with bottom left corner (corner) and width (width).
	We let K be the region inside the square S which fulfills the condition:

	  (corner + i*width) ------------- (corner + (1+i)*width)
						 |    __   S |
						 |   /	\    |
						 |  / K  )   |
						 | |	(    |
						 |  \___/    |
				(corner) ------------- (corner + width)

	Then a point z is in K if and only if z is in S and condition(z) == 1/True
	'''


	minRe = corner.real
	maxRe = minRe + width
	minIm = corner.imag
	maxIm = minIm + width

	xx = np.linspace( minRe, maxRe, N + 1 )
	yy = np.linspace( minIm, maxIm, N + 1 )
	Re, Im = np.meshgrid( xx, yy )
	Grid = Re + Im*1j

	Points = Grid[ condition( Grid ).astype( bool ) ]

	weight = (width / N)**2

	return Points, weight


def innerProduct( u, v, n, Q, K ):

	'''
	Example of usage:

		r = inProd( u, v, n, Q, K )

	Returns:

		r = <u,v> where <u,v> is an inner product on a polynomial space defined in the following way:

			We interpret K to be a region containing the numbers 'Points', each one representing the area 'weight'.

				Define polynomials
						p(z) = sum_j  u[j] * z^j
					and q(z) = sum_j  v[j] * z^j

			<u,v> := integral_K  p * conjugate(q) * exp( -2*n*Q ) dm

	Recieves:

		n 		integer
		u,v 	complex 1d-arrays (representing polynomials)
				len(u) == len(v) == n
		Q 		weight function
		K = ( Points, weight )
				Points 		complex array
				weight 		real
	'''

	Points, weight = K

	p = np.polynomial.polynomial.polyval( Points, u )
	q = np.polynomial.polynomial.polyval( Points, v )

	Int = weight * ( p * np.conjugate(q) * np.exp( -2.*n * Q( Points ) ) )

	return Int.sum()


def Bergman( z, B ):

	'''
	Example of usage:

		S = Bergman( z, B )

	Revieves:

		z 		complex array (or number)
		B 		n*n complex 2d-array
				B[j] coefficients of a polynomial p_j

	Returns:

		S = sum_j  |p_j(z)|^2
	'''

	S = 0

	for j in range( len(B) ):

		p_j = np.polynomial.polynomial.polyval( z, B[j] )

		S += p_j*np.conjugate(p_j)

	# S has no imaginary part. Cast to real.
	return S.real

def Ingmar( z, B ):
	'''
	Example of usage:

		S = Ingmar( z, B )

	Revieves:

		z 		complex array (or number)
		B 		n*n complex 2d-array
				B[j] coefficients of a polynomial p_j

	Returns:

		S = |sum_j  a_j*p_j(z)|
where
		a_j 	normal distributed complex numbers with mean 0
	'''

	S = 0

	for j in range( len(B) ):
		p_j = np.polynomial.polynomial.polyval(z,B[j])
		a_j = (np.random.normal(0,1,1)+np.random.normal(0,1,1)*1j)
		S += a_j*p_j
		L = abs(S)

	# L has no imaginary part. Cast to real.
	return L.real


def Green( z, n, Q = lambda z: 0*z, K = regionPoints() ):

	'''
	g = Green( z, n, Q, K )


	z 		complex array (or number)
	n 		integer
	Q 		weight function				(default: constant function 0)
	K = ( Points, weight )
			Points 		complex array 	(default: unit square [0,1] + [0,1]i )
			weight 		real			(default: 10^-4)


	g 		The n-th approximation of the weighted Green function
			G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K }
			evaluated at z.
	'''

	inProd = lambda u,v: innerProduct( u, v, n, Q, K )

	V = InnerProductSpace.InnerProductSpace( n, inProd )

	B = V.GramSchmidt()

	return np.log( Bergman( z, B ) ) / (2.*n)


def George( z, n, Q = lambda z: 0*z, K = regionPoints() ):
	'''
	Returns the n-th approximation of the weighted Green function
	G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K } evaluated at z.

	Usage: 	g = George( z, n, Q, K )

	Revives:

		z 		complex array (or number)
		n 		integer
		Q 		weight function				(default: constant function 0)
		K = ( Points, weight )
				Points 		complex array 	(default: unit square [0,1] + [0,1]i )
				weight 		real			(default: 10^-4)
	'''

	inProd = lambda u,v: innerProduct( u, v, n, Q, K )

	V = InnerProductSpace.InnerProductSpace( n, inProd )

	B = V.GramSchmidt()

	return np.log( Ingmar( z, B ) ) / (1.*n)

def drawGreen( n, Q = lambda z: 0*z, corner = 0j, width = 1., condition = lambda z: 0*z + 1, N = 100, show = True, save = True, drawBoth = True):
	'''
	Example of usage:

		fig = drawGreen( n, Q, K )

	Revieves:

		n 			integer
		Q 			weight function			(default: constant function 0)
		corner		complex numbers			(default: 0 + 0i)
		width		real numbers			(default: 1)
		condition	lambda function			(default: constant function 1)
		N   		integer					(default: 100)
		show 		boolean 				(default: True)
		save 		boolean					(default: True)
		drawBoth	boolean					(default: True)

	Calculates:

			K = ( Points, weight )
				Points 	complex array 			(default: unit square [0,1] + [0,1]i)
				weight 	real					(default: 10^-4)

		by using the function regionPoints( corner, width, condition, N ):

	Returns:

		figure which shows the n-th approximation of the weighted Green function
				G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K }
		in a neighborhood of K defined by the corner and width given.

		If show == True, plt.show() is called.
		If save == True, the figure is saved to a file in the same folder.
		If drawBoth == True, the method draw both approimations and their corresponding contour plots.
	'''
	K = regionPoints( corner, width, condition, N )

	Points = K[0]

	minRe = corner.real
	maxRe = corner.real + width
	minIm = corner.imag
	maxIm = corner.imag + width

	xx = np.linspace(minRe,maxRe,N+1)
	yy = np.linspace(minIm,maxIm,N+1)
	Re, Im = np.meshgrid( xx, yy )
	Z = Re + Im*1j

	green = Green( Z, n, Q, K )

	fig1 = plt.figure(1)
	plt.xlim(minRe, maxRe)
	plt.ylim(minIm, maxIm)
	plt.gca().set_aspect('equal', adjustable='box')
	CS = plt.contour(Re, Im, green, cmap=cm.coolwarm)
	plt.clabel(CS, inline=1, fontsize=8)

	fig2 = plt.figure(2)
	ax = fig2.add_subplot(111, projection='3d')
	surf = ax.plot_surface(Re, Im, green, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	if drawBoth:
		george = George( Z, n, Q, K )
		fig3 = plt.figure(3)
		plt.gca().set_aspect('equal', adjustable='box')
		CS = plt.contour(Re, Im, george, cmap=cm.coolwarm)
		plt.clabel(CS, inline=1, fontsize=8)

		fig4 = plt.figure(4)
		ax = fig4.add_subplot(111, projection='3d')
		surf = ax.plot_surface(Re, Im, george, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	if save:
		fig1.savefig('Greenfig1.pdf', bbox_inches='tight')
		fig2.savefig('Greenfig2.pdf', bbox_inches='tight')
		if drawBoth:
			fig3.savefig('Georgefig1.pdf', bbox_inches='tight')
			fig4.savefig('Georgefig2.pdf', bbox_inches='tight')

	if show:
		plt.show()

	return fig1, fig2, fig3, fig4


def main():

	# Max degree of polynomials
	n = 50

	# Lower left corner and width of square S
	corner = -2.0 - 2.0*1j
	width = 4.0
	N = 100

	# z is in K iff z is in S and condition(z) == 1/True
	condition = lambda z: (z.real)**2 + (z.imag)**2 < (0.9)**2

	Q = lambda z: np.log(abs(1/(1-z)))

	drawGreen( n, Q, corner, width, condition, N, save = False )


if __name__ == '__main__':
    main()
