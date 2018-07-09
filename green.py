

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



# A user interface to adjust the variables of drawGreen. Returns the variables.
def interface() :
	# Choosing n
	print 'How many polynomials? Between zero and 10000. Press enter for automatic n = 50.'
	while True :
		n = raw_input()
		if n == '' :
			n = 50
			break
		n = int(n)

		# in case of a bad choice of n
		if (n > 0 and n < 10000):
			break
		print 'Too small or too large. Try again.'
	
	# Choosing corner
	print 'Type the lower left corner point of graph (complex number of the form 4+4j). Automatic is 0.'
	while True:
		s = raw_input()
		if s == '' :
			corner = 0+0j
			break
		try :
			corner = complex(s.replace(" ",""))
			break
		except ValueError:
			print('Try again. The complex number should be of the form 1+2j.')


	#Choosing width
	print "Type the width of the graph (same for each dimention). Automatic is 4.0."
	while True:
		s = raw_input()
		if s == '' :
			width = 4.0
			break
		width = float(s)
		if width > 0 and width < 100:
			break
		if width <= 0:
			print "The width can't be negative or zero."
		else :
			print "Too large."

	# The user chooses how he/she adjusts the condtions and the Q
	print "You have the following choices for the conditions and Q.\n\"t\"or\"T\" for typing manually \n\"w\"or\"W\" for typing manually and writing them in a file \n\"r\"or\"R\" for reading a file already created \n\"a\"or\"A\" for automatic conditions (z.real)**2 + (z.imag)**2 < (0.9)**2 and Q as np.log(abs(1/(1-z)))."
	while True:
		tfwa = raw_input( )
		if tfwa == "a" or tfwa == "A":
			condition = lambda z: (z.real)**2 + (z.imag)**2 < (0.9)**2
			Q = lambda z: np.log(abs(1/(1-z)))
			break
		if tfwa == "t" or tfwa == "T":
			print "Type desired conditions for the region of the function."
			while True:
				tempcond = raw_input()
				if tempcond != '' :
					condition = lambda z: eval(tempcond)
					break
			break
			print "Type desired function Q."
			while True:
				tempcond = raw_input()
				if tempcond != '' :
					condition = lambda z: eval(tempcond)
					break
			break
		if tfwa == "w" or tfwa == "W":
			print "Specify the name of the name of the file you would like to create. Make sure you do not overwrite any existing file. No need for .txt."
			name = raw_input() + ".txt"
			cond = raw_input( "Now type the conditions.")
			qtemp = raw_input("Now type Q.")
			with open(name, 'w') as file:
				file.write(cond+"\n" +qtemp)
			break
		if tfwa == "r" or tfwa == "R":
			nafni  = raw_input( "Type the name of your file or press i for instructions.") 
			if nafni == "i" or nafni =="I":
				print "Make sure your desired file is saved in the folder greenfuction. Write the name of the file without \".txt\" at the end. The file should have the equation for the conditions in the first line, and the formula for Q in the second line. No need for \"lambda z:\". Now type the name of your file."
				filename = raw_input() +".txt"
			else :
				filename = nafni +".txt"
			file = open(filename, "r") 
			print "Condition:"+ file.readline(1)
			print "Q:"+ file.readline(2)
			condition = lambda z: eval(file.readline(1))
			Q = lambda z: eval(file.readline(2))
			break
		print "What do you mean? Pick \"t\", \"w\", \"r\" or \"a\"."
	return (n, corner, width, condition, Q)



	#Main function vill ask you for n, corner and width, and the user has four choices about determening condition and Q
def main():
	#For convenience, it is possible to define the variables in the progrem and not have to go through the user interface

	#Automatic variables can be changed here 
	n = 50
	corner = -2-2j
	width = 4.0
	condition = lambda z: (z.real)**2 + (z.imag)**2 < (0.9)**2
	Q = lambda z: np.log(abs(1/(1-z)))

	#Here you choose whether or not to use the interface. 
	yn = raw_input("Would you like to adjust the variables through the user interface? (y/n)\n")
	if yn == "n" or yn == "N":
		drawGreen( n, Q, corner, width, condition, 100, save = False )
	else :
		(n, corner, width, condition, Q) = interface()
		drawGreen( n, Q, corner, width, condition, 100, save = False )
	

if __name__ == '__main__':
    main()
