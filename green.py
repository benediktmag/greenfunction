

# Python 3.14.5
#
#	TG
#	11/08/2015
#
#	MBH
#	22/06/2018
#
# 	MAH
#	27/05/2026


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def regionPoints( corner = 0j, width = 1., condition = lambda z: 0*z + 1, N = 100 ):

	r'''
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


def boundaryPoints(gamma=lambda t: 0j + 0.9 * np.exp(2*np.pi*1j * t),N=10000):
	r'''
	Example of usage:

		Points, weight = boundaryPoints(lambda t: 0j + 0.9 * np.exp(2*np.pi*1j * t), 10000)

	Returns:

		Points:		A complex 1d-array of equally spaced points on the image of the curve
		weight:		A real 1d-array of the weight of each segment of the curve has

	Recieves:

		gamma		a curve from [0,1] to C 			(default: circle with center 0 and radius 0.9)
		N 			integer 							(default 10000)

	Note that this works pretty much the same as the regionPoints() function, except it is used to get the Szegö kernel instead
	of the Bergman kernel. In particular OrthogonalBasis() accepts either function.

	'''
	t=np.linspace(0,2*np.pi,N,endpoint=False)
	dt=1/N

	Points = gamma(t)

	gamma_der = (gamma(t+dt/2)-gamma(t-dt/2))/dt
	weight = np.abs(gamma_der) * dt

	return Points, weight

def OrthogonalBasis(grid, n, Q, K):
    '''
    Generates orthogonal polynomials on K and simultaneously evaluates 
    them on the grid using Arnoldi iteration.
    '''
    Points, weight = K
    W = weight * np.exp(-2. * n * Q(Points))
    
    def inner_product(p_eval, q_eval):
        return np.sum(p_eval * np.conjugate(q_eval) * W)

    P_K = np.zeros((n, len(Points)), dtype=complex)
    
    P_grid = np.zeros((n,) + np.shape(grid), dtype=complex)
    
    P_K[0] = np.ones_like(Points)
    P_grid[0] = np.ones_like(grid)
    
    norm = np.sqrt(inner_product(P_K[0], P_K[0]).real)
    P_K[0] /= norm
    P_grid[0] /= norm
    
    for d in range(1, n):
        p_d_K = Points * P_K[d-1]
        p_d_grid = grid * P_grid[d-1]
        
        for j in range(d):
            h = inner_product(p_d_K, P_K[j])
            
            p_d_K -= h * P_K[j]
            p_d_grid -= h * P_grid[j]  
            
        norm = np.sqrt(inner_product(p_d_K, p_d_K).real)
        
        if norm > 1e-14:
            P_K[d] = p_d_K / norm
            P_grid[d] = p_d_grid / norm
        else:
            print(f"Warning: Norm collapsed at degree {d}.")
            break
            
    return P_grid

def Bergman(B ):

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

	S = np.sum(B*np.conjugate(B), axis=0)

	# S has no imaginary part. Cast to real.
	return S.real

def Ingmar(B ):
	'''
	Example of usage:

		S = Ingmar( z, B )

	Receives:

		z 		complex array (or number)
		B 		n*n complex 2d-array
				B[j] coefficients of a polynomial p_j

	Returns:

		S = |sum_j  a_j*p_j(z)|
where
		a_j 	normal distributed complex numbers with mean 0
	'''

	S = 0

	realParts = np.random.normal(0, 1, len(B))
	imagParts = np.random.normal(0, 1, len(B))
	a=realParts+imagParts*1j

	S = np.tensordot(a, B, axes=1)
	L = np.abs(S)

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

	B = OrthogonalBasis(z, n, Q, K)

	return np.log( Bergman(B ) ) / (2.*n)


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

	B = OrthogonalBasis(z, n, Q, K)

	return np.log( Ingmar(B ) ) / (1.*n)

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
	#K = regionPoints( corner, width, condition, N )
	gamma_circle = lambda t: 0j + 0.9 * np.exp(1j * t)
	K=boundaryPoints(gamma_circle)

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
	print('How many polynomials? Between zero and 10000. Press enter for automatic n = 50.')
	while True :
		n = input()
		if n == '' :
			n = 50
			break
		n = int(n)

		# in case of a bad choice of n
		if (n > 0 and n < 10000):
			break
		print('Too small or too large. Try again.')# Choosing corner
	print('Type the lower left corner point of graph (complex number of the form 4+4j). Automatic is 0.')
	while True:
		s = input()
		if s == '' :
			corner = 0+0j
			break
		try :
			corner = complex(s.replace(" ",""))
			break
		except ValueError:
			print('Try again. The complex number should be of the form 1+2j.')


	#Choosing width
	print("Type the width of the graph (same for each dimention). Automatic is 4.0.")
	while True:
		s = input()
		if s == '' :
			width = 4.0
			break
		width = float(s)
		if width > 0 and width < 100:
			break
		if width <= 0:
			print("The width can't be negative or zero.")
		else :
			print("Too large.")# The user chooses how he/she adjusts the condtions and the Q
	print('You have the following choices for the conditions and Q.\n"t" or "T" for typing manually \n"w" or "W" for typing manually and writing them in a file \n"r" or "R" for reading a file already created \n"a" or "A" for automatic conditions (z.real)**2 + (z.imag)**2 < (0.9)**2 and Q as np.log(abs(1/(1-z))).')
	while True:
		tfwa = input()
		if tfwa == "a" or tfwa == "A":
			condition = lambda z: (z.real)**2 + (z.imag)**2 < (0.9)**2
			Q = lambda z: np.log(abs(1/(1-z)))
			break
		if tfwa == "t" or tfwa == "T":
			print("Type desired conditions for the region of the function.")
			while True:
				tempcond = input()
				if tempcond != '' :
					condition = lambda z: eval(tempcond)
					break
			
			print("Type desired function Q.")
			while True:
				tempcond = input()
				if tempcond != '' :
					condition = lambda z: eval(tempcond)
					break
			break
		if tfwa == "w" or tfwa == "W":
			print("Specify the name of the name of the file you would like to create. Make sure you do not overwrite any existing file. No need for .txt.")
			name = input() + ".txt"
			cond = input("Now type the conditions.")
			qtemp = input("Now type Q.")
			with open(name, 'w') as file:
				file.write(cond+"\n" +qtemp)
			break
		if tfwa == "r" or tfwa == "R":
			nafni  = input("Type the name of your file or press i for instructions.") 
			if nafni == "i" or nafni =="I":
				print('Make sure your desired file is saved in the folder greenfuction. Write the name of the file without ".txt" at the end. The file should have the equation for the conditions in the first line, and the formula for Q in the second line. No need for "lambda z:". Now type the name of your file.')
				filename = input() +".txt"
			else :
				filename = nafni +".txt"
			file = open(filename, "r") 
			conditionStr = file.readline().strip()
			QStr = file.readline().strip()
			print("Condition:"+ conditionStr)
			print("Q:"+ QStr)
			condition = lambda z: eval(conditionStr)
			Q = lambda z: eval(QStr)
			break
		print('What do you mean? Pick "t", "w", "r" or "a".')
	return (n, corner, width, condition, Q)



	#Main function vill ask you for n, corner and width, and the user has four choices about determening condition and Q
def main():
	#For convenience, it is possible to define the variables in the progrem and not have to go through the user interface

	#Automatic variables can be changed here 
	n = 100
	corner = -2-2j
	width = 4.0
	#condition = lambda z: (z.real)**2 + (z.imag)**2 < (0.9)**2
	#condition = lambda z: abs((z.real) + (z.imag)) < 1
	condition = lambda z: (abs(z.imag) <= 1) & (abs(z.real) <= 1)
	# Q = lambda z: np.log(abs(1/(1-z)))
   # Q = lambda z: abs(z)
	Q = lambda z: 0 * z

	#Here you choose whether or not to use the interface. 
	yn = input("Would you like to adjust the variables through the user interface? (y/n)\n")
	if yn == "n" or yn == "N":
		drawGreen( n, Q, corner, width, condition, 100, save = False )
	else :
		(n, corner, width, condition, Q) = interface()
		drawGreen( n, Q, corner, width, condition, 100, save = False )
	

if __name__ == '__main__':
    main()
