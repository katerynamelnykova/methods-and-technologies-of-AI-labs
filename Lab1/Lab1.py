import matplotlib.pyplot as plt
import skfuzzy as fuzz
import numpy as np

# creates a range of x values
x = np.linspace(0, 10, 1000)

# triangle
triangle = fuzz.trimf(x, [3, 7, 9])
plt.figure(figsize=(7, 5))
plt.plot(x, triangle, label='Triangular MF') 
plt.legend()
plt.grid(True)
plt.title('Triangular Membership Function Graph') 
plt.show()

# trapezoidal
trapeze = fuzz.trapmf(x, [3, 5, 7, 9])
plt.figure(figsize=(7, 5))
plt.plot(x, trapeze, label='Trapezoidal MF') 
plt.legend()
plt.grid(True)
plt.title('Trapezoidal Membership Function Graph') 
plt.show()

# simple Gaussian membership function
gauss1 = fuzz.gaussmf(x, 6, 1)
plt.figure(figsize=(7, 5))
plt.plot(x, gauss1, label='Simple Gaussian MF') 
plt.legend()
plt.grid(True)
plt.title('Simple Gaussian Membership Function Graph') 
plt.show()

# two-sided Gaussian membership functions
gauss21 = fuzz.gauss2mf(x, 5, 1.7, 7, 1)
gauss22 = fuzz.gaussmf(x, 5, 1.7)
gauss23 = fuzz.gaussmf(x, 7, 1)
plt.figure(figsize=(7, 5))
plt.plot(x, gauss21, label='(5, 1.7, 7, 1)')
plt.plot(x, gauss22, linestyle='dotted', label='(5, 1.7)') 
plt.plot(x, gauss23, linestyle='dotted', label='(7, 1)') 
plt.legend()
plt.grid(True)
plt.title('Two-sided Gaussian Membership Function Graph') 
plt.show()

# Generalized Bell membership function
gbellmf_values = fuzz.gbellmf(x, 2, 3, 5)
plt.figure(figsize=(7, 5))
plt.plot(x, gbellmf_values, label='Generalized Bell MF') 
plt.legend()
plt.grid(True)
plt.title('Generalized Bell Membership Function Graph') 
plt.show()

# Basic sigmoidal one-sided MF
sigm = fuzz.sigmf(x, 1, 3)
plt.figure(figsize=(7, 5))
plt.plot(x, sigm, label='Basic One-sided Sigmoidal MF') 
plt.legend()
plt.grid(True)
plt.title('Basic One-sided Sigmoidal Membership Function Graph') 
plt.show()

# Additional sigmoidal two-sided MF
sigm_dif = fuzz.dsigmf(x, 1, 3, 5, 1) 
plt.figure(figsize=(7, 5))
plt.plot(x, sigm_dif, label='Additional Two-sided Sigmoidal MF')
plt.legend()
plt.grid(True)
plt.title('Additional Two-sided Sigmoidal Membership Function Graph') 
plt.show()

# Additional asymmetric sigmoidal MF
psigm = fuzz.psigmf(x, 1, 3, 5, 1)
plt.figure(figsize=(7, 5))
plt.plot(x, psigm, label='Asymmetric Sigmoidal MF)')
plt.legend()
plt.grid(True)
plt.title('Additional Asymmetric Sigmoidal Membership Function Graph') 
plt.show()

# Polynomial Z-membership function
z = fuzz.zmf(x, 3, 5)
plt.figure(figsize=(7, 5))
plt.plot(x, z, label='Polynomial Z-membership function')
plt.legend()
plt.grid(True)
plt.title('Polynomial Z-membership Function Graph') 
plt.show()

# Polynomial S-membership function
s = fuzz.smf(x, 6, 8)
plt.figure(figsize=(7, 5))
plt.plot(x, s, label='Polynomial S-membership function')
plt.legend()
plt.grid(True)
plt.title('Polynomial S-membership Function Graph') 
plt.show()

# Polynomial PI-membership function
pi = fuzz.pimf(x, 3, 5, 6, 8)
plt.figure(figsize=(7, 5))
plt.plot(x, pi, label='Polynomial PI-membership function:')
plt.legend()
plt.grid(True)
plt.title('Polynomial PI-membership Function Graph') 
plt.show()

# Minimax interpretation
A = fuzz.trimf(x, [0, 7, 9]) # set A
B = fuzz.trapmf(x, [0, 2, 5, 7]) # set B

# AND (minimum)
min_values = np.fmin(A, B)
plt.figure(figsize=(7, 5))
plt.plot(x, min_values, label='Minimum')
plt.plot(x, A, linestyle='dotted', label='Set A') 
plt.plot(x, B, linestyle='dotted', label='Set B') 
plt.legend()
plt.grid(True)
plt.title('Minimax Interpretation of Logical Operator AND (minimum)') 
plt.show()

# OR (maximum))
max_values = np.fmax(A, B)
plt.figure(figsize=(7, 5))
plt.plot(x, max_values, label='Maximum')
plt.plot(x, A, linestyle='dotted', label='Set A') 
plt.plot(x, B, linestyle='dotted', label='Set B') 
plt.legend()
plt.grid(True)
plt.title('Minimax Interpretation of Logical Operator OR (maximum)') 
plt.show()

# Probabilistic interpretation
# AND (minimum)
values_and = A*B
plt.figure(figsize=(7, 5))
plt.plot(x, values_and, label='Minimum')
plt.plot(x, A, linestyle='dotted', label='Set A') 
plt.plot(x, B, linestyle='dotted', label='Set B') 
plt.legend()
plt.grid(True)
plt.title('Probabilistic Interpretation of Logical Operator AND (minimum)') 
plt.show()

# OR (maximum))
values_or = A+B-A*B
plt.figure(figsize=(7, 5))
plt.plot(x, values_or, label='Maximum')
plt.plot(x, A, linestyle='dotted', label='Set A') 
plt.plot(x, B, linestyle='dotted', label='Set B') 
plt.legend()
plt.grid(True)
plt.title('Probabilistic Interpretation of Logical Operator OR (maximum)') 
plt.show()

# Complement of a fuzzy set (NOT)
not_values = 1 - gauss1
plt.figure(figsize=(7, 5))
plt.plot(x, not_values, label='Complement')
plt.plot(x, gauss1, linestyle='dotted', label='Gaussian Function')
plt.legend()
plt.grid(True)
plt.title('Complement Graph (NOT Logical Operator Interpretation)') 
plt.show()
