import sympy as sp

# Defining the symbols
a, b, t, x, y = sp.symbols("a b t x y")
X_1, X_2, M_1, M_2, C_1, C_2, Y_1, Y_2, m_1, m_2, A_1, A_2 = sp.symbols("X_1 X_2 M_1 M_2 C_1 C_2 Y_1 Y_2 m_1 m_2 A_1 A_2")

# Making matrices
A = sp.Matrix([[3, 0], [2, 6]])
B = sp.Matrix([[a, a], [b, b]])
C = sp.Matrix([[(a+b), (a-b)], [(a-b), (a+b)]])
D = sp.Matrix([[3**t, 2**t], [3**(t-1), 2**(t-1)]])
E = sp.Matrix([[3, -1], [1, -2]])
F = sp.Matrix([[8], [5]])

# Define equations correctly
A1 = Y_1 * (1 + m_1 - C_1) - m_2 * Y_2
A2 = Y_2 * (1 - C_2 - m_2) - m_1 * Y_1

# Solve for Y_1, Y_2
eq1 = sp.Eq(A1, A_1)
eq2 = sp.Eq(A2, A_2)
sol1 = sp.solve([eq1, eq2], (Y_1, Y_2))

#Determinants
print("Det(A):", A.det())
print("Det(B):", B.det())
print("Det(C):", C.det())
print("Det(D):", D.det())

# Solve E * X = F
X = E.LUsolve(F)
print("Solution to E * X = F:", X)

# Print solution for Y_1, Y_2
print("Solution for Y_1, Y_2:", sol1)
