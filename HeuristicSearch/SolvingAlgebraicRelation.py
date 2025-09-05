from constraint import *


problem = Problem()
problem.addVariable('a', range(10))
problem.addVariable('b', range(10))
problem.addConstraint(lambda a, b: a*2 == b)
solutions = problem.getSolutions()
print(solutions)

#MAGIC SQUARE: an arrangement of distinct numbers, generally integers, in a square grid, where the
#              numbers in each row, and in each column, and the numbers in the diagonal, all add up to the same number called the “magic constant”.

def magic_square(matrix_ms):
    isSize = len(matrix_ms[0])
    sum_list = []

    #sum of columns
    for col in range(isSize):
        sum_list.append(sum(row[col] for row in matrix_ms))

    #sum of rows
    sum_list.extend([sum(lines) for lines in matrix_ms])

    #diagonale
    dlResult = 0
    for i in range (0, isSize):
        dlResult += matrix_ms[i][i]
    sum_list.append(dlResult)

    #diagonale inversa
    dlResult = 0
    for i in range (isSize):
        dlResult += matrix_ms[i][isSize-i-1]
    sum_list.append(dlResult)

    if len(set(sum_list))>1:
        return False
    return True

print(magic_square([[1,2,3], [4,5,6], [7,8,9]]))
print(magic_square([[2,7,6],[9,5,1],[4,3,8]]))





