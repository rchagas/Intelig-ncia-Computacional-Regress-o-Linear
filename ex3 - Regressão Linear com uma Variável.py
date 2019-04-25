import numpy as np
import matplotlib.pyplot as plot
import gradient_descent as gd

#inicializando dados do arquivo
data = np.loadtxt('ex1data1.txt', delimiter=',')

#define numero de iterações e o valor de alpha
iterations = 1500
alpha = 0.01

#iniciando X e Y
X = [[1 for i in data],[i[0] for i in data]]
Y = [i[1] for i in data]


#inciando valores de theta e m

theta, jPrev = gd.gradientDescent(X,Y,alpha,iterations)

print("J = ", min(jPrev))
print("theta_0 = ", theta[0])
print("theta_1 = ", theta[1])

#monta a hipotese no gráfico
xo = list( range(0,25))
Hx = [theta[0]+theta[1]*i for i in xo]

fig, ax = plot.subplots()
fig2, j = plot.subplots()

#Plota as amostras
ax.scatter(X[1], Y, label='Lucro Alcançado por População')

#Plota reta Hipótese
ax.plot(xo,Hx,label='Hipótese',color='red', dashes=[6, 2])

fig.suptitle('Exercicio 1 - Regressão Linear com uma váriavel')
ax.set_xlabel('População na cidade (por 10mil)')
ax.set_ylabel('Lucro por Food Truck (US$ 10.000)')

#Gráfico JCusto x Iterações
j.plot(list(range(iterations)),jPrev,label='Custo J x Iterações',color='green')
fig2.suptitle('Exercicio 1 - Regressão Linear com uma váriavel')
j.set_xlabel('Iterações')
j.set_ylabel('J Custo')

ax.legend()
plot.show()
