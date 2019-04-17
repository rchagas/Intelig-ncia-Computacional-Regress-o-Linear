import numpy as np
import matplotlib.pyplot as plot

def computeCost (X,Y,theta):
	m = len(Y)
	J = 0
	pred = [(theta[0]+theta[1]*i) for i in X]
	sqrErro = [(i-j)**2 for i,j in zip(Y,pred)]
	J = 1/(2*m)*sum(sqrErro)
	return J

def gradientDescent(X,Y,theta,alpha,iteracoes):
	m = len(Y)
	jPrev = [0 for i in range(iteracoes)]
	cols = 5
	rows = int((iteracoes/100)//cols)
	fig, axs = plot.subplots(rows, cols, figsize=(12, 7))
	for i in range(iteracoes):
		h = [theta[0]+theta[1]*i for i in X]
		erro = [h-y for h,y in zip(h,Y)]
		thetaNew = [theta[0] - alpha*(1/m) * sum(erro),
					theta[1] - alpha*(1/m) * sum([erro*x for x,erro in zip(X,erro)])]
		theta = thetaNew
		jPrev[i] = computeCost(X,Y,theta)
		if(i%100 == 0) and (i<1500):
			col_aux = int((i/100)%cols)
			row_aux = int((i/100)//cols)
			axs[row_aux][col_aux].scatter(X, Y)
			Hx = [theta[0]+theta[1]*i for i in list(range(25))]
			axs[row_aux][col_aux].plot(list(range(25)),Hx,label='Hipótese',color='red', dashes=[6, 2])

	return theta, jPrev
		 
	

#inicializando dados do arquivo
data = np.loadtxt('ex1data1.txt', delimiter=',')

#define numero de iterações e o valor de alpha
iterations = 1500
alpha = 0.01

#iniciando X e Y
X = [i[0] for i in data]
Y = [i[1] for i in data]


#inciando valores de theta e m
theta = [20,0]
m = len(X)

theta, jPrev = gradientDescent(X,Y,theta,alpha,iterations)

print("J = ", min(jPrev))
print("theta_0 = ", theta[0])
print("theta_1 = ", theta[1])

#monta a hipotese no gráfico
xo = list( range(0,25))
Hx = [theta[0]+theta[1]*i for i in xo]

fig, ax = plot.subplots()
fig2, j = plot.subplots()

#Plota as amostras
ax.scatter(X, Y, label='Lucro Alcançado por População')

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
