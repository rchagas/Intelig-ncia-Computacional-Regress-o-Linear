import numpy as np
import matplotlib.pyplot as plot
import gradient_descent as gd
 
from mpl_toolkits.mplot3d import axes3d, Axes3D

#inicializando dados do arquivo
data = np.loadtxt('ex1data2.txt', delimiter=',')

#define numero de iterações e o valor de alpha
iterations = 2500
alpha = 0.5

#iniciando X com mesma escala 0 a 1 
X = [[1 for i in data],[i[0]/5000 for i in data],[i[1]/5 for i in data]]
Y = [i[2] for i in data]

#inciando valores de theta e m

theta, J = gd.gradientDescent(X,Y,alpha,iterations)

print("J (Custo) = ", min(J))
print("theta_0 = ", theta[0])
print("theta_1 = ", theta[1])
print("theta_2 = ", theta[2])

fig, amostras = plot.subplots(1, 2, figsize=(12, 4))
fig2, resultado = plot.subplots()

#Grafico Tamanho x Preço
amostras[0].scatter([i[0] for i in data], [i[2]/1000 for i in data])

#Grafico Num de Quartos x Preço
amostras[1].scatter([i[1] for i in data], [i[2]/1000 for i in data])

#Grafico J Custo x Iterações
resultado.plot(list(range(len(J))),[i for i in J], color='red')

amostras[0].set_xlabel('Área (pés²)')
amostras[0].set_ylabel('Preço (US$ 1000)')
amostras[1].set_xlabel('Numero de Quartos')
resultado.set_xlabel('Iterações')
resultado.set_ylabel('J - Custo')

fig.suptitle('Exercicio 2 - Dipersão Amostral')
fig2.suptitle('Exercicio 2 - Regressão Linear Multiváriavel')
#plot do plano resultado leve em consideração x1 e x2 de 0 a 1

fig3 = plot.figure()
rl = Axes3D(fig3) #<-- Note the difference from your original code...

x1 = np.arange(0, 1, 0.1)
x2 = np.arange(0, 1, 0.1)
x1, x2 = np.meshgrid(x1, x2)
H = (theta[0]+theta[1]*x1+theta[2]*x2)/1000


surf = rl.plot_surface(x1, x2, H, color='green')
rl.set_zlim(0, 800)
rl.set_xlabel('X¹ - Área')
rl.set_ylabel('X² - Número de Quartos')
rl.set_zlabel('H - Preço Estimado (US$ 1000)')

plot.show() 

#axs.legend()
plot.show()
