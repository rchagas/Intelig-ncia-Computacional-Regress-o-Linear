def computeCost (X,Y,theta):
	m = len(Y)
	J = 0
	pred = [0 for i in range(m)]
	for i in range(m):
		for j in range(len(theta)):
			pred[i] += theta[j]*X[j][i]
	sqrErro = [(y-h)**2 for y,h in zip(Y,pred)]
	J = 1/(2*m)*sum(sqrErro)
	return J

def gradientDescent(X,Y,alpha,iteracoes):
	theta = [0 for i in range(len(X))]
	m = len(Y)
	jPrev = [0 for i in range(iteracoes)]
	thetaNew = [0 for i in range(len(theta))]
	for i in range(iteracoes):
		h = [0 for i in range(m)]
		for j in range(m):
			for z in range(len(theta)):
				h[j] += theta[z]*X[z][j]
		erro = [h-y for h,y in zip(h,Y)]
		for j in range(len(theta)):
			thetaNew[j] = theta[j] - alpha*(1/m) * sum([erro*x for x,erro in zip(X[j],erro)])
		theta = thetaNew
		
		jPrev[i] = computeCost(X,Y,theta)
	return theta, jPrev
