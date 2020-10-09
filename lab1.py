#!/usr/bin/env python3

import math # math.pi, ...
import matplotlib.pyplot as plt # visualization
import random # random

def metricDef(x1, x2):
	return abs(x1 - x2)

def kernelGauss(x):
	#return 1 / (2 * math.pi) * math.exp(x * x * (-0.5))
	return math.exp(x * x * (-0.5))

def kernelQuartic(x):
	return math.pow(1 - x*x, 2) if abs(x) < 1 else 0

def kernelEpach(x):
	return 3./4*(1 - x*x) if abs(x) < 1 else 0

def sse(Y, A):
	assert len(Y) == len(A)
	res = 0
	for i in range(len(Y)):
		res += (Y[i] - A[i]) ** 2
	return res

def NadarayaVatsona(X, Y, kernelFunc, metric, h):
	assert len(X) == len(Y)
	l = len(X)
	res = []
	for xi in X:
		nom = 0
		denom = 0
		for i in range(l):
			k = kernelFunc(metric(xi, X[i]) / h)
			nom += Y[i] * k
			denom += k
		res.append(nom / denom)
	return res

def lowess(X, Y, kernelFunc, metric, h, smoothKernelFunc):
	assert len(X) == len(Y)
	l = len(X)
	smoothing = [1] * l
	res = []

	# iteration count
	for _ in range(10):
		res = []
		for i in range(l):
			nom = 0
			denom = 0
			for j in range(l):
				k = kernelFunc(metric(X[i], X[j]) / h)
				nom += Y[j] * k * smoothing[j]
				denom += k * smoothing[j]
			res.append(nom / denom)

		newSmoothing = []
		for i in range(l):
			nom = 0
			denom = 0
			for j in range(l):
				if i == j:
					continue
				k = kernelFunc(metric(X[i], X[j]) / h)
				nom += Y[j] * k * smoothing[j]
				denom += k * smoothing[j]
			err = abs((nom / denom) - Y[i])
			newSmoothing.append(smoothKernelFunc(err))
		smoothing = newSmoothing
	return res

def bruteVar(begin, end, step, func):
	mi = begin
	mv = 9999
	while begin < end:
		cv = func(begin)
		if cv < mv:
			mi = begin
			mv = cv
		begin += step
	return mi

def main():
	l = 20
	f = lambda x: 3 - x / 3
	coef = [0] * l
	coef[10] = - 10 # выброс
	X = list(range(0, l))
	Y = [f(X[i] + random.random() * 2 - 1) + coef[i] for i in range(l)]
	#X = [1, 2, 2.8, 3, 3.2, 3.6, 4,   5, 5.2, 5.4, 5.8, 6,   7, 8,   9, 10, 11, 12, 13, 14]
	#Y = [2, 1, 2,   2, 2,   2.1, 2.2, 2, 1.9, 1.8, 1.6, 1.5, 1, 1.5, 2, 2,  3,  2,  2,  1]

	#h1 = bruteVar(1, 9, 0.1, lambda x: sse(Y, NadarayaVatsona(X, Y, kernelGauss, metricDef, x)))
	#print(h1)

	resG4 = NadarayaVatsona(X, Y, kernelGauss, metricDef, 4)
	resG9 = NadarayaVatsona(X, Y, kernelGauss, metricDef, 9)
	resQ4 = NadarayaVatsona(X, Y, kernelQuartic, metricDef, 4)
	resQ9 = NadarayaVatsona(X, Y, kernelQuartic, metricDef, 9)
	resE4 = NadarayaVatsona(X, Y, kernelEpach, metricDef, 4)
	resE9 = NadarayaVatsona(X, Y, kernelEpach, metricDef, 9)

	resGQ4 = lowess(X, Y, kernelGauss, metricDef, 4, kernelQuartic)
	resGQ9 = lowess(X, Y, kernelGauss, metricDef, 9, kernelQuartic)
	resGE4 = lowess(X, Y, kernelGauss, metricDef, 4, kernelEpach)
	resGE9 = lowess(X, Y, kernelGauss, metricDef, 9, kernelEpach)	
	
	plt.plot(X, Y, 'bo', label = 'Xs')
	plt.plot(X, resG4, color = "#00FF00", label = "G4")
	plt.plot(X, resG9, color = "#8EFF9D", label = "G9")
	plt.plot(X, resQ4, color = "#FF0000", label = "Q4")
	plt.plot(X, resQ9, color = "#FF8B89", label = "Q9")
	plt.plot(X, resE4, color = "#0000FF", label = "E4")
	plt.plot(X, resE9, color = "#8296FF", label = "E9")
	plt.plot(X, resGQ4, color = "#FF00FF", label = "GQ4")
	plt.plot(X, resGQ9, color = "#B14CFF", label = "GQ9")
	plt.plot(X, resGE4, color = "#00FFFF", label = "GE4")
	plt.plot(X, resGE9, color = "#59A6FF", label = "GE9")
	plt.legend(loc = 'upper right')
	plt.show()

	print(f"G4 err: {sse(Y, resG4)}")
	print(f"G9 err: {sse(Y, resG9)}")
	print(f"Q4 err: {sse(Y, resQ4)}")
	print(f"Q9 err: {sse(Y, resQ9)}")
	print(f"E4 err: {sse(Y, resE4)}")
	print(f"E9 err: {sse(Y, resE9)}")
	print(f"GQ4 err: {sse(Y, resGQ4)}")
	print(f"GQ9 err: {sse(Y, resGQ9)}")
	print(f"GE4 err: {sse(Y, resGE4)}")
	print(f"GE9 err: {sse(Y, resGE9)}")

if __name__ == '__main__':
	main()