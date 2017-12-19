import numpy as np
import matplotlib.pyplot as plt

def qFunc(x, moyenne, ecart):
    return 1/(ecart*np.sqrt(np.pi * 2)) * np.exp(-0.5* ((x-moyenne) / ecart)**2)

def fFunc(z, m, e):
    return qFunc(z, m,e)

def f(x, a,b):
    return x if a <=x and x <= b else -9999 

def monteCarlo1DCarre(N):
    a = -1
    b = 1
    x = np.zeros(N);
    q = np.zeros(N);
    i = 0
    while(i < N):
        z = tirage(b)
        mq = 2 * qFunc(z,0,1)
        u = np.random.uniform(0,mq)
        mpi = fFunc(z, 0,0.5)
        if u <= mpi:
            x[i] = z
            q[i] = u#mq
            i = i + 1
    return (x,q)

#N = 1000
#x,u = monteCarlo1DCarre(N);
#plt.plot(x, u, "go")
#plt.show()

def tirage(m):
    r1 = np.random.uniform(-1,1) * m;
    r2 = np.random.uniform(-1,1) * m;
    return r1 * r2;

def monteCarlo(N):
    a = -1
    b = 1
    x = np.zeros(N);
    y = np.zeros(N);
    i = 0
    j = 0
    while(i < N or j < N):
        #(1) tirer un nombre z selon q(·) (pre-´echantillonnage)
        Z1 = tirage(b+5)
        Z2 = tirage(b+5)
        #(2) calculer mq = k · q(z)
        mq1 = qFunc(Z1, a,b)
        mq2 = qFunc(Z2, a,b)
        #(3) tirer un nombre u selon
        #la distribution uniforme sur [0, mq]
        u1 = tirage(3)
        u2 = tirage(3)
        #(4) accepter z comme ´echantillon si u ≤ f(x).
        
        if u1 <= f(Z1, a, b) and i < N:
            x[i] = Z1
            i = i + 1
        if u2 <= f(Z2, a, b) and j < N:
            y[j] = Z2
            j = j + 1
    ind = indicatrice(x,y)
    pi = 4 * esperance(x[ind],y[ind])
    return (pi,x,y)

def indicatrice(X,Y):
    return np.sqrt(X**2 + Y**2) <= 1

def esperance(X,Y):
    return (np.mean(X) + np.mean(Y)) / 2

plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par Monte Carlo
pi, x, y = monteCarlo(int(3*1e3))

# trace les points dans le cercle et hors du cercle
dist = x*x + y*y 
plt.plot(x[dist <=1], y[dist <=1], "go")
plt.plot(x[dist>1], y[dist>1], "ro")
plt.show()


