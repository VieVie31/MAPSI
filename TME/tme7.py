import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import warnings
# truc pour un affichage plus convivial des matrices numpy
np.random.seed(0)
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

#warnings.filterwarnings("ignore")

with open("TME6_lettres.pkl", "rb") as f:
    data = pkl.load(f, encoding="latin1")
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))
nCl = 26


def discretise(x, d):
    # Discretise x en d etats differents.
    intervalle = 360 / d
    temp = x / intervalle
    return np.array([np.int64(np.floor(x)) for x in temp])


def initGD(X, N):
    return [np.floor(np.linspace(0,N-.00000001,len(x))) for x in X]

def learnHMM(X, Q, N, K, epsi=1e-8):
    A = np.ones((N, N)) * epsi
    B = np.ones((N, K)) * epsi
    Pi = np.ones(N) * epsi

    # A
    d_grams = []
    for s in Q:
        d_grams += list(zip(*[s[i:] for i in range(2)]))

    d_grams = list(map(lambda a: tuple(map(int, a)), d_grams))
    for i in d_grams:
        A[i] += 1
    s = A.sum(1).reshape(N, 1)
    s[s == 0] = 1
    A = A / s

    # B
    for q, x in zip(Q, X):
        for i in range(len(x)):
            B[q[i], x[i]] += 1
    s = B.sum(1).reshape(N, 1)
    s[s == 0] = 1
    B = B / s

    # Pi
    for x in Q:
        Pi[int(x[0])] += 1

    Pi = Pi / Pi.sum()

    return A, B, Pi



argmax = lambda T: np.array(T).argmax()

def viterbi(O, Pi, A, B):
    delta = np.zeros((len(A), len(O)))
    psi   = np.zeros((len(A), len(O)))

    # Initialization
    delta[:, 0] = np.log(B[:, O[0]]) + np.log(Pi)
    psi[:, 0] = -1

    # Recursion
    for t in range(1, len(O)):
        for j in range(len(A)):
            tmp = delta[:, t-1] + np.log(A[:, j])
            delta[j, t] = max(tmp) + np.log(B[j, O[t]])
            psi[j, t] = argmax(tmp)

    # Termination
    S = max(delta[:, -1])

    # Path
    T = len(O)
    s = [0] * T
    s[T - 1] = argmax(delta[:, t])
    for t in range(T - 2, 1, - 1):
        s[t] = psi[:, t + 1][int(s[t + 1])]

    return np.int64(s), S


all_letters = 'abcdefghijklmnopqrstuvwxyz'
def baum_welch(X, Y, N, K, epsi=1e-4):
    Q = {c: np.array(list(map(np.int64, initGD(X[Y == c], N)))) for c in all_letters}

    #histo = []
    converged = False
    A, B, Pi = {}, {}, {}
    probs = {}
    likelihoods = [1]
    while not converged:
        for c in all_letters:
            probs[c] = []
            a, b, pi = learnHMM(X[Y == c], Q[c], N, K)

            qtemp, ptemp = [], []
            for x in X[Y == c]:
                qt, pt = viterbi(x, pi, a, b)
                qtemp.append(qt)
                ptemp.append(pt)

            A[c] = a
            B[c] = b
            Pi[c] = pi
            Q[c] = qtemp
            probs[c] = ptemp

        lik = 0
        for c in all_letters:
            for i in range(len(X[Y == c])):
                lik += probs[c][i]

        converged = ((likelihoods[-1] - lik) / likelihoods[-1] < epsi)
        likelihoods.append(lik)
        #histo.append({'a': A, 'b': B, 'pi': Pi, 'p': probs, 'q': Q})

    return A, B, Pi, likelihoods


def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest

def predict(x, Pi, A, B):
    return all_letters[argmax([viterbi(x, Pi[c], A[c], B[c])[1] for c in A.keys()])]

def predict_all(X, Pi, A, B):
    return [predict(x, Pi, A, B) for x in X]

def eval(X, Y, Pi, A, B):
    temp = np.array(predict_all(X, Pi, A, B))
    return (Y == temp).mean()

def first_part():
    K = 10 # discrétisation (=10 observations possibles)
    N = 5  # 5 états possibles (de 0 à 4 en python)
    Xd = np.array(list(map(np.int64, discretise(X, K))))
    Q = np.array(list(map(np.int64, initGD(X, N))))
    A, B, Pi = learnHMM(Xd[Y == 'a'], Q[Y == 'a'], N, K)
    print(A)
    print(B)
    print(Pi)

    print(viterbi(Xd[0], Pi, A, B))
    A, B, Pi, liks = baum_welch(Xd, Y, N, K)
    plt.plot(liks[1:])
    plt.show()

def evaluation():
    K = 10 # discrétisation (=10 observations possibles)
    N = 5  # 5 états possibles (de 0 à 4 en python)
    Xd = np.array(list(map(np.int64, discretise(X, K))))
    A, B, Pi, liks = baum_welch(Xd, Y, N, K)
    print("Baum-Welch :", eval(Xd, Y, Pi, A, B))

    Q = {c: np.array(list(map(np.int64, initGD(X[Y == c], N)))) for c in all_letters}
    A, B, Pi = {}, {}, {}
    for c in all_letters:
        a, b, pi = learnHMM(Xd[Y == c], Q[c], N, K)

        qtemp = []
        for x in Xd[Y == c]:
            qt, pt = viterbi(x, pi, a, b)
            qtemp.append(qt)

        A[c] = a
        B[c] = b
        Pi[c] = pi
        Q[c] = qtemp
    print("Gauche Droite :", eval(Xd, Y, Pi, A, B))

evaluation()
