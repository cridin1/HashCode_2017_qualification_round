
import numpy as np
from math import floor
from random import randint
import time
alpha = 1
beta = 1

def checkSolution(): #funzione per verificare la soluzione
    global bestSolution
    global Matrix

    score = 0  #nuovo score
    check=0
    nRequest=0 #richieste totali per calcolare lo score
    delayMatrix=np.zeros((E,V)) #costruisco la matrice dei ritardi da utilizare nel calcolo della soluzione
    savedMatrix=np.zeros((E,V)) #costruisco la matrice del tempo salvato per ogni video


    for i in range(E):
        for k in range(V):
            j_max = -1 
            for j in range(C):
                if(x[k][j]==1 and requestMatrix[i][k] > 0 and delayMatrix[i][k] < pingMatrix[i][j] and y[i][j]==1):
                    #se il video è presente nella cache, c'è una richiesta per quel video e la latenza di quella cache è minore della latenza fin'ora trovata e l'enpoint i fa riferimento alla cache j
                    delayMatrix[i][k] = pingMatrix[i][j] #aggiorno il delay per il video con il ping per la cache
                    savedMatrix[i][k] = latencies[i] - delayMatrix[i][k]  #aggiorno il valore della latenza salvata
                    check = 1  #variabile che serve a capire se c'è un collegamento ad una cache per il nodo i
                    j_max = j  #salvo l'indice della cache ottima
            if(j_max != -1):
                Matrix[i][j_max][k] = 1 #aggiorno il valore in matrix
            if(check == 0 and requestMatrix[i][k] > 0):  #se il nodo i non è legato ad alcuna cache per il video K, ma ci sono richieste per quel video
                delayMatrix[i][k]=latencies[i]   #allora la matrice dei ritardi conterrà il ritardo dal dataCenter
                savedMatrix[i][k]=0      #il ritardo risparmiato è 0
            check=0



    for k in range(E): #per ogni end-point
        for i in range(V): #per ogni video
            score+= requestMatrix[k][i] * savedMatrix[k][i]  
            nRequest += requestMatrix[k][i]
    score=score/nRequest
    score=score*1000

    if(score > bestSolution):
        print("solBest:",score)
        bestSolution=score
        return 1
    else:
        return 0





def greedy_best():     
#per ogni endpoint calcolo le differenze tra gli archi verso i datacenter e le cache 
#scelgo le differenze massime
#riempio la cache con video piu richiesti
# ripeto finche tutte le cache non sono piene

    memoryUsed[:] = T #istanzio vettore che tiene traccia, per ogni cache, della memoria disponibile
    requestMatrix_c = requestMatrix.copy() #copia della matrice delle richieste
                                           #la copia è necessaria dato che tale matrice, in questa funzione, verrà modificata 
    
    assorbimento = np.zeros((C))  
    assorbimento[:] = 1
    num_iteration = 0

    while(num_iteration < C):

        massimo = 0
        i_max = 0
        j_max = 0

        for i in range(E):
            for j in range(C):
                if (latencies[i] - pingMatrix[i][j] > massimo and y[i][j] != 0 and np.any(requestMatrix_c[i]) and assorbimento[j] == 1):     #scelgo la cache rispetto all'endpoint i se esiste un arco tra i due 
                                                                                                                        #ed esistono dei video da usare
                                                                                                                         #e la cache è vuota
                    massimo = latencies[i] - y[i][j]
                    i_max = i #endpoint buono
                    j_max = j #cache buona

        coda_video = np.zeros((2,V))
        
        coda_video[0] = requestMatrix_c[i_max].copy()
        coda_video[1] = size
        coda_video = coda_video.T
        #print(coda_video)
        while(memoryUsed[j_max] > 0 and coda_video.size != 0 and np.any(coda_video[:,0])):  #knapsack greedy se sono rimasti degli elementi da checkare allora continua

            video_max = np.argmax(coda_video[:,0])  #trovo il video piu richiesto rispetto all endpoint i
            #print(coda_video[:,0])
            size_video = coda_video[video_max][1]

            if(size_video <= memoryUsed[j_max]):   #se ci entra lo inserisco

                memoryUsed[j_max] -= size_video    #aggiorno la capacità
                requestMatrix_c[i_max][video_max] = 0  #aggiorno EV, ho soddisfatto la richiesta
                x[video_max][j_max] = 1

             #prova ad inserire il video in una delle cache ancora non piene collegate all'endpoint
                
                

            coda_video = np.delete(coda_video,video_max,axis = 0)    #se il video non entra lo elimino dalla coda

        #cache j_max assorbita
        
        #print(memoryCapacity[j_max])
        assorbimento[j_max] = 0
        num_iteration += 1
        #print(np.any(assorbimento))
        
    checkSolution()  

def knapsack01(latency_gains,cacheIndex):
    ''' Solves the 0/1–knapsack problem in quadratic time and quadratic space using a bottom–up DP approach. '''
    W = int(T - memoryUsed[cacheIndex])
    n = len(latency_gains)
    video_sizes = size[:floor(V * alpha)]

    weights, v = np.hstack(([0], video_sizes)), np.hstack(([0], latency_gains))
    dp = np.zeros((n + 1, W + 1), dtype = np.int64)
    for i in range(1, n + 1):
        for j in range(W + 1):
            if weights[i] > j:
                dp[i, j] = dp[i - 1, j]
            else:
                dp[i, j] = max(dp[i - 1, j - weights[i]] + v[i], dp[i - 1, j])

    # Now we need to do the backtracking to come up with the actual optimal configuration
    i, j, videos, total_weight, total_value = n, W, [], 0, 0
    while i > 0 and j > 0:
        while dp[i, j] == dp[i - 1, j] and i >= 0:
            i -= 1

        if i <= 0:
            break

        # Video i must be part of the solution
        videos.append(i - 1)
        total_value += v[i]
        total_weight += weights[i]

        j -= weights[i]
        i -= 1

    assert len(videos) == len(set(videos)), 'Invalid solution: Items should only appear at most once in the cache.'
    assert total_value == dp[-1, -1], 'Invalid solution: Optimal total value should be %d, but is %d instead.' % (dp[-1, -1], total_value)
    assert total_weight <= W, 'Invalid solution: Total weight of items can not exceed cache capacity...'

    return videos

def video_Score(cacheIndex):
    score = [(0,j) for j in range(V)]
    
    for k in range(V):
        if(x[k][cacheIndex] == 0): #se il video non è presente in quella cache
            for i in range(E):
                if(requestMatrix[i][k] != 0 and y[i][cacheIndex] == 1):
                    temp = score[k][0]
                    score[k] = (temp + requestMatrix[i][k] * (latencies[i] - pingMatrix[i][cacheIndex]),k)

    #score.sort(key = key_get)
    return score[:floor(V)]


def key_get(tupla):
    return tupla[0]

def cache_Score():
    score = [(0,j) for j in range(C)]  #vocabolario con chiave lo score e valore la cache


    for i in range(E):
        for k in range(V):
            for j in range(C):
                if(Matrix[i,j,k] == 1):  #se l'endpoint i fa riferimento alla cache j per il video k
                 score[j] = (score[j][0]+requestMatrix[i][k]*(latencies[i] - pingMatrix[i][j]),j)
                    
#lo score di una cache viene calcolato in base al numero di richieste che soddisfa e che miglioramento porta
#rispetto alla richiesta al data point
                    
    score.sort(key = key_get)  #vengono ordinate le cache in base all'indice
    score2 = [e[1] for e in score]  #prelevo il vettore con lo score per ogni indice
    return score2[:floor(C/10)] #restituisco solo la metà del vettore, per valutare solo le cache che mi forniscono un certo miglioramento della soluzione



def localSearch_knapsack():
    global x
        
    scoreCache = cache_Score() #calcolo lo score per ogni cache. L'indice del vettore rappresenta la cache
    n = len(scoreCache)
    index= 0
    while(index<n): #ciclo esterno che itera finchè non vedo tutte le cache
        cacheIndex = scoreCache[index]  #prelevo l'indice della cache con score più alto

        video_gains = video_Score(cacheIndex)
        
        video_gains2 = [e[0] for e in video_gains]

        video_cache = knapsack01(video_gains2,cacheIndex)


        if(len(video_cache) == 0):
            index += 1
        else:
            for video in video_cache:
                index_video = video_gains[video][1]
                x[index_video][cacheIndex] = 1
                memoryUsed[cacheIndex] += size[index_video] 
            
            scoreCache = cache_Score()
            index = 0
    checkSolution()




start = time.time()
bestSolution=0
fileName = "./trending_today.in"
f = open(fileName)
content = f.readlines() #lista di tutte le linee del file
info = content[0].split() #divido la riga in tutte parole

#salvo ogni parola con il significato che gli è stato attribuito

V = int(info[0]) #numero di video
E = int(info[1]) #numero di end-point
R = int(info[2]) #numero di descrizioni di richieste
C = int(info[3]) #numero di cache server
T = int(info[4]) #capacità di ogni cache server in MB

#siccome la seconda riga contiene le size dei vari video, le salvo

size = [int(a) for a in content[1].split()]

latencies = []  #salvo la latenza di ogni end-point dal dataStore

numCacheServer = [] #salvo il numero di cache server a cui ogni nodo è collegato

pingMatrix = np.zeros((E,C)) #matrice dei ritardi da ogni end-point al cache server

y = np.zeros((E,C))

k=0



for a in range(E):
    row = content[2+k].split()
    latencies.append(int(row[0]))  #salvo la latenza di ogni end-point dal dataStore
    numCacheServer.append(int(row[1])) #salvo il numero di cache a cui è legato ogni end-point
    k+=1
    for i in range(numCacheServer[a]):  #per ogni cache a cui è collegato l'end-point a
        row = content[2+k].split()
        j = int(row[0])    #leggo identificativo cache-center
        pingMatrix[a][j] = int(row[1]) #salvo nella matrice le latenze indicate
        k+=1

    for i in range(C):
        if (pingMatrix[a][i] == 0) :
            pingMatrix[a][i] = 50000000000 #se l'end-point a non è legato alla cache i, introduco nella matrice un valore elevato, in modo da non scegliere quel server
            
        else :
            y[a][i]=1

requestMatrix = np.zeros((E,V))#matrice delle richieste dell'end-point per ogni video

for a in range(R):
    row = content[2+k].split()
    requestMatrix[int(row[1])][int(row[0])] = int(row[2]) #righe per end-point, colonne per video
    k+=1
    
#finelettura dati input  
#inizio euristica greedy
#partendo dall'endpoint con latenza maggiore, provo a salavre ogni video richiesto nel cache server
memoryUsed = np.zeros(C) #salvo la memoria usata da ogni cache
x = np.zeros((V,C))  #con Xi,j =1 indico che il video i è presente nella cache j

n_request = sum(sum(requestMatrix))
Matrix = np.zeros((E,C,V))  #matrice che indica se l'endpoint E fa riferimento al video V traemite la cache C


greedy_best()
localSearch_knapsack()
#print(memoryUsed)
end = time.time()
print("finish")
print("tempo esecuzione:",end - start)