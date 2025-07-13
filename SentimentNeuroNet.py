"""
*** SENTIMENT DETECTOR NEURAL NETWORK ***

Il programma è una rete neurale (scritta senza l'uso di librerie esterne come PyTorch o TensorFlow) per classificare il sentiment di una frase in lingua italiana in una delle seguenti quattro categorie: positivo, negativo, neutro, arrabbiato.



DESCRIZIONE

Il modello è un classificatore multi-classe: analizza frasi testuali, genera un vettore di embedding per ciascuna, lo passa attraverso due hidden layer e infine produce una distribuzione di probabilità tra le classi, assegnando l’etichetta più probabile.

La rete neurale ha la seguente architettura: 
- Embedding layer: vettore medio dei pesi associati alle parole della frase
- Primo hidden layer: 64 neuroni con attivazione ReLU
- Secondo hidden layer: 32 neuroni con attivazione ReLU
- Output layer: 4 neuroni (uno per classe) con softmax finale

L'addestramento della rete avviene ogni volta su un dataset incluso nel codice, mediante l'uso di Stochastic Gradient Descent (SGD).
Il dataset è stato creato manualmente, con frasi distribuite sulle 4 emozioni.
La rappresentazione dei testi è basata su bag-of-words mediato con embedding casuali.



LIMITAZIONI

1) La rete non utilizza nessun NLP avanzato: niente stopword, stemming, né tokenizzazione robusta.

2) La rete apprende su una sola frase per ciclo, non c'è nessuna normalizzazione né batch training.

3) Non c'è nessuna regolarizzazione del dizionario: niente dropout o controllo sull'overfitting.

4) L'embedding è molto approssimativo: i vettori sono inizializzati a caso e aggiornati via backpropagation, ma non catturano un significato linguistico reale.

5) La generalizzazione è molto limitata: il vocabolario è creato solo dal training set, quindi le parole sconosciute sono ignorate nei test.

6) Il dataset è decisamente troppo piccolo per lo spazio semantico oggetto della rete.

7) Le prestazioni sono fortemente dipendenti dalla casualità iniziale.


"""


# (0) importazione dei moduli di base necessari

import random, math
from collections import Counter



# (1) DATASET (~200 frasi equamente distribuite tra i 4 sentiment definiti)

full_dataset = [
    ("mi sento felice oggi", "positivo"),
    ("è stata una giornata meravigliosa", "positivo"),
    ("sono soddisfatto dei miei progressi", "positivo"),
    ("ho ricevuto una notizia fantastica", "positivo"),
    ("mi sento motivato e pieno di energia", "positivo"),
    ("tutto sta andando bene", "positivo"),
    ("mi sento fortunato per ciò che ho", "positivo"),
    ("la passeggiata mi ha rilassato", "positivo"),
    ("ho dormito bene e sono riposato", "positivo"),
    ("mi sento amato e apprezzato", "positivo"),
    ("questa esperienza è stata fantastica", "positivo"),
    ("sono grato per la mia famiglia", "positivo"),
    ("oggi splende il sole e sono felice", "positivo"),
    ("ho trascorso un momento bellissimo", "positivo"),
    ("sono orgoglioso di me stesso", "positivo"),
    ("mi sento in pace con me stesso", "positivo"),
    ("è bello sentirsi compresi", "positivo"),
    ("mi sento sereno", "positivo"),
    ("sto imparando cose nuove", "positivo"),
    ("mi sento ispirato", "positivo"),
    ("tutto sta andando secondo i piani", "positivo"),
    ("mi sento pieno di speranza", "positivo"),
    ("sono riuscito in ciò che volevo", "positivo"),
    ("la musica mi ha reso allegro", "positivo"),
    ("ho aiutato qualcuno oggi", "positivo"),
    ("oggi è una giornata perfetta", "positivo"),
    ("mi sento valorizzato", "positivo"),
    ("ogni cosa sta andando per il meglio", "positivo"),
    ("ho avuto una piacevole conversazione", "positivo"),
    ("sono entusiasta del mio lavoro", "positivo"),
    ("mi sento rinnovato", "positivo"),
    ("sono riuscito a rilassarmi", "positivo"),
    ("mi sento felice con me stesso", "positivo"),
    ("ho ricevuto un bel complimento", "positivo"),
    ("è stata una giornata produttiva", "positivo"),
    ("tanta gratitudine per oggi", "positivo"),
    ("mi sento pieno di energia positiva", "positivo"),
    ("è bello poter sorridere così", "positivo"),
    ("oggi ho avuto un bel momento di gioia", "positivo"),
    ("la natura mi ha fatto sentire bene", "positivo"),
    ("ho trovato una soluzione importante", "positivo"),
    ("ho ricevuto supporto e mi ha fatto piacere", "positivo"),
    ("oggi mi sento carico", "positivo"),
    ("tutto è filato liscio oggi", "positivo"),
    ("mi sento finalmente soddisfatto", "positivo"),
    ("la mia autostima è cresciuta", "positivo"),
    ("ho avuto una giornata positiva", "positivo"),
    ("sono fiero del mio impegno", "positivo"),
    ("oggi mi sento realizzato", "positivo"),
    ("ho affrontato tutto con serenità", "positivo"),
    ("una gioia semplice ma intensa", "positivo"),
    ("mi sento triste e solo", "negativo"),
    ("è stata una giornata pesante", "negativo"),
    ("mi sento vuoto dentro", "negativo"),
    ("oggi è stato un disastro", "negativo"),
    ("non ho voglia di fare nulla", "negativo"),
    ("mi sento frustrato", "negativo"),
    ("è tutto così difficile", "negativo"),
    ("ho fallito ancora una volta", "negativo"),
    ("mi sento bloccato e confuso", "negativo"),
    ("non riesco a reagire", "negativo"),
    ("sento solo dolore", "negativo"),
    ("è stato un giorno da dimenticare", "negativo"),
    ("non riesco a sorridere", "negativo"),
    ("mi sento inutile", "negativo"),
    ("sento di aver deluso tutti", "negativo"),
    ("ho avuto solo problemi", "negativo"),
    ("la stanchezza mi schiaccia", "negativo"),
    ("non ho concluso niente", "negativo"),
    ("mi sento sopraffatto", "negativo"),
    ("sono stanco di tutto", "negativo"),
    ("oggi va tutto storto", "negativo"),
    ("mi sento scoraggiato", "negativo"),
    ("niente sembra avere senso", "negativo"),
    ("sento di non farcela", "negativo"),
    ("sono stato ignorato", "negativo"),
    ("ho perso interesse in tutto", "negativo"),
    ("mi sento completamente svuotato", "negativo"),
    ("sto male senza motivo", "negativo"),
    ("non vedo una via d’uscita", "negativo"),
    ("è stato un fallimento totale", "negativo"),
    ("ho perso la motivazione", "negativo"),
    ("mi sento fuori luogo", "negativo"),
    ("la giornata è stata grigia", "negativo"),
    ("mi sento spezzato dentro", "negativo"),
    ("ho rovinato tutto", "negativo"),
    ("sento solo fatica", "negativo"),
    ("mi sento spento", "negativo"),
    ("ogni tentativo è inutile", "negativo"),
    ("sento solo silenzio", "negativo"),
    ("mi manca ogni stimolo", "negativo"),
    ("mi sento sotto pressione", "negativo"),
    ("ho fallito me stesso", "negativo"),
    ("mi manca la forza", "negativo"),
    ("sto annegando nei pensieri", "negativo"),
    ("ogni cosa mi pesa", "negativo"),
    ("sento di non appartenere", "negativo"),
    ("mi sento oppresso", "negativo"),
    ("niente è andato come speravo", "negativo"),
    ("ho pianto oggi", "negativo"),
    ("mi sento in trappola", "negativo"),
    ("ho fatto colazione", "neutro"),
    ("sono uscito a fare la spesa", "neutro"),
    ("ho sistemato la scrivania", "neutro"),
    ("sto leggendo un articolo", "neutro"),
    ("ho acceso il computer", "neutro"),
    ("sto aspettando il corriere", "neutro"),
    ("ho fatto una telefonata", "neutro"),
    ("sto preparando la cena", "neutro"),
    ("mi sono svegliato presto", "neutro"),
    ("ho preso un caffè", "neutro"),
    ("ho camminato un po’", "neutro"),
    ("sto stirando i vestiti", "neutro"),
    ("sto ricaricando il telefono", "neutro"),
    ("ho controllato la posta", "neutro"),
    ("sto riordinando la stanza", "neutro"),
    ("ho spento la tv", "neutro"),
    ("ho scritto un appunto", "neutro"),
    ("ho aggiornato il calendario", "neutro"),
    ("sto seduto alla scrivania", "neutro"),
    ("sto facendo il bucato", "neutro"),
    ("sto camminando in casa", "neutro"),
    ("ho sistemato i documenti", "neutro"),
    ("sto bevendo acqua", "neutro"),
    ("ho pulito la cucina", "neutro"),
    ("sto leggendo un manuale", "neutro"),
    ("sto ascoltando un podcast", "neutro"),
    ("sto osservando il cielo", "neutro"),
    ("ho cucinato del riso", "neutro"),
    ("ho sistemato il frigo", "neutro"),
    ("sto finendo una presentazione", "neutro"),
    ("ho acceso la luce", "neutro"),
    ("sto riordinando il mobile", "neutro"),
    ("ho fatto una lista della spesa", "neutro"),
    ("sto aggiornando il diario", "neutro"),
    ("sto compilando un modulo", "neutro"),
    ("ho risposto ai messaggi", "neutro"),
    ("sto camminando in silenzio", "neutro"),
    ("ho messo in ordine la libreria", "neutro"),
    ("sto cercando un file", "neutro"),
    ("sto scrivendo un promemoria", "neutro"),
    ("sto leggendo un documento", "neutro"),
    ("sto bevendo un caffè", "neutro"),
    ("sto osservando la pioggia", "neutro"),
    ("sto chiudendo le finestre", "neutro"),
    ("sto cercando di concentrarmi", "neutro"),
    ("ho sistemato i cavi", "neutro"),
    ("sto aprendo una finestra", "neutro"),
    ("sto preparando il pranzo", "neutro"),
    ("sto riordinando le idee", "neutro"),
    ("sto aggiornando la lista delle cose da fare", "neutro"),
    ("mi hai fatto arrabbiare", "arrabbiato"),
    ("non sopporto questa situazione", "arrabbiato"),
    ("sono davvero furioso", "arrabbiato"),
    ("basta, ne ho abbastanza", "arrabbiato"),
    ("mi sento preso in giro", "arrabbiato"),
    ("è inaccettabile", "arrabbiato"),
    ("sto perdendo la pazienza", "arrabbiato"),
    ("non è giusto!", "arrabbiato"),
    ("sono stufo di tutto", "arrabbiato"),
    ("non voglio sentire scuse", "arrabbiato"),
    ("non ho più pazienza", "arrabbiato"),
    ("ogni cosa mi irrita", "arrabbiato"),
    ("non ci vedo più dalla rabbia", "arrabbiato"),
    ("non mi ascolta mai nessuno", "arrabbiato"),
    ("ho alzato la voce", "arrabbiato"),
    ("non posso tollerarlo", "arrabbiato"),
    ("sono indignato", "arrabbiato"),
    ("ho completamente perso la pazienza", "arrabbiato"),
    ("questa situazione mi infastidisce", "arrabbiato"),
    ("sono esasperato", "arrabbiato"),
    ("mi hanno mancato di rispetto", "arrabbiato"),
    ("non mi danno retta", "arrabbiato"),
    ("sono stato ignorato", "arrabbiato"),
    ("non sopporto più nulla", "arrabbiato"),
    ("mi hanno provocato troppo", "arrabbiato"),
    ("mi sento escluso", "arrabbiato"),
    ("è stato davvero frustrante", "arrabbiato"),
    ("ho perso il controllo", "arrabbiato"),
    ("non accetto questo comportamento", "arrabbiato"),
    ("mi hanno ferito con le parole", "arrabbiato"),
    ("non ho più voglia di discutere", "arrabbiato"),
    ("non è possibile andare avanti così", "arrabbiato"),
    ("sono molto irritato", "arrabbiato"),
    ("sono esausto per colpa loro", "arrabbiato"),
    ("nessuno mi capisce", "arrabbiato"),
    ("mi stanno mettendo alla prova", "arrabbiato"),
    ("sono stanco di essere sottovalutato", "arrabbiato"),
    ("mi sento maltrattato", "arrabbiato"),
    ("sono infastidito da tutto", "arrabbiato"),
    ("non voglio continuare così", "arrabbiato"),
    ("sento solo critiche", "arrabbiato"),
    ("mi hanno deluso profondamente", "arrabbiato"),
    ("sono arrabbiato con tutti", "arrabbiato"),
    ("non sopporto il loro atteggiamento", "arrabbiato"),
    ("nessuno rispetta i miei limiti", "arrabbiato"),
    ("non riesco a contenere la rabbia", "arrabbiato"),
    ("ho sbottato", "arrabbiato"),
    ("sono stato trattato malissimo", "arrabbiato"),
    ("non voglio più avere a che fare con loro", "arrabbiato"),
    ("mi hanno fatto perdere la calma", "arrabbiato")
]


# le frasi vengono mescolate casualmente per evitare bias dovuti all'ordine delle classi
random.shuffle(full_dataset)


# il dataset viene diviso in due parti: TRAIN e TEST
train = full_dataset[:160]
test = full_dataset[160:]



# (2) DEFINIZIONE DELLE CLASSI E CONVERSIONE STRINGA-NUMERO E NUMERO-STRINGA

labels = ["positivo", "negativo", "neutro", "arrabbiato"]
label_to_i = {l: i for i, l in enumerate(labels)}
i_to_label = {i: l for l, i in label_to_i.items()}
num_classes = len(labels)



# (3) COSTRUZIONE DEL VOCABOLARIO

# il dizionario è fatto di parole uniche presenti nel dataset di training
# ogni parola è mappata a un indice
word_set = {w for frase, _ in train for w in frase.lower().split()}
word_to_i = {w: i for i, w in enumerate(word_set)}
V = len(word_to_i)



# (4) ARCHITETTURA DELLA RETE

# parametri
d_emb = 32
hidden1 = 64
hidden2 = 32
lr = 0.005
epochs = 3000


# inizializzazione dei pesi

# embedding delle pearole
W_emb = [[random.uniform(-0.5, 0.5) for _ in range(d_emb)] for _ in range(V)] 
# peso dei neuroni del primo hidden layer
W1 = [[random.uniform(-1, 1) for _ in range(hidden1)] for _ in range(d_emb)] 
# peso dei neuroni del secondo hidden layer
W2 = [[random.uniform(-1, 1) for _ in range(hidden2)] for _ in range(hidden1)] 
# peso dei neuroni del layer di output
W_out = [[random.uniform(-1, 1) for _ in range(num_classes)] for _ in range(hidden2)]  


# Funzioni di attivazione 

#ReLU
def relu(x):
    return [max(0, i) for i in x]

# Softmax
def softmax(x):
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    s = sum(exps)
    return [e / s for e in exps]


# Funzione per la generaizone dell'embedding delle parole

# trasforma una frase in un singolo vettore sommando i vettori delle parole presenti, poi ne fa la media. 
# Se nessuna parola è nota, restituisce un vettore nullo.
def compute_embedding(parole):
    parole = [w for w in parole if w in word_to_i]
    if not parole:
        return [0.0] * d_emb
    emb = [0.0] * d_emb
    for w in parole:
        idx = word_to_i[w]
        for k in range(d_emb):
            emb[k] += W_emb[idx][k]
    return [e / len(parole) for e in emb]



# (5) TRAINING DELLA RETE NEURALE

# per 3000 iterazioni la rete:
# prende una frase casuale; calcola embedding → hidden layer 1 → hidden layer 2 → output;
# calcola l'errore dell'output rispetto alla classe vera
# propaga l’errore all’indietro aggiornando i pesi (W_out, W2, W1, W_emb)
# la tecnica è Stochastic Gradient Descent senza batch, né momentum, né adaptive learning rate.

for epoch in range(epochs):
    frase, lab = random.choice(train)
    parole = frase.lower().split()
    emb = compute_embedding(parole)
    y_idx = label_to_i[lab]

    h1 = relu([sum(emb[k] * W1[k][j] for k in range(d_emb)) for j in range(hidden1)])
    h2 = relu([sum(h1[k] * W2[k][j] for k in range(hidden1)) for j in range(hidden2)])
    logits = [sum(h2[k] * W_out[k][j] for k in range(hidden2)) for j in range(num_classes)]
    y_hat = softmax(logits)

    err = [y_hat[i] - (1 if i == y_idx else 0) for i in range(num_classes)]

    for i in range(hidden2):
        for j in range(num_classes):
            W_out[i][j] -= lr * err[j] * h2[i]

    dh2 = [sum(err[j] * W_out[i][j] for j in range(num_classes)) for i in range(hidden2)]
    drelu2 = [1 if v > 0 else 0 for v in h2]

    for i in range(hidden1):
        for j in range(hidden2):
            W2[i][j] -= lr * dh2[j] * drelu2[j] * h1[i]

    dh1 = [sum(dh2[j] * drelu2[j] * W2[i][j] for j in range(hidden2)) for i in range(hidden1)]
    drelu1 = [1 if v > 0 else 0 for v in h1]

    for i in range(d_emb):
        for j in range(hidden1):
            W1[i][j] -= lr * dh1[j] * drelu1[j] * emb[i]

    for w in parole:
        if w in word_to_i:
            idx = word_to_i[w]
            for k in range(d_emb):
                W_emb[idx][k] -= lr * sum(dh1[j] * drelu1[j] * W1[k][j] for j in range(hidden1)) / len(parole)

# comunicazione del completamento del training
print("✅ Training completato") 



# (6) VALUTAZIONE DELL'OUTPUT DELLA RETE

# valuta la performance della rete su un set (train o test), 
# calcolando accuracy e una matrice di confusione 4x4

def eval_set(data):
    corr = 0
    mat = [[0] * num_classes for _ in range(num_classes)]
    for frase, lab in data:
        parole = frase.lower().split()
        emb = compute_embedding(parole)

        h1 = relu([sum(emb[k] * W1[k][j] for k in range(d_emb)) for j in range(hidden1)])
        h2 = relu([sum(h1[k] * W2[k][j] for k in range(hidden1)) for j in range(hidden2)])
        logits = [sum(h2[k] * W_out[k][j] for k in range(hidden2)) for j in range(num_classes)]
        y_hat = softmax(logits)
        pred = y_hat.index(max(y_hat))

        true_idx = label_to_i[lab]
        if pred == true_idx:
            corr += 1
        mat[true_idx][pred] += 1
    return corr / len(data), mat

acc_train, mat_train = eval_set(train)
acc_test, mat_test = eval_set(test)

# esibizione dei risultati della valutazione
# accuracy
print(f"Accuracy: TRAIN: {acc_train:.2f}, TEST: {acc_test:.2f}")

# matrice di confusione
print("Matrice di confusione (TEST):")
col_width = 18  # larghezza fissa per ciascuna colonna
# intestazione
header = "".join(label.ljust(col_width) for label in [""] + labels)
print(header)
# righe con etichette, valori e conteggio totale per riga
for i, row in enumerate(mat_test):
    total = sum(row)
    label_with_total = f"{labels[i]} ({total})"
    row_str = label_with_total.ljust(col_width)
    row_str += "".join(str(x).ljust(col_width) for x in row)
    print(row_str)




# (7) CLASSIFICAZIONE: APPLICAZIONE DELLA RETE 

# la funzione restituisce: 
# 1) la classe di sentiment della frase isnerita; 
# 2) il valore della probabilità assegnata a questa classficazione; 
# 3) la distribuzione delle probabilità tra tutte e 4 le classi

def pred(frase):
    parole = frase.lower().split()
    emb = compute_embedding(parole)
    h1 = relu([sum(emb[k] * W1[k][j] for k in range(d_emb)) for j in range(hidden1)])
    h2 = relu([sum(h1[k] * W2[k][j] for k in range(hidden1)) for j in range(hidden2)])
    logits = [sum(h2[k] * W_out[k][j] for k in range(hidden2)) for j in range(num_classes)]
    y_hat = softmax(logits)
    idx = y_hat.index(max(y_hat))
    return labels[idx], max(y_hat), y_hat



# (8) INPUT E INTERAZIONE CON L'UTENTE

while True:
    user_input = input("\nScrivi una frase per analizzarne il sentiment (o 'fine' per uscire): ")
    if user_input.lower() == 'fine':
        break
    sentimento, confidenza, distribuzione = pred(user_input)
    print(f"  Sentimento rilevato: << {sentimento} >> ({confidenza:.2f})\n")
    print("  Distribuzione di probabilità:")
    for i, prob in enumerate(distribuzione):
        print(f"   {labels[i]}: {prob:.2f}")

