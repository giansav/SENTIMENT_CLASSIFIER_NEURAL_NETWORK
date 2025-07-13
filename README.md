# SENTIMENT CLASSIFIER NEURAL NETWORK


Il programma è una rete neurale minimale (scritta senza l'uso di librerie esterne come PyTorch o TensorFlow) per classificare il sentiment di una frase in lingua italiana in una delle seguenti quattro categorie: positivo, negativo, neutro, arrabbiato.
Il suo scopo è meramente didattico: esso consente di familiarizzare con l'idea dell'addestramento di una rete neurale. 

<br>
<br>

# Descrizione

Il modello è un classificatore multi-classe: analizza frasi testuali, genera un vettore di embedding per ciascuna, lo passa attraverso due hidden layer e infine produce una distribuzione di probabilità tra le classi, assegnando l’etichetta più probabile.
Completato l'addestramento della rete neurale, il programma consente all'utente di inserire una frase e testare la capacità della rete di classificarne il sentiment.


        
# Caratteristiche tecniche

La rete neurale ha la seguente architettura: 
- Embedding layer: vettore medio dei pesi associati alle parole della frase
- Primo hidden layer: 64 neuroni con attivazione ReLU
- Secondo hidden layer: 32 neuroni con attivazione ReLU
- Output layer: 4 neuroni (uno per classe) con softmax finale

L'addestramento della rete avviene ogni volta su un dataset incluso nel codice, mediante l'uso di Stochastic Gradient Descent (SGD).
Il dataset è stato creato manualmente, con frasi distribuite sulle 4 emozioni.
La rappresentazione dei testi è basata su bag-of-words mediato con embedding casuali.



# Limitazioni

1) La rete non utilizza nessun NLP avanzato: niente stopword, stemming, né tokenizzazione robusta.

2) La rete apprende su una sola frase per ciclo, non c'è nessuna normalizzazione né batch training.

3) Non c'è nessuna regolarizzazione del dizionario: niente dropout o controllo sull'overfitting.

4) L'embedding è molto approssimativo: i vettori sono inizializzati a caso e aggiornati via backpropagation, ma non catturano un significato linguistico reale.

5) La generalizzazione è molto limitata: il vocabolario è creato solo dal training set, quindi le parole sconosciute sono ignorate nei test.

6) Il dataset è decisamente troppo piccolo per lo spazio semantico oggetto della rete.

7) Le prestazioni sono fortemente dipendenti dalla casualità iniziale.
