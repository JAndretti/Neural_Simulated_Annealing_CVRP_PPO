### Algorithme de Répartition Optimale pour le VRP

Voici l'algorithme détaillé étape par étape pour répartir les clients dans des camions de manière optimale, en respectant les contraintes de capacité, avec un ordre de visite fixé.

#### **Prérequis et Notations**
- **Séquence fixe** : \(0 \rightarrow v_1 \rightarrow v_2 \rightarrow \dots \rightarrow v_n\) (hub = 0).
- **Demandes** : \(d_i\) = demande du client \(v_i\) (avec \(d_i \leq C\)).
- **Capacité** : \(C\) (capacité maximale d'un camion).
- **Coûts** : 
  - \( \text{cost}(a, b) \) = coût de l'arc entre \(a\) et \(b\).
  - **Coût de base total** (sans découpage) :
    \[
    \text{base\_total} = \text{cost}(0, v_1) + \sum_{j=1}^{n-1} \text{cost}(v_j, v_{j+1}) + \text{cost}(v_n, 0)
    \]
  - **Coût d'insertion** (pour un découpage entre \(v_{i-1}\) et \(v_i\)) :
    \[
    \Delta(i) = \text{cost}(v_{i-1}, 0) + \text{cost}(0, v_i) - \text{cost}(v_{i-1}, v_i) \quad \text{pour } i \geq 2
    \]

#### **Objectif**
Minimiser le coût total en insérant des retours au hub (découpages) dans la séquence fixe, tout en respectant \(C\) :
\[
\text{coût\_total} = \text{base\_total} + \text{coût\_supplémentaire}
\]
où \(\text{coût\_supplémentaire}\) est minimisé par programmation dynamique.

---

### **Étapes de l'Algorithme**

#### **1. Précalculs**
- **Vérifier la faisabilité** : \(\forall i,  d_i \leq C\). Si non, problème infaisable.
- **Calculer \(\text{base\_total}\)**.
- **Précalculer \(\Delta(i)\) pour \(i = 2 \dots n\)**.
  - Exemple : \(\Delta(2) = \text{cost}(v_1, 0) + \text{cost}(0, v_2) - \text{cost}(v_1, v_2)\).

#### **2. Initialisation des Variables DP**
- **État** : 
  - \( \text{dp}[k][c] \) = coût supplémentaire minimal pour les \(i\) premiers clients, avec \(k\) camions, et capacité \(c\) utilisée dans le dernier camion.
  - \( \text{best}[k] \) = \(\min_{c} \text{dp}[k][c]\).
- **Initialisation (pour \(i = 0\))** :
  - \(\text{dp}_{\text{prev}}[1][0] = 0\) (1 camion vide).
  - \(\text{best}_{\text{prev}}[1] = 0\).
  - Pour \(k \geq 2\) ou \(c > 0\) : \(\infty\).

#### **3. Boucle Principale (pour chaque client \(i = 1 \dots n\))**
- **Entrée** : \(\text{dp}_{\text{prev}}\) et \(\text{best}_{\text{prev}}\) après le client \(i-1\).
- **Sortie** : \(\text{dp}_{\text{curr}}\) et \(\text{best}_{\text{curr}}\) après le client \(i\).
- **Pour chaque** \(k = 1 \dots n\) (nombre de camions) et \(c = 0 \dots C\) (capacité utilisée) :
  - **Option 1 (pas de découpage)** :
    - Si \(c \geq d_i\) et \(k \geq 1\) :
      \[
      \text{candidate}_1 = \text{dp}_{\text{prev}}[k][c - d_i]
      \]
    - *Explication* : Ajouter \(v_i\) au camion courant sans insertion de hub.
  - **Option 2 (découpage avant \(v_i\))** :
    - Si \(c = d_i\), \(k \geq 2\), et \(i \geq 2\) :
      \[
      \text{candidate}_2 = \text{best}_{\text{prev}}[k-1] + \Delta(i)
      \]
    - *Explication* : Démarrer un nouveau camion à \(v_i\), coût \(\Delta(i)\) pour l'insertion du hub.
  - **Mise à jour** :
    \[
    \text{dp}_{\text{curr}}[k][c] = \min(\text{candidate}_1, \text{candidate}_2)
    \]
- **Calculer \(\text{best}_{\text{curr}}[k] = \min_{c} \text{dp}_{\text{curr}}[k][c]\)**.

#### **4. Résultat Final (après \(i = n\))**
- **Coût supplémentaire minimal** :
  \[
  \text{min\_additional} = \min_{k, c} \text{dp}_{\text{prev}}[k][c]
  \]
- **Coût total** :
  \[
  \text{coût\_total} = \text{base\_total} + \text{min\_additional}
  \]
- Si \(\text{min\_additional} = \infty\), problème infaisable.

---

### **Complexité**
- **Temps** : \(O(n^2 \cdot C)\) (pour \(n\) clients, \(k \leq n\), capacité \(C\)).
- **Espace** : \(O(n \cdot C)\) (utilisation de deux matrices \(k \times C\)).

---

### **Exemple Illustratif**
#### **Données** :
- Clients : \(v_1, v_2\) avec demandes \(d_1=5\), \(d_2=7\), \(C=10\).
- Coûts :  
  \(\text{cost}(0,v_1)=1\), \(\text{cost}(v_1,v_2)=2\), \(\text{cost}(v_2,0)=3\),  
  \(\text{cost}(v_1,0)=4\), \(\text{cost}(0,v_2)=5\).

#### **Calculs** :
1. **base_total** = \(1 + 2 + 3 = 6\).
2. **\(\Delta(2)\)** = \(\text{cost}(v_1,0) + \text{cost}(0,v_2) - \text{cost}(v_1,v_2) = 4 + 5 - 2 = 7\).
3. **Initialisation** :  
   \(\text{dp}_{\text{prev}}[1][0] = 0\), \(\text{best}_{\text{prev}}[1] = 0\).
4. **Client \(i=1\)** :  
   - \(k=1\), \(c=5\) : \(\text{dp}_{\text{curr}}[1][5] = 0\).
5. **Client \(i=2\)** :  
   - \(k=1\) : aucune option (capacité \(5+7=12 > 10\)).
   - \(k=2\), \(c=7\) : \(\text{dp}_{\text{curr}}[2][7] = 0 + 7 = 7\).
6. **Résultat** :  
   \(\text{min\_additional} = 7\), \(\text{coût\_total} = 6 + 7 = 13\).

#### **Interprétation** :
- **Solution optimale** : 
  - Camion 1 : \(0 \rightarrow v_1 \rightarrow 0\) (coût = \(1 + 4 = 5\)).
  - Camion 2 : \(0 \rightarrow v_2 \rightarrow 0\) (coût = \(5 + 3 = 8\)).
  - Total = \(5 + 8 = 13\) (valide, capacité 5 et 7 ≤ 10).

---

### **Remarques**
- **Pas de découpage en \(i=1\)** : Interdit (pas deux hubs consécutifs).
- **Gestion des demandes nulles** : Traitées naturellement (\(c\) inchangé).
- **Backtracking** : Pour obtenir les positions de découpage, stocker les décisions prises.

Cet algorithme garantit une répartition optimale avec un ordre de visite fixé, en respectant les contraintes de capacité.