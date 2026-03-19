# Projektauftrag: Machine Learning Project

**Team:** 1–2 Personen  
**Datenset:** Frei wählbar, in Absprache mit dem Ausbildner  
**Abgabe:** 15.30 Uhr PowerPoint oder Jupyter Notebook  
**Präsentation:** 5–10 Minuten  

---

## Aufgaben

### 1. Datensatz auswählen
Wähle einen öffentlich verfügbaren Datensatz (z.B. von Kaggle, UCI Machine Learning Repository, OpenML usw.) oder einen eigenen Datensatz. Stelle sicher, dass die Daten umfangreich genug sind, um eine aussagekräftige Fragestellung zu untersuchen und zu beantworten.

### 2. Forschungsfrage oder Hypothese formulieren
Formuliere eine klare, datengestützte Frage oder Hypothese, die du mithilfe von ML beantworten möchtest. Ordne dein Ziel einer der gängigen ML-Aufgaben zu:
- **Klassifikation**: Vorhersage von Kategorien (z.B. Spam vs. kein Spam)
- **Regression / Vorhersage**: Vorhersage numerischer Ergebnisse (z.B. Immobilienpreise)
- **Clustering**: Erkennung natürlicher Gruppierungen in Daten (z.B. Kundensegmentierung)

### 3. Geeigneten ML-Ansatz wählen und begründen
Wähle eine geeignete ML-Methode (z.B. Entscheidungsbaum, Logistische Regression, K-Means, Random Forest, SVM, Neuronales Netz). Begründe deine Wahl anhand der Beschaffenheit deiner Daten und deiner Fragestellung. Wähle mindestens eine geeignete Evaluationsmetrik (z.B. Genauigkeit, F1-Score, RMSE, Silhouettenkoeffizient) und erläutere, warum diese zu deiner Aufgabe passt.

### 4. Lösung implementieren (Pipeline)
Deine Pipeline sollte idealerweise umfassen:
- **Datenexploration und Visualisierung**
- **Datenvorverarbeitung** (z.B. Umgang mit fehlenden Werten, Normalisierung, Kodierung kategorialer Variablen)
- **Datenaufteilung** (Train/Test oder k-fache Kreuzvalidierung)
- **Modelltraining und -optimierung**
- **Evaluation und Diskussion der Ergebnisse**

### 5. Ergebnisse kommunizieren
Bereite eine kurze Präsentation (5–10 Minuten) vor, in der du folgende Punkte vorstellst: 
- Fragestellung und ihre Relevanz
- Überblick über deinen Datensatz
- Gewählte Methode und deren Begründung
- Wichtigsten Schritte deines Arbeitsablaufs
- Ergebnisse, Interpretationen und Einschränkungen
- Was du gelernt hast

---

## Example Project Ideas

### 1. Iris Flower Classification (Biological Data)
- **Hypothesis**: Different species of iris flowers can be identified based on petal and sepal measurements.
- **Task Type**: Classification
- **ML Methods**: k-Nearest Neighbors (k-NN), Decision Tree, Logistic Regression
- **Metric**: Accuracy, Confusion Matrix
- **Access**:
  ```python
  from sklearn.datasets import load_iris
  iris = load_iris()
  ```

### 2. Penguins
Measurements of several penguin species (Chinstrap, Gentoo, Adélie).
- **Task**: Classification or Regression
- **Access**:
  ```python
  import seaborn as sns
  penguins = sns.load_dataset("penguins")
  ```

### 3. Titanic
The sinking of the RMS Titanic. Predict who survived based on demographic information.

### 4. Predict Housing Prices (Real Estate Data)
- **Hypothesis**: Housing prices can be predicted using features like number of rooms, location, and distance from the CBD.
- **Task Type**: Regression
- **ML Methods**: Linear Regression, Random Forest Regressor
- **Metric**: RMSE, R² Score
- **Access**: `fetch_california_housing` from `sklearn.datasets`

### 5. Tips
Information about restaurant customers and tips.
- **Access**:
  ```python
  import seaborn as sns
  tips = sns.load_dataset("tips")
  ```

### 6. Handwritten Digits (MNIST)
Large database of handwritten digits for training image processing systems.
- **Access**:
  ```python
  import tensorflow.keras as ks
  mnist = ks.datasets.mnist
  ```

### 7. Cats vs Dogs
- **Task**: Convolutional Neural Networks (CNN)
- **Access**: [Microsoft Download](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765)

### 8. Customer Segmentation (Mall Data)
- **Hypothesis**: Customers can be grouped into clusters based on income and spending score.
- **Task Type**: Clustering
- **ML Methods**: K-Means Clustering
- **Metric**: Silhouette Score, Visual cluster separation

---

## Reinforcement Learning & Tutorials
- [AI Gym Library](https://www.gymlibrary.dev/)
- [HuggingFace Course](https://huggingface.co/learn)
