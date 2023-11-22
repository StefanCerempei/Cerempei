import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

start = time.time()
df = pd.read_csv('C:/Users/Admin/Desktop/Programare/Healthcare-Diabetes.csv')
print(df.info())
print(df.describe())

X = df.drop('Outcome', axis=1)  # Caracteristici
y = df['Outcome']  # Variabila dependentă (rezultatul)

# Divizați datele în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizați datele
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Alte metrici și matrice de confuzie
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Setăm un stil de afișare pentru vizualizare mai plăcută
sns.set(style="whitegrid")

# Histogramă pentru vârsta
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title('Distribuția Vârstei')
plt.xlabel('Vârstă')
plt.ylabel('Frecvență')
plt.show()

# Diagramă de dispersie pentru Glucoză și BMI
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, palette='viridis')
plt.title('Glucoză vs BMI cu marcarea Rezultatului')
plt.xlabel('Glucoză')
plt.ylabel('BMI')
plt.show()

# Diagramă bară pentru Outcome (rezultat)
plt.figure(figsize=(8, 5))
sns.countplot(x='Outcome', data=df, hue='Outcome', palette='Set2')
plt.title('Distribuția Rezultatelor')
plt.xlabel('Outcome')
plt.ylabel('Număr de cazuri')
plt.show()


# Corelații între caracteristici
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de Corelație')
plt.show()

end = time.time()
print('Time for running this is: ', end - start, ' seconds')
