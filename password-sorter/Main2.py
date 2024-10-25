import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Cargar los datos
data = pd.read_csv("base-datos-extendido.csv")

# Verificar balance de clases
print(data['label'].value_counts())

# Separar características y etiquetas
X = data['password']
y = data['label']

# Vectorizar las contraseñas con n-gramas
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_transformed = vectorizer.fit_transform(X)

# Dividir los datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Entrenar el modelo con SVM (linear kernel para este caso)
model = SVC(kernel='linear', random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Prueba con nuevas contraseñas
nuevas_contrasenas = ["12345", "7S$k@9Jv", "password123", "U+LMj1@3"]
nuevas_vectorizadas = vectorizer.transform(nuevas_contrasenas)
predicciones = model.predict(nuevas_vectorizadas)

print("\nResultados de predicción para nuevas contraseñas:")
for contraseña, pred in zip(nuevas_contrasenas, predicciones):
    print(f"{contraseña}: {pred}")
