import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pyfiglet

def mostrar_titulo(titulo):
    ascii_art = pyfiglet.figlet_format(titulo)
    print("Password-Sorter")

# Ejemplo
mostrar_titulo("Mi Programa")
class PasswordStrengthClassifier:
    def __init__(self)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
        self.smote = SMOTE(random_state=42)
        self.classifier = SVC(kernel='linear', random_state=42)
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self.param_grid = {
            'vectorizer__ngram_range': [(1, 3), (1, 4)],
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
        
    def extract_password_features(self, password):
        """Extrae características adicionales de las contraseñas."""
        return {
            'length': len(password),
            'has_upper': any(c.isupper() for c in password),
            'has_lower': any(c.islower() for c in password),
            'has_digit': any(c.isdigit() for c in password),
            'has_special': any(not c.isalnum() for c in password),
            'char_diversity': len(set(password)) / len(password) if password else 0
        }
    
    def fit(self, X, y):
        X_transformed = self.vectorizer.fit_transform(X)
        X_resampled, y_resampled = self.smote.fit_resample(X_transformed, y)
        grid_search = GridSearchCV(self.classifier, {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }, cv=5, scoring='accuracy')
        
        grid_search.fit(X_resampled, y_resampled)
        self.classifier = grid_search.best_estimator_
        print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
        return self
    
    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        return self.classifier.predict(X_transformed)
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo con métricas detalladas."""
        X_transformed = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_transformed)
        
        print("\nMétricas de evaluación:")
        print(f"Precisión: {accuracy_score(y_test, y_pred):.3f}")
        print("\nInforme de clasificación detallado:")
        print(classification_report(y_test, y_pred))
        cv_scores = cross_val_score(self.classifier, X_transformed, y_test, cv=5)
        print(f"\nPrecisión de validación cruzada: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    def predict_with_confidence(self, passwords):
        """Realiza predicciones con nivel de confianza."""
        predictions = []
        
        for password in passwords:
            pred = self.predict([password])[0]
            features = self.extract_password_features(password)
            strength_score = sum([
                features['length'] >= 8,
                features['has_upper'],
                features['has_lower'],
                features['has_digit'],
                features['has_special'],
                features['char_diversity'] > 0.5
            ]) / 6.0
            
            predictions.append({
                'password': password,
                'prediction': pred,
                'strength_score': strength_score,
                'features': features
            })
            
        return predictions
if __name__ == "__main__":
    data = pd.read_csv("base-datos-extendido.csv")
    X = data['password']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    classifier = PasswordStrengthClassifier()
    classifier.fit(X_train, y_train)
    classifier.evaluate(X_test, y_test)
    test_passwords = [
        "12345",
        "7S$k@9JvP2",
        "password123",
        "U+LMj1@3#kL"
    ]
    
    results = classifier.predict_with_confidence(test_passwords)
    
    print("\nAnálisis de nuevas contraseñas:")
    for result in results:
        print(f"\nContraseña: {result['password']}")
        print(f"Predicción: {'Fuerte' if result['prediction'] == 1 else 'Débil'}")
        print(f"Puntuación de fortaleza: {result['strength_score']:.2f}")
        print("Características:")
        for feature, value in result['features'].items():
            print(f"- {feature}: {value}")
