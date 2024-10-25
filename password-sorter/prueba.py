import pandas as pd
import random
import string

# Cargar tu dataset original (asegúrate de que el archivo esté en la misma carpeta)
try:
    data = pd.read_csv("base-datos.csv")
except FileNotFoundError:
    print("El archivo 'base-datos.csv' no se encontró. Asegúrate de que esté en el mismo directorio.")
    exit()

# Función para generar contraseñas fuertes
def generar_password_fuerte(longitud=16):
    caracteres = string.ascii_letters + string.digits + string.punctuation
    password = [
        random.choice(string.ascii_lowercase),  # Al menos una letra minúscula
        random.choice(string.ascii_uppercase),  # Al menos una letra mayúscula
        random.choice(string.digits),           # Al menos un dígito
        random.choice(string.punctuation)       # Al menos un símbolo
    ]
    password += random.choices(caracteres, k=longitud - 4)  # Completar la longitud
    random.shuffle(password)  # Mezclar los caracteres
    return ''.join(password)

# Generar 100 contraseñas fuertes
contraseñas_fuertes = [generar_password_fuerte() for _ in range(100)]

# Crear un DataFrame con las nuevas contraseñas
df_fuertes = pd.DataFrame(contraseñas_fuertes, columns=['password'])
df_fuertes['label'] = 'fuerte'  # Etiquetar como 'fuerte'

# Etiquetar las contraseñas originales como 'débiles'
data['label'] = 'débil'

# Combinar ambos datasets
data_extendido = pd.concat([data, df_fuertes], ignore_index=True)

# Guardar el nuevo archivo CSV
data_extendido.to_csv("base-datos-extendido.csv", index=False)

print("Archivo guardado como 'base-datos-extendido.csv'")
