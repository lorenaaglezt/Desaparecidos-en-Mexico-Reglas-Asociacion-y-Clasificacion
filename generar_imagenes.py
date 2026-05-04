import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Asegurar que el directorio de salida exista
os.makedirs('heatmaps', exist_ok=True)

# 1. Cargar el dataset procesado
print("Cargando dataset...")
frame = pd.read_csv("../DS_procesado.csv")
print(f"Dataset cargado: {frame.shape[0]} registros, {frame.shape[1]} columnas")

# 2. Separar features (X) y target (y)
columnas_excluir = ['ESTATUS_VICTIMA', 'MUNICIPIO']
X = frame.drop(columns=columnas_excluir)
y = frame['ESTATUS_VICTIMA']

# 3. One-Hot Encoding para variables categóricas en X
X = pd.get_dummies(X, drop_first=False)

# 4. Label Encoding para la variable objetivo
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Mapeo de clases:")
for i, clase in enumerate(label_encoder.classes_):
    print(f"  {clase} = {i}")

# 5. División train/test 80-20 con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# 6. Entrenar Árbol de Decisión
print("\n--- Entrenando Árbol de Decisión ---")
clf_tree = DecisionTreeClassifier(max_depth=5,
                                   class_weight='balanced',
                                   random_state=42)
t0 = time.time()
clf_tree.fit(X_train, y_train)
t1 = time.time()
y_pred_tree = clf_tree.predict(X_test)
print(classification_report(y_test, y_pred_tree,
      target_names=label_encoder.classes_))
print(f"Tiempo de entrenamiento: {round(t1-t0, 2)}s")

# 7. Entrenar Random Forest
print("\n--- Entrenando Random Forest ---")
clf_rf = RandomForestClassifier(n_estimators=100,
                                 class_weight='balanced',
                                 random_state=42,
                                 n_jobs=-1)
t0 = time.time()
clf_rf.fit(X_train, y_train)
t1 = time.time()
y_pred_rf = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf,
      target_names=label_encoder.classes_))
print(f"Tiempo de entrenamiento: {round(t1-t0, 2)}s")

# 8. Generar matrices de confusión
print("\nGenerando confusion_matrices.png")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, y_pred, titulo in zip(axes,
    [y_pred_tree, y_pred_rf],
    ['Árbol de Decisión (max_depth=5)', 'Random Forest (100 árboles)']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig('heatmaps/confusion_matrices.png', dpi=150)
plt.close()
print("heatmaps/confusion_matrices.png generada")

# 8.5. Generar visualizacion del arbol de decision
print("\nGenerando decision_tree.png")
fig, ax = plt.subplots(figsize=(28, 14))
plot_tree(clf_tree,
          feature_names=X_train.columns.tolist(),
          class_names=label_encoder.classes_.tolist(),
          filled=True,
          rounded=True,
          fontsize=7,
          proportion=True,
          ax=ax)
ax.set_title('Arbol de Decision (max_depth=5)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmaps/decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("heatmaps/decision_tree.png generada")

# Imprimir representacion textual del arbol
print("\nRepresentacion textual del arbol:")
tree_text = export_text(clf_tree,
                        feature_names=X_train.columns.tolist(),
                        class_names=label_encoder.classes_.tolist(),
                        max_depth=3)
print(tree_text)

# 9. Generar feature importance
print("Generando feature_importance.png")
importances = pd.Series(clf_rf.feature_importances_,
                         index=X_train.columns)
top10 = importances.nlargest(10).sort_values()

plt.figure(figsize=(10, 6))
top10.plot(kind='barh', color='#301934')
plt.title('Top 10 Variables más Importantes — Random Forest',
          fontweight='bold')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig('heatmaps/feature_importance.png', dpi=150)
plt.close()
print("heatmaps/feature_importance.png generada")

print("\nTodas las imagenes fueron generadas exitosamente.")
