from sklearn.metrics import classification_report
import pandas as pd

arquivo_entrada = "resultados_zero_shot.jsonl"

df = pd.read_json(arquivo_entrada, lines=True)

results = classification_report(df['class_real'], df['class_previsto'])
print(results)