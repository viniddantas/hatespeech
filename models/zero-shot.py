import ollama
from pydantic import BaseModel
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import emoji

df = pd.read_csv("./datasets/HateBR.csv")
arquivo_saida = "./results/resultados_zero_shot.jsonl"

indices_originais = list(range(len(df)))
x = df['comentario']
y = df['label_final']

x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
    x, y, indices_originais, test_size=0.2, random_state=42
)

class HateSpeech(BaseModel):
    classificacao: int
    motivo: str

system_prompt = f"""
    Você é um moderador imparcial.
    Sua tarefa é analisar e classificar as frases delimitadas por ''' de acordo com a seguinte regra:

    1 = Contém discurso de ódio (linguagem que ataca, humilha, ameaça ou incita violência contra uma pessoa ou grupo com base em raça, religião, etnia, orientação sexual, deficiência ou gênero).
    0 = NÃO contém discurso de ódio (pode ser uma crítica, opinião dura, ou texto neutro, mas não atinge os critérios acima).

    Regra de formatação: Você deve responder ÚNICA e EXCLUSIVAMENTE com o número "1" ou "0". Não adicione nenhuma palavra, explicação ou pontuação extra.
"""

with open(arquivo_saida, 'a', encoding='utf-8') as f:
    for idx, (frase, label_final, id_test) in enumerate(zip(x_test, y_test, id_test)):
        frase = emoji.demojize(frase, language='pt')
        frase = frase.replace(':', '').replace('_', ' ')
        try:
            response = ollama.chat(
                model='mistral:7b',
                options={
                    'temperature': 0,
                    'seed': 42
                    },
                messages=[
                    {
                        'role': 'system', 
                        'content': system_prompt
                    },
                    { 
                        'role': 'user', 
                        'content': f"Frase: '''{frase}'''"
                    },
                ],
                format=HateSpeech.model_json_schema()
            )
            reponse_json = json.loads(response.message.content)

            resultado_final = {
                'id': id_test,
                'comentario': frase,
                'class_previsto': reponse_json.get('classificacao'),
                'class_real': int(label_final),
                'motivo': reponse_json.get('motivo')
            }

            f.write(json.dumps(resultado_final, ensure_ascii=False) + '\n')

            f.flush()

            print(f"Progresso: {idx} de {len(x_test)} concluídos...")
        except Exception as e:
            print(f"Erro no item {id_test}: {e}")
            continue

print("=" *60, "Imprimindo JSON", "=" *60)
df = pd.read_json("resultados_zero_shot.jsonl", lines=True) 
print(df.head())