import ollama
from datasets import load_dataset
from pydantic import BaseModel
import json
import pandas as pd

ds = load_dataset("franciellevargas/HateBR", split="train")
arquivo_saida = "resultados_zero_shot.jsonl"

subset = ds.select(range(5))

class HateSpeech(BaseModel):
    classificacao: int
    motivo: str

with open(arquivo_saida, 'a', encoding='utf-8') as f:
    for idx, frase in enumerate(subset):
        prompt = f"""
            Você é um moderador imparcial.
            Sua tarefa é analisar e classificar as frases delimitadas por ''' de acordo com a seguinte regra:

            1 = Contém discurso de ódio (linguagem que ataca, humilha, ameaça ou incita violência contra uma pessoa ou grupo com base em raça, religião, etnia, orientação sexual, deficiência ou gênero).
            0 = NÃO contém discurso de ódio (pode ser uma crítica, opinião dura, ou texto neutro, mas não atinge os critérios acima).

            Regra de formatação: Você deve responder ÚNICA e EXCLUSIVAMENTE com o número "1" ou "0". Não adicione nenhuma palavra, explicação ou pontuação extra.

            Frase: '''{frase['comentario']}'''
        """
        try:
            response = ollama.chat(
                model='mistral:7b',
                options={'temperature': 0},
                messages=[
                    {
                        'role': 'user', 
                        'content': prompt
                    },
                ],
                format=HateSpeech.model_json_schema()
            )
            reponse_json = json.loads(response.message.content)

            resultado_final = {
                'id': idx,
                'comentario': frase['comentario'],
                'class_previsto': reponse_json.get('classificacao'),
                'class_real': frase['label_final'],
                'motivo': reponse_json.get('motivo')
            }

            f.write(json.dumps(resultado_final, ensure_ascii=False) + '\n')

            f.flush()
            print(response.message.content)
        except Exception as e:
            print(f"Erro no item{idx}: {e}")
            
            continue

print("=" *60, "Imprimindo JSON", "=" *60)
df = pd.read_json("resultados_zero_shot.jsonl", lines=True) 
print(df.head())