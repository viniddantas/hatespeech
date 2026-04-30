# Definindo a variável de ambiente USER_AGENT para corrigir warnings
import os
os.environ["USER_AGENT"] = "HateSpeechDetectWithRAG"

from vector import vector_store
from vector import embeddings
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import emoji

# Instanciação do modelo com langchain
model = ChatOllama(
    model="mistral:7b", 
    temperature=0, 
    seed=42
)

# =========================== Definindo um template de resposta ===========================
class HateSpeech(BaseModel):
    classificacao: int = Field(description='Retorne 1 se for discurso de ódio e 0 se for Neutro')
    motivo: str = Field(description='A justificativa da classificação baseada exlusivamente no contexto recuperado')
parser = PydanticOutputParser(pydantic_object=HateSpeech)

# =========================== Cria o template que sera enviado para a LLM ===========================
template = """
    Você é um moderador imparcial.
    Sua tarefa é analisar e classificar as frases delimitadas por ''' de acordo com a seguinte regra:

    1 = Contém discurso de ódio
    0 = NÃO contém discurso de ódio

    Além disso, você deve classificar com base nas diretrizes abaixo:
    {context}
    
    Regra de formatação: {format_instructions}

    '''{input}'''
"""
prompt = ChatPromptTemplate.from_template(template)

# =========================== Cria a corrente(chain) usado pelo langchain  ===========================

prompt = prompt.partial(format_instructions=parser.get_format_instructions()) # Repassa as instruções de formato para o prompt
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Busca pelos 2 documentos mais relevantes na VectorStore
document_chain = create_stuff_documents_chain(llm = model, prompt = prompt, output_parser = parser) # Concatena os k documentos da linha anterior em um único context
retrieval_chain = create_retrieval_chain(retriever, document_chain) # Responsável por passar os documentos e o prompt para a LLM

# =========================== Separa os dados de treino e teste ===========================
df = pd.read_csv("./datasets/HateBR.csv")
arquivo_saida = "./results/resultados_rag.jsonl"

indices_originais = list(range(len(df)))
x = df['comentario']
y = df['label_final']

x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
    x, y, indices_originais, test_size=0.2, random_state=42
)

# =========================== Adiciona as classificações no jsonl ===========================

with open(arquivo_saida, 'a', encoding='utf-8') as f:
    for idx, (frase, label_final, id_test) in enumerate(zip(x_test, y_test, id_test)):
        frase = emoji.demojize(frase, language='pt')
        frase = frase.replace(':', '').replace('_', ' ')
        try:
            response = retrieval_chain.invoke({'input': frase})
            
            results = response['answer']
            retrieval_documents = response['context']

            context = []
            for item in retrieval_documents:
                context.append({
                    'source': item.metadata['source'],
                    'content': item.page_content
                })
            
            
            classificacao = results.classificacao
            motivo = results.motivo

            resultado_final = {
                'id': id_test,
                'comentario': frase,
                'class_previsto': classificacao,
                'class_real': int(label_final),
                'motivo': motivo,
                'documents': context
            }
            

            f.write(json.dumps(resultado_final, ensure_ascii=False) + '\n')

            f.flush()

            print(f"Progresso: {idx} de {len(x_test)} concluídos...")
        except Exception as e:
            print(f"Erro no item {id_test}: {e}")
            continue