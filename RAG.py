from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


    
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="mistral:7b", 
    temperature=0, 
    seed=42
)

system_prompt = f"""
    Você é um moderador imparcial.
    Sua tarefa é analisar e classificar as frases delimitadas por ''' de acordo com a seguinte regra:

    1 = Contém discurso de ódio (linguagem que ataca, humilha, ameaça ou incita violência contra uma pessoa ou grupo com base em raça, religião, etnia, orientação sexual, deficiência ou gênero).
    0 = NÃO contém discurso de ódio (pode ser uma crítica, opinião dura, ou texto neutro, mas não atinge os critérios acima).

    Regra de formatação: Você deve responder ÚNICA e EXCLUSIVAMENTE com o número "1" ou "0". Não adicione nenhuma palavra, explicação ou pontuação extra.
"""


QUERY_PROMPT = PromptTemplate(
    input_variables=["frase"],
    template=f"""{system_prompt} Frase: '''{{frase}}'''""",
)

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from vector import vector_store

retriever = MultiQueryRetriever.from_llm(
    vector_store.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
    query_prompt=QUERY_PROMPT,
)

