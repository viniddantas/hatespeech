from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# =========================== Criando instancias de embeddings e vector store ===========================

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = InMemoryVectorStore(embeddings) # In-Memory Vector Store

# =========================== Carregando documentos da web usando WebBaseLoader ===========================

# filtro específico para extrair apenas o conteúdo relevante da página
bs4_strainer = bs4.SoupStrainer(class_=("mw-content-ltr mw-parser-output", "mw-page-title-main"))

loader = WebBaseLoader(
  web_path="https://pt.wikipedia.org/wiki/Discurso_de_%C3%B3dio",
  bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

print(docs[0].page_content)

# =========================== Realizando o split do documento ===========================

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200,
  add_start_index=True, 
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# =========================== Armazendo os documentos ===========================

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

