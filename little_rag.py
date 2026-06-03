# ---------- 2. Imports ----------
import os
import faiss
import langchain_community.vectorstores.faiss as FAISS
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed
import json
import requests
import re
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------- 5. FAISS index ----------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
  """
  Build a FAISS index for fast nearest‑neighbour search.

  Parameters
  ----------
  embeddings : np.ndarray
  2‑D array of document embeddings (float32 or float64).

  Returns
  -------
  faiss.IndexFlatL2
  A FAISS index that can answer distance‑based queries.
  """
  print("Building FAISS index...")
  if embeddings.ndim != 2 or embeddings.shape[0] == 0:
    raise ValueError(
      f"Embeddings must be a non‑empty 2‑D array. Received shape: {embeddings.shape}"
    )
  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)
  index.add(embeddings)
  return index

# ---------- 6. Retrieval ----------
def retrieve(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 3) -> tuple:
 """
 Find the top‑k most similar document chunks to a query vector.

 Parameters
 ----------
 index : faiss.IndexFlatL2
 The pre‑built FAISS index.
 query_embedding : np.ndarray
 Embedding of the user question, shape (1, dim).
 k : int, optional
 How many neighbours to return (default 3).

 Returns
 -------
 indices : np.ndarray of int
 1‑D array of the top‑k document indices.
 distances : np.ndarray of float
 Corresponding L2 distances – useful for debugging.
 """
 print(f"Retrieving top-{k} documents...")
 distances, indices = index.search(query_embedding, k)
 return indices[0], distances[0]

# ---------- 7. Prompt construction ----------
def build_prompt(context_docs: list, user_query: str) -> str:
 """
 Assemble the final prompt that will be fed to the language model.

 Parameters
 ----------
 context_docs : list of str
 The text chunks that were retrieved for the query.
 user_query : str
 The original user question.

 Returns
 -------
 prompt : str
 A single string that follows the format required by the
 generation step: “Context:\n<docs>\n\nQuestion: <q>\nAnswer:”
 """
 print("Building prompt...")
 context = "\n\n".join(context_docs)
 prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
 return prompt

# ---------- 4. Embeddings ----------
def embed_documents(docs: list, model_name: str = "all-MiniLM-L6-v2") -> tuple:
 """
 Convert text chunks into dense vector representations.

 Parameters
 ----------
 docs : list of str
 The list of document chunks to embed.
 model_name : str, optional
 The sentence‑transformer model to use. The default
 “all-MiniLM-L6-v2” is lightweight and works well for quick demos.

 Returns
 -------
 embeddings : np.ndarray
 2‑D array of shape (num_chunks, embedding_dim).
 model : SentenceTransformer
 The loaded embedding model – kept for re‑encoding queries later.
 """
 print("Embedding documents...")
 model = SentenceTransformer(model_name)
 embeddings = model.encode(docs, convert_to_numpy=True)
 return embeddings, model

# ---------- 8. Generation ----------
def generate_answer(context_docs: list, user_query: str) -> str:
 """
 Generate a deterministic answer using a causal language model.

 The generation step is *deterministic* because we fix the random seed
 and set `do_sample=False`. This makes the output reproducible – a
 crucial property for teaching labs.

 Parameters
 ----------
 prompt : str
 Prompt produced by :func:`build_prompt`.
 model_name : str, optional
 Name of the Hugging‑Face transformer model to use.
 `"gpt2-large"` is a good trade‑off between quality and speed.
 max_new_tokens : int, optional
 Maximum number of tokens to generate beyond the prompt.
 dtype : torch.dtype, optional
 Data type for model tensors – `torch.float16` reduces GPU memory
 usage when a GPU is available.

 Returns
 -------
 answer : str
 The generated text after the last “Answer:” marker.
 """
 # Define prompt templates (no need for separate Runnable chains)
 prompt = ChatPromptTemplate.from_template ("Create a response from the context{context} for the question: {question}.")
 context = "\n\n".join(context_docs)

 parser = StrOutputParser()

 # 3. Create the pipeline (chain) using the initialized model
 chain = prompt | llm | parser
 # Run it
 response = chain.invoke(input={"context": context, "question": user_query})
 return response

def load_processed_files(index_file_path):
    """
    Loads a set of processed file paths from a JSON file.
    """
    if index_file_path.exists():
        with open(index_file_path, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files_set, index_file_path):
    """
    Saves a set of processed file paths to a JSON file.
    """
    with open(index_file_path, 'w', encoding='utf-8') as f:
        json.dump(list(processed_files_set), f, indent=4)

from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/50 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def find_and_follow(content, output_dir):
        links = re.findall(r'(https?://\S+)', content)

        print(f"Found {len(links)} links. Following...\n\n")

        for link in links:
            # Ensure the output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Changed output_filename to be relative to the output_dir
            output_filename = Path(output_dir) / f"output_{timestamp}.txt"

            try:
              with open(output_filename, 'w', encoding='utf-8') as f:
                # Regex to find http/https links
                # Follow the link
                response = requests.get(link, headers=headers, timeout=5)
                if (response.status_code == 200):
                  # Parse HTML and extract text
                  soup = BeautifulSoup(response.text, 'html.parser')
                  plain_text = soup.get_text(separator=' ', strip=True)
                  f.write(f"Content:\n{plain_text}\n\n") # Add this line to write the response text
                else:
                    f.write(f"Failed to retrieve content from {link}. Status Code: {response.status_code}\n\n")
            except Exception as e:
              print(f"Exception {e.__class__.__name__} while processing {link}")
            print(f"Output successfully written to {output_filename}")

# ---------- 3. Document collector ----------
def collect_documents(doc_dir: str):
  """
  Load and preprocess all plain‑text documents in a directory.
  It uses an index to avoid reprocessing already processed files.

  Parameters
  ----------
  doc_dir : str
  Path to a folder that contains one or more *.txt files.

  Returns
  -------
  docs : list of str
  A flat list where each element is a chunk of 50 words taken from the
  original files, with a 10‑word overlap between consecutive chunks.

  Notes
  -----
  • Chunking at a small granularity (50 words) allows the retriever to
  identify highly relevant snippets rather than whole paragraphs.
  • The 10‑word overlap ensures that the boundary words of a chunk
  are not lost when we split a document – this improves semantic
  continuity for the embedding model.
  """
  print("Collecting documents...")
  INDEX_FILENAME = "processed_files_index.json"
  OUTPUT_DIR_FOR_LINKS = Path(doc_dir) / "link_outputs"

  # Ensure the directory for linked outputs and the index file exists
  OUTPUT_DIR_FOR_LINKS.mkdir(parents=True, exist_ok=True)

  index_file_path = OUTPUT_DIR_FOR_LINKS / INDEX_FILENAME
  print(f"index file path {index_file_path}")
  processed_files = load_processed_files(index_file_path)
  print(f"processed files count: {len(processed_files)}")
  files_to_process = []
  for file_path in Path(doc_dir).glob("*.txt"):
    print(f"Checking file: {file_path}")
    if file_path.name == INDEX_FILENAME:
      continue # Skip the index file itself
    if str(file_path) not in processed_files:
      files_to_process.append(file_path)

  if not files_to_process:
    print("No new files to process.")
    return

  for file_path in files_to_process:
    with open(file_path, "r", encoding="utf-8") as f:
      content = f.read()
      print(f"Processing new file: {file_path}")
      find_and_follow(content, OUTPUT_DIR_FOR_LINKS) # Corrected output directory for linked files
      processed_files.add(str(file_path))

  save_processed_files(processed_files, index_file_path)
  print(f"Updated processed files index at {index_file_path}")
  print(f"Processed {len(files_to_process)} new documents.")

# ---------- 3. Document loader ----------
def load_documents(doc_dir: str) -> list:
  INDEX_FILENAME = "processed_files_index.json"
  """
  Load and preprocess all plain‑text documents in a directory.

  Parameters
  ----------
  doc_dir : str
  Path to a folder that contains one or more *.txt files.
  output_link_filename : str, optional
  If provided, links found will be written to this file. Defaults to a timestamped file.

  Returns
  -------
  docs : list of str
  A flat list where each element is a chunk of 50 words taken from the
  original files, with a 10‑word overlap between consecutive chunks.

  Notes
  -----
  • Chunking at a small granularity (50 words) allows the retriever to
  identify highly relevant snippets rather than whole paragraphs.
  • The 10‑word overlap ensures that the boundary words of a chunk
  are not lost when we split a document – this improves semantic
  continuity for the embedding model.
  """
  print("Loading documents...")
  # Helper that splits a single string into overlapping chunks
  def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list:
    """
    Split a block of text into overlapping word‑based chunks.

    Parameters
    ----------
    text : str
    Raw document text.
    chunk_size : int, optional
    Number of words per chunk (default 50).
    overlap : int, optional
    Number of words that consecutive chunks share (default 10).

    Returns
    -------
    list of str
    List of chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
      end = start + chunk_size
      chunk = " ".join(words[start:end])
      chunks.append(chunk)
      start += chunk_size - overlap
    return chunks
  docs = []
  #Processing the main files
  for file_path in Path(doc_dir).glob("*.txt"):
    if file_path.name == INDEX_FILENAME:
      continue # Skip the index file itself
    with open(file_path, "r", encoding="utf-8") as f:
      content = f.read()
      # Extend the master list with the new chunks
      docs.extend(chunk_text(content))
  #Processing the linked files
  for file_path in Path(f"{doc_dir}/link_outputs").glob("*.txt"):
    if file_path.name == INDEX_FILENAME:
      continue # Skip the index file itself
    with open(file_path, "r", encoding="utf-8") as f:
      content = f.read()
      # Extend the master list with the new chunks
      docs.extend(chunk_text(content))
  print(f"Loaded: Number of docs: {len(docs)}")
  return docs

from langsmith import traceable

@traceable()
def run_rag_pipeline(user_query: str) -> tuple[str, list]:
  """
  Runs the full RAG pipeline for a given user query.

  Parameters
  ----------
  user_query : str
      The user's question.

  Returns
  -------
  tuple[str, list]
      A tuple containing the generated answer and the list of retrieved documents.
  """

  # 1️⃣ Load documents
  docs = load_documents("/content/drive/MyDrive/Colab Notebooks/docs")
  # 2️⃣ Create embeddings
  #embeddings, embed_model = embed_documents(docs)
  embed_model = SentenceTransformer("all-MiniLM-L6-v2")
  # # 3️⃣ Build FAISS index -try later
  # faiss_index = FAISS.load_local(
  #       "faiss_index",
  #       embeddings,
  #       allow_dangerous_deserialization=True
  #   )
  #faiss_index = faiss.read_index("faiss_index")
  model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = model.encode(docs, convert_to_numpy=True)
  faiss_index = build_faiss_index(embeddings)
  # 4️⃣ Encode the user query
  query_vec = embed_model.encode([user_query], convert_to_numpy=True)

  # 5️⃣ Retrieve top‑k contexts
  top_k_indices, _ = retrieve(faiss_index, query_vec, k=3)
  retrieved_docs = [docs[i] for i in top_k_indices]

  # 6️⃣ Build prompt
  #prompt = build_prompt(retrieved_docs, user_query)
  #print (f"prompt {prompt}")
  # 7️⃣ Generate answer
  answer = generate_answer(
      context_docs=retrieved_docs,
      user_query=user_query
  )
  return answer, retrieved_docs

from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI # Assuming ChatOpenAI is used for retrieval_relevance_llm

# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[
        int,
        ...,
        "A relevance score between 1 and 5. 1 if none or little relevant information, 2-5 for increasing relevance.",
    ]

# Grade prompt
retrieval_relevance_instructions = """You are a analyist grading an answer. You will be given a Product Market QUESTION and a set of FACTS provided by the researcher. Your goal is to assign a relevance score to the FACTS based on the QUESTION.

Scoring Criteria:
- Score 1-2: FACTS contain very little or vaguely related information, or contain some keywords but lack semantic meaning.
- Score 3-4: FACTS contain some relevant information, addressing parts of the question, or have a clear semantic connection.
- Score 5: FACTS are highly relevant and directly answer the QUESTION comprehensively.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0 # Changed model to gpt-4o-mini as used elsewhere
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

print("RetrievalRelevanceGrade and retrieval_relevance_llm updated to provide a numeric score.")

def retrieval_relevance(inputs: dict, outputs: dict) -> int:
    """An evaluator for document relevance, returning a numeric score from 0-5."""
    # Ensure outputs['documents'] is a list of strings if it contains page_content, otherwise handle it
    if outputs and 'documents' in outputs and outputs['documents']:
        if hasattr(outputs['documents'][0], 'page_content'): # Check if it's a list of Document objects
            doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        else: # Assume it's already a list of strings
            doc_string = "\n\n".join(outputs["documents"])
    else:
        doc_string = "No documents retrieved."

    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": answer}
    ])
    return grade["score"]

from langsmith import traceable

@traceable()
def rag_bot(user_query: str) -> dict:
    # langchain Retriever will be automatically traced
  answer, retrieved_docs = run_rag_pipeline(user_query) # user_query is a string here

  return {"outputs": answer, "documents": retrieved_docs}

def target(inputs: dict) -> dict:
  # Ensure inputs["question"] is a string before passing to rag_bot
  question_input = inputs["question"]
  print(f"question_input in target: {question_input}")
  actual_question_str = ""
  if isinstance(question_input, list) and len(question_input) > 0:
      actual_question_str = question_input[0]
  elif isinstance(question_input, str):
      actual_question_str = question_input
  else:
      raise ValueError(f"Question input has unexpected type: {type(question_input)}. Expected str or list[str].")

  return rag_bot(actual_question_str) # Pass the string to rag_bot

# class FeatureImportanceInput(TypedDict):
#     competitor_name: Annotated[str, ..., "The name of the competitor to be evaluated"]
#     features: Annotated[list[str], ..., "A list of features or criteria to analyze"]

# class FeatureImportanceOutput(TypedDict):
#     analysis_summary: Annotated[str, ..., "The synthesized analysis output from the LLM"]
#     feature_scores: Annotated[dict[str, int], ..., "Mapping of features to their evaluated scores (1-5)"]

# class CompetitorState(TypedDict):
#     feature_importance_input: FeatureImportanceInput
#     feature_importance_output: FeatureImportanceOutput
#     retrieved_docs: Annotated[list[str], ..., "The document chunks used for this analysis"]

# @traceable()
# def query_competition_agent(state: CompetitorState) -> dict:
#     """
#     Agent function to perform competition analysis using the RAG pipeline.
#     """
#     input_data = state.get("feature_importance_input")
#     if not input_data:
#         raise ValueError("feature_importance_input is missing from state")

#     comp_name = input_data.get("competitor_name", "Unknown Competitor")
#     features = input_data.get("features", [])

#     query = f"Analyze {comp_name} focusing on: {', '.join(features)}"

#     # Utilize the existing RAG pipeline logic
#     answer, docs = run_rag_pipeline(query)

#     return {
#         "feature_importance_output": {"analysis_summary": answer, "feature_scores": {}},
#         "retrieved_docs": docs
#     }
