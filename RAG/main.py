import faiss
import numpy as np
from model import QwenChatbot
from sentence_transformers import SentenceTransformer


class RAGChatbot:
    def __init__(
        self,
        facts_file: str = "RAG/cat-facts.txt",
        embed_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        self.facts = RAGChatbot.load_txt_file(facts_file)
        self.embedder = SentenceTransformer(embed_model)
        self.top_k = top_k
        self.index = self.build_faiss_index(self.facts)
        self.generator = QwenChatbot()

    @staticmethod
    def load_txt_file(file_path: str):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def build_faiss_index(self, sentences):
        embeddings = self.embedder.encode(sentences, convert_to_tensor=False)
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index

    def retrieve_context(self, query: str):
        query_embedding = self.embedder.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), self.top_k)
        return [self.facts[i] for i in indices[0]]

    def generate_answer(self, user_query: str):
        context = self.retrieve_context(user_query)
        context_str = "\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}"
        return self.generator.generate(prompt)


if __name__ == "__main__":
    rag_bot = RAGChatbot()

    question = "When was the fist cat show and where it was held ?"
    print("🔍 Query:", question)
    answer = rag_bot.generate_answer(question)
    print("🤖 Answer:", answer)
