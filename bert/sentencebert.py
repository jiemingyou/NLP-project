from sentence_transformers import SentenceTransformer, util

class SentenceBert:
    "SentenceBert class to encode and rank sentences using SentenceBert model."
    def __init__(self, model_name="all-distilroberta-v1"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, sentence):
        return self.model.encode(sentence)
    
    def cos_sim(self, embedding1, embedding2):
        return util.cos_sim(embedding1, embedding2)
    
    def rank_sentences(self, sentence, sentences):
        embedding = self.encode(sentence)
        embeddings = self.encode(sentences)
        return self.cos_sim(embedding, embeddings)
    
    def top_k_sentences(self, sentence, sentences, k=5):
        scores = self.rank_sentences(sentence, sentences)
        sorted_scores, sorted_indices = scores.sort(descending=True)
        top_k_indices = sorted_indices[0][:k]
        top_k_sentences = [(sentences[i], scores[0][i].item()) for i in top_k_indices]
        return top_k_sentences
    


