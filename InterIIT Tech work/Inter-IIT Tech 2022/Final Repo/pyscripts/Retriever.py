from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import numpy as np
from tqdm import tqdm
import torch

class Retriever():
	def __init__(self, train_df, query_encoder = "facebook/dpr-question_encoder-single-nq-base", passage_encoder = "facebook/dpr-ctx_encoder-single-nq-base", top_k = 10):
		self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_encoder)
		self.ctx_model = DPRContextEncoder.from_pretrained(passage_encoder)
		self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_encoder)
		self.query_model = DPRQuestionEncoder.from_pretrained(query_encoder)
		self.top_k = top_k
		self.train_df = train_df
	
	def get_ctx_embeddings(self, passage):
		passage_input_ids = self.ctx_tokenizer(passage, return_tensors="pt")["input_ids"]
		passage_embeddings = self.ctx_model(passage_input_ids).pooler_output.detach().numpy()
		return passage_embeddings
	
	def get_query_embeddings(self, query):
		query_input_ids = self.query_tokenizer(query, return_tensors="pt")["input_ids"]
		query_embeddings = self.query_model(query_input_ids).pooleer_output.detach().numpy()
		return query_embeddings

	def similarity(self, query_embeddings, passage_embeddings):
		return np.dot(query_embeddings, passage_embeddings)
	
	def get_top_k(self, query_embeddings):
		paras = []
		sim_vector = np.dot(self.passage_embeddings, query_embeddings.T)
		a = np.argpartition(sim_vector, -self.top_k)[-self.top_k:]
		for i in range(len(paras)):
			paras[i] = self.train_df.iloc[a[i]]['paargraph']
		
		sim_dict = [{
			"passage": paras[i],
			"similarity": sim_vector[a[i]],
			"theme": self.train_df.iloc[a[i]]['theme'],
		} for i in range(len(paras))]
		return sim_dict

	def update_embeddings(self):
		self.passage_embeddings = self.get_ctx_embeddings(self.train_df.iloc[0]['passage'])
		for i in tqdm(range(1,len(self.train_df)), desc = "Fetching Embeddings"):
			passage = self.train_df.iloc[i]['passage']
			passage_embeddings = self.get_ctx_embeddings(passage)
			self.passage_embeddings = np.row_stack((self.passage_embeddings, passage_embeddings))

	def retrieve(self, query):
		query_embeddings = self.get_query_embeddings(query)
		return self.get_top_k(query_embeddings)