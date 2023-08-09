from transformers import DPRReader, DPRReaderTokenizer
import torch
import numpy as np
from tqdm import tqdm

class Reader():
    def __init__(self, reader = "facebook/dpr-reader-single-nq-base"):
        self.reader = DPRReader.from_pretrained(reader)
        self.tokenizer = DPRReaderTokenizer.from_pretrained(reader)
    
    def get_encoded_inputs(self, query, theme , context):
        encoded_inputs = self.tokenizer(
            questions=[query],
            titles=[theme],
            texts=[context],
            return_tensors="pt",
        )
        return encoded_inputs

    def logistic(self, x):
        return 1/(1+np.exp(-x))
    
    def get_output(self, query, theme, context):
        encoded_inputs= self.get_encoded_inputs(query, theme, context)
        outputs= self.reader(**encoded_inputs)
        return outputs, encoded_inputs

    def get_answer(self, query, theme, context):
        outputs, encoded_inputs = self.get_output(query, theme, context)
        start_scores = outputs.start_logits.detach().numpy()
        end_scores = outputs.end_logits.detach().numpy()
        start_index = np.argmax(start_scores)
        end_index = np.argmax(end_scores)
        score = self.logistic(outputs.relevance_logits.detach().numpy()[0])
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'][0][start_index:end_index+1]))
        dict = {"answer": answer, "score": score}
        return dict

    def get_answers(self, query, theme, context):
        answers = []
        for i in range(len(context)):
            answers.append(self.get_answer(query, theme, context[i]))
        return answers

    def get_best_answer(self, query, theme, context):
        answers = self.get_answers(query, theme, context)
        best_answer = answers[0]
        for i in range(len(answers)):
            if answers[i]["score"] > best_answer["score"]:
                best_answer = answers[i]
        return best_answer
    
    