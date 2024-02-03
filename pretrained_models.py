import nltk
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

conversational_models = [
    'nisten/BigCodeLlama-169b',
    'alpindale/goliath-120b',
    'TheBloke/goliath-120b-GGUF',
    'karakuri-ai/karakuri-lm-70b-chat-v0.1',
    'microsoft/DialoGPT-medium'
]

prompts = [
    "How does photosynthesis work?",
    "Tell me a joke.",
    "What is the capital of France?",
    "Explain the theory of relativity.",
    "Any book recommendations?"
]

references = [
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
    "Why did the chicken cross the road? To get to the other side!",
    "The capital of France is Paris.",
    "The theory of relativity, formulated by Albert Einstein, describes the relationships between space, time, and gravity.",
    "It depends on your interests. What genres do you like?"
]

def initialize_model_and_tokenizer(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_beams=5)
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_response

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score


def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    rouge_n_score = scores[0]['rouge-1']['f']
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_n_score, rouge_l_score


def calculate_meteor(reference, candidate):
    meteor_score_val = meteor_score([reference], candidate)
    return meteor_score_val

def calculate_f1(reference, candidate):
    reference_set = set(reference.split())
    candidate_set = set(candidate.split())

    precision = len(reference_set.intersection(candidate_set)) / len(candidate_set)
    recall = len(reference_set.intersection(candidate_set)) / len(reference_set)

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1


results = dict()

def evaluate_models(model_names, prompts, references):
    

    for model_name in model_names:
        model, tokenizer = initialize_model_and_tokenizer(model_name)
        bleu_scores = []
        rouge_n_scores = []
        rouge_l_scores = []
        meteor_scores = []
        f1_scores = []

        for prompt, reference in zip(prompts, references):
            model_response = generate_response(prompt, model, tokenizer)

            bleu_score = calculate_bleu(reference, model_response)
            bleu_scores.append(bleu_score)

            rouge_n_score, rouge_l_score = calculate_rouge(reference, model_response)
            rouge_n_scores.append(rouge_n_score)
            rouge_l_scores.append(rouge_l_score)


            meteor_score_val = calculate_meteor(reference, model_response)
            meteor_scores.append(meteor_score_val)

            f1 = calculate_f1(reference, model_response)
            f1_scores.append(f1)

        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        avg_rouge_n_score = sum(rouge_n_scores) / len(rouge_n_scores)
        avg_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)
        avg_meteor_score = sum(meteor_scores)/len(meteor_scores)
        avg_f1_score = sum(f1_scores)/len(f1_scores)
       

            

        results[model_name] = {
            "BLEU": avg_bleu_score,
            "ROUGE-N": avg_rouge_n_score,
            "ROUGE-L": avg_rouge_l_score,
            "METEOR" : avg_meteor_score,
            "F1" : avg_f1_score
        }

    return results


model_results = evaluate_models(conversational_models,prompts,references)
model_results = pd.Dataframe(model_results) 
model_results.to_csv('Input.csv',index = False)
    
    
    
