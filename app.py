# app.py
import streamlit as st
import altair as alt

# st.title('MindMate Prototype')
# st.write('Welcome to MindMate, your mental health companion chatbot.')

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "I'm feeling really stressed lately because of work."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate a response
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
response_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(response_text)

