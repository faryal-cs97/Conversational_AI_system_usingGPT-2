# Conversational_AI_system_usingGPT-2

This repository contains the code for building a Conversational AI System using GPT-2, a powerful transformer-based language model. The system can generate context-aware responses to user input and handle multi-turn conversations.

# Features
Pre-trained GPT-2 model for text generation.
Built-in dataset support (AG News) for diverse conversational context.
Text preprocessing for cleaning and tokenizing input data.
Configurable parameters like temperature, top-k sampling, and repetition penalty for better response quality.
Multi-turn conversation system to maintain context and coherence throughout a chat.

# Requirements
Python 3.x
transformers library by Hugging Face
nltk for text tokenization
datasets for built-in datasets
torch for model loading and inference

# Installation
Clone the repository:

git clone https://github.com/your-username/conversational-ai-gpt2.git
cd conversational-ai-gpt2

# Install required packages:

pip install -r requirements.txt

# How to Use
# 1. Load the GPT-2 model
The system uses the pre-trained GPT-2 model from Hugging Face's model hub.

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# 2. Preprocess User Input
The input text is cleaned and tokenized for better text generation quality.

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    sentences = sent_tokenize(text)
    return sentences

# 3. Generate a Response
The generate_response function handles generating conversational responses based on user input.

def generate_response(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        early_stopping=True,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# 4. Run the Conversational AI System
Run the system and start interacting with the bot in a multi-turn conversation:

def conversational_system():
    print("Conversational AI System (GPT-2)")
    print("Type 'exit' to end the conversation.\n")
    conversation_context = f"Dataset Prompt: {load_and_sample_dataset()}\n"

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Conversation ended. Goodbye!")
            break

        conversation_context += f"User: {user_input}\nBot: "
        bot_response = generate_response(conversation_context)
        print(f"Bot: {bot_response}")
        conversation_context += bot_response + "\n"

if __name__ == "__main__":
    conversational_system()

# Example Questions for the Model
Here are a few questions to ask the model:

"What is your name?"
"Tell me about AI."
"Can you explain the theory of relativity?"
"How does climate change affect the planet?"
"Who was the first president of the United States?"
