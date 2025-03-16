from flask import Flask, render_template, request, redirect, url_for, session
import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

app = Flask(__name__)

app.secret_key = 'your_secure_random_key_here'  # Replace with a secure string 

# Load models and vector store once when the app starts
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prompt Template
prompt = ChatPromptTemplate.from_template("""\
You are Elon Musk. Based on the following context, provide a concise answer to the question as Elon Musk would. 
If the answer isn't explicitly in the context, make an educated guess using your knowledge of Elon Musk's public persona.

Context: {context}

Question: {question}

Answer:
""")

# Load the embedding model
model_name = 'sentence-transformers/all-mpnet-base-v2'    
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

# Load the vector store
vector_path = 'vector-store'
db_file_name = 'elon_musk_vector_store'
vector_store = FAISS.load_local(
    os.path.join(vector_path, db_file_name), embedding_model, index_name='elon',
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) 


# Load the generator model
model_id = "meta-llama/Llama-2-7b-chat-hf"  # 7B Llamna 2 model
quantization_config = BitsAndBytesConfig(load_in_4bit=True)  
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

text_gen_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,  
    temperature=0.5,  # Lower means more focused responses
    truncation=True,
    max_length=2048,
    return_full_text=False 
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Set up the memory and conversational chain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=False
)

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form['question'].strip()
        # Get chat history from session (default to empty list)
        chat_history = session.get('chat_history', [])
        
        # Convert chat history to list of (human, ai) tuples for the chain
        chat_history_for_chain = [(exchange['question'], exchange['answer']) for exchange in chat_history]
        
        # Call the chain with the question and history
        response = chain({"question": question, "chat_history": chat_history_for_chain})
        answer = response['answer'].strip()
        sources = response['source_documents']
        
        # Format source displays
        source_displays = []
        for doc in sources:
            if 'page' in doc.metadata:
                source_displays.append(f"{doc.metadata['source']}, page {doc.metadata['page']}")
            else:
                source_displays.append(doc.metadata['source'])
        
        # Add the new exchange to chat history
        new_exchange = {
            "question": question,
            "answer": answer,
            "sources": source_displays
        }
        chat_history.append(new_exchange)
        session['chat_history'] = chat_history
        
        return redirect(url_for('chat'))
    
    # GET request: Display the chat interface
    chat_history = session.get('chat_history', [])
    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)