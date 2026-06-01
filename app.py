import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Keys
load_dotenv()

# --- NEW: Smart History Management ---
HISTORY_FILE = "chat_history.json"

def load_history():
    """Loads all previously asked questions and answers from the file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # NEW: Only load items that actually have the "question" and "answer" keys. 
                # This ignores any old or corrupted formats!
                valid_history = [item for item in data if "question" in item and "answer" in item]
                return valid_history
        except Exception:
            return []
    return []

def save_history(history_list):
    """Saves the Q&A list securely to a JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_list, f, indent=4)
# -------------------------------------

# 2. Setup Streamlit Page (The UI)
st.set_page_config(page_title="NanoPhysics AI Assistant", page_icon="🔬", layout="wide")
st.title("🔬 NanoPhysics & Nanoelectronics AI Assistant")
st.markdown("Ask me any question, and I will search across your nanophysics textbooks to write a highly detailed, essay-style answer!")

# 3. Cache the heavy AI models
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = PineconeVectorStore(index_name="textbook-index", embedding=embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.3, 
        max_tokens=3000
    )

vectorstore = load_vectorstore()
llm = load_llm()

# 4. Setup the AI Brain
prompt = ChatPromptTemplate.from_template("""
You are an expert, highly detailed, and professional Physics Professor and academic tutor. 
Your goal is to write a comprehensive, long-form academic essay to answer the user's question, using the retrieved textbook context as your primary foundation.

Guidelines for your response:
1. **Essay Format & Length:** Write in a detailed, exhaustive, and expansive format. Your response should read like a thorough textbook chapter or academic essay. Do NOT give brief or basic summaries. 
2. **Elaborate Context:** Use the retrieved context as your factual anchor. However, you are highly encouraged to use your internal expert physics knowledge to deeply explain, elaborate on, and contextualize the formulas, principles, and derivations found in the context.
3. **Logical Structure:** Use bold headings, bullet points, and numbered lists where appropriate, but ensure there are thick, detailed paragraphs of explanatory text connecting the ideas.
4. **No Dead Ends:** If the context only has partial information, thoroughly explain what you do have, and naturally fill in the physics gaps to ensure the user gets a complete conceptual understanding.

Context:
{context}

Question: 
{input}

Detailed Essay Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8}) 
rag_chain = create_retrieval_chain(retriever, document_chain)

# 5. Initialize State Variables
if "history" not in st.session_state:
    # 'history' holds ALL saved Q&As from the database
    st.session_state.history = load_history()

if "current_messages" not in st.session_state:
    # 'current_messages' only holds what is currently visible on the screen
    st.session_state.current_messages = []

# --- NEW: ChatGPT-Style Sidebar ---
with st.sidebar:
    st.header("📚 Chat History")
    
    # Button to clear screen for a new question
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_messages = []
        st.rerun()
        
    st.divider()
    st.markdown("**Previous Questions:**")
    
    # Loop through history backwards (so newest questions are at the top)
    for idx, item in enumerate(reversed(st.session_state.history)):
        q_text = item["question"]
        # Shorten the question so it fits nicely on the sidebar button
        btn_text = (q_text[:35] + "...") if len(q_text) > 35 else q_text
        
        # If the user clicks a history button, load ONLY that Q&A onto the screen!
        if st.button(f"📝 {btn_text}", key=f"hist_{idx}", help=q_text):
            st.session_state.current_messages = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
            st.rerun()
# ----------------------------------

# 6. Display ONLY the currently active conversation
for message in st.session_state.current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. React to the User's Question
if user_input := st.chat_input("Ask a physics question..."):
    
    # --- SMART SEARCH: Check if this question was already asked! ---
    # We check if the exact typed question exists in our history
    existing_item = next((item for item in st.session_state.history if item["question"].strip().lower() == user_input.strip().lower()), None)
    
    if existing_item:
        # If it was asked before, instantly show the saved answer! (Saves time & API usage)
        st.session_state.current_messages = [
            {"role": "user", "content": existing_item["question"]},
            {"role": "assistant", "content": existing_item["answer"]}
        ]
        st.rerun()
        
    else:
        # It's a brand NEW question. 
        # Clear the screen and show the user's new question
        st.session_state.current_messages = [{"role": "user", "content": user_input}]
        with st.chat_message("user"):
            st.markdown(user_input)

        # Show a loading spinner while AI writes the new essay
        with st.chat_message("assistant"):
            with st.spinner("Writing a detailed academic essay based on your textbooks..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                
                # Source Citations
                sources = []
                for doc in response["context"]:
                    if 'page' in doc.metadata and 'source' in doc.metadata:
                        book_name = os.path.basename(doc.metadata['source'])
                        page_num = doc.metadata['page'] + 1
                        sources.append(f"**{book_name}** (Page {page_num})")
                
                if sources:
                    unique_sources = list(set(sources))
                    answer += f"\n\n---\n**Sources Used:** {', '.join(unique_sources)}"
                
                st.markdown(answer)
        
        # 1. Update the screen
        st.session_state.current_messages.append({"role": "assistant", "content": answer})
        
        # 2. Auto-Save to the global history list
        st.session_state.history.append({
            "question": user_input,
            "answer": answer
        })
        
        # 3. Permanently save to the JSON file so everyone can see it
        save_history(st.session_state.history)