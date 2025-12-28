import streamlit as st
import requests
import time

# --- THEME & STYLE ---
st.set_page_config(page_title="RecruitAI | Dashboard", page_icon="🎯", layout="centered")

# Custom CSS to soften the UI and make it look "Hand-Coded"
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #FDFDFD; }
    
    /* Soften the chat bubbles */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #EAEAEA !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F7F9 !important;
        border-right: 1px solid #E0E0E0;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Clean Divider */
    hr { margin-top: 1rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR (The "Workspace") ---
with st.sidebar:
    st.subheader("📁 Documents")
    uploaded_files = st.file_uploader(
        "Drop resumes here", 
        accept_multiple_files=True,
        type=['pdf', 'txt'],
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("Add to Knowledge Base", use_container_width=True):
            with st.status("Reading files...", expanded=False) as status:
                files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
                try:
                    res = requests.post("http://localhost:8000/upload", files=files)
                    if res.status_code == 200:
                        status.update(label="Index updated!", state="complete", expanded=False)
                        st.toast("Resumes added successfully!", icon="✅")
                    else:
                        st.error("Upload failed.")
                except Exception as e:
                    st.error(f"Backend offline: {e}")

    st.divider()
    
    # Human touch: A way to reset without digging into settings
    if st.button("🗑️ Reset Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CONTENT ---
# A human dev would put a clear, welcoming header
st.title("Hi there! 👋")
st.markdown("I'm your **Hiring Assistant**. Ask me anything about the candidates you've uploaded.")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Suggested Prompts (Human-like shortcuts)
if not st.session_state.messages:
    st.info("Try asking: 'Who has the most experience in Python?' or 'Summarize Samira’s profile.'")

# Chat Input
if prompt := st.chat_input("Message your assistant..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Response Logic
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Human touch: A slight "thinking" delay feels more natural
        with st.spinner("Thinking..."):
            try:
                res = requests.post("http://localhost:8000/ask", json={"query": prompt})
                if res.status_code == 200:
                    data = res.json()
                    full_response = data.get("answer", "I'm not sure.")
                    sources = data.get("sources", [])
                    
                    # Display response
                    message_placeholder.markdown(full_response)
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                else:
                    st.error("My backend is having a moment. Please try again.")
            except:
                st.error("I can't reach the server right now.")