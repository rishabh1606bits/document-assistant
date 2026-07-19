import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Assistant", page_icon="📄")
st.title("📄 Document Assistant")

if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# --- File upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Process document"):
        with st.spinner("Uploading and processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)

            if response.status_code == 200:
                result = response.json()
                st.session_state.doc_processed = True
                st.success(f"Processed {result['chunks_created']} chunks from {result['filename']}")
            else:
                st.error(f"Upload failed: {response.text}")

# --- Chat interface ---
if st.session_state.doc_processed:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching and generating answer..."):
            response = requests.post(f"{BACKEND_URL}/chat", json={"question": question})

            if response.status_code == 200:
                result = response.json()
                st.markdown("### Answer")
                st.write(result["answer"])

                with st.expander("Sources used"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(source)
            else:
                st.error(f"Chat failed: {response.text}")

    # --- Chat history ---
    with st.expander("📜 Past questions"):
        history_response = requests.get(f"{BACKEND_URL}/history")
        if history_response.status_code == 200:
            history = history_response.json()
            for record in history:
                st.markdown(f"**Q:** {record['question']}")
                st.markdown(f"**A:** {record['answer']}")
                st.caption(record['created_at'])
                st.divider()
else:
    st.info("Upload a PDF and click 'Process document' to get started.")