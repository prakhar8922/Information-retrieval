import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    # Check if the conversation object exists
    if st.session_state.conversation is None:
        st.write("❌ Conversation chain is not initialized. Please upload a PDF and process it first.")
        return

    # Call the conversation chain with the user's question
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']

    # Display the conversation history
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)

def main():
    st.set_page_config(page_title="Information Retrieval")
    st.header("Information Retrieval System")

    # Input field for the user's question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    # Handle user input
    if user_question:
        user_input(user_question)

    # Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process the uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)

                # Initialize the conversation chain
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

if __name__ == "__main__":
    main()
