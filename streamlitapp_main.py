import streamlit as st
import os
from query_rag_application import index_data, query_file_and_invoke_llm

st.set_page_config(page_title='RAG Text Summarization', layout='centered')

st.title('RAG Text Summarization App')

# Upload PDF
st.header('Upload PDF for Indexing')
uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf','txt',"docs"])

if uploaded_file is not None:
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)  # Ensure the directory exists
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded successfully: {uploaded_file.name}")
    try:
        index_data(file_path)
    except Exception as e:
        st.error(f"Error while indexing: {str(e)}")
    st.success("Data indexed successfully!")

    # Indexing the file
    with st.spinner('Indexing the PDF file...'):
        try:
            index_data(file_path)
            st.success('Data indexed successfully')
        except Exception as e:
            st.error(f'Error while indexing: {str(e)}')

# Query Section
st.header('Query the Indexed Data')
user_query = st.text_input('Enter your query')

if st.button('Get Answer'):
    if user_query:
        with st.spinner('Generating answer...'):
            try:
                answer = query_file_and_invoke_llm(user_query)
                st.success('Answer:')
                st.write(answer)
            except Exception as e:
                st.error(f'Error while querying: {str(e)}')
    else:
        st.warning('Please enter a query')
