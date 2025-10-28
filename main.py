import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

#LLM and key loading function
def load_LLM(openai_api_key):
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm


#Page title and header
st.set_page_config(page_title="Resumir archivos de texto con IA")
st.header("Resume tus archivos con IA")


#Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT no puede resumir textos largos. Ahora puedes hacerlo con esta aplicación")

with col2:
    st.write("Contacta conmigo [aquí](https://autmatix.es) para construir tus proyectos de IA personalizados.")


#Input OpenAI API Key
st.markdown("## Escribe tu OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", type="password")
    return input_text

openai_api_key = get_openai_api_key()


# Input
st.markdown("## Sube el archivo de texto que quieres resumir")

uploaded_file = st.file_uploader("Elige tu archivo", type="txt")

       
# Output
st.markdown("### Aquí está el resumen de tu archivo:")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    file_input = string_data

    if len(file_input.split(" ")) > 20000:
        st.write("Porfavor, sube un archivo con menos de 20,000 palabras.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning('Porfavor escribe tu OpenaAI API KEY \
            Instrucciones [aquí](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
        )

    splitted_documents = text_splitter.create_documents([file_input])

    llm = load_LLM(openai_api_key=openai_api_key)

    summarize_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce"
        )

    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)
