from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from transformers import pipeline
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from pytube import YouTube


load_dotenv()

#session_state is memeory of streamlit and retriver is container that conatin value

if 'retriver' not in st.session_state:
     st.session_state.retriver=None

if 'loader' not in st.session_state:
     st.session_state.loader=False


st.header('Lets make your learning easy ')

st.subheader('Here is your friend learningSathi: ')

youtube_id=st.text_input("Paste your url")

load = st.button("Load Video")


if load:
       try:
           yt=YouTube(youtube_id)
       except Exception:
             st.error("invalid url")
             st.stop()


# LOADER

       try:
              transcript = YouTubeTranscriptApi().fetch(
              yt.video_id,
              languages=['en','hi']
           )

       except TranscriptsDisabled:
              st.error("Transcript is disabled.")
              st.stop()

       text=" ".join(doc.text for doc in transcript)

       st.success("Video loaded successfully!")

       # print(text)

# SPLITTER

       spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

       docs=spliter.create_documents([text])

       # print(docs[0])

       # vectorstor

       emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


       vectorstore=FAISS.from_documents(docs,emb)


# RETREVER

       st.session_state.retriver=vectorstore.as_retriever(search_type='similarity',search_kwargs={'k':5})
       st.session_state.loader = True

#      print(retriver.invoke('I love'))



# Augmantation

# ----close source llm-----

chatmodel=ChatGoogleGenerativeAI(  model="gemini-2.5-flash",)

# print(chatmodel.invoke("how are you bro").content)

prompt=PromptTemplate(template=''' you are helper assistance.Answer only from provided transcript context if context is insufficient just say don't know {context} question {question}''',
                      input_variables=['context','question'])


query=st.text_input("Write your query: ")

parser=StrOutputParser()

def raw_text(docs):
    context_text="\n\n".join(doc.page_content for doc in docs)
    return context_text



# lets gerenertae multiple version of query


def mutqueries(query):

       prompt2 = PromptTemplate(
       template="""
       You are an AI assistant.

       Generate 5 different versions of the given user question.
       Each version should have the same meaning but use different wording.
       These variations will be used to retrieve relevant documents from a vector database.

       Original Question:
       {query}

       Return only the 5 rewritten questions, one per line.
       """,
       input_variables=["query"]
       )

       prompt_res=prompt2|chatmodel|parser

       multiversion = prompt_res.invoke({"query": query})
       # return string 

       queries=multiversion.split("\n")

       queries.append(query)

       return queries


def retrevive_all_context(queries):
     all_docs=[]
     for q in queries:
          docs=st.session_state.retriver.invoke(q)
          all_docs.extend(docs)
     
     return all_docs


def remove_duplicate(docs):
     unique_doc={}

     for d in docs:
          unique_doc[d.page_content]=d
     
     return list(unique_doc.values())


          


if st.button("Ask"):

    if st.session_state.retriver is None:
        st.warning("Please load a YouTube video first.")

    elif query == "":
        st.warning("Please enter a question.")

    else:
       queries = mutqueries(query)

       docs = retrevive_all_context(queries)

       docs = remove_duplicate(docs)

       context = raw_text(docs)

       final_prompt = prompt.invoke({
       "context": context,
       "question": query
       })

       answer = chatmodel.invoke(final_prompt)

       result = parser.invoke(answer)

       st.write(result)




















