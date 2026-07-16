from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from pytube import YouTube
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric, ContextualPrecisionMetric,ContextualRecallMetric
from  deepeval import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models import GeminiModel

# IMP ->(TinyLlama) has a maximum context window of 2048 tokens.

# LOADER

input_api=input("Enter your URL: ")

video_id=YouTube(input_api)
# print(type(video_id)) #object
video_id=video_id.video_id

try:
    transcript=YouTubeTranscriptApi().fetch(video_id=video_id,languages=['hi','en'])
    # print(transcript)
    raw_text=[]
    for trans in transcript:
        raw_text.append(trans.text)
    
    raw_text=" ".join(raw_text)
    
except TranscriptsDisabled:
    print("INVALID URL YOU HAVE ENTER")
    exit()
  
except Exception:
    print("INVALID URL YOU HAVE ENTER")
    exit()


# SPLITTING

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

docs=splitter.create_documents([raw_text])

# print(docs[0])

# VECTORSTOR

emb_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vectorstor=Chroma.from_documents(documents=docs,embedding=emb_model,collection_name='default')


#RETRIVER

retriver=vectorstor.as_retriever(search_type='similarity',search_kwargs={'k':3})

# user_query=input("Enter your query user: ")

# Augmentation

prompt=PromptTemplate(template=''' you are helper assistance.Answer only from provided transcript context if context is insufficient just say don't know {context} question {question}''',
                      input_variables=['context','question'])


# GENERATION

pipe=pipeline(task='text-generation',
              model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
              temperature=0.1,
              return_full_text=False)

llm=HuggingFacePipeline(pipeline=pipe)  # we can directly use this because it made llm pipeline

chatmodel=ChatHuggingFace(llm=llm)  #it is use for directly interact with api

parser=StrOutputParser()
# -----------------------------------------------------
# print(chatmodel.invoke('hello model').content)

# context=retriver.invoke(user_query)

# print(context[0])

# context_raw="\n\n".join([doc.page_content for doc in context])


# final_prompt=prompt.invoke({'context':context,'question':user_query})


# result=chatmodel.invoke(final_prompt)

# parser_res=parser.invoke(result)

# print(parser_res)

# -----------------------------------------------

# TILL HERE EVERYTHING IS MANUAL LETS MAKE IT AUTO BY USING CHAIN


def clean_reteriver(context):
    context_raw="\n\n".join([doc.page_content for doc in context])
    return context_raw

paralll_chain=RunnableParallel({
    'context':retriver|RunnableLambda(clean_reteriver),
    'question':RunnablePassthrough()
})

seq_chain=paralll_chain|prompt|chatmodel|parser

# response = seq_chain.invoke(user_query)

# print(response)


# Evaluate(MOST IMP)

# ground truth ->means actual correct ans

questions = [
    "What is Retrieval-Augmented Generation?",
    "Why can't ChatGPT answer private company HR policy questions?",
    "Why do we chunk large documents?",
    "What analogy is used to explain RAG?",
    "What does the microbiology book represent?"
]

ground_truths = [
    "Retrieval-Augmented Generation (RAG) is a technique that combines an LLM with external knowledge retrieval to answer questions.",
    "Because ChatGPT is trained on general internet knowledge and does not know private company HR policies.",
    "Large documents are chunked to avoid context window limitations and reduce token costs while retrieving only relevant information.",
    "The speaker compares an LLM to a student named Mira taking an open-book microbiology exam.",
    "The microbiology book represents an external knowledge source, such as an HR policy document."
]


generated=[]

for q in questions:
    res=seq_chain.invoke(q)
    generated.append(res)

contexts=[]

for q in questions:
    res=retriver.invoke(q)

    retrieved_chunks = []

    for doc in res:
        retrieved_chunks.append(doc.page_content)

    contexts.append(retrieved_chunks)


# LLMTestCase as a container that stores one evaluation
# test_case = LLMTestCase(
#     input=,
#     actual_output=,
#     expected_output=,
#     retrieval_context=[
#     ]
# )


# ALERT ->A stronger (often closed-source) LLM is used as the judge because it provides more accurate and reliable evaluation of answer quality, relevance, and faithfulness than smaller models.

judge = GeminiModel(model="gemini-2.5-flash")


test_cases=[]

for i in range(len(questions)):
    test_case=LLMTestCase(
        input=questions[i],
        actual_output=generated[i],
        expected_output=ground_truths[i],
        retrieval_context=contexts[i]
    )

    test_cases.append(test_case)



answer_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge

)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=judge

)

context_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=judge

)

context_recall = ContextualRecallMetric(
    threshold=0.7,
    model=judge

)


evaluate(
    test_cases=test_cases,
    metrics=[
        answer_metric,
        faithfulness_metric,
        context_precision,
        context_recall
    ]
)















