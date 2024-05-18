from openai import OpenAI
from config import Config
from langchain_openai import OpenAIEmbeddings

system_prompt = Config.SYSTEM_PROMPT

def chat(user_prompt, model, vector_store=None, max_tokens=200, temp=0.7):
    client = OpenAI()

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # If context is available, prepend it to the user message
    if vector_store:
        context = get_relevant_context(user_prompt, vector_store)
        messages.append({"role": "system", "content": context})

    # Add user message
    messages.append({"role": "user", "content": user_prompt})


     # https://platform.openai.com/docs/api-reference/chat/create
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        stream=True
    )

    return completion

def stream_parser(stream):
    for chunk in stream:
        if chunk.choices[0].delta.content != None:
            yield chunk.choices[0].delta.content

def get_relevant_context(user_prompt, vector_store, top_k=3):
    # Assuming vector_store can provide similarity search
    docs = VectorStore.similarity_search(query=query, k=3)
    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
    st.write(response)