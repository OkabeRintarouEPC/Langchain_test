from fastapi import Body, Request, Query
from fastapi.responses import JSONResponse
from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs

from fastapi.responses import JSONResponse

async def knowledge_base_chat_once(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["test1"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=1),
                              history: List[History] = Body([],
                                                        description="历史对话",
                                                        examples=[[
                                                            {"role": "user",
                                                             "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                            {"role": "assistant",
                                                             "content": "虎头虎脑"}]]
                                                        ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: int = Query(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                              prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                          ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    callback = AsyncIteratorCallbackHandler()
    model = get_ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[callback],
    )

    docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
    context = "\n".join([doc.page_content for doc in docs])

    prompt_template = get_prompt_template("knowledge_base_chat_once", prompt_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])

    chain = LLMChain(prompt=chat_prompt, llm=model)

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        chain.acall({"context": context, "question": query}),
        callback.done),
    )

    source_documents = []
    model_responses = []  # New list to accumulate model responses

    for inum, doc in enumerate(docs):
        filename = os.path.split(doc.metadata["source"])[-1]
        parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
        url = f"/knowledge_base/download_doc?" + parameters
        text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
        source_documents.append(text)

    async for token in callback.aiter():
        # Accumulate model responses in the list
        model_responses.append(token)

    # After completing the generation, create a JSON response
    answer = "".join(model_responses)
    response_data = {"answer": answer, "docs": source_documents}
    
    await task
    # Use JSONResponse to send the accumulated response to the client
    return JSONResponse(content=response_data)


