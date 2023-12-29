from fastapi import Body, Query
from fastapi.responses import StreamingResponse
from configs import LLM_MODEL, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List
from server.chat.utils import History
from server.utils import get_prompt_template


async def chat_abstract(query: str = Body(..., description="用户输入", examples=["软件测试方法主要有黑箱测试方法与白箱测试两类。黑箱测试又称功能测试、数据驱动测试或基于规格说明的测试，是在完全不考虑程序内部结构和内部特性的情况下，检查输入与输出之间关系是否符合要求。白箱测试又称结构测试、逻辑驱动测试或基于程序的测试，是在已知程序内部结构的情况下设计测试用例的测试方法。显然，白箱测试适合在单元测试中运用，而在独立测试阶段多采用黑箱测试方法。"]),
                history: List[History] = Body([],
                                       description="历史对话"),
                stream: bool = Body(False, description="流式输出"),
                model_name: str = Body(LLM_MODEL, description="LLM 模型名称"),
                temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                max_tokens: int = Query(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                prompt_name: str = Body("test2-b", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
         ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
                            model_name: str = LLM_MODEL,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        prompt_template = get_prompt_template("llm_abstract", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")
