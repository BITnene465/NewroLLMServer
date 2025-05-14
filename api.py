from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import time
from modelServer import ModelServer

app = FastAPI(title="Newro LLM Server", description="Newro轻量级LLM服务器")
PORT = 10721
MODEL_DIR = "./serve_models"  
DEFAULT_MODEL = "Qwen3-1.7B"  # 默认模型

# 创建ModelServer实例
model_server = ModelServer(models_dir=MODEL_DIR)
model_server.load_model(DEFAULT_MODEL)  # 默认加载的模型

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    top_p: float = Field(default=0.9, ge=0, le=1.0)
    enable_thinking: bool = Field(default=False)

class ChatCompletionResponse(BaseModel):
    content: str
    inference_time: float
    model: str

class ModelSwitchRequest(BaseModel):
    model_name: str

class ModelInfoResponse(BaseModel):
    current_model: Optional[str]

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    """
    聊天补全API，类OpenAI接口
    """
    if model_server.get_current_model() is None:
        raise HTTPException(status_code=400, detail="未加载模型，请先加载模型")   # 会把错误发送到客户端
    
    # 转换消息格式
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # 调用模型生成回复
    content, inference_time = model_server.generate_response(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        enable_thinking=request.enable_thinking
    )
    
    return ChatCompletionResponse(
        content=content,
        inference_time=inference_time,
        model=model_server.get_current_model()
    )

@app.post("/models/switch", response_model=ModelInfoResponse)
async def switch_model(request: ModelSwitchRequest):
    """
    切换模型API
    """
    success = model_server.load_model(request.model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"模型 '{request.model_name}' 未找到")
    print(get_current_model())
    return ModelInfoResponse(current_model=model_server.get_current_model())

@app.get("/models/current", response_model=ModelInfoResponse)
async def get_current_model():
    """
    获取当前加载的模型信息
    """
    return ModelInfoResponse(current_model=model_server.get_current_model())

@app.get("/models/list", response_model=Dict[str, Any])
async def list_models():
    """
    列出所有可用的模型
    """
    models = model_server.get_available_models()
    return {"models": models}

@app.get("/health")
async def health_check():
    """
    健康检查API
    """
    return {"ready": True}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=PORT, reload=True)  # 开发环境允许 reload
