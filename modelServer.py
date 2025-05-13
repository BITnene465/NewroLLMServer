import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from typing import Dict, List, Optional, Tuple, Set

class ModelServer:
    """
    LLM模型服务器，负责加载、切换模型和处理生成请求
    """
    def __init__(self, models_dir: str = "./serve_models"):
        self.models_dir = models_dir
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
    def load_model(self, model_name: str) -> bool:
        """
        加载指定的模型
        Args:
            model_name: 模型名称（文件夹名）
        Returns:
            bool: 是否成功加载
        """
        try:
            return self._load_model_sync(model_name)
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            return False
    
    def _load_model_sync(self, model_name: str) -> bool:
        """
        同步加载模型的内部实现
        """
        try:
            # 释放之前的模型资源
            if self.model is not None:
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()
            
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_path):
                return False
            
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
            self.model_name = model_name
            return True
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            return False
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 0.7, 
                         max_tokens: int = 1024, 
                         top_p: float = 0.9,
                         enable_thinking: bool = False) -> Tuple[str, float]:
        """
        生成回复文本
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            top_p: Top-p采样参数
            enable_thinking: 是否启用思维链
        Returns:
            Tuple[str, float]: (生成的文本, 推理时间)
        """
        if self.model is None or self.tokenizer is None:
            return "模型未加载，请先加载模型", 0
        
        # 直接调用同步方法
        return self._generate_response(
            messages,
            temperature,
            max_tokens,
            top_p,
            enable_thinking
        )
    
    def _generate_response(self, 
                              messages: List[Dict[str, str]], 
                              temperature: float = 0.7, 
                              max_tokens: int = 1024, 
                              top_p: float = 0.9,
                              enable_thinking: bool = False) -> Tuple[str, float]:
        """
        生成回复
        """
        # 准备模型输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # 控制是否使用思维链
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 记录开始时间
        start_time = time.time()
        
        # 生成文本
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # 计算生成时间
        inference_time = time.time() - start_time
        # 提取生成的ID
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # 如果使用思维链，解析思维内容和最终内容
        if enable_thinking:
            try:
                # 尝试找到</think>标记 (151668)  -- 目前适配 qwen，todo：之后修改
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            # 返回包含思维过程的完整输出
            return f"思考过程: {thinking_content}\n回答: {content}", inference_time
        else:
            # 直接返回生成的文本
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            return content, inference_time
    
    def get_current_model(self) -> Optional[str]:
        """获取当前加载的模型名称"""
        return self.model_name
    
    def get_available_models(self) -> List[str]:
        """获取所有可用的模型列表"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for item in os.listdir(self.models_dir):
            if os.path.isdir(os.path.join(self.models_dir, item)):
                models.append(item)
        return models
    
    def close(self):
        """关闭并清理资源"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()