"""
LLM集成模块
支持多种大语言模型的集成，用于RAG系统的生成部分
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from ..utils.interfaces import Document, Query


class BaseLLM(ABC):
    """LLM基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def generate(self, query: str, context: List[Document], **kwargs) -> str:
        """基于查询和上下文生成回答"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT模型集成"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=config.get('api_key'),
            base_url=config.get('base_url')
        )
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.max_tokens = config.get('max_tokens', 1000)
        self.temperature = config.get('temperature', 0.7)
    
    def generate(self, query: str, context: List[Document], **kwargs) -> str:
        """使用OpenAI API生成回答"""
        
        # 构建上下文
        context_text = "\n\n".join([
            f"文档{i+1}: {doc.title}\n{doc.text[:500]}..."
            for i, doc in enumerate(context[:5])  # 限制上下文长度
        ])
        
        # 构建prompt
        prompt = f"""基于以下上下文信息回答问题：

上下文：
{context_text}

问题：{query}

请基于上下文信息给出准确、详细的回答。如果上下文中没有相关信息，请说明。
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，请基于提供的上下文信息准确回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"生成回答时出错: {str(e)}"


class HuggingFaceLLM(BaseLLM):
    """HuggingFace模型集成"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = config.get('device', 'cpu')
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def generate(self, query: str, context: List[Document], **kwargs) -> str:
        """使用HuggingFace模型生成回答"""
        
        # 构建输入文本
        context_text = " ".join([doc.text[:200] for doc in context[:3]])
        input_text = f"Context: {context_text} Question: {query} Answer:"
        
        # Tokenize
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        if self.device == 'cuda':
            inputs = inputs.cuda()
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案部分
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response
        
        return answer


class LLMManager:
    """LLM管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> BaseLLM:
        """初始化LLM"""
        llm_type = self.config.get('type', 'openai')
        
        if llm_type == 'openai':
            return OpenAILLM(self.config)
        elif llm_type == 'huggingface':
            return HuggingFaceLLM(self.config)
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")
    
    def generate_answer(self, query: Query, retrieved_docs: List[Document]) -> str:
        """生成最终答案"""
        return self.llm.generate(query.text, retrieved_docs)


# 使用示例配置
EXAMPLE_CONFIG = {
    "openai": {
        "type": "openai",
        "model": "gpt-3.5-turbo",
        "api_key": "your-api-key",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "huggingface": {
        "type": "huggingface",
        "model_name": "microsoft/DialoGPT-medium",
        "device": "cpu"
    }
}
