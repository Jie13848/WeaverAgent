"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI

logger = logging.getLogger('weaveragent.llm_client')

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        is_gpt5 = bool(re.match(r'gpt-5', self.model, re.IGNORECASE))

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if is_gpt5:
            # GPT-5 系列是 reasoning 模型，API 限制：
            #   1. 不接受自定义 temperature（只能用默认 1）
            #   2. 用 max_completion_tokens 而不是 max_tokens
            #   3. reasoning_effort 控制思考量：minimal/low/medium(默认)/high
            #      - minimal: 几乎不思考，速度/单价接近 gpt-4o-mini，但 ReACT 工具选择会失败
            #      - low: 仅做必要决策，比默认快 3-5 倍，仍能正确触发工具调用
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = kwargs.pop("reasoning_effort", "low")
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        if response_format:
            kwargs["response_format"] = response_format

        # 各家思考型模型默认开启 reasoning，与 response_format=json 不兼容且浪费大量 tokens
        # 通过 extra_body 关闭：
        #   - Qwen3:  enable_thinking=False
        #   - GLM-5+: thinking.type=disabled
        if re.match(r'qwen3', self.model, re.IGNORECASE):
            kwargs["extra_body"] = {"enable_thinking": False}
        elif re.match(r'glm', self.model, re.IGNORECASE):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()

                usage = getattr(response, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", 0) or 0
                    ct = getattr(usage, "completion_tokens", 0) or 0
                    rt = 0
                    details = getattr(usage, "completion_tokens_details", None)
                    if details is not None:
                        rt = getattr(details, "reasoning_tokens", 0) or 0
                    logger.info(
                        f"[usage] model={self.model} prompt={pt} completion={ct} "
                        f"reasoning={rt} total={pt+ct}"
                    )

                # GPT-5 reasoning 模型可能把所有 token 都用在思考上、content 为空
                # 自动加大 max_completion_tokens 重试一次
                if not content and is_gpt5 and attempt < 2:
                    finish = response.choices[0].finish_reason
                    usage = getattr(response, "usage", None)
                    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) if usage else 0
                    new_budget = max(int(kwargs["max_completion_tokens"] * 2), reasoning * 4 + 1024)
                    logger.warning(
                        f"GPT-5 空响应 (finish={finish}, reasoning_tokens={reasoning}), "
                        f"max_completion_tokens {kwargs['max_completion_tokens']} → {new_budget} 重试"
                    )
                    kwargs["max_completion_tokens"] = new_budget
                    continue

                return content
            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                is_rate_limit = (
                    "429" in err_str
                    or "rate_limit" in err_lower
                    or "rate limit" in err_lower
                    or "too many requests" in err_lower
                )
                if is_rate_limit and attempt < max_retries - 1:
                    delay = 2.0 * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit ({type(e).__name__}: {err_str[:200]}), "
                        f"retry {attempt+1}/{max_retries} after {delay:.0f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM 调用失败 ({type(e).__name__}): {err_str[:500]}"
                    )
                    raise
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

