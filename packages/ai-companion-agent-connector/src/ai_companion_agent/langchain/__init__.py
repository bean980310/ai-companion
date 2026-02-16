import os
import warnings

try:
    from langchain_integrator import LangchainIntegrator
    from langchain.tools import tool
    from langchain.agents import create_agent, AgentState
    from langchain.agents.middleware import SummarizationMiddleware, ModelRetryMiddleware, ToolRetryMiddleware, PIIMiddleware, HumanInTheLoopMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from langgraph.runtime import Runtime
except ImportError:
    warnings.warn("ai-companion-langchain-integrator is not installed. Please install it to use langchain features.", UserWarning)

class LangchainAgentsIntegrator(LangchainIntegrator):
    def __init__(self, provider, model_name = None, lora_model_name = None, model = None, tokenizer = None, processor = None, enable_thinking = False, tools: list[str] | None = None, **kwargs):
        super().__init__(provider, model_name = None, lora_model_name = None, model = None, tokenizer = None, processor = None, enable_thinking = False, **kwargs)
        self.tools = tools
        self.interrupt_on = {}
        for t in self.tools:
            self.interrupt_on.update(t=True)
        self.redact_email, self.mask_credit_card, self.block_api_key = self.get_pii_middle_ware()
        self.agent = create_agent(model=self.chat, tools=self.tools, middleware = [
            SummarizationMiddleware(
                model = self.chat,
                trigger = {"fraction": 0.8},
            ),
            ModelRetryMiddleware(),
            ToolRetryMiddleware(max_retries=3,tools=self.tools,backoff_factor=1.5),
            self.redact_email, self.mask_credit_card, self.block_api_key,
            HumanInTheLoopMiddleware(interrupt_on=self.interrupt_on)
        ])

    def get_pii_middle_ware(self):
        return PIIMiddleware("email", strategy='redact'), PIIMiddleware('credit_card', strategy='mask'), PIIMiddleware('api_key', detector=fr"sk-[a-zA-Z0-9]{32}", strategy='block')
