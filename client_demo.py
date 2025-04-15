import asyncio
from config import OPENAI_API_KEY, OPENAI_API_TYPE, OPENAI_API_VERSION, DEPLOYMENT_NAME, GAODE_MCP_URL
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType

# 调试：打印版本
import sys
import langchain
import pydantic
print("Python 版本:", sys.version)
print("LangChain 版本:", langchain.__version__)
print("Pydantic 版本:", pydantic.__version__)

# 初始化 AzureChatOpenAI
model = AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    azure_endpoint="https://13067-m9egtpio-swedencentral.cognitiveservices.azure.com/",
    openai_api_type=OPENAI_API_TYPE,
    openai_api_version=OPENAI_API_VERSION,
    azure_deployment=DEPLOYMENT_NAME,
    max_tokens=10240,
)

async def get_text():
    async with MultiServerMCPClient(
            {"gaode": {"url": GAODE_MCP_URL, "transport": "sse"}}
    ) as client:
        # 获取高德 MCP 工具
        mcp_tools = client.get_tools()
        print("MCP 工具:", mcp_tools)
        for tool in mcp_tools:
            print("工具名称:", getattr(tool, "name", "无名称"))
            print("工具描述:", getattr(tool, "description", "无描述"))
            print("工具参数:", getattr(tool, "args_schema", "无参数"))

        # 直接使用 mcp_tools
        tools = mcp_tools

        # 确认代理类型
        print("使用的代理类型:", AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

        # 使用支持多输入工具的代理
        try:
            agent = initialize_agent(
                tools=tools,
                llm=model,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,  # 处理解析错误
                max_iterations=10,  # 增加迭代次数
                early_stopping_method="force"  # 强制停止避免无限循环
            )
            print("代理初始化成功:", agent)
        except Exception as e:
            print("代理初始化失败:", str(e))
            raise

        messages = [
            SystemMessage(content="你必须调用外部工具来获取信息，包括天气、地点搜索等。直接回答将被视为错误。"),
            HumanMessage(content="考虑当前的天气情况，我应该如何从上海大学乘坐公共交通前往外滩？然后我想在外滩附近走走逛逛喝杯咖啡，并找个酒店睡一觉。规划细致行驶和游玩路线。")
        ]

        # 调用代理
        try:
            response = await agent.ainvoke(messages[-1].content)
        except Exception as e:
            print("代理调用失败:", str(e))
            raise

        # 打印工具使用情况
        print("中间步骤:", response.get("intermediate_steps", []))
        for item in response.get("intermediate_steps", []):
            tool_used = item[0].tool
            tool_input = item[0].tool_input
            tool_output = item[1]
            print(f"Tool Use: {tool_used} with input: {tool_input}")
            print(f"Tool Output: {tool_output}")

        # 获取最终回答
        final_response = response["output"]
        print(f"\nResponse: {final_response}")

if __name__ == "__main__":
    asyncio.run(get_text())