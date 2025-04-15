from config import OPENAI_API_KEY, OPENAI_API_TYPE, OPENAI_API_BASE, OPENAI_API_VERSION, DEPLOYMENT_NAME
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain_core.tools import StructuredTool
from tools.weather_tool import get_weather_info_sync
from tools.attraction_tool import recommend_attractions
from tools.route_tool import plan_routes

llm = AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME
)

# 定义工具
tools = [
    StructuredTool.from_function(
        func=get_weather_info_sync,
        name="WeatherTool",
        description="通过高德MCP获取指定城市的天气信息。输入城市名称（如 '上海'）。"
    ),
    StructuredTool.from_function(
        func=recommend_attractions,
        name="AttractionTool",
        description="基于知识库推荐指定城市的旅游景点。输入城市名称（如 '上海'）。"
    ),
    StructuredTool.from_function(
        func=plan_routes,
        name="RouteTool",
        description="通过高德MCP为指定城市规划推荐景点之间的路线。输入城市名称（如 '上海'）。"
    )
]

# 初始化代理
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    question = "我打算下周去上海旅游，帮我推荐景点、安排路线，并告诉我天气"
    result = agent.run(question)
    print("\n🧳 旅游方案：\n", result)