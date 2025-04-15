from config import OPENAI_API_KEY, OPENAI_API_TYPE, OPENAI_API_BASE, OPENAI_API_VERSION, DEPLOYMENT_NAME
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain_core.tools import StructuredTool
from tools.weather_tool import get_weather_info_sync
from tools.attraction_tool import recommend_attractions
from tools.route_tool import plan_routes
from pydantic import BaseModel, Field


# 定义工具输入 schema
class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，例如 '上海'")


class AttractionInput(BaseModel):
    query: str = Field(description="查询字符串，例如 '上海'")


class RouteInput(BaseModel):
    query: str = Field(description="路线查询，例如 '上海'")


# 初始化 LLM
llm = AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME
)

# 定义工具
weather_tool = StructuredTool.from_function(
    func=get_weather_info_sync,
    name="WeatherTool",
    description="通过高德MCP获取指定城市的天气信息。输入城市名称（如 '上海'）。",
    args_schema=WeatherInput
)

attraction_tool = StructuredTool.from_function(
    func=recommend_attractions,
    name="AttractionTool",
    description="基于知识库推荐指定城市的旅游景点。输入城市名称（如 '上海'）。",
    args_schema=AttractionInput
)

route_tool = StructuredTool.from_function(
    func=plan_routes,
    name="RouteTool",
    description="通过高德MCP为指定城市规划推荐景点之间的路线。输入城市名称（如 '上海'）。",
    args_schema=RouteInput
)

# 初始化多个 agent
weather_agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

attraction_agent = initialize_agent(
    tools=[attraction_tool],
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

route_agent = initialize_agent(
    tools=[route_tool],
    llm=llm,
    agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


# 协调多个 agent
def coordinate_agents(question):
    try:
        # 提取城市名称
        city = question.split("去")[-1].split("旅游")[0].strip()

        # Step 1: 获取天气
        weather_result = "无法获取天气信息，请稍后重试"
        try:
            print(f"\n=== 天气 Agent 处理: {city} ===")
            weather_result = weather_agent.run(f"获取{city}的天气")
        except Exception as e:
            print(f"天气 Agent 失败: {str(e)}")

        # Step 2: 推荐景点
        attractions = "暂无景点推荐"
        try:
            print(f"\n=== 景点 Agent 处理: {city} ===")
            attractions = attraction_agent.run(f"推荐{city}的景点")
        except Exception as e:
            print(f"景点 Agent 失败: {str(e)}")

        # Step 3: 规划路线
        routes = "暂无路线规划"
        try:
            print(f"\n=== 路线 Agent 处理: {city} ===")
            routes = route_agent.run(f"规划{city}的路线")
        except Exception as e:
            print(f"路线 Agent 失败: {str(e)}")

        # 整合结果
        result = f"🧳 旅游方案 - {city}\n\n" \
                 f"🌤️ 天气信息:\n{weather_result}\n\n" \
                 f"🏛️ 推荐景点:\n{attractions}\n\n" \
                 f"🗺️ 路线规划:\n{routes}"
        return result

    except Exception as e:
        return f"生成旅游方案失败: {str(e)}"


if __name__ == "__main__":
    question = "我打算下周去上海旅游，帮我推荐景点、安排路线，并告诉我天气"
    result = coordinate_agents(question)
    print("\n", result)