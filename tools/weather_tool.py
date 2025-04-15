import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_weather_info(city: str) -> str:
    client = MultiServerMCPClient()
    try:
        # 获取工具列表
        tools = client.get_tools()
        print("天气工具列表（原始）:", tools)
        # 适配可能的工具格式
        weather_tool = None
        for tool in tools:
            try:
                if isinstance(tool, dict):
                    name = tool.get("name", "").lower()
                    if name in ["weather_forecast", "weatherforecast"]:
                        weather_tool = tool.get("func")
                        break
                elif hasattr(tool, "name"):
                    if tool.name.lower() in ["weather_forecast", "weatherforecast"]:
                        weather_tool = tool
                        break
                else:
                    print("未知工具格式:", tool)
            except Exception as e:
                print(f"解析工具错误: {str(e)}")
        if not weather_tool:
            return f"未找到天气预报工具，无法获取 {city} 的天气信息。确保 MCP 服务器运行在 http://localhost:8000。"
        # 调用天气预报工具
        weather_data = await weather_tool(city)
        if isinstance(weather_data, list) and weather_data:
            weather = weather_data[0].get("weather", "未知")
            temp = weather_data[0].get("temperature", "未知")
            return f"{city} 的天气: {weather}, {temp}°C"
        return f"{city} 的天气: 未知"
    except Exception as e:
        return f"获取 {city} 天气时出错: {str(e)}"

def get_weather_info_sync(city: str) -> str:
    return asyncio.run(get_weather_info(city))
