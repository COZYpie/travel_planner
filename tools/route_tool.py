import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.attraction_tool import recommend_attractions

async def get_route_info(query: str) -> str:
    client = MultiServerMCPClient()
    try:
        # 提取城市
        city = query.split()[0]
        # 获取推荐景点
        attractions = recommend_attractions(city)
        if not attractions or "暂无" in attractions.lower():
            return f"{city} 没有足够的景点推荐，无法规划路线"
        # 解析景点：提取地名
        attraction_list = []
        for line in attractions.split("。"):
            if "：" in line:
                name = line.split("：")[0].strip()
                if name:
                    attraction_list.append(name)
        if len(attraction_list) < 2:
            return f"{city} 景点不足（仅找到 {attraction_list}），无法规划路线"
        start, end = attraction_list[:2]
        # 获取工具
        tools = client.get_tools()
        print("路线工具列表（原始）:", tools)
        poi_tool = None
        route_tool = None
        for tool in tools:
            try:
                if isinstance(tool, dict):
                    name = tool.get("name", "").lower()
                    if name in ["search_poi", "searchpoi"]:
                        poi_tool = tool.get("func")
                    elif name in ["route_plan", "routeplan"]:
                        route_tool = tool.get("func")
                elif hasattr(tool, "name"):
                    name = tool.name.lower()
                    if name in ["search_poi", "searchpoi"]:
                        poi_tool = tool
                    elif name in ["route_plan", "routeplan"]:
                        route_tool = tool
                else:
                    print("未知工具格式:", tool)
            except Exception as e:
                print(f"解析工具错误: {str(e)}")
        if not poi_tool or not route_tool:
            return f"未找到 POI 或路线规划工具，无法规划从 {start} 到 {end} 的路线。确保 MCP 服务器运行在 http://localhost:8000。"
        # 调用工具
        start_poi = await poi_tool(start, city=city)
        end_poi = await poi_tool(end, city=city)
        route_data = await route_tool(start_poi[0] if isinstance(start_poi, list) else start_poi,
                                    end_poi[0] if isinstance(end_poi, list) else end_poi)
        distance = route_data.get("distance", "未知")
        steps = route_data.get("paths", [{}])[0].get("instruction", "步行前往")
        return f"从 {start} 到 {end} 的路线: 距离 {distance}米，{steps}"
    except Exception as e:
        return f"规划 {query} 路线时出错: {str(e)}"

def plan_routes(query: str) -> str:
    return asyncio.run(get_route_info(query))
