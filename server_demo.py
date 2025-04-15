# from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP
import httpx

app = FastMCP('gaode')

# 高德API配置
AMAP_KEY = "d9aaf03856e11f50e121a504a55f6efd"
AMAP_BASE_URL = "https://restapi.amap.com/v3"


@app.tool()
async def search_poi(query: str, city: str = "") -> list:
    """
    地点搜索API（POI搜索）

    Args:
        query: 要搜索的关键词 (例如 "咖啡馆")
        city: 城市名称 (例如 "北京")

    Returns:
        POI搜索结果列表
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AMAP_BASE_URL}/place/text",
            params={
                "key": AMAP_KEY,
                "keywords": query,
                "city": city,
                "output": "json"
            }
        )
        return response.json()["pois"]


@app.tool()
async def weather_forecast(city: str) -> dict:
    """
    天气情况 API

    Args:
        city: 城市名称

    Returns:
        当地天气情况
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{AMAP_BASE_URL}/weather/weatherInfo",
            params={
                "key": AMAP_KEY,
                "city": city,
                "output": "json"
            }
        )
        return response.json()["lives"]


if __name__ == "__main__":
    app.run(transport="sse")