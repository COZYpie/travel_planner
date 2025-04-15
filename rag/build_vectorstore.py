import shutil
import os
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.vectorstore_loader import embedding
import time

PERSIST_DIRECTORY = "C:\\Users\\13067\\PycharmProjects\\travel_planner\\vectordb"

# 清空现有数据库
if os.path.exists(PERSIST_DIRECTORY):
    shutil.rmtree(PERSIST_DIRECTORY)
    print(f"已清空现有数据库目录: {PERSIST_DIRECTORY}")

os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# 地点列表
places = [
    {"name": "上海博物馆", "city": "上海", "description": "人民广场的文化地标，展示中国古代青铜器、陶瓷、书法等珍品，免费参观。"},
    {"name": "外滩", "city": "上海", "description": "黄浦江边历史建筑群，对岸是陆家嘴摩天大楼，夜景令人叹为观止。"},
    {"name": "豫园", "city": "上海", "description": "明代园林，假山、池塘、古建筑精致，城隍庙小吃街热闹非凡。"},
    {"name": "东方明珠", "city": "上海", "description": "浦东标志性电视塔，观景台俯瞰全城，内有历史博物馆。"},
    {"name": "田子坊", "city": "上海", "description": "老弄堂改造的艺术区，汇集创意店铺、咖啡馆，充满文艺气息。"},
    {"name": "故宫", "city": "北京", "description": "紫禁城，世界最大宫殿群，收藏明清文物，需提前预约。"},
    {"name": "天坛", "city": "北京", "description": "明清祭天圣地，祈年殿蓝顶壮观，圜丘坛设计独特。"},
    {"name": "颐和园", "city": "北京", "description": "清代皇家园林，昆明湖与十七孔桥构成湖山美景。"},
    {"name": "长城", "city": "北京", "description": "世界奇迹，八达岭段最著名，登山俯瞰雄伟景观。"},
    {"name": "西湖", "city": "杭州", "description": "人间天堂，断桥、苏堤、三潭印月等十景闻名，四季皆美。"},
    {"name": "雷峰塔", "city": "杭州", "description": "西湖边的历史名塔，白蛇传故事发源地，夜景灯光璀璨。"},
    {"name": "灵隐寺", "city": "杭州", "description": "佛教名刹，飞来峰石刻与古树环绕，禅意盎然。"},
    {"name": "乌镇", "city": "嘉兴", "description": "江南水乡，石桥、古宅、运河夜游，保留千年风貌。"},
    {"name": "南浔古镇", "city": "湖州", "description": "低调的水乡，中西合璧建筑，历史文化底蕴深厚。"}
]

# 增强备用描述
fallback_data = {
    "上海博物馆": "位于人民广场，展出青铜器、陶瓷、书法等中国古代艺术珍品，免费开放，适合文化爱好者。馆内常设展览包括青铜馆、陶瓷馆和书法绘画馆，是了解中国历史的好去处。",
    "外滩": "上海的象征，沿黄浦江排列历史建筑，夜晚灯光与浦东 skyline 交相辉映，游客必访。步行可欣赏万国建筑博览群，感受上海百年历史变迁。",
    "豫园": "明代私家园林，亭台楼阁与假山池塘布局精妙，周边城隍庙提供上海特色小吃，如南翔小笼包和梨膏糖，适合半日游。",
    "东方明珠": "上海最高地标，旋转餐厅和观景台提供全景体验，底部有城市历史展览。夜晚的灯光秀吸引众多摄影爱好者。",
    "田子坊": "石库门弄堂改造成的创意园区，汇集手工艺品店、画廊和特色餐厅，适合漫步。夜晚的酒吧和咖啡馆充满国际氛围。",
    "故宫": "明清皇宫，收藏百万文物，午门、太和殿、御花园展现皇家气派，需提前购票。冬季雪景尤为壮观，吸引历史爱好者。",
    "天坛": "皇帝祭天的场所，祈年殿和圜丘坛建筑精巧，象征天人合一，文化意义深远。园区安静，适合了解中国古代礼制。",
    "颐和园": "清代避暑胜地，长廊彩绘、昆明湖游船、佛香阁远眺，风景如画。适合全天游览，体验皇家园林的恢弘。",
    "长城": "中国象征，八达岭长城交通便利，城墙蜿蜒，登高可感历史厚重。建议穿舒适鞋，春秋季游览最佳。",
    "西湖": "杭州名片，十景如平湖秋月、花港观鱼各有韵味，泛舟湖上别有情趣。环湖步道适合骑行或散步，免费开放。",
    "雷峰塔": "西湖边重建古塔，与白娘子传说相关，顶层可俯瞰西湖全景，夜景迷人。内部电梯方便游客登塔。",
    "灵隐寺": "杭州佛教胜地，飞来峰石刻群与古刹氛围宁静，适合祈福与游览。周边有北高峰可登山，俯瞰杭州。",
    "乌镇": "水乡古镇，青石板路、木结构民居、夜游运河，展现江南生活方式。东栅和西栅各有特色，适合深度游。",
    "南浔古镇": "湖州的水乡名镇，小莲庄等私家园林与古桥老街，文化气息浓厚。游客较少，适合安静的古镇体验。"
}

# 维基百科抓取
def fetch_wikipedia_data(place_name, retries=3):
    for attempt in range(retries):
        try:
            search_url = f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={place_name}&format=json"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
            response = requests.get(search_url, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data["query"]["search"]:
                title = data["query"]["search"][0]["title"]
                content_url = f"https://zh.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&explaintext=&titles={title}&format=json"
                response = requests.get(content_url, headers=headers, timeout=20)
                response.raise_for_status()
                pages = response.json()["query"]["pages"]
                for page_id in pages:
                    if "extract" in pages[page_id]:
                        return pages[page_id]["extract"][:2500]
            print(f"未找到 {place_name} 的维基百科内容")
            return fallback_data.get(place_name)
        except Exception as e:
            print(f"获取 {place_name} 维基百科数据时出错 (尝试 {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(3)
    return fallback_data.get(place_name)

# 百度抓取
def fetch_search_engine_data(place_name):
    try:
        search_url = f"https://www.baidu.com/s?wd={place_name}+旅游+信息"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9"
        }
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        snippets = []
        for result in soup.select("div[class*='c-container'], div[class*='result']")[:6]:
            snippet = result.select_one("div, span, p[class*='content'], div[class*='abstract']")
            if snippet:
                snippets.append(snippet.get_text(strip=True)[:1000])
        return " | ".join(snippets) if snippets else None
    except Exception as e:
        print(f"百度抓取 {place_name} 数据时出错: {str(e)}")
        return None

# 去哪儿抓取
def fetch_travel_site_data(place_name):
    try:
        search_url = f"https://piao.qunar.com/ticket/list.htm?keyword={place_name}&region=&from=mpshouye_hotcity&sort=pp"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
            "Accept": "text/html,application/xhtml+xml"
        }
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        snippets = []
        for item in soup.select("div[class*='sight_item'], li[class*='item']")[:4]:
            desc = item.select_one("div[class*='intro'], p[class*='desc'], div[class*='content']")
            if desc:
                snippets.append(desc.get_text(strip=True)[:800])
        return " | ".join(snippets) if snippets else None
    except Exception as e:
        print(f"去哪儿抓取 {place_name} 数据时出错: {str(e)}")
        return None

# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = []
for place in places:
    wiki_content = fetch_wikipedia_data(place["name"])
    search_content = fetch_search_engine_data(place["name"])
    travel_content = fetch_travel_site_data(place["name"])

    text_parts = [f"{place['name']} - {place['description']}"]
    if wiki_content:
        text_parts.append(f"维基百科: {wiki_content}")
    if search_content:
        text_parts.append(f"百度: {search_content}")
    if travel_content:
        text_parts.append(f"去哪儿: {travel_content}")
    text = " | ".join(text_parts)

    chunks = text_splitter.split_text(text)
    print(f"分块 ({place['name']}): {len(chunks)} 块")
    for chunk in chunks:
        docs.append(Document(
            page_content=chunk,
            metadata={
                "name": place["name"],
                "city": place["city"],
                "source": "combined",
                "wiki_included": bool(wiki_content),
                "search_included": bool(search_content),
                "travel_included": bool(travel_content)
            }
        ))

# 放宽去重
unique_docs = {}
for doc in docs:
    key = (doc.page_content[:200], doc.metadata["name"])
    if key not in unique_docs:
        unique_docs[key] = doc
docs = list(unique_docs.values())

print("文档数量:", len(docs))
for doc in docs:
    print("文档内容:", doc.page_content[:100], "...", "元数据:", doc.metadata)

# 构建数据库
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=PERSIST_DIRECTORY,
    collection_name="places"
)

print("数据库中文档数量:", vectorstore._collection.count())

# 测试查询
queries = ["上海博物馆", "北京景点", "西湖", "杭州景点"]
for query in queries:
    results = vectorstore.similarity_search(query, k=10)
    print(f"\n查询 '{query}' 结果数量:", len(results))
    for i, res in enumerate(results):
        print(f"结果 {i + 1}:", res.page_content[:100], "...", res.metadata)

print("✅ 向量数据库构建完成")