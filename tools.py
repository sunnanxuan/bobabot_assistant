# -*- coding: utf-8 -*-
"""
tools.py
—— BobaBot 的 LangChain 工具集合：
- quote_price：计算报价（含小料与冷热校验）
- list_menu：列出全菜单或某一类别
- can_make_hot：查询是否可做热饮
- list_extras：列出所有可加小料与价格
"""

from typing import List, Optional, Literal, Dict, Any
from langchain_core.tools import tool

from menu_config import PRICES, CATEGORY, EXTRAS, hot_allowed

# 枚举
Size = Literal["小杯", "中杯", "大杯"]
Sugar = Literal["全糖", "半糖", "少糖", "无糖"]
Ice = Literal["正常冰", "少冰", "去冰", "热饮"]  # “热饮”会转换为 hot=True

def _normalize_hot_ice(drink: str, hot: Optional[bool], ice: Optional[Ice]) -> Dict[str, Any]:
    """将 hot/ice 归一化：若出现热饮，则 hot=True 且 ice='热饮'；并校验是否允许热饮。"""
    is_hot = bool(hot) or (ice == "热饮")
    ice_final: Optional[Ice] = "热饮" if is_hot else (ice if ice in ("正常冰", "少冰", "去冰") else None)

    if is_hot and not hot_allowed(drink):
        return {"ok": False, "error": f"{drink} 为果茶类，只能做冷饮"}

    return {"ok": True, "hot": is_hot, "ice": ice_final}

@tool
def quote_price(
    drink: str,
    size: Size,
    extras: Optional[List[str]] = None,
    hot: Optional[bool] = None,
    sugar: Optional[Sugar] = None,
    ice: Optional[Ice] = None,
) -> dict:
    """计算某杯饮品的价格，校验冷热可做与小料合法性，并返回明细。"""
    if drink not in PRICES:
        return {"ok": False, "error": f"未找到饮品：{drink}"}
    if size not in PRICES[drink]:
        return {"ok": False, "error": f"规格错误：{size}"}

    # 统一冷热逻辑 + 校验
    norm = _normalize_hot_ice(drink, hot, ice)
    if not norm["ok"]:
        return norm
    is_hot = norm["hot"]
    ice_final = norm["ice"]

    base = PRICES[drink][size]
    total = base
    ext_list = []

    for e in (extras or []):
        if e not in EXTRAS:
            return {"ok": False, "error": f"未知小料：{e}"}
        price = EXTRAS[e]
        ext_list.append({"name": e, "price": price})
        total += price

    return {
        "ok": True,
        "drink": drink,
        "category": CATEGORY.get(drink),
        "size": size,
        "hot_allowed": hot_allowed(drink),
        "hot": is_hot,                # 最终冷热
        "sugar": sugar,
        "ice": ice_final,             # 若为热饮，这里固定为 '热饮'
        "base_price": base,
        "extras": ext_list,
        "total": total,
    }

@tool
def list_menu(category: Optional[Literal["奶茶类", "纯茶类", "茶拿铁类", "果茶类"]] = None) -> dict:
    """列出全菜单或某一类别的饮品及定价。"""
    items = []
    for d, sizes in PRICES.items():
        if category and CATEGORY.get(d) != category:
            continue
        row = {"name": d, "category": CATEGORY.get(d)}
        row.update(sizes)  # 展平规格价格
        items.append(row)
    # 可按类别+名称排序，便于阅读
    items.sort(key=lambda x: (x["category"] or "", x["name"]))
    return {"ok": True, "items": items}

@tool
def can_make_hot(drink: str) -> dict:
    """查询某饮品是否可做热饮。"""
    if drink not in PRICES:
        return {"ok": False, "error": f"未找到饮品：{drink}"}
    return {"ok": True, "drink": drink, "hot_allowed": hot_allowed(drink)}

@tool
def list_extras() -> dict:
    """列出所有可加小料及价格，便于咨询/推荐。"""
    items = [{"name": k, "price": v} for k, v in EXTRAS.items()]
    # 可按价格再按名称排序
    items.sort(key=lambda x: (x["price"], x["name"]))
    return {"ok": True, "extras": items}

# 一次性绑定给模型
TOOLS = [quote_price, list_menu, can_make_hot, list_extras]
