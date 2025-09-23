# -*- coding: utf-8 -*-
"""
tools.py
—— BobaBot 的 LangChain 工具集合：
- quote_price：计算报价（含小料与冷热校验）
- list_menu：列出全菜单或某一类别
- can_make_hot：查询是否可做热饮
"""

from typing import List, Optional, Literal
from langchain_core.tools import tool
# -*- coding: utf-8 -*-
"""
tools.py
—— BobaBot 的 LangChain 工具集合：
- quote_price：计算报价（含小料与冷热校验）
- list_menu：列出全菜单或某一类别
- can_make_hot：查询是否可做热饮
"""

from typing import List, Optional, Literal
from langchain_core.tools import tool

from menu_config import PRICES, CATEGORY, EXTRAS, hot_allowed

# 规格枚举
Size = Literal["小杯", "中杯", "大杯"]


@tool
def quote_price(
    drink: str,
    size: Size,
    extras: Optional[List[str]] = None,
    hot: Optional[bool] = None,
    sugar: Optional[Literal["全糖", "半糖", "少糖", "无糖"]] = None,
    ice: Optional[Literal["正常冰", "少冰", "去冰", "热饮"]] = None,
) -> dict:
    """计算某杯饮品的价格，校验冷热可做与小料合法性，并返回明细。"""
    if drink not in PRICES:
        return {"ok": False, "error": f"未找到饮品：{drink}"}
    if size not in PRICES[drink]:
        return {"ok": False, "error": f"规格错误：{size}"}

    if hot is None and ice == "热饮":
        hot = True
    if hot and not hot_allowed(drink):
        return {"ok": False, "error": f"{drink} 为果茶类，只能做冷饮"}

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
        "hot": bool(hot),
        "sugar": sugar,
        "ice": ice,
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
        row.update(sizes)
        items.append(row)
    return {"ok": True, "items": items}


@tool
def can_make_hot(drink: str) -> dict:
    """查询某饮品是否可做热饮。"""
    if drink not in PRICES:
        return {"ok": False, "error": f"未找到饮品：{drink}"}
    return {"ok": True, "drink": drink, "hot_allowed": hot_allowed(drink)}


# 一次性绑定到模型
TOOLS = [quote_price, list_menu, can_make_hot]
