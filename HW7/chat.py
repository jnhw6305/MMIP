"""Multi-turn state management and user-facing response generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from router import REQUIRED, RouteResult, route
from services import dispatch


@dataclass
class Session:
    history: list[dict[str, Any]] = field(default_factory=list)
    topic: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)

    def reply(self, text: str, image_name: str = "", method: str = "constraint") -> dict[str, Any]:
        result = route(text, image_name, method)
        result.entities = {**self.entities, **result.entities}
        result.missing_fields = [f for f in REQUIRED.get(result.route, []) if f not in result.entities]
        result.needs_clarification = bool(result.missing_fields) or result.route == "other"
        topic_changed = bool(self.topic and result.route not in {self.topic, "other", "security"})
        if result.safety["blocked"]:
            message = "偵測到可能的提示注入或敏感資訊要求。為保護系統與個資，我不會執行該指令；若是正常客服需求，請只描述商品或訂單問題。"
        elif result.route == "other":
            message = "這個問題目前與 3C 購物客服的脈絡不一致。你想詢問商品、訂單、退貨、維修，還是付款／發票？"
        elif result.missing_fields:
            labels = {"order_id": "訂單編號", "reason": "退貨原因", "product": "商品名稱或照片"}
            message = "我已理解你的需求。為了正確分流，還需要：" + "、".join(labels.get(x, x) for x in result.missing_fields) + "。"
        else:
            message = f"已確認為「{result.intent}」，我會轉交 {result.route} 處理佇列。"
        if topic_changed:
            message = "看起來你開始了新的問題；我已保留上一個案件並切換主題。" + message
        self.topic = result.route if result.route not in {"other", "security"} else self.topic
        self.entities = result.entities
        self.history.extend([{"role":"user", "text":text, "image":image_name}, {"role":"assistant", "text":message, "route":result.to_dict()}])
        dispatched=None
        if not result.needs_clarification:
            dispatched=dispatch(result.route,{"entities":result.entities,"intent":result.intent}).__dict__
        return {"message": message, "routing": result.to_dict(), "dispatch":dispatched,"topic_changed": topic_changed,
                "reading_delay_ms": min(2400, 450 + len(message) * 18)}
