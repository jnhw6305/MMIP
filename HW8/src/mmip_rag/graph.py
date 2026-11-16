import json
from collections import defaultdict, deque
from pathlib import Path

from .rag import tokens


class GraphRAG:
    ALIASES = {"編碼器": "Encoder", "解碼器": "Decoder", "轉換器": "Transformer", "多頭注意力": "Multi-Head Attention", "縮放點積注意力": "Scaled Dot-Product Attention", "前饋網路": "Feed-Forward Network", "前饋": "Feed-Forward Network", "殘差連接": "Residual Connection", "英德": "WMT14 EN-DE", "英法": "WMT14 EN-FR", "優化器": "Adam", "最佳化器": "Adam", "optimizer": "Adam"}
    INTENTS = {
        "contains": ("包含", "組成", "元件", "組件", "架構", "堆疊", "位於"),
        "uses": ("使用", "機制", "注意力", "避免", "未來", "順序", "表示"),
        "trained_with": ("訓練", "optimizer", "optimiser", "優化器", "最佳化器"),
        "uses_schedule": ("warmup", "排程", "schedule"),
        "regularized_by": ("正則", "regular", "dropout", "smoothing"),
        "evaluated_on": ("資料集", "benchmark", "評估於", "翻譯"),
        "achieves": ("bleu", "分數", "表現", "成績", "結果"),
        "has": ("多少", "幾個", "維度", "heads", "head", "d_model", "d_ff"),
        "implements": ("實作", "implement", "原理"),
        "attends_to": ("關注", "注意", "attend"),
        "followed_by": ("後接", "接著", "follow"),
    }

    def __init__(self, data):
        self.nodes = dict(data["nodes"])
        self.edges = data["edges"]
        self.adj = defaultdict(list)
        for a, relation, b in self.edges:
            self.adj[a].append((relation, b))
            self.adj[b].append(("inverse_" + relation, a))
        self.history = []

    def entities(self, question):
        q = question.lower()
        found = []
        for alias, node in self.ALIASES.items():
            if alias.lower() in q and node not in found:
                found.append(node)
        for node in self.nodes:
            if node.lower() in q and node not in found:
                found.append(node)
        return found

    def seeds(self, question):
        explicit = self.entities(question)
        if explicit:
            return explicit[:3]
        q_tokens = set(tokens(question))
        scored = []
        for node in self.nodes:
            score = len(set(tokens(node)) & q_tokens)
            if score:
                scored.append((score, node))
        found = [node for _, node in sorted(scored, reverse=True)[:3]]
        # Generic model-level questions refer to the paper's central Transformer node.
        if not found and self.relations(question):
            return ["Transformer"]
        return found

    def relations(self, question):
        q = question.lower()
        return {relation for relation, words in self.INTENTS.items() if any(word in q for word in words)}

    def shortest_path(self, start, goal, max_hops=4):
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for relation, target in self.adj[node]:
                next_path = path + [(node, relation, target)]
                if target == goal:
                    return next_path
                if target not in seen:
                    seen.add(target)
                    queue.append((target, next_path))
        return []

    def query(self, question, hops=2):
        seeds = self.seeds(question)
        q = question.lower()
        paths = []
        if len(seeds) >= 2 and any(word in q for word in ("如何連", "關係", "路徑", "connect")):
            paths = self.shortest_path(seeds[0], seeds[1], max_hops=max(4, hops))
        elif seeds:
            wanted = self.relations(question)
            for a, relation, b in self.edges:
                if relation in wanted and (a in seeds or b in seeds):
                    paths.append((a, relation, b))
            reached = {b for _, _, b in paths}
            for a, relation, b in self.edges:
                if a in reached and relation in wanted and (a, relation, b) not in paths:
                    paths.append((a, relation, b))
        answer = "；".join(f"{a} --{relation}--> {b}" for a, relation, b in paths)
        if not answer:
            answer = "知識圖譜中沒有足夠資訊回答此問題。"
        self.history.append((question, answer))
        return {"answer": answer, "seeds": seeds, "paths": paths}


def load_graph(root):
    data = json.loads((Path(root) / "data/paper_graph.json").read_text(encoding="utf-8"))
    return GraphRAG(data)
