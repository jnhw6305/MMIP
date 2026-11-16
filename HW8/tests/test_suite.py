import json, tempfile, unittest
from pathlib import Path
from src.mmip_rag.rag import embed, similarity, load_rag
from src.mmip_rag.graph import load_graph
from src.mmip_rag.evaluate import evaluate_rag, evaluate_graph, evaluate_video

ROOT=Path(__file__).resolve().parents[1]
class TestSuite(unittest.TestCase):
    def test_embedding(self): self.assertGreater(similarity(embed("注意力"),embed("注意力機制")),0)
    def test_rag_reference(self): self.assertIn("[D",load_rag(ROOT).answer("head 有幾個？")["answer"])
    def test_rag_refuses_out_of_domain(self):
        rag=load_rag(ROOT)
        for question in ["今天天氣好嗎？","Transformer 能預測明天股價嗎？","論文作者住在哪個城市？","請說一個笑話"]:
            result=rag.answer(question)
            self.assertIn("語料不足",result["answer"])
            self.assertEqual(result["references"],[])
    def test_video_refuses_out_of_domain(self):
        from src.mmip_rag.video import load_video
        result=load_video(ROOT).answer("今天天氣好嗎？")
        self.assertIn("語料不足",result["answer"])
        self.assertEqual(result["references"],[])
    def test_video_recorded_sidecars_work_without_mp4(self):
        from src.mmip_rag.video import ingest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); side=root/"video_corpus/sidecars"; side.mkdir(parents=True)
            for name in ["silent_operation.json","spoken_transformer.json"]:
                (side/name).write_text((ROOT/"video_corpus/sidecars"/name).read_text(encoding="utf-8"),encoding="utf-8")
            docs=ingest(root)
            self.assertEqual(len(docs),14)
            self.assertEqual({doc["modality"] for doc in docs},{"speech","visual"})
    def test_multiturn(self):
        r=load_rag(ROOT); r.answer("什麼是 Transformer？")
        self.assertIn("Transformer",r.rewrite("它有幾層？"))
        self.assertIn("六層",r.answer("它有幾層？")["answer"])
    def test_graph(self):
        graph=load_graph(ROOT)
        components=graph.query("Transformer 包含哪些組件？")
        connection=graph.query("Decoder 如何連到 Scaled Dot-Product Attention？")
        self.assertTrue(components["paths"])
        self.assertNotEqual(components["answer"],connection["answer"])
        self.assertTrue(all(relation=="contains" for _,relation,_ in components["paths"]))
        self.assertEqual(connection["paths"][-1][2],"Scaled Dot-Product Attention")
    def test_graph_non_demo_and_unknown(self):
        graph=load_graph(ROOT)
        self.assertIn("Adam",graph.query("模型使用什麼優化器？")["answer"])
        self.assertIn("BLEU 28.4",graph.query("英德翻譯 BLEU 分數是多少？")["answer"])
        self.assertFalse(graph.query("論文作者住在哪個城市？")["paths"])
    def test_dataset_sizes(self):
        self.assertGreaterEqual(len(json.loads((ROOT/"data/rag_qa.json").read_text(encoding="utf-8"))),20)
        self.assertGreaterEqual(len(json.loads((ROOT/"data/graph_benchmark.json").read_text(encoding="utf-8"))),20)
    def test_all_evaluators(self):
        self.assertEqual(len(evaluate_rag(ROOT)),25)
        self.assertEqual(len(evaluate_graph(ROOT)),22)
        self.assertEqual(len(evaluate_video(ROOT)),4)
if __name__=="__main__": unittest.main()

