from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/pdf/transformer_spoken_video_screen.pdf"
OUT.parent.mkdir(parents=True,exist_ok=True)
W,H=960,540

font_path=Path(r"C:\Windows\Fonts\msjh.ttc")
if font_path.exists():
    pdfmetrics.registerFont(TTFont("ZH",str(font_path),subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ZH-B",str(Path(r"C:\Windows\Fonts\msjhbd.ttc")),subfontIndex=0))
else:
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    FONT=FONT_B="STSong-Light"
FONT=locals().get("FONT","ZH"); FONT_B=locals().get("FONT_B","ZH-B")

c=canvas.Canvas(str(OUT),pagesize=(W,H))
navy=HexColor("#0B1F3A"); blue=HexColor("#2563EB"); cyan=HexColor("#22D3EE")
panel=HexColor("#132B4F"); muted=HexColor("#B7C7DD"); yellow=HexColor("#FBBF24")
c.setFillColor(navy); c.rect(0,0,W,H,fill=1,stroke=0)
c.setFillColor(blue); c.rect(0,H-8,W,8,fill=1,stroke=0)

c.setFont(FONT_B,30); c.setFillColor(white); c.drawString(42,482,"Transformer：Attention Is All You Need")

def round_box(x,y,w,h,title,lines,accent=cyan):
    c.setFillColor(panel); c.roundRect(x,y,w,h,14,fill=1,stroke=0)
    c.setFillColor(accent); c.roundRect(x+18,y+h-42,112,26,7,fill=1,stroke=0)
    c.setFillColor(navy); c.setFont(FONT_B,14); c.drawCentredString(x+74,y+h-34,title)
    c.setFillColor(white); c.setFont(FONT,15)
    ty=y+h-70
    for line in lines:
        c.drawString(x+20,ty,line); ty-=27

round_box(40,268,280,160,"核心概念",[
    "• 2017 年由 Vaswani 等人提出",
    "• 以 Attention 取代循環與卷積",
    "• 可平行處理序列，提升訓練效率",
])
round_box(340,268,280,160,"Q / K / V",[
    "Query：目前要尋找的資訊",
    "Key：用來比對的特徵",
    "Value：依權重彙整的內容",
],yellow)
round_box(640,268,280,160,"多頭與位置",[
    "Multi-Head：學習不同關係與子空間",
    "Positional Encoding：補充詞元順序",
    "基礎模型：8 heads，d_k = 64",
])

c.setFillColor(HexColor("#F8FAFC")); c.roundRect(40,112,880,126,16,fill=1,stroke=0)
c.setFillColor(navy); c.setFont(FONT_B,16); c.drawString(62,205,"Scaled Dot-Product Attention")
c.setFont("Helvetica-Bold",25); c.setFillColor(blue)
c.drawCentredString(480,160,"Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V")
c.setFont(FONT,13); c.setFillColor(HexColor("#475569"))
c.drawCentredString(480,132,"除以 sqrt(d_k)：避免內積隨維度變大，使 softmax 過度集中，讓訓練更穩定")

c.setFillColor(cyan); c.circle(52,64,5,fill=1,stroke=0)
c.setFont(FONT_B,15); c.setFillColor(white); c.drawString(66,59,"一句話總結")
c.setFont(FONT,14); c.setFillColor(muted); c.drawString(176,59,"Transformer 用注意力理解詞元關係，用多頭捕捉多種模式，用位置編碼保留順序。")
c.setFont("Helvetica",9); c.setFillColor(HexColor("#7890AE")); c.drawRightString(918,24,"MMIP HW8 • Spoken Video Corpus")
c.showPage(); c.save()
print(OUT)
