import reportlab
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import CMYKColor

def gerar_macbeth_chart_cmyk(nome_arquivo="macbeth_chart_cmyk.pdf"):
    # Tabela com os 24 patches do Macbeth ColorChecker em proporções CMYK (0.0 a 1.0)
    macbeth_patches = [
        # Linha 1
        {"id": 1,  "name": "Dark Skin",     "cmyk": (0.38, 0.56, 0.55, 0.29)},
        {"id": 2,  "name": "Light Skin",    "cmyk": (0.16, 0.35, 0.40, 0.02)},
        {"id": 3,  "name": "Blue Sky",      "cmyk": (0.56, 0.30, 0.15, 0.12)},
        {"id": 4,  "name": "Foliage",       "cmyk": (0.56, 0.42, 0.77, 0.25)},
        {"id": 5,  "name": "Blue Flower",   "cmyk": (0.44, 0.44, 0.10, 0.05)},
        {"id": 6,  "name": "Bluish Green",  "cmyk": (0.76, 0.16, 0.38, 0.05)},
        # Linha 2
        {"id": 7,  "name": "Orange",        "cmyk": (0.00, 0.58, 0.88, 0.02)},
        {"id": 8,  "name": "Purplish Blue", "cmyk": (0.75, 0.58, 0.10, 0.08)},
        {"id": 9,  "name": "Moderate Red",  "cmyk": (0.20, 0.78, 0.55, 0.12)},
        {"id": 10, "name": "Purple",        "cmyk": (0.68, 0.88, 0.20, 0.20)},
        {"id": 11, "name": "Yellow Green",  "cmyk": (0.38, 0.08, 0.88, 0.02)},
        {"id": 12, "name": "Orange Yellow", "cmyk": (0.00, 0.38, 0.95, 0.00)},
        # Linha 3
        {"id": 13, "name": "Blue",          "cmyk": (0.92, 0.75, 0.10, 0.10)},
        {"id": 14, "name": "Green",         "cmyk": (0.78, 0.22, 0.88, 0.15)},
        {"id": 15, "name": "Red",           "cmyk": (0.12, 0.95, 0.88, 0.08)},
        {"id": 16, "name": "Yellow",        "cmyk": (0.05, 0.18, 0.95, 0.00)},
        {"id": 17, "name": "Magenta",       "cmyk": (0.28, 0.88, 0.18, 0.05)},
        {"id": 18, "name": "Cyan",          "cmyk": (0.88, 0.28, 0.18, 0.05)},
        # Linha 4 (Escala de Cinzas / Neutros)
        {"id": 19, "name": "White 0.05",    "cmyk": (0.05, 0.04, 0.04, 0.00)},
        {"id": 20, "name": "Neutral 8",     "cmyk": (0.18, 0.14, 0.14, 0.05)},
        {"id": 21, "name": "Neutral 6.5",   "cmyk": (0.32, 0.26, 0.26, 0.12)},
        {"id": 22, "name": "Neutral 5",     "cmyk": (0.48, 0.40, 0.40, 0.25)},
        {"id": 23, "name": "Neutral 3.5",   "cmyk": (0.62, 0.54, 0.54, 0.48)},
        {"id": 24, "name": "Black 2",       "cmyk": (0.75, 0.68, 0.65, 0.85)},
    ]

    # Criar canvas em orientação Paisagem (A4)
    c = canvas.Canvas(nome_arquivo, pagesize=landscape(A4))
    largura, altura = landscape(A4)

    # Fundo escuro (moldura de calibração)
    c.setFillColor(CMYKColor(0, 0, 0, 0.02))  # K=85%
    c.rect(0, 0, largura, altura, fill=1, stroke=0)

    # Dimensões da grade
    cols, rows = 6, 4
    patch_w, patch_h = 80, 80
    gap = 12
    chart_w = cols * patch_w + (cols - 1) * gap
    chart_h = rows * patch_h + (rows - 1) * gap

    # Centralização no papel
    start_x = (largura - chart_w) / 2
    start_y = (altura - chart_h) / 2 + 15

    # Placa base clara/branca
    border_margin = 20
    c.setFillColor(CMYKColor(0.75, 0.68, 0.65, 0.85))
    c.rect(start_x - border_margin, start_y - border_margin, 
           chart_w + 2*border_margin, chart_h + 2*border_margin, fill=1, stroke=0)

    # Cabeçalho
    c.setFillColor(CMYKColor(0, 0, 0, 1.0)) # Preto puro (K=100%)
    c.setFont("Helvetica-Bold", 16)
    # c.drawCentredString(largura / 2, altura - 35, "MACBETH COLORCHECKER (CMYK - CALIBRATION TARGET)")

    # Desenho dos 24 patches
    for idx, patch in enumerate(macbeth_patches):
        r = idx // cols
        col = idx % cols

        x = start_x + col * (patch_w + gap)
        y = start_y + (rows - 1 - r) * (patch_h + gap)

        c_val, m_val, y_val, k_val = patch["cmyk"]

        # Define a cor nativa em CMYK
        c.setFillColor(CMYKColor(c_val, m_val, y_val, k_val))
        c.setStrokeColor(CMYKColor(0, 0, 0, 0.2))
        c.rect(x, y, patch_w, patch_h, fill=1, stroke=1)

        # Rótulo de identificação e valores numéricos CMYK
        c.setFillColor(CMYKColor(0, 0, 0, 1.0))
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(x + patch_w/2, y + 5, f"{patch['id']}. {patch['name']}")
        c.setFont("Helvetica", 6)
        c.drawCentredString(x + patch_w/2, y - 8, f"C{int(c_val*100)} M{int(m_val*100)} Y{int(y_val*100)} K{int(k_val*100)}")

    c.showPage()
    c.save()
    print(f"Arquivo '{nome_arquivo}' gerado com sucesso!")

if __name__ == "__main__":
    gerar_macbeth_chart_cmyk()