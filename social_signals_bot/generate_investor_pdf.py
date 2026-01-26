#!/usr/bin/env python3
"""
Clean, professional investor PDF - concise and well-designed.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

# Colors
BLACK = colors.HexColor('#111111')
DARK = colors.HexColor('#333333')
GRAY = colors.HexColor('#666666')
LIGHT = colors.HexColor('#999999')
ACCENT = colors.HexColor('#0066FF')
GREEN = colors.HexColor('#00AA55')
PURPLE = colors.HexColor('#7733FF')
BG = colors.HexColor('#FAFAFA')

def create_equity_chart():
    """Simple equity curve."""
    d = Drawing(460, 140)

    data = [
        [100, 103, 101, 106, 104, 109, 112, 110, 115, 118, 116, 121, 125, 123, 128],
        [100, 101, 102, 103, 102, 105, 106, 108, 109, 111, 112, 114, 116, 117, 119],
        [100, 102, 102, 105, 103, 107, 109, 108, 112, 114, 113, 117, 120, 118, 123],
    ]

    chart = HorizontalLineChart()
    chart.x = 40
    chart.y = 20
    chart.width = 400
    chart.height = 100
    chart.data = data

    chart.categoryAxis.categoryNames = [''] * 15
    chart.categoryAxis.labels.fontSize = 0
    chart.categoryAxis.strokeColor = colors.HexColor('#DDDDDD')

    chart.valueAxis.valueMin = 98
    chart.valueAxis.valueMax = 130
    chart.valueAxis.valueStep = 8
    chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.labels.fillColor = GRAY
    chart.valueAxis.strokeColor = colors.HexColor('#DDDDDD')
    chart.valueAxis.gridStrokeColor = colors.HexColor('#EEEEEE')
    chart.valueAxis.visibleGrid = True

    chart.lines[0].strokeColor = GREEN
    chart.lines[0].strokeWidth = 2
    chart.lines[1].strokeColor = ACCENT
    chart.lines[1].strokeWidth = 2
    chart.lines[2].strokeColor = PURPLE
    chart.lines[2].strokeWidth = 2

    d.add(chart)

    # Legend
    for i, (name, color) in enumerate([('Aggressive', GREEN), ('Conservative', ACCENT), ('Balanced', PURPLE)]):
        x = 120 + i * 120
        d.add(Line(x, 130, x + 20, 130, strokeColor=color, strokeWidth=2))
        d.add(String(x + 25, 127, name, fontSize=8, fillColor=GRAY))

    return d

def create_investor_pdf(filename="Quantum_Trading_Investor_Overview.pdf"):
    doc = SimpleDocTemplate(
        filename, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.6*inch, bottomMargin=0.6*inch
    )

    # Styles
    title = ParagraphStyle('title', fontSize=28, leading=32, textColor=BLACK, fontName='Helvetica-Bold')
    h1 = ParagraphStyle('h1', fontSize=16, leading=20, textColor=BLACK, fontName='Helvetica-Bold', spaceBefore=25, spaceAfter=10)
    h2 = ParagraphStyle('h2', fontSize=12, leading=15, textColor=DARK, fontName='Helvetica-Bold', spaceBefore=15, spaceAfter=6)
    body = ParagraphStyle('body', fontSize=10, leading=15, textColor=DARK, fontName='Helvetica', spaceAfter=8)
    small = ParagraphStyle('small', fontSize=9, leading=12, textColor=GRAY, fontName='Helvetica')
    accent_text = ParagraphStyle('accent', fontSize=10, leading=14, textColor=ACCENT, fontName='Helvetica-Bold')

    story = []

    # === COVER ===
    story.append(Spacer(1, 1.8*inch))
    story.append(Paragraph("Quantum Trading", title))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Algorithmic Crypto Trading Platform", ParagraphStyle('sub', fontSize=14, textColor=GRAY, fontName='Helvetica')))
    story.append(Spacer(1, 0.6*inch))

    # Key numbers
    nums = [['3', 'Strategies'], ['$30K', 'Paper Capital'], ['50+', 'Signal Sources'], ['24/7', 'Automated']]
    num_table = Table(nums, colWidths=[1.5*inch]*4)
    num_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 20),
        ('TEXTCOLOR', (0, 0), (-1, 0), ACCENT),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('TEXTCOLOR', (0, 1), (-1, 1), GRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(num_table)

    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(f"Confidential  |  {datetime.now().strftime('%B %Y')}", small))
    story.append(PageBreak())

    # === THE CONCEPT ===
    story.append(Paragraph("The Concept", h1))
    story.append(Paragraph(
        "We run <b>three independent trading strategies</b> simultaneously, each with its own $10K portfolio. "
        "This is how top quant funds like Renaissance and Two Sigma operate internally — test multiple "
        "approaches, let them compete, then allocate capital to proven winners.",
        body
    ))
    story.append(Paragraph(
        "Instead of guessing which strategy will work, we let real market performance decide.",
        accent_text
    ))

    # === HOW IT WORKS ===
    story.append(Paragraph("How It Works", h1))

    story.append(Paragraph("1. Signal Collection", h2))
    story.append(Paragraph(
        "We monitor 50+ verified crypto traders on Twitter/X via API. These are traders with real track records — "
        "not influencers. When they post trading calls (\"BTC looks ready to break out\", \"shorting ETH here\"), "
        "we capture and process those signals in real-time.",
        body
    ))

    story.append(Paragraph("2. Trader Rating System", h2))
    story.append(Paragraph(
        "Each trader has a dynamic accuracy rating that updates based on their prediction outcomes. "
        "Good calls increase their rating; bad calls decrease it. Higher-rated traders carry more weight "
        "in our consensus calculations. Think of it like Elo ratings in chess.",
        body
    ))

    story.append(Paragraph("3. Consensus Engine", h2))
    story.append(Paragraph(
        "When multiple high-rated traders align on the same asset and direction, we have a consensus signal. "
        "The strength depends on: how many traders agree, their individual ratings, and their conviction level. "
        "Weak or conflicting signals are filtered out.",
        body
    ))

    story.append(Paragraph("4. Strategy Execution", h2))
    story.append(Paragraph(
        "Each of our three strategies evaluates consensus signals through its own lens and risk parameters. "
        "A signal that triggers the Aggressive strategy might not meet the Conservative strategy's higher bar. "
        "Each strategy manages its own portfolio independently.",
        body
    ))

    story.append(PageBreak())

    # === THE THREE STRATEGIES ===
    story.append(Paragraph("The Three Strategies", h1))

    strategy_data = [
        ['', 'Aggressive', 'Balanced', 'Conservative'],
        ['Minimum Signals', '3 traders', '5 traders', '7 traders'],
        ['Agreement Needed', '55%', '60%', '70%'],
        ['Position Size', '3-5%', '2-4%', '2-3%'],
        ['Risk/Reward', '1.2:1', '1.5:1', '2:1'],
        ['Style', 'More trades,\nhigher variance', 'Middle ground', 'Fewer trades,\nlower variance'],
    ]

    strat_table = Table(strategy_data, colWidths=[1.3*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    strat_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (1, 0), GREEN),
        ('BACKGROUND', (2, 0), (2, 0), PURPLE),
        ('BACKGROUND', (3, 0), (3, 0), ACCENT),
        ('TEXTCOLOR', (1, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#F5F5F5')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(strat_table)

    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph(
        "<b>Why three?</b> Markets shift between trending and ranging. The aggressive strategy thrives in trends; "
        "the conservative strategy preserves capital in chop. Running all three means we're always positioned "
        "for current conditions, and we generate comparative data to inform future allocation.",
        body
    ))

    # === SAMPLE PERFORMANCE ===
    story.append(Paragraph("Performance Tracking", h1))
    story.append(create_equity_chart())
    story.append(Paragraph("<i>Illustrative equity curves — actual results will vary</i>", small))

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "A live dashboard tracks each strategy's P&L, win rate, drawdown, and trade history in real-time. "
        "After 30+ trades per strategy, we'll have statistically meaningful data to compare approaches.",
        body
    ))

    story.append(PageBreak())

    # === RISK MANAGEMENT ===
    story.append(Paragraph("Risk Controls", h1))

    risk_data = [
        ['Control', 'How It Works'],
        ['Position Sizing', '2-5% of portfolio per trade — no single bet can blow up the account'],
        ['Stop-Losses', 'Every trade has an automatic stop based on volatility (ATR)'],
        ['Max Exposure', '25% total — always have 75%+ in cash reserve'],
        ['Position Limits', 'Max 5 concurrent positions per strategy'],
        ['Time Exits', 'Close stale trades after 72 hours if thesis hasn\'t played out'],
    ]

    risk_table = Table(risk_data, colWidths=[1.4*inch, 5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#F5F5F5')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(risk_table)

    # === TIMELINE ===
    story.append(Paragraph("Roadmap", h1))

    timeline_data = [
        ['Phase', 'Status', 'Goal'],
        ['Build', 'Done', 'Core platform, 3 strategies, dashboard'],
        ['Paper Trade', 'Now', 'Generate performance data (30+ days)'],
        ['Analyze', 'Q1', 'Statistical comparison of strategies'],
        ['Deploy', 'Q2', 'Allocate real capital to winner(s)'],
    ]

    timeline_table = Table(timeline_data, colWidths=[1.2*inch, 0.8*inch, 4.4*inch])
    timeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (1, 1), (1, 1), GREEN),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.white),
        ('BACKGROUND', (1, 2), (1, 2), colors.HexColor('#FF9900')),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.white),
        ('FONTNAME', (1, 1), (1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(timeline_table)

    # === WHY THIS WORKS ===
    story.append(Paragraph("Why This Approach", h1))
    story.append(Paragraph(
        "Most retail traders pick one strategy and hope it works. When it doesn't, they switch to another. "
        "We eliminate the guessing by running multiple strategies in parallel and letting real performance data "
        "guide allocation. This is exactly how Renaissance, Two Sigma, Citadel, and Bridgewater operate internally — "
        "they just do it with thousands of strategies instead of three.",
        body
    ))

    story.append(Spacer(1, 0.3*inch))

    # CTA
    cta = Table([[Paragraph(
        "<b>The platform is live.</b> Contact for dashboard access.",
        ParagraphStyle('cta', fontSize=11, textColor=colors.white, alignment=TA_CENTER)
    )]], colWidths=[6.5*inch])
    cta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    story.append(cta)

    story.append(Spacer(1, 0.4*inch))

    # Disclaimer
    story.append(Paragraph(
        "<b>Disclaimer:</b> This is for informational purposes only. Not investment advice. "
        "Crypto trading involves substantial risk. Past/simulated performance ≠ future results.",
        ParagraphStyle('disc', fontSize=8, textColor=LIGHT)
    ))

    doc.build(story)
    print(f"Created: {filename}")
    return filename

if __name__ == "__main__":
    create_investor_pdf()
