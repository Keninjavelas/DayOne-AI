import re

def apply_fixes():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Mascot Orbit -> Halo and Sparks
    new_mascot_css = """            .mascot-container {{
                position: relative;
                width: 140px;
                height: 140px;
                margin: 0 auto 1.5rem auto;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .mascot-halo {{
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                width: 140px; height: 140px;
                border-radius: 50%;
                box-shadow: 0 0 60px 20px rgba(124, 58, 237, 0.4), 0 0 100px 40px rgba(59, 130, 246, 0.2);
                animation: pulseHalo 4s ease-in-out infinite;
                z-index: 1;
            }}
            @keyframes pulseHalo {{
                0% {{ opacity: 0.6; transform: translate(-50%, -50%) scale(0.95); }}
                50% {{ opacity: 1; transform: translate(-50%, -50%) scale(1.05); }}
                100% {{ opacity: 0.6; transform: translate(-50%, -50%) scale(0.95); }}
            }}
            .spark {{
                position: absolute;
                width: 4px; height: 4px;
                background: #fff;
                border-radius: 50%;
                box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8);
                opacity: 0;
                animation: floatSpark 3s ease-in infinite;
                z-index: 2;
            }}
            .spark-1 {{ top: 10%; left: 20%; animation-delay: 0s; }}
            .spark-2 {{ top: 80%; left: 80%; animation-delay: 1.5s; width: 3px; height: 3px; }}
            .spark-3 {{ top: 70%; left: 10%; animation-delay: 2.2s; }}
            .spark-4 {{ top: 20%; left: 85%; animation-delay: 0.7s; width: 5px; height: 5px; }}
            @keyframes floatSpark {{
                0% {{ transform: translateY(0) scale(0.5); opacity: 0; }}
                20% {{ opacity: 1; }}
                80% {{ opacity: 1; }}
                100% {{ transform: translateY(-40px) scale(1.2); opacity: 0; }}
            }}"""
    content = re.sub(r'            \.mascot-container \{\{.*?@keyframes spin \{\{ 100% \{\{ transform: translate\(-50%, -50%\) rotate\(360deg\); \}\} \}\}', new_mascot_css, content, flags=re.DOTALL)

    # Remove old hero-title
    content = re.sub(r'            \.hero-title \{\{.*?\}\}\s*\}\}', '', content, flags=re.DOTALL)

    # 2. Unified Hero Card CSS
    new_hero_css = """            .unified-hero {{
                background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
                backdrop-filter: blur(40px);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-top: 1px solid rgba(124, 58, 237, 0.5);
                border-top-left-radius: 24px;
                border-top-right-radius: 24px;
                border-bottom: none;
                box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1), 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 40px rgba(124, 58, 237, 0.1);
                padding: 3rem 2rem 1.5rem 2rem;
                max-width: 700px;
                margin: 0 auto;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            .unified-hero::before {{
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; height: 100%;
                background: radial-gradient(circle at top center, rgba(124, 58, 237, 0.15), transparent 60%);
                pointer-events: none;
            }}
            .unified-hero .hero-title {{
                font-size: 3.5rem !important;
                font-weight: 800 !important;
                letter-spacing: -0.05em !important;
                margin-bottom: 0.25rem !important;
                line-height: 1.1 !important;
                background: linear-gradient(135deg, #ffffff, #94a3b8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative; z-index: 10;
            }}
            .unified-hero .hero-context {{
                font-size: 1.15rem;
                color: #94a3b8;
                font-weight: 500;
                margin-bottom: 2.5rem;
                letter-spacing: 0.02em;
                position: relative; z-index: 10;
            }}
            .unified-hero .insight-box {{
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 1.5rem;
                text-align: left;
                margin-bottom: 0.5rem;
                position: relative; z-index: 10;
            }}
            .unified-hero .insight-header {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.85rem;
                font-weight: 700;
                color: #A78BFA;
                margin-bottom: 0.75rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .unified-hero .insight-body {{
                font-size: 1.15rem;
                color: #f8fafc;
                line-height: 1.6;
                font-weight: 500;
                margin: 0;
            }}
            
            /* Primary button snaps to unified hero card */
            .stButton > button[kind="primary"] {{
                border-top-left-radius: 0 !important;
                border-top-right-radius: 0 !important;
                border-bottom-left-radius: 24px !important;
                border-bottom-right-radius: 24px !important;
                background: rgba(124, 58, 237, 0.15) !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-top: 1px solid rgba(255, 255, 255, 0.03) !important;
                color: #fff !important;
                text-align: center !important;
                justify-content: center !important;
                padding: 1.2rem !important;
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5) !important;
                margin-top: -16px !important;
                max-width: 700px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                transition: background 0.2s;
            }}"""
    content = re.sub(r'            /\* ── DayOne Insight Centerpiece.*?\}\}', new_hero_css, content, flags=re.DOTALL, count=1)

    # 3. Add Color-coded Proactive Cards
    new_proactive_css = """            /* ── Proactive Action Cards (Lower Weight) ── */
            .proactive-card {{
                background: rgba(255, 255, 255, 0.01);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-bottom: none;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                padding: 1.25rem;
                margin-bottom: -1rem; /* Collapse with button */
                box-shadow: inset 0 1px 1px rgba(255,255,255,0.02);
            }}
            .proactive-card-red {{ border-top: 2px solid rgba(244, 63, 94, 0.5); background: linear-gradient(180deg, rgba(244, 63, 94, 0.05) 0%, transparent 100%); }}
            .proactive-card-purple {{ border-top: 2px solid rgba(168, 85, 247, 0.5); background: linear-gradient(180deg, rgba(168, 85, 247, 0.05) 0%, transparent 100%); }}
            .proactive-card-green {{ border-top: 2px solid rgba(16, 185, 129, 0.5); background: linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%); }}
            .proactive-card-blue {{ border-top: 2px solid rgba(59, 130, 246, 0.5); background: linear-gradient(180deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%); }}"""
    content = re.sub(r'            /\* ── Proactive Action Cards \(Lower Weight\) ── \*/.*?box-shadow: inset 0 1px 1px rgba\(255,255,255,0\.02\);\n            \}\}', new_proactive_css, content, flags=re.DOTALL)

    kicker_css = """
            .section-kicker {{
                font-size: 0.85rem;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 1rem;
            }}"""
    content = content.replace('/* Glassmorphism Cards */', kicker_css + '\n            /* Glassmorphism Cards */')

    # Unified Hero HTML
    new_hero_html = """
    mascot_b64 = get_base64_image(MASCOT_PATH)
    mascot_html = f'''
    <div class="mascot-container">
        <div class="mascot-halo"></div>
        <div class="spark spark-1"></div>
        <div class="spark spark-2"></div>
        <div class="spark spark-3"></div>
        <div class="spark spark-4"></div>
        <img src="data:image/png;base64,{mascot_b64}" class="hero-mascot" alt="Mascot">
    </div>
    ''' if mascot_b64 else '<div class="hero-mascot-placeholder">🤖</div>'

    st.markdown(
        f'''
        <div class="unified-hero">
            {mascot_html}
            <div class="hero-title">Good Morning, {first_name} 👋</div>
            <div class="hero-context">14 PTO days • 2 tasks due</div>
            
            <div class="insight-box">
                <div class="insight-header">🤖 DayOne Insight</div>
                <div class="insight-body">
                    You have <strong style="color: #A78BFA;">enough PTO</strong> for a long weekend. 
                    However, your compliance training is <strong style="color: #FFB020;">overdue</strong>.
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    if st.button("Take Action", key="insight_btn", type="primary", use_container_width=True):
        st.session_state.pending_prompt = "Help me finish my compliance training and request PTO."

    st.markdown('<div class="section-kicker">Your Workspace</div>', unsafe_allow_html=True)
    cols = st.columns([2, 1, 1])
"""
    content = re.sub(r'    mascot_b64 = get_base64_image\(MASCOT_PATH\).*?cols = st\.columns\(3\)', new_hero_html, content, flags=re.DOTALL)

    # Coverage Score
    new_coverage = """        <div class="glass-card stat-card" style="align-items: center; justify-content: flex-start; padding: 1.5rem; display: flex; flex-direction: row; gap: 1rem;">
            <svg viewBox="0 0 36 36" class="circular-chart" style="stroke: #00D084; min-width: 60px;">
                <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path class="circle" stroke-dasharray="92, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <text x="18" y="20.35" class="percentage">92%</text>
            </svg>
            <div style="display: flex; flex-direction: column;">
                <div class="stat-title" style="margin-bottom: 0.25rem;">Coverage</div>
                <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4; font-weight: 500;">
                    <span style="color: #00D084;">✓</span> Medical<br>
                    <span style="color: #00D084;">✓</span> Dental<br>
                    <span style="color: #00D084;">✓</span> Vision
                </div>
            </div>
        </div>"""
    content = re.sub(r'        <div class="glass-card stat-card" style="align-items: center; justify-content: center; padding: 1rem 1.5rem;">.*?</div>', new_coverage, content, flags=re.DOTALL)

    # Color-coded suggestions
    content = content.replace('<div class="proactive-card">', '<div class="proactive-card proactive-card-purple">', 1)
    content = content.replace('<div class="proactive-card">', '<div class="proactive-card proactive-card-green">', 1)
    content = content.replace('<div class="proactive-card">', '<div class="proactive-card proactive-card-blue">', 1)
    content = content.replace('<div class="proactive-card">', '<div class="proactive-card proactive-card-red">', 1)

    content = content.replace('st.markdown("##### AI Suggestions")', 'st.markdown(\'<div class="section-kicker">AI Suggestions</div>\', unsafe_allow_html=True)')
    content = content.replace('st.markdown("##### Recent Activity")', 'st.markdown(\'<div class="section-kicker">Recent Activity</div>\', unsafe_allow_html=True)')
    content = content.replace('st.markdown("<br>", unsafe_allow_html=True)', '')

    content = content.replace('    st.write("")\n    st.write("")\n    st.write("")', '')
    content = content.replace('    st.write("")\n    st.write("")', '')

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    apply_fixes()
