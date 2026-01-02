import fitz  # PyMuPDF
import pandas as pd
import re
import os
from openai import OpenAI

# =========================
# Configuration & Constants
# =========================
EXAM_ID_RE = re.compile(r'^\d{7,11}$')
AWARD_SET = {"一等奖", "二等奖", "三等奖"}
ADVANCE_SET = {"是", "否"}

# =========================
# Core Logic Functions
# =========================

def parse_pdf_to_df(pdf_path):
    """
    Parses the competition PDF and returns a pandas DataFrame.
    """
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return None, f"无法打开 PDF 文件: {e}"

    for page_index, page in enumerate(doc):
        if page_index == 0:
            continue  # 第一页不是正式表格

        blocks = page.get_text("blocks")

        # 收集文本
        lines = []
        for b in blocks:
            if b[6] == 0 and b[4]:
                lines.extend(
                    l.strip() for l in b[4].splitlines() if l.strip()
                )

        current = []

        for token in lines:
            # 准考证号 = 一条记录的起点
            if EXAM_ID_RE.match(token):
                if current:
                    rows.append(current)
                current = [token]
            else:
                current.append(token)

        if current:
            rows.append(current)

    # ===== 结构化 =====
    data = []

    for r in rows:
        # r: [准考证号, 学校..., 姓名, 科目, 奖项, 是否进决赛]
        if len(r) < 5:
            continue

        exam_id = r[0]
        rest = r[1:]

        # 找奖项
        award_idx = None
        for i, x in enumerate(rest):
            if x in AWARD_SET:
                award_idx = i
                break
        if award_idx is None:
            continue

        award = rest[award_idx]
        advance = (
            rest[award_idx + 1]
            if award_idx + 1 < len(rest) and rest[award_idx + 1] in ADVANCE_SET
            else ""
        )

        front = rest[:award_idx]
        if len(front) < 3:
            continue

        subject = front[-1]
        name = front[-2]
        school = "".join(front[:-2]).strip()

        data.append({
            "省份": "安徽",
            "准考证号": exam_id,
            "学校名称": school,
            "考生姓名": name,
            "科目名称": subject,
            "奖项": award,
            "是否进入决赛": advance
        })

    if not data:
        return None, "未提取到有效数据，请检查 PDF 格式。"

    df = pd.DataFrame(data)
    return df, None

def generate_analysis_report(df, api_key, base_url):
    """
    Analyzes the DataFrame and calls the AI to generate a report.
    """
    if df is None or df.empty:
        return None, "数据为空，无法分析。"

    # --- 1. Statistical Analysis ---
    
    # Summary: Total, Awards distribution
    total_participants = len(df)
    award_counts = df['奖项'].value_counts()
    
    df_summary = pd.DataFrame({
        "指标": ["参赛人数", "一等奖", "二等奖", "三等奖"],
        "数值": [
            total_participants, 
            award_counts.get("一等奖", 0),
            award_counts.get("二等奖", 0),
            award_counts.get("三等奖", 0)
        ]
    })
    
    # Subject Strength
    if '科目名称' in df.columns:
        subject_counts = df['科目名称'].value_counts().reset_index()
        subject_counts.columns = ['竞赛方向', '获奖人数']
        # Simple heuristic for strength
        def judge(count):
            if count > 30: return "强"
            if count > 20: return "较强"
            return "有潜力"
        subject_counts['优势判断'] = subject_counts['获奖人数'].apply(judge)
        df_subject_strength = subject_counts.head(10) # Top 10
    else:
        df_subject_strength = pd.DataFrame(columns=['竞赛方向', '获奖人数', '优势判断'])

    # School/Unit Trend
    # The user prompt mentions "College" (学院). If '学校名称' contains different universities, this is a inter-school report.
    # If '学校名称' is constant or contains department names, it's intra-school.
    # We will treat '学校名称' as the unit of analysis.
    if '学校名称' in df.columns:
        school_counts = df['学校名称'].value_counts().reset_index()
        school_counts.columns = ['单位名称', '获奖人数']
        df_school_trend = school_counts.head(10)
    else:
        df_school_trend = pd.DataFrame(columns=['单位名称', '获奖人数'])

    # --- 2. Format Evidence for AI ---
    def df_to_md(d):
        try:
            return d.to_markdown(index=False)
        except:
            return d.to_string(index=False)

    evidence_md = f"""
### 总体情况统计
{df_to_md(df_summary)}

### 各单位/学校获奖情况 (Top 10)
{df_to_md(df_school_trend)}

### 优势学科与竞赛方向 (Top 10)
{df_to_md(df_subject_strength)}
"""

    # --- 3. Call AI ---
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        prompt = f"""
你是一名高校竞赛管理与数据分析助手。
请【仅基于下方给出的数据证据】，撰写一份
《蓝桥杯竞赛总结报告》。

要求：
1. 输出格式：Markdown
2. 正文不少于 800 字
3. 报告需包含以下部分：
   - 概述
   - 总体获奖情况分析
   - 各参赛单位/学院获奖情况分析
   - 优势学科与优势方向分析
   - 存在问题与不足（没有数据支撑的地方请写“需进一步统计”）
   - 改进建议与工作展望
4. 不允许虚构数据、学院、专业或结论。只能基于提供的【数据证据】进行分析。
5. 语言正式、逻辑清晰，可直接作为学校/组委会总结材料。

=========================
【数据证据】
{evidence_md}
=========================

请直接输出完整 Markdown 报告正文。
"""
        response = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[
                {"role": "system", "content": "你是严谨的竞赛总结报告写作助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2600
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"AI 生成报告失败: {e}"

def save_uploaded_file(uploaded_file, temp_dir="temp_data"):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
