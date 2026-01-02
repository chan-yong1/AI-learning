import streamlit as st
import os
import pandas as pd
import backend_logic

# Page Config
st.set_page_config(
    page_title="竞赛数据智能分析助手",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "report_md" not in st.session_state:
    st.session_state.report_md = None

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = "sk-ca2yd4btbfyaknb4lcvkk1gx9sv9ny586r3gby6cha2d2iqz"
    base_url = st.text_input("Base URL", value="https://api.xiaomimimo.com/v1")
    
    st.markdown("---")
    st.markdown("""
    **使用说明**：
    1. **步骤一**：上传 PDF 获奖名单，系统自动解析为 Excel。
    2. **步骤二**：确认数据无误后，点击生成分析报告。
    """)

# Main Title
st.title("🏆 竞赛数据智能解析与报告生成系统")
st.markdown("---")

# Layout: Two main columns or Tabs? Tabs might be cleaner for the "Sequence" logic.
tab1, tab2 = st.tabs(["1️⃣ PDF 解析与数据提取", "2️⃣ AI 智能分析与报告"])

# --- Tab 1: PDF Parsing ---
with tab1:
    st.subheader("📄 上传 PDF 获奖名单")
    pdf_file = st.file_uploader("请上传 .pdf 文件", type=["pdf"], key="pdf_uploader")
    
    if pdf_file is not None:
        if st.button("开始解析 PDF", type="primary"):
            with st.spinner("正在解析 PDF 内容，请稍候..."):
                # Save and Parse
                temp_pdf_path = backend_logic.save_uploaded_file(pdf_file)
                df, error = backend_logic.parse_pdf_to_df(temp_pdf_path)
                
                if error:
                    st.error(error)
                else:
                    st.session_state.df = df
                    st.success(f"解析成功！共提取到 {len(df)} 条记录。")
                    
                    # Show Preview
                    st.dataframe(df.head())
                    
                    # Create Excel for download
                    excel_path = os.path.join("temp_data", "获奖名单_解析结果.xlsx")
                    df.to_excel(excel_path, index=False)
                    
                    with open(excel_path, "rb") as f:
                        st.download_button(
                            label="📥 下载 Excel 结果",
                            data=f,
                            file_name="获奖名单_解析结果.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

# --- Tab 2: Analysis & Report ---
with tab2:
    st.subheader("📊 数据分析与报告生成")
    
    # Input Source: From Session State (just parsed) OR Upload new Excel
    data_source = st.radio("选择数据来源", ["使用刚才解析的数据", "上传已有的 Excel 文件"])
    
    target_df = None
    
    if data_source == "使用刚才解析的数据":
        if st.session_state.df is not None:
            st.info("已加载刚才解析的数据。")
            target_df = st.session_state.df
            st.dataframe(target_df.head(3))
        else:
            st.warning("暂无解析数据，请先在【步骤 1】解析 PDF，或选择上传 Excel。")
            
    else: # Upload Excel
        excel_file = st.file_uploader("上传 Excel 文件 (.xlsx)", type=["xlsx"], key="excel_uploader")
        if excel_file is not None:
            try:
                target_df = pd.read_excel(excel_file)
                st.success(f"Excel 加载成功！共 {len(target_df)} 条记录。")
                st.dataframe(target_df.head(3))
            except Exception as e:
                st.error(f"Excel 读取失败: {e}")

    st.markdown("### 🤖 AI 报告生成")
    
    if st.button("🚀 开始生成分析报告", type="primary", disabled=(target_df is None)):
        if not api_key:
            st.error("请在左侧侧边栏输入 API Key！")
        elif target_df is None:
            st.error("没有可用的数据！")
        else:
            with st.spinner("正在进行数据统计并调用 AI 生成报告（这可能需要几十秒）..."):
                report, error = backend_logic.generate_analysis_report(target_df, api_key, base_url)
                
                if error:
                    st.error(error)
                else:
                    st.session_state.report_md = report
                    st.success("✅ 报告生成成功！")

    # Display Report
    if st.session_state.report_md:
        st.markdown("---")
        st.subheader("📝 报告预览")
        st.markdown(st.session_state.report_md)
        
        st.download_button(
            label="� 下载 Markdown 报告",
            data=st.session_state.report_md,
            file_name="竞赛总结报告.md",
            mime="text/markdown"
        )
