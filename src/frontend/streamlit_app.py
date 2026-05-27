from __future__ import annotations
import os
import io
import json
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Cyberbullying Detector", layout="wide")

st.title("AI Cyberbullying Detection")

text = st.text_area("Enter text to analyze", height=150)
col1, col2 = st.columns([1,1])
with col1:
    source_lang = st.text_input("Source language (optional, e.g., en, es, fr)")
with col2:
    if st.button("Analyze", type="primary") and text.strip():
        with st.spinner("Analyzing..."):
            resp = requests.post(f"{API_BASE}/analyze", json={"text": text, "source_lang": source_lang or None})
            if resp.ok:
                data = resp.json()
                st.session_state['last_analysis'] = data
            else:
                st.error(f"API error: {resp.status_code}")

if 'last_analysis' in st.session_state:
    data = st.session_state['last_analysis']
    st.subheader("Prediction")
    color = {"low": "#8BC34A", "medium": "#FFC107", "high": "#F44336"}.get(data['severity'], "#9E9E9E")
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:{color}'>Prediction: <b>{data['prediction']}</b>, Category: <b>{data['category']}</b>, Severity: <b>{data['severity']}</b>, Confidence: <b>{data['confidence']:.3f}</b></div>", unsafe_allow_html=True)
    if data.get('translated'):
        st.info(f"Translated text used for inference: {data['translated']}")

    st.write("Top contributing tokens (LIME):")
    if data.get('explanations'):
        dfexp = pd.DataFrame(data['explanations'])
        st.bar_chart(dfexp.set_index('token'))
    else:
        st.write("Explainability unavailable.")

    st.subheader("Rephrasing suggestion")
    if st.button("Suggest safer rephrasing"):
        rr = requests.post(f"{API_BASE}/rephrase", json={"text": text})
        if rr.ok:
            rj = rr.json()
            st.code(rj['suggestion'])
        else:
            st.error("Rephrase API error")

    st.subheader("Feedback")
    fbcol1, fbcol2 = st.columns(2)
    with fbcol1:
        if st.button("Was this correct? Yes ✅"):
            requests.post(f"{API_BASE}/feedback", json={"text": text, "correct": True})
            st.success("Thanks for the feedback!")
    with fbcol2:
        if st.button("Was this correct? No ❌"):
            requests.post(f"{API_BASE}/feedback", json={"text": text, "correct": False})
            st.success("Thanks for the feedback!")

st.divider()

st.subheader("Dashboard")
if st.button("Refresh dashboard"):
    dash = requests.get(f"{API_BASE}/dashboard").json()
    st.session_state['dash'] = dash

if 'dash' in st.session_state:
    dash = st.session_state['dash']
    c1, c2 = st.columns(2)
    with c1:
        st.write("Category distribution")
        df = pd.DataFrame([dash['category_distribution']]).T
        df.columns = ['count']
        st.bar_chart(df)
    with c2:
        st.write("Severity distribution")
        df2 = pd.DataFrame([dash['severity_distribution']]).T
        df2.columns = ['count']
        st.bar_chart(df2)
    c3, c4 = st.columns(2)
    with c3:
        st.write("Timeline")
        tl = pd.DataFrame(dash['timeline'])
        if not tl.empty:
            tl['date'] = pd.to_datetime(tl['date'])
            tl = tl.set_index('date')
            st.line_chart(tl)
        else:
            st.write("No data yet.")
    with c4:
        st.write("Word cloud")
        if dash.get('wordcloud_path') and os.path.exists(dash['wordcloud_path']):
            st.image(dash['wordcloud_path'])
        else:
            st.write("No word cloud available.")

st.divider()

st.subheader("Reports")
ccol1, ccol2 = st.columns(2)
with ccol1:
    if st.button("Download CSV report"):
        r = requests.get(f"{API_BASE}/report/csv").json()
        st.success(f"CSV generated at: {r['path']}")
with ccol2:
    if st.button("Download PDF report"):
        r = requests.get(f"{API_BASE}/report/pdf").json()
        st.success(f"PDF generated at: {r['path']}")

st.divider()

st.title("📊 Model Performance Comparison")

if st.button("Load Model Comparison", type="primary"):
    try:
        comp_resp = requests.get(f"{API_BASE}/models/comparison")
        if comp_resp.ok:
            st.session_state['model_comparison'] = comp_resp.json()
        else:
            st.error("Failed to load model comparison data")
    except Exception as e:
        st.error(f"Error: {e}")

if 'model_comparison' in st.session_state:
    comp_data = st.session_state['model_comparison']
    
    st.subheader("Overall Performance Metrics")
    
    if comp_data.get('metrics'):
        metrics = comp_data['metrics']
        
        # Create summary table
        summary_df = pd.DataFrame([{
            "Model": m['model_name'],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision_macro']:.4f}",
            "Recall": f"{m['recall_macro']:.4f}",
            "F1 Score": f"{m['f1_macro']:.4f}"
        } for m in metrics])
        
        # Sort by F1 score
        summary_df['F1_numeric'] = [m['f1_macro'] for m in metrics]
        summary_df = summary_df.sort_values('F1_numeric', ascending=False).drop('F1_numeric', axis=1)
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Best model highlight
        best_model = metrics[0] if metrics else None
        if best_model:
            for m in metrics:
                if m['f1_macro'] == max([x['f1_macro'] for x in metrics]):
                    best_model = m
                    break
            st.success(f"🏆 Best Model: **{best_model['model_name']}** with F1 Score: {best_model['f1_macro']:.4f}")
        
        st.divider()
        
        # Comparison plots
        st.subheader("Performance Comparison Plots")
        if comp_data.get('comparison_plots'):
            for plot in comp_data['comparison_plots']:
                if os.path.exists(plot['path']):
                    st.image(plot['path'], caption=plot['filename'].replace('.png', '').replace('_', ' ').title(), use_container_width=True)
        
        st.divider()
        
        # Confusion Matrices
        st.subheader("Confusion Matrices")
        
        # Model selector
        model_names = list(set([cm['model'] for cm in comp_data.get('confusion_matrices', [])]))
        if model_names:
            selected_model = st.selectbox("Select model to view confusion matrix:", model_names)
            
            # Show confusion matrices for selected model
            col_cm1, col_cm2 = st.columns(2)
            
            for cm in comp_data.get('confusion_matrices', []):
                if cm['model'] == selected_model and os.path.exists(cm['path']):
                    if 'normalized' in cm['filename']:
                        with col_cm2:
                            st.image(cm['path'], caption="Normalized Confusion Matrix", use_container_width=True)
                    else:
                        with col_cm1:
                            st.image(cm['path'], caption="Confusion Matrix", use_container_width=True)
        
        st.divider()
        
        # Detailed per-class metrics
        st.subheader("Per-Class Performance")
        
        selected_model_detail = st.selectbox("Select model for detailed metrics:", [m['model_name'] for m in metrics])
        
        selected_metrics = next((m for m in metrics if m['model_name'] == selected_model_detail), None)
        if selected_metrics and 'per_class_metrics' in selected_metrics:
            per_class_df = pd.DataFrame([
                {
                    "Class": class_name,
                    "Precision": f"{class_metrics['precision']:.4f}",
                    "Recall": f"{class_metrics['recall']:.4f}",
                    "F1 Score": f"{class_metrics['f1_score']:.4f}",
                    "Support": class_metrics['support']
                }
                for class_name, class_metrics in selected_metrics['per_class_metrics'].items()
            ])
            st.dataframe(per_class_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Download links
        st.subheader("Download Detailed Reports")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            if comp_data.get('csv_path') and os.path.exists(comp_data['csv_path']):
                st.info(f"📄 Summary CSV: `{comp_data['csv_path']}`")
        
        with col_dl2:
            if comp_data.get('detailed_csv_path') and os.path.exists(comp_data['detailed_csv_path']):
                st.info(f"📄 Detailed CSV: `{comp_data['detailed_csv_path']}`")
        
        with col_dl3:
            if comp_data.get('report_path') and os.path.exists(comp_data['report_path']):
                st.info(f"📄 Text Report: `{comp_data['report_path']}`")
    
    else:
        st.warning("No model comparison data available. Please run training first to generate metrics.")
