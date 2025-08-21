import streamlit as st
import sys, os

# ✅ Add parent folder (task5) to Python path so Python can find src/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.infer import predict

st.title('Task 5 — Support Ticket Auto-Tagging')

text = st.text_area('Ticket text', 'I cannot login to my account')

if st.button('Tag'):
    out = predict(text)
    st.write('Predicted tags and confidences:')
    for k, v in out.items():
        st.write(f"- {k}: {v:.2f}")
