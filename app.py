import streamlit as st

st.set_page_config(layout="wide", page_title="Computer Price Explorer")
st.title("💻 Computer Price Explorer")

page = st.sidebar.selectbox("Go to", ["Descriptive","Predictive","Prescriptive"])
if page == "Descriptive":
    from modules.descriptive import app as show_desc; show_desc()
elif page == "Predictive":
    from modules.predictive import app as show_pred; show_pred()
else:
    from modules.prescriptive import app as show_presc; show_presc()