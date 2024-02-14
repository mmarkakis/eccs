import streamlit as st
import sys
sys.path.append('../')
from utils import logo, hide_default, refresh_page, reset_session_button, display_current_info
from streamlit import components
import os
from io import StringIO
import pandas as pd
from eccs_ui import ECCSUI
from eccs.graph_renderer import GraphRenderer


st.set_page_config(
    page_title='Home',
    page_icon='🏠',
    layout='wide'
)

st.title('ECCS: Exposing Critical Causal Structures')

logo()

def reset_experiment():
    for key in st.session_state.keys():
        del st.session_state[key]

st.sidebar.header("Reset Experiment:")
st.sidebar.button("Reset", on_click=reset_experiment)

def display_current_info(header_text: str, variable: str):
    st.sidebar.header(header_text)

    if variable not in st.session_state:
        st.sidebar.text("None")
    else:
        st.sidebar.text(st.session_state[variable])

if "eccs_ui" not in st.session_state:
    st.session_state["eccs_ui"] = ECCSUI()
eccs_ui = st.session_state["eccs_ui"]


eccs_ui.prompt_select_file()

if "is_file_chosen" in st.session_state and st.session_state["is_file_chosen"]:
    # Row about deciding on edges
    st.subheader("Decide whether to include an edge to the causal graph")
    col_1, col_2 = st.columns([0.3, 0.7])
    with col_1:
        eccs_ui.prompt_edit_graph()
        eccs_ui.prompt_fix_edge()
    with col_2:
        with st.expander("Causal graph", expanded=True):
            if "graph" in st.session_state:  
                components.v1.html(GraphRenderer.graph_string_to_html(st.session_state["graph"])._repr_html_(), height=500)

    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)
