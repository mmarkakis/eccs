import streamlit as st
import sys

sys.path.append("../")
from utils import (
    logo,
    hide_default,
    refresh_page,
    reset_session_button,
    display_current_info,
)
import os
from io import StringIO
import pandas as pd
from eccs_ui import ECCSUI


st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

st.title("ECCS: Exposing Critical Causal Structures")

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

with st.sidebar:
    eccs_ui.prompt_select_file()

if "is_file_chosen" in st.session_state and st.session_state["is_file_chosen"]:

    # Row about deciding on edges
    st.subheader("Refine your causal graph:")
    col_1, col_2, col_3 = st.columns([0.3, 0.3, 0.3])
    with col_1:
        eccs_ui.prompt_edit_graph()
        eccs_ui.prompt_fix_edge()
    with col_2:
        eccs_ui.prompt_ban_edge()
        with st.expander("Banlist", expanded=True):
            st.dataframe(eccs_ui.eccs.banlist_df)
    with col_3:
        eccs_ui.show_current_graph()
       
    st.markdown('<hr style="border:1px solid lightgray">', unsafe_allow_html=True)

    st.subheader("Get suggested modifications:")

    col_1, col_2, col_3 = st.columns([0.3, 0.3, 0.3])
    with col_1:
        eccs_ui.prompt_select_treatment()
        eccs_ui.prompt_select_outcome()
    with col_2:
        eccs_ui.prompt_calculate_current_ate()
        if "ate" in st.session_state:
            eccs_ui.prompt_press_eccs()
        if "ate" in st.session_state and "future_ate" in st.session_state:
            eccs_ui.show_eccs_findings()
    with col_3:
        if "ate" in st.session_state and "future_graph" in st.session_state:
            eccs_ui.show_future_graph()
        if "ate" in st.session_state and "future_modifications" in st.session_state:
            with st.expander("Graph modifications", expanded=True):
                st.write(st.session_state["future_modifications"])
