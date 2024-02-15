import pandas as pd 
import sys
sys.path.append('../')
from eccs.eccs import ECCS
import streamlit as st

DATASET_INFO = {
    "XYZ": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-14_21:07:53.csv",
    }
}

GRAPH_INFO = {
    "XYZ_simple": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-14_21:07:53_simple.dot",
    }, 
    "XYZ_correct": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-14_21:07:53_correct.dot",
    }, 
    "XYZ_incorrect": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-14_21:07:53_incorrect.dot",
    }
}


class ECCSUI:
    def __init__(self):
        pd_options = [
            ("display.max_rows", None),
            ("display.max_columns", None),
            ("expand_frame_repr", False),
            ("display.max_colwidth", None),
        ]
        for option, config in pd_options:
            pd.set_option(option, config)

    def prompt_select_file(self):
        def on_click_select_file():
            pass

        with st.form("select_file_form"):
            st.subheader("Choose a data file to analyze:")

            left, right = st.columns(2)

            with left:
                file_selection = st.selectbox(
                    "Select a data file:", list(DATASET_INFO.keys()), key="file_choice"
                )
            with right:
                graph_selection = st.selectbox(
                    "Select a graph file (optional)", list(GRAPH_INFO.keys()), key="graph_choice", placeholder=""
                )

            submitted = st.form_submit_button(
                "Select file", on_click=on_click_select_file
            )
            if submitted:
                with st.spinner("Selecting file..."):
                    self.eccs = ECCS(
                        data_path=DATASET_INFO[file_selection]["path"],
                        graph_path=GRAPH_INFO[graph_selection]["path"] if graph_selection else None,
                    )
                    st.session_state["is_file_chosen"] = True

    def prompt_select_treatment_outcome(self):
        def on_click_select_treatment_outcome():
            pass

        with st.form("select_treatment_outcome_form"):
            st.subheader("Select the treatment and outcome variables:")

            left, right = st.columns(2)

            with left:
                treatment = st.selectbox(
                    "Select the treatment variable:", self.eccs.data.columns, key="treatment"
                )
            with right:
                outcome = st.selectbox(
                    "Select the outcome variable:", self.eccs.data.columns, key="outcome"
                )

            submitted = st.form_submit_button(
                "Select treatment and outcome", on_click=on_click_select_treatment_outcome
            )
            if submitted:
                with st.spinner("Selecting treatment and outcome..."):
                    self.eccs.set_treatment(treatment)
                    self.eccs.set_outcome(outcome)
                    st.session_state["is_treatment_outcome_chosen"] = True

    
    def prompt_edit_graph(self):
        pass

    def prompt_fix_edge(self):
        pass

