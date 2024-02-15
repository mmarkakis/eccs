import pandas as pd
import sys

sys.path.append("../")
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
    },
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

    def clear_next(self, variables):
        for var in variables:
            if var in st.session_state:
                del st.session_state[var]

    def prompt_select_file(self):
        def on_click_select_file():
            pass

        with st.form("select_file_form"):
            st.subheader("Choose a data file to analyze:")

            file_selection = st.selectbox(
                "Select a data file:", list(DATASET_INFO.keys()), key="file_choice"
            )

            graph_selection = st.selectbox(
                "Select a graph file (optional)",
                list(GRAPH_INFO.keys()),
                key="graph_choice",
                placeholder="",
            )

            submitted = st.form_submit_button(
                "Select file",
                on_click=on_click_select_file,
                disabled=(
                    "is_file_chosen" in st.session_state
                    and st.session_state["is_file_chosen"]
                ),
            )
            if submitted:
                with st.spinner("Selecting file..."):
                    self.eccs = ECCS(
                        data_path=DATASET_INFO[file_selection]["path"],
                        graph_path=(
                            GRAPH_INFO[graph_selection]["path"]
                            if graph_selection
                            else None
                        ),
                    )
                    st.session_state["is_file_chosen"] = True
                    st.session_state["graph"] = self.eccs.draw_graph()

    def prompt_select_treatment(self):
        def on_click_select_treatment():
            pass

        with st.form("select_treatment_form"):
            st.subheader("Select the treatment variable:")

            treatment = st.selectbox(
                "Select the treatment variable:",
                self.eccs.data.columns,
                key="treatment",
            )
            submitted = st.form_submit_button(
                "Select treatment",
                on_click=on_click_select_treatment,
                disabled=(
                    "is_treatment_chosen" in st.session_state
                    and st.session_state["is_treatment_chosen"]
                ),
            )

            if submitted:
                with st.spinner("Selecting treatment ..."):
                    self.eccs.set_treatment(treatment)
                    st.session_state["is_treatment_chosen"] = True

    def prompt_select_outcome(self):
        def on_click_select_outcome():
            pass

        with st.form("select_outcome_form"):
            st.subheader("Select the outcome variable:")

            outcome = st.selectbox(
                "Select the outcome variable:",
                self.eccs.data.columns,
                key="outcome",
            )
            submitted = st.form_submit_button(
                "Select outcome",
                on_click=on_click_select_outcome,
                disabled=(
                    "is_outcome_chosen" in st.session_state
                    and st.session_state["is_outcome_chosen"]
                ),
            )

            if submitted:
                with st.spinner("Selecting outcome ..."):
                    self.eccs.set_outcome(outcome)
                    st.session_state["is_outcome_chosen"] = True

    def prompt_edit_graph(self):
        """
        Prompts the user with options for editing the graph.
        """

        def on_click_add():
            self.eccs.add_edge(
                src=st.session_state["edit_source_node"],
                dst=st.session_state["edit_destination_node"],
            )
            self.clear_next(["edit_source_node", "edit_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        def on_click_remove():
            self.eccs.remove_edge(
                src=st.session_state["edit_source_node"],
                dst=st.session_state["edit_destination_node"],
            )
            self.clear_next(["edit_source_node", "edit_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        with st.form("edit_graph_form"):
            st.subheader("Edit the graph:")
            st.markdown(
                "You cannot add an edge that is in the `banlist` or remove an edge that is `fixed` (green)."
            )
            source_col, destination_col = st.columns(2)

            with source_col:
                selected_source = st.selectbox(
                    "Source node:",
                    self.eccs.data.columns,
                    key="edit_source_node",
                )

            with destination_col:
                selected_destination = st.selectbox(
                    "Destination node:",
                    self.eccs.data.columns,
                    key="edit_destination_node",
                )

            acc_col, rej_col, _ = st.columns([0.2, 0.3, 0.5])
            with acc_col:
                st.form_submit_button("Add", on_click=on_click_add)
            with rej_col:
                st.form_submit_button("Remove", on_click=on_click_remove)

    def prompt_fix_edge(self):
        """
        Prompts the user with options for fixing an edge.
        """

        def on_click_fix():
            self.eccs.fix_edge(
                src=st.session_state["fix_source_node"],
                dst=st.session_state["fix_destination_node"],
            )
            self.clear_next(["fix_source_node", "fix_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        def on_click_unfix():
            self.eccs.unfix_edge(
                src=st.session_state["fix_source_node"],
                dst=st.session_state["fix_destination_node"],
            )
            self.clear_next(["fix_source_node", "fix_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        with st.form("fix_edge_form"):
            st.subheader("Fix an edge in graph:")
            st.markdown("You cannot fix an edge that is in the `banlist`.")
            source_col, destination_col = st.columns(2)

            with source_col:
                selected_source = st.selectbox(
                    "Source node:",
                    self.eccs.data.columns,
                    key="fix_source_node",
                )

            with destination_col:
                selected_destination = st.selectbox(
                    "Destination node:",
                    self.eccs.data.columns,
                    key="fix_destination_node",
                )

            acc_col, rej_col, _ = st.columns([0.2, 0.3, 0.5])
            with acc_col:
                st.form_submit_button("Fix", on_click=on_click_fix)
            with rej_col:
                st.form_submit_button("Unfix", on_click=on_click_unfix)

    def prompt_ban_edge(self):
        """
        Prompts the user with options for banning an edge.
        """

        def on_click_ban():
            self.eccs.ban_edge(
                src=st.session_state["ban_source_node"],
                dst=st.session_state["ban_destination_node"],
            )
            self.clear_next(["ban_source_node", "ban_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        def on_click_unban():
            self.eccs.unban_edge(
                src=st.session_state["ban_source_node"],
                dst=st.session_state["ban_destination_node"],
            )
            self.clear_next(["ban_source_node", "ban_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        with st.form("ban_edge_form"):
            st.subheader("Ban an edge from the graph:")
            st.markdown("You cannot ban an edge that is `fixed`, or unban its reverse.")
            source_col, destination_col = st.columns(2)

            with source_col:
                selected_source = st.selectbox(
                    "Source node:",
                    self.eccs.data.columns,
                    key="ban_source_node",
                )

            with destination_col:
                selected_destination = st.selectbox(
                    "Destination node:",
                    self.eccs.data.columns,
                    key="ban_destination_node",
                )

            acc_col, rej_col, _ = st.columns([0.2, 0.3, 0.5])
            with acc_col:
                st.form_submit_button("Ban", on_click=on_click_ban)
            with rej_col:
                st.form_submit_button("Unban", on_click=on_click_unban)
