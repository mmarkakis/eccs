import pandas as pd
import sys

sys.path.append("../")
from eccs.eccs import ECCS
import streamlit as st
from eccs.graph_renderer import GraphRenderer
from streamlit import components


DATASET_INFO = {
    "XYZ": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-20_20:42:28.csv",
    }
}

GRAPH_INFO = {
    "XYZ_simple": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-20_20:42:28_simple.dot",
    },
    "XYZ_correct": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-20_20:42:28_correct.dot",
    },
    "XYZ_incorrect": {
        "path": "../../datasets/xyz_extended/dataset_2024-02-20_20:42:28_incorrect.dot",
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
            self.clear_next(["ate"])

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
            )

            if submitted:
                with st.spinner("Selecting treatment ..."):
                    self.eccs.set_treatment(treatment)
                    st.session_state["is_treatment_chosen"] = True

    def prompt_select_outcome(self):
        def on_click_select_outcome():
            self.clear_next(["ate"])

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
            self.clear_next(["ate", "edit_source_node", "edit_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        def on_click_remove():
            self.eccs.remove_edge(
                src=st.session_state["edit_source_node"],
                dst=st.session_state["edit_destination_node"],
            )
            self.clear_next(["ate", "edit_source_node", "edit_destination_node"])
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
            self.clear_next(["ate", "fix_source_node", "fix_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        def on_click_unfix():
            self.eccs.unfix_edge(
                src=st.session_state["fix_source_node"],
                dst=st.session_state["fix_destination_node"],
            )
            self.clear_next(["ate", "fix_source_node", "fix_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        with st.form("fix_edge_form"):
            st.subheader("Fix an edge in the graph:")
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
            self.clear_next(["ate", "ban_source_node", "ban_destination_node"])
            st.session_state["graph"] = self.eccs.draw_graph()

        def on_click_unban():
            self.eccs.unban_edge(
                src=st.session_state["ban_source_node"],
                dst=st.session_state["ban_destination_node"],
            )
            self.clear_next(["ate", "ban_source_node", "ban_destination_node"])
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

    def show_current_graph(self):
        with st.expander("Current Causal graph", expanded=True):
            if "graph" in st.session_state:
                components.v1.html(
                    GraphRenderer.graph_string_to_html(
                        st.session_state["graph"]
                    )._repr_html_(),
                    height=300,
                )

    def prompt_calculate_current_ate(self):

        def on_click_calculate_current_ate():
            with st.spinner("Calculating ATE ..."):
                st.session_state["ate"] = self.eccs.get_ate()
                self.clear_next(["future_ate", "future_graph", "future_modifications"])

        with st.form("calculate_current_ate_form"):
            st.subheader("Calculate the current ATE:")
            submitted = st.form_submit_button(
                "Calculate ATE",
                on_click=on_click_calculate_current_ate,
                disabled=not (
                    "is_treatment_chosen" in st.session_state
                    and "is_outcome_chosen" in st.session_state
                ),
            )

            if "ate" in st.session_state:
                st.markdown(f"The current ATE is **{st.session_state['ate']:.3f}**")

    def prompt_press_eccs(self):
        def on_click_press_eccs():
            pass

        with st.form("press_eccs_form"):
            st.subheader("Press ECCS to Doubt!")

            # Have a dropdown for the different ECCS methods
            eccs_method = st.selectbox(
                "Select a method:", ECCS.EDGE_SUGGESTION_METHODS, key="eccs_method"
            )

            submitted = st.form_submit_button(
                "ECCS",
                on_click=on_click_press_eccs,
                disabled=not ("ate" in st.session_state),
            )

            if submitted:
                with st.spinner("Calculating..."):
                    ate, graph, modifications = self.eccs.suggest(method=eccs_method)
                    st.session_state["future_ate"] = ate
                    st.session_state["future_graph"] = graph
                    st.session_state["future_modifications"] = modifications

    def show_eccs_findings(self):
        with st.container(border=True):
            st.subheader("EECS Findings")
            st.markdown(
                f"For the graph on the right, the ATE could be **{st.session_state['future_ate']:.3f}**"
            )
            is_increase = (
                "an increase"
                if st.session_state["future_ate"] > st.session_state["ate"]
                else "a decrease"
            )
            diff = abs(st.session_state["future_ate"] - st.session_state["ate"])
            st.markdown(
                f"This is {is_increase} of **{diff:.3f}** from the current ATE!"
            )

    def show_future_graph(self):
        with st.expander("Alternative Causal Graph", expanded=True):
            components.v1.html(
                GraphRenderer.graph_string_to_html(
                    st.session_state["future_graph"]
                )._repr_html_(),
                height=300,
            )
