"""Module 5 - SAD Integration: requirements, actors, use case diagram, decision workflow."""
import streamlit as st


def show():
    st.header("System Design (SAD Integration)")

    st.subheader("Functional requirements (from the midterm findings)")
    st.markdown(
        "| ID | Requirement | Evidence | Where in this app |\n|---|---|---|---|\n"
        "| FR1 | Show SLA breach rate per assigned office | Registrar 94.0% (n=67) highest; Dean's Office 89.3% (n=75) lowest | Data Analysis, Dashboard |\n"
        "| FR2 | Show open requests and queue category; alert when queue > 15 | Low 65.1%, Medium 96.4%, High 100.0% | Dashboard (queue chart) |\n"
        "| FR3 | Calculate a delay-risk flag from submission-time data, show reasons, send to staff for review | Tree F1 0.964 vs baseline 0.947, recall 1.000 | Prediction, Dashboard |\n"
        "| FR4 | Let Office Staff confirm or override a flag with a written reason | Precision 0.931 - the model can be wrong | Dashboard (Staff decision) |\n"
        "| FR5 | Let users upload a CSV, validate its columns and summarize nulls/duplicates | 12 missing Queue_Length values, 3 duplicate IDs found in midterm | Dataset |")

    st.subheader("Non-functional requirements")
    st.markdown(
        "- **NFR1 Explainability:** every prediction shows risk %, main reasons, similar past requests and a limits note.\n"
        "- **NFR2 Privacy and safety:** only synthetic or authorized data; no names or passwords; no leakage columns used.\n"
        "- **NFR3 Performance and reliability:** the saved model loads once (cached) and is never retrained on refresh; "
        "invalid files show clear error messages.")

    st.subheader("Actors and permissions")
    st.markdown(
        "| Actor | View risk results | Record final decision |\n|---|---|---|\n"
        "| Requester (Student/Faculty/Staff) | No (submit and track only) | No |\n"
        "| Office Staff | Yes | **Yes** (confirm or override with reason) |\n"
        "| Administrator | Yes (SLA performance, queue status) | No |")

    st.subheader("Updated use case diagram")
    try:
        st.image("use_case_diagram.png", use_container_width=True)
    except Exception:
        st.error("Use case diagram not found: use_case_diagram.png")
    st.caption("Blue = new analytics use cases. Grey = existing use cases.")

    st.subheader("Analytical process / decision workflow")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style=rounded];
      A [label="1. Upload / load\\nrequest data"] -> B [label="2. Validate and clean"];
      B -> C [label="3. Build submission-time\\nfeatures (no leakage)"];
      C -> D [label="4. Saved Decision Tree\\npredicts delay risk"];
      D -> E [label="5. Dashboard shows\\nrisk + reasons + warnings"];
      E -> F [label="6. Office Staff reviews\\n(human decision)"];
      F -> G [label="7. Confirm / Override\\n+ reason saved"];
    }""")
    st.info("Predictions are decision-support outputs, not automatic authorization to reject or "
            "deny any service. A staff member always makes the final decision.")
