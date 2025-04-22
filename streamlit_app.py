"""Streamlit frontend for user login and launching the Diabetes Predictor Flask app."""

import webbrowser  # Standard library imports should come first
import streamlit as st  # Third-party libraries
from auth import login_register_ui, is_authenticated, logout  # Local module imports
from database import create_user_table


def main():
    """Main function to handle user authentication and app launching."""
    st.set_page_config(page_title="Login Portal", page_icon="🔐")
    create_user_table()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not is_authenticated():
        login_register_ui()
    else:
        st.success(f"✅ Welcome {st.session_state['username']} 🎉")
        if st.button("Launch Diabetes Predictor App"):
            webbrowser.open_new_tab("http://127.0.0.1:5000")  # your Flask app URL

        if st.button("Logout"):
            logout()
            st.rerun()


if __name__ == "__main__":
    main()
