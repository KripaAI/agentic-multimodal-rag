"""Sign-in and change-password screens (spec §7.6, LLD §5.8)."""

from __future__ import annotations

import streamlit as st

from mmrag.auth import service
from mmrag.auth.passwords import PolicyError
from mmrag.config import Settings


def client_ip(settings: Settings) -> str | None:
    """The client IP for throttling. Proxy headers are trusted only behind our own proxy."""
    if settings.auth.trust_proxy_headers:
        forwarded = st.context.headers.get("X-Forwarded-For")
        if isinstance(forwarded, str) and forwarded.split(",")[0].strip():
            return forwarded.split(",")[0].strip()
    ip = getattr(st.context, "ip_address", None)
    return ip if isinstance(ip, str) and ip else "127.0.0.1"  # local run, or no address available


def sign_in_screen(settings: Settings) -> None:
    st.title("Document assistant")
    st.caption("Sign in to ask questions about the PDF library.")
    with st.form("sign_in", clear_on_submit=False):
        email = st.text_input("Email", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if st.session_state.pop("_signed_out_message", None):
        st.info("Password changed. Please sign in with your new password.")
    if submitted:
        result = service.login(settings, email, password, client_ip(settings))
        if result.ok:
            st.session_state.clear()
            st.session_state["token"] = result.token  # only in this browser tab (D10)
            st.rerun()
        st.error(result.message)  # the same message for every failure


def change_password_screen(settings: Settings, forced: bool) -> None:
    st.title("Change your password")
    if forced:
        st.info("You are signed in with a temporary password. Choose your own to continue.")
    st.caption(f"At least {settings.auth.min_password_length} characters. A long passphrase is best; "
               "common passwords are refused.")
    with st.form("change_password"):
        current = st.text_input("Current password", type="password", autocomplete="current-password")
        new = st.text_input("New password", type="password", autocomplete="new-password")
        again = st.text_input("New password again", type="password", autocomplete="new-password")
        submitted = st.form_submit_button("Change password", type="primary")
    if not forced and st.button("Cancel"):
        st.session_state.pop("view", None)
        st.rerun()
    if submitted:
        if new != again:
            st.error("The two new passwords are different.")
            return
        try:
            service.change_password(settings, st.session_state.get("token"), current, new)
        except (ValueError, PolicyError) as e:
            st.error(str(e))
            return
        except PermissionError:
            st.session_state.clear()
            st.rerun()
        st.session_state.clear()  # every session was revoked: sign in again with the new password
        st.session_state["_signed_out_message"] = True
        st.rerun()
