# ========================= UI (SIDEBAR) =========================

cfg0 = default_config()
_init_state()

with st.sidebar:
    st.header("Įvestys")

    # --- layout-changing controls OUTSIDE form (instant updates) ---
    data_source = st.radio(
        "Duomenų šaltinis",
        options=["Integruoti scenarijai", "Įkelti failus"],
        index=0,
        help="Jeigu pasirinksite integruotus scenarijus, naudotojui nereikės įkelti failų.",
    )

    uploads = []
    if data_source == "Įkelti failus":
        st.markdown("Įkelkite scenarijų failus")
        uploads = st.file_uploader(
            "Scenarijų failai",
            type=["csv", "txt"],
            accept_multiple_files=True,
            help="Pasirinkite *.csv / *.txt failus. Galite pažymėti kelis failus iš karto.",
        )
    else:
        st.markdown("Naudojami integruoti scenarijai iš sistemos.")

    mode = st.radio("Peržiūros režimas", options=["Scenarijus", "Įvestis"], index=0)

    # ✅ IMPORTANT: don't pass `value=` when using `key=` (prevents "reverting")
    saving_custom_enabled = st.checkbox(
        "Taikyti sutaupymo vertę (€/100NM)",
        key="saving_custom_enabled",
    )

    # Optional: reset the custom value when unchecked (keeps state clean)
    if not bool(st.session_state.get("saving_custom_enabled", False)):
        st.session_state["saving_custom_value"] = float(st.session_state.get("saving_custom_value", 2.0) or 2.0)

    # --- stable controls INSIDE form (no jitter/shadow; only submits on click) ---
    with st.form("sidebar_inputs", clear_on_submit=False):
        fuel_price = st.number_input(
            "Degalų kaina (€/kg)",
            min_value=0.0,
            value=float(cfg0.fuel_price_eur_per_kg),
            step=0.01,
        )
        tc_op = st.number_input(
            "Laiko sąnaudos (€/h)",
            min_value=0.0,
            value=float(cfg0.time_cost_operational),
            step=100.0,
        )
        epsilon_pct = st.number_input(
            "ECSR epsilon (%)",
            min_value=0.0,
            value=float(cfg0.epsilon_break_even) * 100.0,
            step=0.1,
        )

        # ✅ show only when checked, read from session_state (bulletproof)
        saving_custom = float(st.session_state.get("saving_custom_value", 2.0))
        if bool(st.session_state.get("saving_custom_enabled", False)):
            saving_custom = st.number_input(
                "Sutaupymas (€/100NM)",
                min_value=0.0,
                value=float(st.session_state.get("saving_custom_value", 2.0)),
                step=1.0,
                key="saving_custom_value",
            )

        run_btn = st.form_submit_button("Generuoti", type="primary", use_container_width=True)
