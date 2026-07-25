"""AI Litigation Copilot - Streamlit frontend.

Pure API client: talks to the FastAPI backend over HTTP and imports nothing
from the backend codebase. If this UI can render it, any client can.

Run:
    streamlit run frontend/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("LITIGATION_API_URL", "http://localhost:8000")
POLL_SECONDS = 1.0

STAGE_LABELS = {
    "index": "Indexando documento (RAG)",
    "classify": "Classificando a acao",
    "extract": "Extraindo dados estruturados",
    "analyze": "Analise juridica",
    "enrich": "Validando no DataJud",
    "risk": "Avaliando riscos",
    "strategy": "Elaborando estrategia",
    "compose": "Montando relatorio",
}
RISK_LABELS = {"low": "Baixo", "medium": "Medio", "high": "Alto", "critical": "Critico"}
PRIORITY_ICONS = {
    "urgent": ":red[URGENTE]",
    "high": ":orange[Alta]",
    "medium": ":blue[Media]",
    "low": ":gray[Baixa]",
}

st.set_page_config(
    page_title="AI Litigation Copilot", page_icon=":material/balance:", layout="wide"
)


# ---------------------------------------------------------------- helpers
def api_get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", timeout=30, **kwargs)


def confidence_badge(conclusion: dict) -> str:
    pct = round(conclusion.get("confidence", 0) * 100)
    color = "green" if pct >= 75 else "orange" if pct >= 50 else "red"
    return f":{color}[{pct}% de confianca]"


def render_conclusion(conclusion: dict, key_prefix: str) -> None:
    st.markdown(f"**{conclusion['statement']}**  {confidence_badge(conclusion)}")
    with st.expander("Por que? (justificativa e fontes)"):
        st.write(conclusion.get("reasoning", ""))
        for citation in conclusion.get("citations", []):
            page = f" (p. {citation['page']})" if citation.get("page") else ""
            st.caption(f'Fonte: "{citation["quote"]}"{page}')


def render_stages(stages: list[dict], container) -> None:
    icons = {
        "done": ":material/check_circle:",
        "running": ":material/progress_activity:",
        "pending": ":material/radio_button_unchecked:",
        "failed": ":material/error:",
    }
    lines = [
        f"{icons[s['state']]} {STAGE_LABELS.get(s['name'], s['name'])}"
        for s in stages
    ]
    container.markdown("\n\n".join(lines))


@st.fragment(run_every=POLL_SECONDS)
def poll_analysis(job_id: str) -> None:
    """Refresh only job progress instead of blocking the whole Streamlit app."""
    try:
        status = api_get(f"/analyses/{job_id}").json()
    except requests.RequestException as exc:
        st.error(f"Falha ao acompanhar a analise: {exc}")
        return

    with st.container(border=True):
        render_stages(status["stages"], st)

    if status["state"] == "failed":
        st.session_state["job_error"] = "; ".join(status["errors"])
        st.rerun()
    if status["state"] in ("succeeded", "partial"):
        try:
            response = api_get(f"/analyses/{job_id}/report")
            response.raise_for_status()
            st.session_state["report"] = response.json()
            st.session_state.pop("job_error", None)
        except requests.RequestException as exc:
            st.session_state["job_error"] = f"Falha ao carregar relatorio: {exc}"
        st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚖️ Litigation Copilot")
    st.caption("Analise inicial de acoes judiciais com IA explicavel")
    try:
        api_get("/health").raise_for_status()
        st.success("API conectada")
        totals = api_get("/runs/totals").json()
        col1, col2 = st.columns(2)
        col1.metric("Analises", totals["runs"])
        col2.metric("Custo total", f"US$ {totals['total_cost_usd']:.3f}")
        if totals["runs"]:
            with st.expander("Historico de execucoes"):
                for run in api_get("/runs").json():
                    status = "✅" if run["success"] else "❌"
                    st.caption(
                        f"{status} {run['filename']} · "
                        f"{run['metrics']['total_tokens']} tokens · "
                        f"US$ {run['metrics']['total_cost_usd']:.4f}"
                    )
    except requests.RequestException:
        st.error(f"API indisponivel em {API_URL}")
        st.caption("Inicie com: `uvicorn app.api.main:app`")
        st.stop()

    st.divider()
    st.caption(
        "Relatorios gerados por IA como apoio a decisao. "
        "Nao substituem a analise de um advogado."
    )

# ---------------------------------------------------------------- main
st.title("Analise de peticao inicial")
with st.form("analysis_upload", border=False):
    uploaded = st.file_uploader("Envie o PDF da peticao inicial", type=["pdf"])
    submitted = st.form_submit_button(
        "Analisar", type="primary", icon=":material/analytics:", width="stretch"
    )

if uploaded and submitted:
    try:
        response = requests.post(
            f"{API_URL}/analyses",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Falha no upload: {exc}")
        st.stop()
    st.session_state["job_id"] = response.json()["job_id"]
    st.session_state.pop("report", None)
    st.session_state.pop("job_error", None)

if job_id := st.session_state.get("job_id"):
    if "report" not in st.session_state:
        st.subheader("Pipeline de agentes")
        if error := st.session_state.get("job_error"):
            st.error("A analise falhou: " + error)
            st.stop()
        poll_analysis(job_id)
        st.stop()

    report = st.session_state["report"]

    # ------------------------------------------------ header metrics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confianca agregada", f"{round(report['confidence_level'] * 100)}%")
    lawsuit_type = (report.get("classification") or {}).get("lawsuit_type", "-")
    col2.metric("Tipo de acao", lawsuit_type)
    col3.metric("Custo", f"US$ {report['metrics']['total_cost_usd']:.4f}")
    col4.metric("Tokens", report["metrics"]["total_tokens"])

    for warning in report.get("warnings", []):
        st.warning(warning)

    # ------------------------------------------------ tabs
    tab_summary, tab_risk, tab_strategy, tab_details, tab_ai = st.tabs(
        ["Resumo", "Riscos", "Estrategia", "Detalhes", "Explicabilidade"]
    )

    with tab_summary:
        st.subheader("Resumo Executivo")
        st.write(report["executive_summary"] or "(indisponivel)")
        if classification := report.get("classification"):
            render_conclusion(classification["conclusion"], "cls")
        if timeline := report.get("timeline"):
            st.subheader("Linha do Tempo")
            for event in timeline:
                st.markdown(f"- **{event.get('date') or 's/ data'}** - {event['description']}")
        if parties := (report.get("parties") or {}).get("parties"):
            st.subheader("Partes")
            st.table(
                [
                    {"Papel": p["role"], "Nome": p["name"], "Advogado": p.get("lawyer") or "-"}
                    for p in parties
                ]
            )

    with tab_risk:
        if risk := report.get("legal_risks"):
            level = risk["overall_level"]
            st.subheader(f"Nivel geral: {RISK_LABELS.get(level, level)}")
            render_conclusion(risk["overall"], "risk-overall")
            for i, item in enumerate(risk.get("risks", [])):
                with st.container(border=True):
                    st.markdown(f"**{item['title']}**")
                    st.caption(
                        f"Risco: {RISK_LABELS.get(item['level'], item['level'])}"
                    )
                    if item.get("financial_exposure"):
                        st.caption(f"Exposicao: {item['financial_exposure']}")
                    render_conclusion(item["conclusion"], f"risk-{i}")
        else:
            st.info("Avaliacao de risco indisponivel nesta execucao.")

    with tab_strategy:
        if strategy := report.get("suggested_strategy"):
            st.subheader("Abordagem recomendada")
            render_conclusion(strategy["overall_approach"], "strat")
            if defenses := strategy.get("defenses"):
                st.subheader("Linhas de defesa")
                for i, defense in enumerate(defenses):
                    basis = f" — {defense['legal_basis']}" if defense.get("legal_basis") else ""
                    st.markdown(f"**{defense['argument']}**{basis}")
                    render_conclusion(defense["assessment"], f"def-{i}")
            if settlement := report.get("possible_settlement"):
                st.subheader("Acordo")
                render_conclusion(settlement, "settle")
            if actions := strategy.get("next_actions"):
                st.subheader("Proximas acoes")
                for action in actions:
                    icon = PRIORITY_ICONS.get(action["priority"], action["priority"])
                    st.markdown(f"- {icon} {action['action']} — {action['rationale']}")
        else:
            st.info("Estrategia indisponivel nesta execucao.")

    with tab_details:
        if claims := report.get("main_claims"):
            st.subheader("Pedidos analisados")
            for i, claim in enumerate(claims):
                basis = f" (base legal: {claim['legal_basis']})" if claim.get("legal_basis") else ""
                st.markdown(f"**{claim['claim']}**{basis}")
                render_conclusion(claim["assessment"], f"claim-{i}")
        if evidence := report.get("evidence_found"):
            st.subheader("Provas identificadas")
            for i, item in enumerate(evidence):
                render_conclusion(item, f"ev-{i}")
        if missing := report.get("missing_information"):
            st.subheader("Informacoes ausentes")
            for item in missing:
                st.markdown(f"- ⚠️ {item}")

    with tab_ai:
        st.subheader("Como a IA chegou a estas conclusoes")
        st.text(report["ai_reasoning"])
        st.subheader("Execucao por agente")
        st.table(
            [
                {
                    "Agente": t["agent"],
                    "Status": t["status"],
                    "Duracao (ms)": round(t["duration_ms"]),
                    "Tokens": (t.get("llm_meta") or {}).get("usage", {}).get(
                        "prompt_tokens", 0
                    )
                    + (t.get("llm_meta") or {}).get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                    "Prompt": (t.get("llm_meta") or {}).get("prompt_version") or "-",
                }
                for t in report.get("traces", [])
            ]
        )

    # ------------------------------------------------ downloads
    st.divider()
    import json

    col_md, col_pdf, col_docx, col_json = st.columns(4)
    doc_id = report["doc_id"]
    col_md.download_button(
        "Baixar Markdown",
        data=api_get(f"/analyses/{job_id}/report.md").text,
        file_name=f"relatorio_{doc_id}.md",
        mime="text/markdown",
        width="stretch",
    )
    col_pdf.download_button(
        "Baixar PDF",
        data=api_get(f"/analyses/{job_id}/report.pdf").content,
        file_name=f"relatorio_{doc_id}.pdf",
        mime="application/pdf",
        width="stretch",
    )
    col_docx.download_button(
        "Baixar DOCX",
        data=api_get(f"/analyses/{job_id}/report.docx").content,
        file_name=f"relatorio_{doc_id}.docx",
        mime="application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document",
        width="stretch",
    )
    col_json.download_button(
        "Baixar JSON",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=f"relatorio_{doc_id}.json",
        mime="application/json",
        width="stretch",
    )
