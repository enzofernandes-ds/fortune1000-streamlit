import logging
logging.disable(logging.WARNING)  # esconde os avisos do Streamlit no "modo bare"

import csv
import io
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors

# LEITURA E LIMPEZA DA BASE DE DADOS

# base de dados hospedada no GitHub (qualquer pessoa consegue rodar sem precisar do arquivo local)
URL_CSV = "https://raw.githubusercontent.com/enzofernandes-ds/fortune1000-streamlit/main/Fortune_1000_Companies_by_Revenue.csv"


@st.cache_data(show_spinner=False)
def baixar_csv(url):
    with urllib.request.urlopen(url) as resposta:
        return resposta.read().decode("utf-8")


leitor_bruto = csv.reader(io.StringIO(baixar_csv(URL_CSV), newline=""))
cabecalho = [c.strip() for c in next(leitor_bruto)]

linhas = []
for linha in leitor_bruto:
    if not linha:
        continue
    # cada linha veio como um unico campo (string CSV aninhada)
    campo_unico = linha[0]
    campos = next(csv.reader(io.StringIO(campo_unico)))

    # correcao de um caso especifico: quando o rank e "1,000",
    # a virgula de milhar quebra o campo em dois pedacos ("1" e "000")
    if len(campos) == len(cabecalho) + 1:
        campos = [campos[0] + campos[1]] + campos[2:]

    linhas.append(campos)
 
 
def limpar_numero(valor: str):
    """Remove $, %, parenteses (negativos), virgulas de milhar e
    espacos, retornando um float. Valores vazios ou '-' viram NaN."""
    if valor is None:
        return np.nan
    v = valor.strip()
    if v in ("", "-", "N/A", "NA"):
        return np.nan
 
    negativo = v.startswith("(") and v.endswith(")")
    v = v.replace("(", "").replace(")", "")
    v = v.replace("$", "").replace("%", "").replace(",", "")
    v = v.strip()
 
    try:
        numero = float(v)
    except ValueError:
        return np.nan
 
    return -numero if negativo else numero

# ==========================================================
# MONTAGEM DO DATAFRAME
# ==========================================================

df = pd.DataFrame(linhas, columns=cabecalho)
df["name"] = df["name"].str.strip()

colunas_numericas = [
    "rank",
    "revenues",
    "revenue_percent_change",
    "profits",
    "profits_percent_change",
    "assets",
    "market_value",
    "employees",
    "change_in_rank",
]
for coluna in colunas_numericas:
    df[coluna] = df[coluna].apply(limpar_numero)

df = df.dropna(subset=["revenues"]).sort_values("revenues", ascending=False).reset_index(drop=True)
df_original = df.copy()  # base completa, usada para treinar o SVM independente dos filtros da sidebar

# paleta simples e consistente para todos os gráficos
COR_PRINCIPAL = "#0B5FFF"
COR_SECUNDARIA = "#D9D9D9"
COR_POSITIVA = "#1E9E5A"
COR_NEGATIVA = "#D64545"


def formatar_bilhoes(x, pos):
    return f"${x / 1000:,.0f} bi"


# ==========================================================
# STREAMLIT - LAYOUT DO DASHBOARD
# ==========================================================

st.set_page_config(page_title="Fortune 1000 - Dashboard", layout="wide")

# ==========================================================
# FILTROS INTERATIVOS (SIDEBAR)
# ==========================================================
st.sidebar.header("Filtros")

receita_min, receita_max = float(df["revenues"].min()), float(df["revenues"].max())
faixa_receita = st.sidebar.slider(
    "Faixa de receita (US$ milhões)",
    min_value=receita_min,
    max_value=receita_max,
    value=(receita_min, receita_max),
)

func_min, func_max = float(df["employees"].min(skipna=True)), float(df["employees"].max(skipna=True))
faixa_funcionarios = st.sidebar.slider(
    "Faixa de número de funcionários",
    min_value=func_min,
    max_value=func_max,
    value=(func_min, func_max),
)

busca_nome = st.sidebar.text_input("Buscar empresa pelo nome")

df_filtrado = df[df["revenues"].between(*faixa_receita) & df["employees"].between(*faixa_funcionarios)]
if busca_nome:
    df_filtrado = df_filtrado[df_filtrado["name"].str.contains(busca_nome, case=False, na=False)]

st.sidebar.caption(f"{len(df_filtrado)} de {len(df)} empresas selecionadas")

if df_filtrado.empty:
    st.warning("Nenhuma empresa encontrada com os filtros selecionados.")
    st.stop()

df = df_filtrado  # a partir daqui, todos os gráficos já usam a base filtrada

st.title("Dashboard Fortune 1000")
st.caption(f"Base com {len(df):,} empresas analisadas".replace(",", "."))

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Receita total (US$ mi)", f"{df['revenues'].sum():,.0f}".replace(",", "."))
col_kpi2.metric("Lucro total (US$ mi)", f"{df['profits'].sum():,.0f}".replace(",", "."))
col_kpi3.metric("Empregos gerados", f"{df['employees'].sum():,.0f}".replace(",", "."))

st.divider()

# ==========================================================
# PAINEL INTERATIVO (EM LINHA)
# ==========================================================
st.markdown("#### 🔎 Explorar empresa")

col_sel, col_rec, col_luc, col_mar, col_func, col_rank = st.columns([2, 1, 1, 1, 1, 1])

empresa_escolhida = col_sel.selectbox(
    "Selecione uma empresa",
    options=df.sort_values("name")["name"],
)
linha_empresa = df[df["name"] == empresa_escolhida].iloc[0]

margem_empresa = (
    linha_empresa["profits"] / linha_empresa["revenues"] * 100
    if pd.notna(linha_empresa["revenues"]) and pd.notna(linha_empresa["profits"]) and linha_empresa["revenues"] != 0
    else None
)

col_rec.metric("Receita (US$ mi)", f"{linha_empresa['revenues']:,.0f}".replace(",", "."))
col_luc.metric("Lucro (US$ mi)", f"{linha_empresa['profits']:,.0f}".replace(",", ".") if pd.notna(linha_empresa["profits"]) else "N/D")
col_mar.metric("Margem de lucro", f"{margem_empresa:.1f}%" if margem_empresa is not None else "N/D")
col_func.metric("Funcionários", f"{linha_empresa['employees']:,.0f}".replace(",", ".") if pd.notna(linha_empresa["employees"]) else "N/D")
col_rank.metric("Var. no ranking", f"{linha_empresa['change_in_rank']:+.0f} pos." if pd.notna(linha_empresa["change_in_rank"]) else "sem registro")

st.markdown("#### 🏆 Líder por métrica")

metricas_disponiveis = {
    "Receita": "revenues",
    "Lucro": "profits",
    "Funcionários": "employees",
    "Valor de mercado": "market_value",
    "Ativos": "assets",
}
col_metrica, col_lider, col_valor = st.columns([2, 1, 1])
metrica_escolhida = col_metrica.selectbox("Classificar líder por", options=list(metricas_disponiveis.keys()))
coluna_metrica = metricas_disponiveis[metrica_escolhida]

df_metrica = df.dropna(subset=[coluna_metrica])
if df_metrica.empty:
    col_lider.info("Sem dados para essa métrica com o filtro atual.")
else:
    lider_metrica = df_metrica.loc[df_metrica[coluna_metrica].idxmax()]
    col_lider.metric(f"Líder em {metrica_escolhida}", lider_metrica["name"])
    col_valor.metric(metrica_escolhida, f"{lider_metrica[coluna_metrica]:,.0f}".replace(",", "."))

st.divider()

linha1_col1, linha1_col2 = st.columns(2)
linha2_col1, linha2_col2 = st.columns(2)
linha3_col1, linha3_col2 = st.columns(2)

# ----------------------------------------------------------
# GRÁFICO 1 — Barras horizontais: Top 15 empresas por receita
# ----------------------------------------------------------
with linha1_col1:
    st.subheader("Top 15 empresas por receita")

    top15 = df.nlargest(15, "revenues").sort_values("revenues")

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.barh(top15["name"], top15["revenues"], color=COR_PRINCIPAL)
    ax1.set_xlabel("Receita (US$ milhões)")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(formatar_bilhoes))
    ax1.tick_params(axis="y", labelsize=9)
    fig1.tight_layout()
    st.pyplot(fig1)

# ----------------------------------------------------------
# GRÁFICO 2 — Pizza: concentração de receita (Top 10 vs Restante)
# ----------------------------------------------------------
with linha1_col2:
    st.subheader("Concentração de receita")

    receita_top10 = df.nlargest(10, "revenues")["revenues"].sum()
    receita_restante = df["revenues"].sum() - receita_top10

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.pie(
        [receita_top10, receita_restante],
        labels=["Top 10 empresas", "Demais ~990 empresas"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[COR_PRINCIPAL, COR_SECUNDARIA],
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax2.axis("equal")
    fig2.tight_layout()
    st.pyplot(fig2)

# ----------------------------------------------------------
# GRÁFICO 3 — Barras: Top 10 empresas por número de funcionários
# ----------------------------------------------------------
with linha2_col1:
    st.subheader("Top 10 empresas por número de funcionários")

    top10_emp = df.nlargest(10, "employees").sort_values("employees")

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.barh(top10_emp["name"], top10_emp["employees"], color=COR_PRINCIPAL)
    ax3.set_xlabel("Número de funcionários")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x / 1000:,.0f} mil"))
    ax3.tick_params(axis="y", labelsize=9)
    fig3.tight_layout()
    st.pyplot(fig3)

# ----------------------------------------------------------
# GRÁFICO 4 — Barras: margem de lucro das mesmas Top 15 (receita x lucratividade)
# ----------------------------------------------------------
with linha2_col2:
    st.subheader("As gigantes de receita também são as mais lucrativas?")

    top15_margem = top15.copy()
    top15_margem["margem"] = top15_margem["profits"] / top15_margem["revenues"] * 100
    top15_margem = top15_margem.sort_values("margem")

    cores_margem = [COR_POSITIVA if m >= 0 else COR_NEGATIVA for m in top15_margem["margem"]]

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    ax4.barh(top15_margem["name"], top15_margem["margem"], color=cores_margem)
    ax4.axvline(0, color="black", linewidth=1)
    ax4.set_xlabel("Margem de lucro (%) = lucro / receita")
    ax4.tick_params(axis="y", labelsize=9)
    fig4.tight_layout()
    st.pyplot(fig4)

    melhor = top15_margem.iloc[-1]
    pior = top15_margem.iloc[0]
    st.caption(
        f"Entre as 15 maiores receitas, {melhor['name']} tem a maior margem "
        f"({melhor['margem']:.1f}%) e {pior['name']} a menor ({pior['margem']:.1f}%) — "
        "receita alta não significa, necessariamente, lucratividade alta."
    )

# ----------------------------------------------------------
# GRÁFICO 5 — Barras divergentes: maiores saltos e quedas no ranking
# ----------------------------------------------------------
with linha3_col1:
    st.subheader("Quem mais subiu e mais caiu no ranking")

    variacao_rank = df.dropna(subset=["change_in_rank"])
    subiram = variacao_rank.nlargest(8, "change_in_rank")
    cairam = variacao_rank.nsmallest(8, "change_in_rank")

    if subiram.empty or cairam.empty:
        st.info("Dados de variação de ranking insuficientes para esse filtro.")
    else:
        kpi5_1, kpi5_2 = st.columns(2)
        kpi5_1.metric("Maior alta", subiram.iloc[0]["name"], f"+{subiram.iloc[0]['change_in_rank']:.0f} posições")
        kpi5_2.metric("Maior queda", cairam.iloc[0]["name"], f"{cairam.iloc[0]['change_in_rank']:.0f} posições")

        movimentos = pd.concat([cairam, subiram]).sort_values("change_in_rank")
        cores_rank = [COR_POSITIVA if v >= 0 else COR_NEGATIVA for v in movimentos["change_in_rank"]]

        fig5, ax5 = plt.subplots(figsize=(6, 5))
        ax5.barh(movimentos["name"], movimentos["change_in_rank"], color=cores_rank)
        ax5.axvline(0, color="black", linewidth=1)
        ax5.set_xlabel("Posições ganhas (+) ou perdidas (-) no ranking")
        ax5.tick_params(axis="y", labelsize=9)
        fig5.tight_layout()
        st.pyplot(fig5)

# ----------------------------------------------------------
# GRÁFICO 6 — Barras: receita por funcionário (eficiência/produtividade)
# ----------------------------------------------------------
with linha3_col2:
    st.subheader("Quem fatura mais por funcionário")

    df_eficiencia = df.dropna(subset=["employees"])
    df_eficiencia = df_eficiencia[df_eficiencia["employees"] > 0].copy()
    df_eficiencia["receita_por_func"] = df_eficiencia["revenues"] * 1_000_000 / df_eficiencia["employees"]
    top10_eficiencia = df_eficiencia.nlargest(10, "receita_por_func").sort_values("receita_por_func")

    if top10_eficiencia.empty:
        st.info("Nenhuma empresa com dados suficientes para esse filtro.")
    else:
        lider = top10_eficiencia.iloc[-1]
        kpi6_1, kpi6_2 = st.columns(2)
        kpi6_1.metric("Maior receita/funcionário", lider["name"], f"${lider['receita_por_func']/1e6:,.1f} mi")
        kpi6_2.metric("Mediana (Top 10)", f"${top10_eficiencia['receita_por_func'].median()/1e6:,.1f} mi")

        fig6, ax6 = plt.subplots(figsize=(6, 5))
        ax6.barh(top10_eficiencia["name"], top10_eficiencia["receita_por_func"], color=COR_PRINCIPAL)
        ax6.set_xlabel("Receita por funcionário (US$)")
        ax6.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"${x/1e6:,.1f} mi"))
        ax6.tick_params(axis="y", labelsize=9)
        fig6.tight_layout()
        st.pyplot(fig6)

st.divider()
st.header("Classificação com SVM — a empresa tem margem de lucro alta?")
st.caption(
    "Modelo treinado sobre a base completa (não filtrada) para prever se uma empresa está "
    "acima ou abaixo da mediana de margem de lucro, com base em porte e desempenho financeiro."
)

features_svm = ["revenues", "assets", "market_value", "employees", "profits_percent_change"]
base_svm = df_original.dropna(subset=features_svm + ["profits"]).copy()
base_svm["margem"] = base_svm["profits"] / base_svm["revenues"] * 100
mediana_margem = base_svm["margem"].median()
base_svm["alta_margem"] = (base_svm["margem"] >= mediana_margem).astype(int)

X = base_svm[features_svm]
y = base_svm["alta_margem"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler_svm = StandardScaler()
X_train_s = scaler_svm.fit_transform(X_train)
X_test_s = scaler_svm.transform(X_test)

modelo_svm = SVC(kernel="rbf", C=10, random_state=42, probability=True, class_weight='balanced')
modelo_svm.fit(X_train_s, y_train)

pred_svm = modelo_svm.predict(X_test_s)
acuracia = accuracy_score(y_test, pred_svm)
matriz_confusao = confusion_matrix(y_test, pred_svm)

col_svm1, col_svm2, col_svm3 = st.columns(3)
col_svm1.metric("Acurácia do modelo", f"{acuracia*100:.1f}%")
col_svm2.metric("Mediana de margem (corte)", f"{mediana_margem:.1f}%")
col_svm3.metric("Empresas no treino/teste", f"{len(base_svm)}")

col_matriz, col_simulador = st.columns(2)

with col_matriz:
    st.subheader("Matriz de confusão")
    fig_svm, ax_svm = plt.subplots(figsize=(5, 4))
    ax_svm.imshow(matriz_confusao, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax_svm.text(j, i, matriz_confusao[i, j], ha="center", va="center", fontsize=14)
    ax_svm.set_xticks([0, 1])
    ax_svm.set_yticks([0, 1])
    ax_svm.set_xticklabels(["Baixa margem", "Alta margem"])
    ax_svm.set_yticklabels(["Baixa margem", "Alta margem"])
    ax_svm.set_xlabel("Previsto")
    ax_svm.set_ylabel("Real")
    fig_svm.tight_layout()
    st.pyplot(fig_svm)

with col_simulador:
    st.subheader("Simular uma empresa")

    receita_sim = st.number_input("Receita (US$ milhões)", min_value=0.0, value=50000.0, step=1000.0)
    ativos_sim = st.number_input("Ativos (US$ milhões)", min_value=0.0, value=50000.0, step=1000.0)
    valor_mercado_sim = st.number_input("Valor de mercado (US$ milhões)", min_value=0.0, value=50000.0, step=1000.0)
    funcionarios_sim = st.number_input("Número de funcionários", min_value=0.0, value=50000.0, step=1000.0)
    crescimento_lucro_sim = st.number_input("Variação do lucro (%)", value=5.0, step=1.0)

    entrada_sim = pd.DataFrame(
        [[receita_sim, ativos_sim, valor_mercado_sim, funcionarios_sim, crescimento_lucro_sim]],
        columns=features_svm,
    )
    entrada_sim_s = scaler_svm.transform(entrada_sim)
    previsao_sim = modelo_svm.predict(entrada_sim_s)[0]

    if previsao_sim == 1:
        st.success(f"O modelo prevê **alta margem de lucro** (acima de {mediana_margem:.1f}%).")
    else:
        st.warning(f"O modelo prevê **baixa margem de lucro** (abaixo de {mediana_margem:.1f}%).")

st.markdown("---")
st.subheader("Avaliação mais completa do modelo")

relatorio = classification_report(
    y_test, pred_svm, target_names=["Baixa margem", "Alta margem"], output_dict=True
)
df_relatorio = pd.DataFrame(relatorio).transpose().round(2)
st.dataframe(df_relatorio, width="stretch")

dummy_svm = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_svm.fit(X_train_s, y_train)
acuracia_baseline = accuracy_score(y_test, dummy_svm.predict(X_test_s))

scores_cv = cross_val_score(
    SVC(kernel="rbf", C=10, random_state=42),
    scaler_svm.fit_transform(X), y, cv=5
)

col_svm4, col_svm5 = st.columns(2)
col_svm4.metric(
    "Modelo vs. baseline ingênuo",
    f"{acuracia*100:.1f}%",
    f"+{(acuracia - acuracia_baseline)*100:.1f} p.p. vs. {acuracia_baseline*100:.1f}%",
)
col_svm5.metric(
    "Validação cruzada (5 divisões)",
    f"{scores_cv.mean()*100:.1f}%",
    f"± {scores_cv.std()*100:.1f} p.p.",
)

col_roc, col_importancia = st.columns(2)

with col_roc:
    st.subheader("Curva ROC")
    proba_svm = modelo_svm.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_svm)
    auc_svm = roc_auc_score(y_test, proba_svm)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color=COR_PRINCIPAL, linewidth=2, label=f"SVM (AUC = {auc_svm:.2f})")
    ax_roc.plot([0, 1], [0, 1], color="#999999", linestyle="--", label="Chute aleatório")
    ax_roc.set_xlabel("Taxa de falsos positivos")
    ax_roc.set_ylabel("Taxa de verdadeiros positivos")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    st.pyplot(fig_roc)

with col_importancia:
    st.subheader("Importância das variáveis")
    importancia = permutation_importance(
        modelo_svm, X_test_s, y_test, n_repeats=20, random_state=42
    )
    df_importancia = pd.DataFrame(
        {"variavel": features_svm, "importancia": importancia.importances_mean}
    ).sort_values("importancia")

    fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
    ax_imp.barh(df_importancia["variavel"], df_importancia["importancia"], color=COR_PRINCIPAL)
    ax_imp.set_xlabel("Queda de acurácia ao embaralhar a variável")
    fig_imp.tight_layout()
    st.pyplot(fig_imp)

st.caption(
    f"O modelo supera o baseline ingênuo ({acuracia_baseline*100:.1f}%) e mantém desempenho "
    f"consistente na validação cruzada ({scores_cv.mean()*100:.1f}% ± {scores_cv.std()*100:.1f} p.p.). "
    "O valor de mercado é a variável que mais influencia a previsão da margem de lucro."
)

st.markdown("---")
st.subheader("Dispersão das previsões do modelo")

dados_dispersao = base_svm.loc[X_test.index].copy()
dados_dispersao["previsto"] = pred_svm
dados_dispersao["erro"] = dados_dispersao["previsto"] != dados_dispersao["alta_margem"]

fig_disp, ax_disp = plt.subplots(figsize=(9, 5))

for classe, cor, rotulo in [
    (0, COR_NEGATIVA, "Previsto: baixa margem"),
    (1, COR_POSITIVA, "Previsto: alta margem"),
]:
    grupo = dados_dispersao[dados_dispersao["previsto"] == classe]
    ax_disp.scatter(
        grupo["market_value"], grupo["margem"],
        c=cor, label=rotulo, alpha=0.75, edgecolors="white", linewidths=0.5,
    )

erros = dados_dispersao[dados_dispersao["erro"]]
ax_disp.scatter(
    erros["market_value"], erros["margem"],
    facecolors="none", edgecolors="black", s=110, linewidths=1.2, label="Previsão errada",
)

ax_disp.axhline(
    mediana_margem, color="black", linestyle="--", linewidth=1,
    label=f"Corte (mediana = {mediana_margem:.1f}%)",
)
ax_disp.set_xscale("log")
ax_disp.set_ylim(-30, 80)
ax_disp.set_xlabel("Valor de mercado (US$ milhões, escala log)")
ax_disp.set_ylabel("Margem de lucro real (%)")
ax_disp.legend(fontsize=8, loc="upper left")
fig_disp.tight_layout()
st.pyplot(fig_disp)

st.caption(
    f"{int(dados_dispersao['erro'].sum())} erros em {len(dados_dispersao)} empresas de teste. "
    "Os erros (círculos pretos) se concentram perto da linha de corte: o modelo acerta "
    "com folga as empresas claramente acima ou abaixo da mediana e hesita nas que estão no limite."
)

# ==========================================================
# DBSCAN 3D — quais empresas se parecem (e quais fogem do padrão)
# ==========================================================
st.divider()
st.header("Segmentação com DBSCAN — quais empresas se parecem?")
st.caption(
    "O DBSCAN junta empresas que estão 'coladas' umas nas outras no espaço de receita, "
    "margem de lucro e funcionários. Quem não tem vizinhos suficientes por perto não entra em "
    "nenhum grupo e é marcada como atípica. Gire o gráfico com o mouse, use a roda para dar zoom "
    "e clique na legenda para esconder ou mostrar grupos."
)

# --- dados: 3 eixos (receita, margem, funcionários) + cor (grupo encontrado) ---
base_cluster = df_original.dropna(subset=["revenues", "profits", "employees"]).copy()
base_cluster = base_cluster[(base_cluster["revenues"] > 0) & (base_cluster["employees"] > 0)]
base_cluster["margem"] = base_cluster["profits"] / base_cluster["revenues"] * 100
base_cluster["margem_plot"] = base_cluster["margem"].clip(-20, 50)  # limita extremos (gráfico e agrupamento)

dados_db = np.column_stack([
    np.log10(base_cluster["revenues"]),
    base_cluster["margem_plot"],
    np.log10(base_cluster["employees"]),
])
dados_db_s = StandardScaler().fit_transform(dados_db)

col_eps, col_min = st.columns(2)
eps_db = col_eps.slider("eps — raio de vizinhança (maior = grupos mais amplos)", 0.20, 1.00, 0.44, 0.01)
min_amostras_db = col_min.slider("min_samples — vizinhos mínimos para formar um grupo", 3, 20, 5, 1)

base_cluster["cluster"] = DBSCAN(eps=eps_db, min_samples=min_amostras_db).fit_predict(dados_db_s)

# nomes dos grupos: com os parâmetros padrão (5 grupos) cada um recebe um nome pela sua característica marcante;
# com outros valores de eps/min_samples, os grupos recebem só a numeração (Grupo 1, Grupo 2, ...)
NOMES_5_GRUPOS = [
    "Núcleo típico",             # maior grupo: perfil médio da base
    "Enxutas de margem baixa",   # margem mediana ~2% e poucos funcionários
    "Grandes empregadoras",      # centenas de milhares de funcionários
    "Em prejuízo",               # margem mediana muito negativa
    "Gigantes de receita",       # receita mediana acima de US$ 100 bi
]
ordem_grupos = [g for g in base_cluster["cluster"].value_counts().index if g != -1]
usar_nomes = len(ordem_grupos) == 5
nomes_grupos = {-1: "Atípicas"}
for posicao, g in enumerate(ordem_grupos):
    nomes_grupos[g] = NOMES_5_GRUPOS[posicao] if usar_nomes else f"Grupo {posicao + 1}"
base_cluster["grupo"] = base_cluster["cluster"].map(nomes_grupos)

# o gráfico respeita os filtros da sidebar, mas os grupos são calculados na base completa
dados_plot = base_cluster[base_cluster.index.isin(df.index)]

if dados_plot.empty:
    st.info("Nenhuma empresa com dados completos para os filtros atuais.")
else:
    n_grupos = len(ordem_grupos)
    n_atipicas = int((dados_plot["cluster"] == -1).sum())

    col_db1, col_db2, col_db3 = st.columns(3)
    col_db1.metric("Grupos encontrados", n_grupos)
    col_db2.metric("Empresas agrupadas", f"{len(dados_plot) - n_atipicas}")
    col_db3.metric(
        "Empresas atípicas", f"{n_atipicas}",
        f"{n_atipicas / len(dados_plot) * 100:.0f}% do total", delta_color="off",
    )

    cores_grupos = ["#0B5FFF", "#1E9E5A", "#8E5BD0", "#D64545", "#17BECF", "#E377C2", "#BCBD22", "#8C564B"]
    COR_ATIPICAS = "#F5A623"    

    fig_db = go.Figure()

    def adicionar_grupo(sub, nome, cor, simbolo, tamanho, opacidade):
        fig_db.add_trace(go.Scatter3d(
            x=sub["revenues"], y=sub["margem_plot"], z=sub["employees"],
            mode="markers",
            name=f"{nome} ({len(sub)})",
            marker=dict(size=tamanho, color=cor, symbol=simbolo, opacity=opacidade),
            customdata=sub[["name", "margem", "revenues", "employees"]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Receita: US$ %{customdata[2]:,.0f} mi<br>"
                "Margem de lucro: %{customdata[1]:.1f}%<br>"
                "Funcionários: %{customdata[3]:,.0f}"
                f"<extra>{nome}</extra>"
            ),
        ))

    for posicao, g in enumerate(ordem_grupos):
        sub = dados_plot[dados_plot["cluster"] == g]
        if not sub.empty:
            adicionar_grupo(sub, nomes_grupos[g], cores_grupos[posicao % len(cores_grupos)], "circle",
                            3 if posicao == 0 else 5, 0.55 if posicao == 0 else 0.9)

    sub_atipicas = dados_plot[dados_plot["cluster"] == -1]
    if not sub_atipicas.empty:
        adicionar_grupo(sub_atipicas, "Atípicas", COR_ATIPICAS, "diamond", 4, 0.85)

    fig_db.update_layout(
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(title="Grupo", itemsizing="constant"),
        uirevision="dbscan_3d",  # mantém câmera e modo de rotação escolhidos quando o Streamlit reexecuta
        scene=dict(
            xaxis=dict(title="Receita (US$ mi, escala log)", type="log"),
            yaxis=dict(title="Margem de lucro (%)"),
            zaxis=dict(title="Funcionários (escala log)", type="log"),
            dragmode="orbit",  # começa em rotação livre; Turntable e Orbital continuam disponíveis na barra do gráfico
        ),
    )
    
    st.plotly_chart(fig_db, width="stretch", key="grafico_dbscan_3d")

    # perfil de cada grupo: o que cada cor significa
    ordem_legenda = [nomes_grupos[g] for g in ordem_grupos] + ["Atípicas"]
    presentes = set(dados_plot["grupo"])
    perfil = (
        dados_plot.groupby("grupo")
        .agg(
            empresas=("name", "size"),
            receita_mediana=("revenues", "median"),
            margem_mediana=("margem", "median"),
            funcionarios_mediano=("employees", "median"),
            exemplos=("name", lambda s: ", ".join(s.head(4))),
        )
        .reindex([n for n in ordem_legenda if n in presentes])
    )
    perfil["receita_mediana"] = perfil["receita_mediana"].round(0)
    perfil["margem_mediana"] = perfil["margem_mediana"].round(1)
    perfil["funcionarios_mediano"] = perfil["funcionarios_mediano"].round(0)
    perfil.columns = [
        "Empresas", "Receita mediana (US$ mi)", "Margem mediana (%)",
        "Funcionários (mediana)", "Maiores exemplos",
    ]
    st.dataframe(perfil, width="stretch")

    st.caption(
        f"Com eps={eps_db:.2f} e min_samples={min_amostras_db}, o DBSCAN encontrou {n_grupos} grupo(s) e deixou "
        f"{n_atipicas} empresas ({n_atipicas / len(dados_plot) * 100:.0f}% do total) como atípicas. "
        "Margens abaixo de -20% ou acima de 50% foram limitadas a esses valores apenas para o gráfico e o agrupamento. "
        "Os nomes dos grupos valem para eps=0.44 e min_samples=5; com outros valores, os grupos são apenas numerados."
    )

# ==========================================================
# COTOVELO (k-distance) E GAP PARA O DBSCAN
# ==========================================================
st.divider()
st.header("Escolha dos parâmetros do DBSCAN — método do cotovelo e gap")
st.caption(
    "Dois testes para checar se o eps escolhido faz sentido: o cotovelo mostra a partir de qual distância "
    "as empresas deixam de ter vizinhos próximos, e o gap compara os grupos encontrados nos dados reais "
    "com os que o DBSCAN acharia em dados aleatórios."
)

# --- 1) cotovelo: distância de cada empresa até o seu vizinho nº min_samples, em ordem crescente ---
distancias, _ = NearestNeighbors(n_neighbors=min_amostras_db).fit(dados_db_s).kneighbors(dados_db_s)
k_dist = np.sort(distancias[:, -1])

# cotovelo = ponto da curva mais distante da reta que liga o primeiro ao último ponto
pos_x = np.linspace(0, 1, len(k_dist))
pos_y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min())
i_cotovelo = int(np.argmax(pos_x - pos_y))
eps_cotovelo = float(k_dist[i_cotovelo])
grupos_no_cotovelo = len(set(DBSCAN(eps=eps_cotovelo, min_samples=min_amostras_db).fit_predict(dados_db_s)) - {-1})


# --- 2) gap adaptado ao DBSCAN: compara a dispersão dos grupos reais com a de dados aleatórios ---
@st.cache_data(show_spinner="Calculando o gap (pode levar alguns segundos)...")
def calcular_gap_dbscan(dados, min_amostras, eps_lista, n_ref=10, semente=42):
    def dispersao(X, rotulos):
        # soma dos quadrados dentro de cada grupo (o ruído conta como um grupo à parte)
        total = 0.0
        for r in np.unique(rotulos):
            g = X[rotulos == r]
            total += ((g - g.mean(axis=0)) ** 2).sum()
        return total

    rng = np.random.default_rng(semente)
    refs = [rng.uniform(dados.min(axis=0), dados.max(axis=0), size=dados.shape) for _ in range(n_ref)]

    linhas = []
    for eps in eps_lista:
        rot = DBSCAN(eps=eps, min_samples=min_amostras).fit_predict(dados)
        log_w = np.log(dispersao(dados, rot))
        log_w_ref = np.array([
            np.log(dispersao(r, DBSCAN(eps=eps, min_samples=min_amostras).fit_predict(r))) for r in refs
        ])
        linhas.append({
            "eps": eps,
            "grupos": len(set(rot) - {-1}),
            "gap": log_w_ref.mean() - log_w,
            "erro": log_w_ref.std() * np.sqrt(1 + 1 / n_ref),
        })
    return pd.DataFrame(linhas)


tabela_gap = calcular_gap_dbscan(dados_db_s, min_amostras_db, np.round(np.arange(0.20, 0.801, 0.02), 2))
linha_max = tabela_gap.loc[tabela_gap["gap"].idxmax()]

col_cotovelo, col_gap = st.columns(2)

with col_cotovelo:
    st.subheader("Método do cotovelo (k-distance)")

    fig_cot, ax_cot = plt.subplots(figsize=(6, 4.5))
    ax_cot.plot(k_dist, color=COR_PRINCIPAL, linewidth=2)
    ax_cot.axhline(eps_cotovelo, color=COR_NEGATIVA, linestyle="--", linewidth=1,
                   label=f"Cotovelo: eps ≈ {eps_cotovelo:.2f}")
    ax_cot.axhline(eps_db, color="#999999", linestyle=":", linewidth=1.5, label=f"eps atual = {eps_db:.2f}")
    ax_cot.scatter([i_cotovelo], [eps_cotovelo], color=COR_NEGATIVA, zorder=3)
    ax_cot.set_ylim(0, np.percentile(k_dist, 99) * 1.3)
    ax_cot.set_xlabel("Empresas ordenadas pela distância")
    ax_cot.set_ylabel(f"k-distance (k = min_samples = {min_amostras_db})")
    ax_cot.legend(fontsize=8, loc="upper left")
    fig_cot.tight_layout()
    st.pyplot(fig_cot)

    posicao_eps = "abaixo" if eps_db < eps_cotovelo else "acima"
    efeito_eps = ("mais sensível: mais grupos pequenos e mais empresas atípicas" if eps_db < eps_cotovelo
                  else "mais tolerante: grupos mais amplos e menos empresas atípicas")
    st.caption(
        f"Até o cotovelo, as distâncias crescem devagar (empresas com vizinhos próximos); depois dele disparam "
        f"(empresas isoladas). O cotovelo indica eps ≈ {eps_cotovelo:.2f}, valor com o qual o DBSCAN encontra "
        f"{grupos_no_cotovelo} grupo(s). Seu eps atual ({eps_db:.2f}) está {posicao_eps} do cotovelo: {efeito_eps}."
    )

with col_gap:
    st.subheader("Estatística de gap (adaptada ao DBSCAN)")

    fig_gap, ax_gap = plt.subplots(figsize=(6, 4.5))
    ax_k = ax_gap.twinx()
    ax_k.step(tabela_gap["eps"], tabela_gap["grupos"], where="mid", color="#BBBBBB", linewidth=1.5)
    ax_k.set_ylabel("Nº de grupos encontrados (cinza)")
    ax_gap.set_zorder(ax_k.get_zorder() + 1)  # gap por cima da linha cinza
    ax_gap.patch.set_visible(False)

    ax_gap.errorbar(tabela_gap["eps"], tabela_gap["gap"], yerr=tabela_gap["erro"], color=COR_PRINCIPAL,
                    marker="o", markersize=4, linewidth=2, capsize=2, label="Gap")
    ax_gap.axvline(linha_max["eps"], color=COR_POSITIVA, linestyle="--", linewidth=1,
                   label=f"Maior gap: eps = {linha_max['eps']:.2f} ({int(linha_max['grupos'])} grupos)")
    ax_gap.axvline(eps_db, color="#999999", linestyle=":", linewidth=1.5, label=f"eps atual = {eps_db:.2f}")
    ax_gap.set_xlabel("eps")
    ax_gap.set_ylabel("Gap (maior = grupos mais compactos que o acaso)")
    ax_gap.legend(fontsize=8, loc="lower left")
    fig_gap.tight_layout()
    st.pyplot(fig_gap)

    gap_atual = tabela_gap.loc[(tabela_gap["eps"] - eps_db).abs().idxmin()]
    st.caption(
        f"O gap é maior em eps = {linha_max['eps']:.2f} ({int(linha_max['grupos'])} grupos); no seu eps atual "
        f"({eps_db:.2f}) o gap é {gap_atual['gap']:.2f} e o DBSCAN acha {int(gap_atual['grupos'])} grupos. "
        "Adaptação: o ruído conta como um grupo à parte e o gap é calculado para vários valores de eps "
        "(e não para vários k, como no k-means), comparando com 10 conjuntos aleatórios."
    )

