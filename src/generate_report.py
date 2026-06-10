from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "model"
OUTPUT_PATH = REPORTS_DIR / "relatorio_atualizado.pdf"


def table_from_df(df: pd.DataFrame, max_rows: int = 8) -> Table:
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: f"{value:.4f}")
    data = [shown.columns.tolist()] + shown.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def paragraph(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    title = styles["Title"]
    heading = styles["Heading2"]
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    validation = pd.read_csv(REPORTS_DIR / "comparacao_validacao.csv")
    cv = pd.read_csv(REPORTS_DIR / "validacao_cruzada.csv")
    test = pd.read_csv(REPORTS_DIR / "resultado_teste.csv")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    coefficients = pd.read_csv(REPORTS_DIR / "coeficientes_logistic_regression.csv")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    story = [
        Paragraph("Relatorio Atualizado - Previsao de Churn de Clientes", title),
        paragraph(
            "Participantes: Nathan Gabriel da Silva - RA: 2077558; "
            "Taina de Souza Alvez - RA: 2041631; "
            "Julia Soares de Azevedo Lombardi - RA: 2032874.",
            body,
        ),
        Spacer(1, 8),
        Paragraph("1. Introducao", heading),
        paragraph(
            "Este projeto tem como objetivo prever a chance de cancelamento de clientes "
            "de uma empresa de telecomunicacoes. A P2 manteve o mesmo tema e dataset da P1, "
            "mas consolidou as melhorias pedidas no feedback: comparacao mais completa dos "
            "modelos, interpretacao da Regressao Logistica, discussao do trade-off entre "
            "precision e recall, tratamento do desbalanceamento e salvamento do modelo final "
            "para uso em uma aplicacao Streamlit.",
            body,
        ),
        Paragraph("2. Metodologia", heading),
        paragraph(
            "O dataset foi carregado a partir do arquivo data/dataset.csv. A coluna "
            "TotalCharges foi convertida para numerica, registros sem valor nessa coluna "
            "foram removidos, customerID foi descartada e a variavel alvo Churn foi "
            "convertida para 0 e 1. Tambem foi criada a variavel tenure_group, agrupando "
            "o tempo de permanencia do cliente em faixas. As variaveis numericas passaram "
            "por StandardScaler e as categoricas por OneHotEncoder dentro de um Pipeline, "
            "evitando vazamento de dados entre treino e teste.",
            body,
        ),
        paragraph(
            "A divisao foi estratificada em treino, validacao e teste. A estratificacao "
            "foi mantida porque a classe churn e menor do que a classe nao churn. Para "
            "lidar melhor com esse desbalanceamento, a Regressao Logistica e a Random "
            "Forest foram treinadas com class_weight='balanced', e o Gradient Boosting "
            "teve o threshold de decisao ajustado com base na curva precision-recall.",
            body,
        ),
        Paragraph("3. Comparacao dos Modelos", heading),
        paragraph("Resultados no conjunto de validacao:", body),
        table_from_df(validation),
        Spacer(1, 10),
        paragraph("Medias da validacao cruzada estratificada com 5 folds:", body),
        table_from_df(cv),
        Paragraph("4. Modelo Final e Threshold", heading),
        paragraph(
            f"O modelo final mantido foi o {metadata['model_name']}. Ele apresentou bom "
            "desempenho geral em AUC-ROC e foi ajustado com threshold "
            f"{metadata['threshold']:.3f}. Esse ajuste reduz a dependencia do limite "
            "padrao 0.50 e deixa mais clara a relacao entre precision e recall. Em churn, "
            "um recall maior ajuda a encontrar mais clientes em risco, mas tende a aumentar "
            "falsos positivos. Ja uma precision maior torna os alertas mais confiaveis, "
            "mas pode deixar passar clientes que realmente cancelariam.",
            body,
        ),
        paragraph("Resultados finais no conjunto de teste:", body),
        table_from_df(test),
        Paragraph("5. Interpretacao dos Modelos", heading),
        paragraph(
            "A Random Forest foi usada para observar importancia das variaveis. As features "
            "com maior peso reforcam que tempo de permanencia, tipo de contrato, valor mensal "
            "e caracteristicas do plano influenciam diretamente a chance de churn.",
            body,
        ),
        table_from_df(importance[["Feature", "Importance"]], max_rows=10),
        Spacer(1, 10),
        paragraph(
            "A Regressao Logistica foi interpretada pelos coeficientes. Coeficientes "
            "positivos aumentam a probabilidade estimada de churn, enquanto coeficientes "
            "negativos reduzem essa probabilidade. Como os dados passam por preprocessing "
            "no Pipeline, a interpretacao considera as variaveis ja transformadas.",
            body,
        ),
        table_from_df(coefficients[["Feature", "Coefficient"]], max_rows=10),
        PageBreak(),
        Paragraph("6. Aplicacao Streamlit", heading),
        paragraph(
            "A aplicacao app.py carrega o modelo salvo em model/modelo_final.joblib e o "
            "arquivo model/metadata.json, que contem o threshold e a lista de colunas de "
            "entrada. O usuario preenche os dados do cliente, o app organiza essas entradas "
            "em um DataFrame compativel com o Pipeline e exibe a probabilidade de churn, "
            "o threshold usado e a classificacao final.",
            body,
        ),
        Paragraph("7. Limitacoes", heading),
        paragraph(
            "O modelo usa dados historicos e pode refletir padroes especificos do dataset. "
            "A predicao deve ser usada como apoio a decisao, nao como decisao automatica "
            "isolada. Alem disso, a aplicacao simula uma entrada manual; em um ambiente real, "
            "seria necessario integrar o modelo a uma base atualizada de clientes.",
            body,
        ),
        Paragraph("8. Conclusao", heading),
        paragraph(
            "A versao final evolui em relacao a P1 porque conecta notebook, modelo salvo, "
            "relatorio e app em um fluxo unico. O projeto passa a demonstrar nao apenas a "
            "analise e o treinamento, mas tambem a aplicacao pratica do modelo em uma "
            "interface funcional para previsao de churn.",
            body,
        ),
    ]

    doc.build(story)
    print(f"Relatorio gerado em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
