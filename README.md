# Previsao de Churn de Clientes

Projeto avaliativo P2 da disciplina, continuando a P1 com melhorias no notebook, modelo salvo, relatorio atualizado e aplicacao em Streamlit.

## Links da entrega

- Notebook no Google Colab: https://colab.research.google.com/drive/1o-zirN4y6HbMgnEoRvQgthzbLdPgmqhG?usp=sharing
- Repositorio GitHub: https://github.com/NathanGSsilva/previsao-de-churn-de-clientes
- App publicado no Streamlit: https://previsao-de-churn-de-clientes-cqvmgvmekvzrrk83gchdng.streamlit.app/

## Integrantes

- Nathan Gabriel da Silva - RA: 2077558
- Taina de Souza Alvez - RA: 2041631
- Julia Soares de Azevedo Lombardi - RA: 2032874

## Problema

O objetivo e prever quais clientes de uma empresa de telecomunicacoes possuem maior chance de cancelar o servico, problema conhecido como churn. Essa previsao pode apoiar a empresa na priorizacao de acoes de retencao.

## Dataset

Foi utilizado o dataset Telco Customer Churn, com informacoes contratuais, demograficas e de servicos contratados por clientes de telecomunicacoes.

Arquivo usado no projeto: `data/dataset.csv`

Fonte publica usada para montar esta entrega: https://github.com/treselle-systems/customer_churn_analysis/blob/master/WA_Fn-UseC_-Telco-Customer-Churn.csv

## Tipo de problema

Classificacao binaria:

- `0`: cliente nao cancelou
- `1`: cliente cancelou

## Metodologia

O notebook realiza limpeza dos dados, conversao de `TotalCharges` para valor numerico, remocao de registros incompletos, exclusao de `customerID`, criacao da variavel `tenure_group` e transformacao da variavel alvo `Churn` para 0 e 1.

O pre-processamento foi feito com `ColumnTransformer`:

- variaveis numericas: `StandardScaler`
- variaveis categoricas: `OneHotEncoder`

Os modelos foram treinados dentro de `Pipeline`, reduzindo risco de vazamento de dados.

## Modelos treinados

- Logistic Regression com `class_weight="balanced"`
- Random Forest com `class_weight="balanced"`
- Gradient Boosting

## Modelo final

O modelo final salvo foi o `Gradient Boosting`, com threshold ajustado a partir da curva precision-recall. O ajuste de threshold foi usado para discutir melhor o trade-off entre precision e recall, um ponto importante em problemas de churn.

Arquivo do modelo: `model/modelo_final.joblib`

Metadados do modelo: `model/metadata.json`

## Metricas

Foram usadas:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Matriz de confusao
- Curva ROC
- Curva precision-recall

Os arquivos CSV em `reports/` guardam as tabelas finais de comparacao, validacao cruzada, teste, feature importance e coeficientes da Regressao Logistica.

## Estrutura

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv
├── model/
│   ├── modelo_final.joblib
│   └── metadata.json
├── notebooks/
│   ├── notebook_p1_original.ipynb
│   └── notebook_atualizado.ipynb
├── reports/
│   ├── relatorio_atualizado.pdf
│   └── arquivos de metricas em CSV
└── src/
    ├── train_model.py
    └── generate_report.py
```

## Como executar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Treine o modelo e gere os arquivos de metricas:

```bash
python src/train_model.py
```

Gere o relatorio atualizado:

```bash
python src/generate_report.py
```

Execute o app:

```bash
streamlit run app.py
```

## Deploy

O app deve ser publicado no Streamlit Community Cloud a partir do repositorio publico no GitHub.

Link do app publicado: https://previsao-de-churn-de-clientes-cqvmgvmekvzrrk83gchdng.streamlit.app/

## Limitacoes

O modelo foi treinado com dados historicos e serve como apoio a decisao. Em uso real, seria importante monitorar mudancas no comportamento dos clientes, atualizar o modelo periodicamente e integrar a aplicacao a uma base operacional.

## Conclusao

A P2 transforma o trabalho da P1 em um projeto mais completo: notebook revisado, tratamento mais claro do desbalanceamento, comparacao completa dos modelos, interpretacao dos resultados, modelo salvo e app Streamlit funcional para predicao.
