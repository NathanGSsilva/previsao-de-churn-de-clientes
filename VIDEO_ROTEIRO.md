# Roteiro para o video individual

Tempo esperado: 9 a 12 minutos.

## 1. Identificacao inicial - 30 segundos

Falar nome completo, curso, grupo e tema:

"Meu nome e Nathan Gabriel da Silva, faco parte do grupo do projeto de previsao de churn de clientes. O objetivo e prever quais clientes de uma empresa de telecomunicacoes possuem maior chance de cancelar o servico."

## 2. Explicacao do notebook - 7 a 9 minutos

Ordem sugerida:

1. Objetivo do projeto
2. Importacao das bibliotecas
3. Carregamento do arquivo `data/dataset.csv`
4. Limpeza dos dados
5. Analise exploratoria
6. Criacao da variavel `tenure_group`
7. Separacao entre `X` e `y`
8. Pre-processamento com `ColumnTransformer`
9. Divisao estratificada em treino, validacao e teste
10. Treinamento dos modelos
11. Comparacao das metricas
12. Interpretacao dos graficos
13. Escolha do modelo final
14. Ajuste do threshold
15. Salvamento do modelo com `joblib.dump`

Pontos importantes para falar:

- `Churn` foi transformado em 0 e 1 porque o problema e de classificacao binaria.
- `TotalCharges` precisou ser convertido para numerico.
- `customerID` foi removido porque e apenas identificador.
- A classe churn e desbalanceada: aproximadamente 26,6% dos clientes cancelaram.
- A divisao usa `stratify` para manter a proporcao de churn nos conjuntos.
- `StandardScaler` padroniza variaveis numericas.
- `OneHotEncoder` transforma variaveis categoricas em colunas numericas.
- `Pipeline` junta pre-processamento e modelo, evitando vazamento de dados.
- A Regressao Logistica e a Random Forest usam `class_weight="balanced"`.
- O Gradient Boosting foi mantido como modelo final por ter bom AUC-ROC.

Metricas finais para comentar:

- Threshold padrao 0.50: accuracy 0.7939, precision 0.6373, recall 0.5214, F1 0.5735, AUC 0.8399.
- Threshold ajustado 0.217: accuracy 0.7178, precision 0.4823, recall 0.8396, F1 0.6127, AUC 0.8399.

Interpretacao do trade-off:

"Com threshold 0.50, o modelo e mais conservador e erra menos falsos positivos, mas deixa passar clientes que poderiam cancelar. Com threshold 0.217, o recall sobe bastante, entao o modelo encontra mais clientes em risco. A precision cai, mas em churn isso pode ser aceitavel se a acao de retencao tiver custo menor do que perder o cliente."

## 3. Demonstracao do app - ate 2 minutos

1. Abrir o app Streamlit.
2. Mostrar os campos de entrada.
3. Preencher um cliente de exemplo.
4. Clicar em `Executar predicao`.
5. Mostrar probabilidade, threshold e resultado.
6. Explicar que o app carrega `model/modelo_final.joblib` e `model/metadata.json`.

Frase util:

"O app pega os dados digitados, monta um DataFrame com as mesmas colunas usadas no treinamento, passa pelo Pipeline salvo e exibe a probabilidade de churn. A classificacao final compara essa probabilidade com o threshold ajustado."

## 4. Fechamento - 30 segundos

Concluir dizendo:

"A P2 evoluiu a P1 porque agora o projeto tem notebook revisado, relatorio atualizado, modelo salvo, app funcional e uma discussao mais clara sobre desbalanceamento, precision, recall e escolha do modelo."
