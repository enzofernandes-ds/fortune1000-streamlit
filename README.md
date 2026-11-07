# Fortune 1000 - Streamlit analysis

Dashboard em Streamlit feito com a base das 1000 maiores empresas da Fortune (receita, lucro, funcionários, ativos...). Ainda estou aprendendo a mexer no Streamlit e no GitHub, então o código deve ter bastante coisa pra melhorar.

## O que tem

- filtros na barra lateral (receita, funcionários e busca por nome)
- 6 gráficos: top 15 por receita, concentração de receita, top 10 por funcionários, margem de lucro, quem mais subiu/caiu no ranking e receita por funcionário
- um SVM que tenta prever se a empresa tem margem de lucro alta ou baixa
- um gráfico 3D com DBSCAN pra ver quais empresas se parecem (dá pra girar com o mouse), com o método do cotovelo e o gap pra escolher os parâmetros

## Como rodar

```
pip install -r requirements.txt
streamlit run "Dashboard - Fortune 1000.py"
```

O código lê a base (`Fortune_1000_Companies_by_Revenue.csv`) direto daqui do GitHub, então precisa de internet.

Feito por Enzo Fernandes.
