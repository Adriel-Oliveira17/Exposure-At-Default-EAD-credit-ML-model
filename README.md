# Modelagem de EAD (*Exposure at Default*) e CCF (*Credit Conversion Factor*)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-HistGradientBoosting-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Risk](https://img.shields.io/badge/Credit%20Risk-Basel%20%2F%20IFRS%209-B10909)

Projeto de Ciência de Dados aplicado a **Risco de Crédito** para estimar a exposição financeira no momento do calote (**EAD — *Exposure at Default***) em uma carteira de crédito rotativo (cartão de crédito). A abordagem utiliza *Machine Learning* (`HistGradientBoostingRegressor`) para prever o fator de conversão de crédito (**CCF — *Credit Conversion Factor***) a partir de variáveis cadastrais e comportamentais observadas 12 meses antes da janela de *default* (> 90 dias de atraso), com validação temporal (*Out-of-Time*).

---

## Visão Geral e Contexto de Negócio

Em produtos de crédito rotativo, o cliente possui um limite aprovado, mas utiliza apenas parte dele em um dado momento. Quando ocorre deterioração financeira e o cliente entra em *default*, ele tende a consumir parte do limite ainda disponível antes do bloqueio definitivo.

Para dimensionar corretamente a perda esperada ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$), estimamos primeiro o **CCF** (proporção do limite livre que será consumida até o *default*) e, em seguida, calculamos o **EAD** em reais:

$$\text{CCF}_{\text{realizado}} = \frac{\text{Saldo}_{\text{default}} - \text{Saldo}}{\text{Limite} - \text{Saldo}}$$

$$\text{EAD}_{\text{previsto}} = \text{Saldo} + \text{CCF}_{\text{previsto}} \times (\text{Limite} - \text{Saldo})$$

---

## Estrutura do Projeto

```text
├── gerador.ipynb      # Simulação da base sintética de clientes e comportamento até o default (Dados.csv)
├── modelo.ipynb       # EDA, engenharia de atributos, treino OOT, avaliação e regra de produção
├── Dados.csv          # Dataset gerado com safras de observação de 2022 a 2025
└── README.md          # Documentação do projeto
