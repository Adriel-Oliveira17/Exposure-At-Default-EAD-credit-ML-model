# Modelagem de EAD (*Exposure at Default*) e CCF (*Credit Conversion Factor*)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-HistGradientBoosting-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Risk](https://img.shields.io/badge/Credit%20Risk-Basel%20%2F%20IFRS%209-B10909)

Projeto de Ciência de Dados aplicado a **Risco de Crédito** para estimar a exposição financeira no momento do calote (**EAD — *Exposure at Default***) em uma carteira de crédito rotativo (cartão de crédito). A abordagem utiliza *Machine Learning* (`HistGradientBoostingRegressor`) para prever o fator de conversão de crédito (**CCF — *Credit Conversion Factor***) a partir de variáveis cadastrais e comportamentais observadas 12 meses antes do evento de *default* (> 90 dias de atraso), com validação temporal (*Out-of-Time*).

---

## Visão Geral e Contexto de Negócio

Em produtos de crédito rotativo, o cliente possui um limite aprovado, mas utiliza apenas parte dele em um dado momento. Quando ocorre deterioração financeira e o cliente entra em *default*, ele tende a consumir parte do limite ainda disponível antes do bloqueio definitivo.

Para dimensionar corretamente a perda esperada ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$), estimamos primeiro o **CCF** (proporção do limite livre que será consumida até o *default*) e, em seguida, calculamos o **EAD** em reais:

$$\text{CCF}_{\text{realizado}} = \frac{\text{Saldo}_{\text{default}} - \text{Saldo}_{12\text{m}}}{\text{Limite}_{12\text{m}} - \text{Saldo}_{12\text{m}}}$$

$$\text{EAD}_{\text{previsto}} = \text{Saldo}_{12\text{m}} + \text{CCF}_{\text{previsto}} \times (\text{Limite}_{12\text{m}} - \text{Saldo}_{12\text{m}})$$

---

## Estrutura do Projeto

```text
├── gerador.ipynb      # Simulação da base sintética de clientes e comportamento até o default (Dados.csv)
├── modelo.ipynb       # EDA, engenharia de atributos, treino OOT, avaliação e regra de produção
├── Dados.csv          # Dataset gerado com 10.000 clientes e safras de 2022 a 2025
└── README.md          # Documentação do projeto
```

---

## Dicionário de Dados

A base simulada contém **10.000 registros** distribuídos em quatro safras anuais (`2022` a `2025`), com variáveis observadas **12 meses antes do *default*** e o saldo efetivo no momento do calote:

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `safra_default` | `int` | Ano em que o cliente atingiu o marco de *default* (`2022`, `2023`, `2024` ou `2025`). |
| `renda_mensal` | `float` | Renda mensal declarada/estimada do cliente (R\$). |
| `limite_12m` | `float` | Limite total de crédito disponível 12 meses antes do *default* (R\$). |
| `saldo_12m` | `float` | Saldo devedor utilizado 12 meses antes do *default* (R\$). |
| `variacao_saldo_3m` | `float` | Variação percentual do saldo nos 3 meses anteriores à janela de observação. |
| `qtd_atrasos_curtos_6m` | `int` | Quantidade de atrasos de curta duração nos 6 meses anteriores. |
| `pagou_minimo_ult_fatura` | `int` | Indicador binário (`0` ou `1`) se o cliente pagou apenas o mínimo na última fatura. |
| `saldo_default` | `float` | Saldo devedor real registrado no dia do *default* (> 90 dias de atraso) (R\$). |

---

## Metodologia e Pipeline de Modelagem

1. **Geração de Dados (`gerador.ipynb`):**
   - Simulação de 10.000 clientes com `seed=42`, incorporando sinais de estresse financeiro (taxa de utilização inicial, variação recente de saldo, atrasos curtos e pagamento mínimo) para compor o CCF verdadeiro e o saldo final no *default*.
2. **Análise Exploratória (`modelo.ipynb`):**
   - Avaliação da estabilidade temporal entre safras (`2022` a `2025`) para variação de saldo, atrasos curtos, taxa de pagamento mínimo, renda média e saldo médio de *default*, além da matriz de correlação entre atributos.
3. **Filtragem e Construção do Alvo (`ccf_realizado`):**
   - Seleção de clientes elegíveis com limite disponível (`limite_12m > saldo_12m`), resultando em **9.917 clientes** e evitando divisão por zero.
   - Aplicação de *clipping* no intervalo $[0.0, 1.0]$ para garantir aderência regulatória ao domínio do CCF.
4. **Engenharia de Atributos (*Feature Engineering*):**
   - `taxa_utilizacao_12m` = `saldo_12m / limite_12m`
   - `comprometimento_renda` = `saldo_12m / renda_mensal`
   - **Features finais do modelo:** `limite_12m`, `saldo_12m`, `qtd_atrasos_curtos_6m`, `pagou_minimo_ult_fatura`, `variacao_saldo_3m`, `taxa_utilizacao_12m` e `comprometimento_renda`.
5. **Validação Temporal (*Out-of-Time* — OOT):**
   - **Treino:** Safras de `2022`, `2023` e `2024` (~75% da base).
   - **Teste OOT:** Safra de `2025` (~25% da base), simulando o desempenho real em produção.
6. **Treinamento do Modelo:**
   - Algoritmo `HistGradientBoostingRegressor` configurado com hiperparâmetros conservadores para mitigar *overfitting*: `max_depth=4`, `learning_rate=0.05`, `max_iter=200` e `random_state=42`.

---

## Resultados e Desempenho (Safra OOT 2025)

O modelo apresentou alta capacidade preditiva e excelente calibração financeira na safra futura de teste (`2025`):

| Métrica | Valor Obtido | Interpretação |
| :--- | :---: | :--- |
| **Base Elegível (`limite_livre > 0`)** | `9.917 / 10.000` | Clientes com margem livre para conversão de limite |
| **MAE — CCF** | `0.0976` | Erro médio absoluto de ~9,8 pontos percentuais no fator de conversão |
| **MAE — EAD** | `R$ 576,45` | Erro médio absoluto financeiro por contrato |
| **RMSE — EAD** | `R$ 917,57` | Raiz do erro quadrático médio financeiro |
| **$R^2$ — EAD** | `0.967` | O modelo explica 96,7% da variância do saldo no *default* |
| **Calibração (EAD Médio Real vs. Previsto)** | `R$ 7.010,14` vs. `R$ 7.013,64` | Desvio agregado de apenas **R\$ 3,50 (+0,05%)** na carteira |

---

## Regra de Produção para Carteira Completa

Para escorar 100% da carteira ativa em produção — incluindo clientes que já estão com o limite totalmente tomado ou estourado (`limite_livre_12m <= 0`) — foi implementada a função `prever_ead_carteira_completa(df_carteira, modelo_ccf)`:

- **Se `limite_livre_12m > 0`:** estima o $\text{CCF}_{\text{previsto}} \in [0, 1]$ via modelo e aplica a equação geral do EAD.
- **Se `limite_livre_12m <= 0`:** assume que não há limite adicional a ser consumido e define `EAD = saldo_12m`.

---

## Como Executar

### 1. Pré-requisitos
Instale as bibliotecas necessárias:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. Ordem de Execução
1. Execute todas as células de **`gerador.ipynb`** para gerar o arquivo `Dados.csv`.
2. Execute todas as células de **`modelo.ipynb`** para reproduzir os gráficos exploratórios, treinar o modelo, avaliar as métricas na safra OOT (`2025`) e testar a função de inferência em produção.
