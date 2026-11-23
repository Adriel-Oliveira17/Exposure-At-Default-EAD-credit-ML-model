# Modelagem de EAD (*Exposure at Default*) e CCF (*Credit Conversion Factor*)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-HistGradientBoosting-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Risk](https://img.shields.io/badge/Credit%20Risk-Basel%20%2F%20IFRS%209-B10909)

Projeto de Ciência de Dados aplicado a **Risco de Crédito** para estimar a exposição financeira no momento do calote (**EAD — *Exposure at Default***) em uma carteira de crédito rotativo (cartão de crédito). A abordagem utiliza *Machine Learning* (`HistGradientBoostingRegressor`) para prever o fator de conversão de crédito (**CCF — *Credit Conversion Factor***) a partir de variáveis cadastrais e comportamentais observadas na data de referência (`safra_obs`), com validação temporal (*Out-of-Time*).

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
├── Dados.csv          # Dataset gerado com 60.000 clientes e safras de observação de 2022 a 2025
└── README.md          # Documentação do projeto
```

---

## Dicionário de Dados

A base simulada contém **60.000 registros** distribuídos em quatro safras anuais de observação (`2022` a `2025`), com atributos cadastrais e comportamentais observados na data de referência, indicador de *default* em 12 meses e saldo efetivo no momento do calote:

| Variável | Tipo | Descrição |
| :--- | :--- | :--- |
| `safra_obs` | `int` | Ano da safra de observação do cliente (`2022`, `2023`, `2024` ou `2025`). |
| `renda_mensal` | `float` | Renda mensal declarada/estimada do cliente (R\$). |
| `limite` | `float` | Limite total de crédito disponível na data de observação (R\$). |
| `saldo` | `float` | Saldo devedor utilizado na data de observação (R\$). |
| `variacao_saldo_3m` | `float` | Variação percentual do saldo nos 3 meses anteriores à data de observação. |
| `qtd_atrasos_curtos_6m` | `int` | Quantidade de atrasos de curta duração nos 6 meses anteriores. |
| `pagou_minimo_ult_fatura` | `int` | Indicador binário (`0` ou `1`) se o cliente pagou apenas o mínimo na última fatura. |
| `saldo_default` | `float` | Saldo devedor real registrado no momento do *default* (> 90 dias de atraso) (R\$). |
| `default_12m` | `int` | Indicador binário (`0` ou `1`) se o cliente entrou em *default* nos 12 meses seguintes à observação. |

---

## Metodologia e Pipeline de Modelagem

1. **Geração de Dados (`gerador.ipynb`):**
   - Simulação de 60.000 clientes com `seed=42`, incorporando sinais de estresse financeiro (taxa de utilização, variação recente de saldo, atrasos curtos e pagamento mínimo) para compor a ocorrência de *default* (`default_12m`), o CCF verdadeiro e o saldo final no *default* (`saldo_default`).
2. **Análise Exploratória (`modelo.ipynb`):**
   - Avaliação da estabilidade temporal entre safras (`safra_obs` de `2022` a `2025`) para variação de saldo, atrasos curtos, taxa de pagamento mínimo, renda média e saldo médio de *default*, além da matriz de correlação entre atributos.
3. **Filtragem e Construção do Alvo (`ccf_realizado`):**
   - Seleção da amostra de modelagem restrita aos clientes que entraram em *default* (`default_12m == 1`) e possuíam limite disponível na observação (`limite > saldo`), evitando divisão por zero.
   - Aplicação de *clipping* (`np.clip(..., 0.0, 1.0)`) para garantir aderência regulatória ao domínio $[0.0, 1.0]$ do CCF.
4. **Engenharia de Atributos (*Feature Engineering*):**
   - `taxa_utilizacao` = `saldo / limite`
   - `comprometimento_renda` = `saldo / renda_mensal`
   - **Features finais do modelo:** `limite`, `saldo`, `qtd_atrasos_curtos_6m`, `pagou_minimo_ult_fatura`, `variacao_saldo_3m`, `taxa_utilizacao` e `comprometimento_renda`.
5. **Validação Temporal (*Out-of-Time* — OOT):**
   - **Treino:** Safras de observação `safra_obs` de `2022`, `2023` e `2024`.
   - **Teste OOT:** Safra futura `safra_obs == 2025`, simulando o desempenho real em produção.
6. **Treinamento do Modelo:**
   - Algoritmo `HistGradientBoostingRegressor` configurado com hiperparâmetros conservadores para mitigar *overfitting*: `max_depth=4`, `learning_rate=0.05`, `max_iter=200` e `random_state=42`.

---

## Resultados e Desempenho (Safra OOT 2025)

O modelo apresentou alta capacidade preditiva e excelente calibração financeira na safra futura de teste (`safra_obs == 2025`):

| Métrica | Valor Obtido | Interpretação |
| :--- | :---: | :--- |
| **MAE — CCF** | `0.0976` | Erro médio absoluto de ~9,8 pontos percentuais no fator de conversão |
| **MAE — EAD** | `R$ 576,45` | Erro médio absoluto financeiro por contrato |
| **RMSE — EAD** | `R$ 917,57` | Raiz do erro quadrático médio financeiro |
| **$R^2$ — EAD** | `0.967` | O modelo explica 96,7% da variância do saldo no *default* |
| **Calibração (EAD Médio Real vs. Previsto)** | `R$ 7.010,14` vs. `R$ 7.013,64` | Desvio agregado de apenas **R\$ 3,50 (+0,05%)** na carteira |

---

## Regra de Produção para Carteira Completa

Para escorar 100% da carteira ativa em produção, incluindo clientes que já estão com o limite totalmente tomado ou estourado (`limite_livre <= 0`), foi implementada a função `prever_ead_carteira_completa(df_carteira, modelo_ccf)`:

- **Se `limite_livre > 0` (`limite > saldo`):** estima o $\text{CCF}_{\text{previsto}} \in [0, 1]$ via modelo e aplica a equação geral do EAD.
- **Se `limite_livre <= 0` (`limite <= saldo`):** assume que não há limite adicional a ser consumido e define `EAD = saldo`.

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
