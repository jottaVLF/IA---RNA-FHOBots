# 🤖 Análise Preditiva de Futebol de Robôs com Redes Neurais

> Engenharia reversa de tomada de decisão em agentes autônomos utilizando Machine Learning.

Este projeto aplica técnicas de **Aprendizado de Máquina Supervisionado** para analisar logs de uma partida de futebol de robôs (5 contra 5). O objetivo é prever a decisão tática (estado) de cada robô em tempo real, baseando-se apenas na configuração espacial do jogo (posições da bola e dos jogadores).

## 📋 Contexto do Projeto

Em competições de futebol robótico, os agentes tomam decisões baseadas em uma máquina de estados finitos ou árvores de comportamento. Sem acesso ao código-fonte original dos robôs, utilizamos um log de dados (`log_jogo5v5Final.csv`) para treinar modelos que realizam a **engenharia reversa** dessas regras.

O desafio principal deste dataset é o **desbalanceamento severo de classes**: estados como `Idle` (Ocioso) são extremamente frequentes, enquanto ações críticas como `GotoBall` (Ir para a Bola) ou `BackOff` (Recuar) são raras, dificultando o aprendizado de modelos tradicionais.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (MLPClassifier, StandardScaler)
* **Dados Desbalanceados:** Imbalanced-learn (SMOTE, Pipeline)
* **Visualização:** Matplotlib, Seaborn

## 🧠 Metodologia

A solução foi estruturada em um pipeline robusto para garantir a validade dos resultados:

1.  **Pré-processamento:** Limpeza de dados, conversão de tipos e normalização de features espaciais (Z-score) usando `StandardScaler`.
2.  **Correção de Viés (SMOTE):** Aplicação da técnica *Synthetic Minority Over-sampling Technique* apenas nos dados de treino. Isso cria exemplos sintéticos de estados raros, forçando o modelo a aprender comportamentos táticos complexos em vez de apenas prever a classe majoritária.
3.  **Modelagem (Redes Neurais):** Utilização de um **Perceptron de Múltiplas Camadas (MLP)** com arquitetura otimizada (camadas ocultas de 100 e 50 neurônios) para capturar relações não-lineares.
4.  **Avaliação Estratificada:** Divisão de treino/teste (70/30) mantendo a proporção original das classes.

## 📊 Principais Resultados

O modelo alcançou uma acurácia média de **~95%**, com melhorias drásticas no *Recall* de classes raras devido ao SMOTE. A análise das Matrizes de Confusão revelou que o time opera sob uma **IA Baseada em Funções (Role-Based AI)**:

* **Goleiro (y0):** Altamente previsível, toma decisões baseadas quase exclusivamente em sua própria posição no eixo X (defesa de área).
* **Zagueiros (y1, y2):** Comportamento híbrido entre manter posição e reagir à bola.
* **Atacantes (y3, y4):** Comportamento altamente reativo e dinâmico, focado na posição instantânea da bola (`bx`, `by`), apresentando maior complexidade na transição entre `GotoBall` e `Attack`.

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter as bibliotecas necessárias instaladas:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn# IA---RNA-FHOBots
Código inicial para previsão do jogo de futebol FHObots
