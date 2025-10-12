# Hackathon Code4Cancer --- IA-Maria 🤖🧠

## LEMA: Detectar cedo é hackear o câncer

## OBJETIVO CENTRAL DO HACKATHON: Como ampliar o acesso ao rastreamento e à detecção precoce do câncer?

## IA-Maria

### Sistema inteligente para rastreamento à detecção precoce do câncer e encaminhamento oncológico

A IA-Maria é uma assistente clínica inteligente desenvolvida para apoiar profissionais da Atenção Primária à Saúde no rastreamento, pré-diagnóstico e orientação de condutas relacionadas aos cânceres colorretal e pulmonar.

## Sobre o projeto

IA-RaDi é uma inteligência artificial voltada para auxiliar profissionais da saúde no rastreamento de possíveis casos de câncer.
O sistema utiliza bases de dados confiáveis, com recomendações de organizações reconhecidas da área, para oferecer suporte clínico rápido e assertivo.

## Objetivo

Aumentar a detecção precoce de câncer e agilizar o encaminhamento adequado, reduzindo diagnósticos tardios e otimizando recursos médicos.

## Como funciona

- Entrada de dados do paciente
  O profissional de saúde informa histórico, sintomas e outros dados clínicos relevantes.

- Análise pela IA
  A IA calcula a probabilidade de o paciente ter câncer com base nos dados e referências médicas confiáveis.

- Sugestão de exames
  O sistema indica quais exames são apropriados para investigação inicial ou complementar.

- Encaminhamento automatizado
  Se os exames indicarem suspeita ou confirmação, a IA gera um documento com todas as informações necessárias para o oncologista e sugere que o paciente seja

## Fontes e Base de Conhecimento

- guideline INCA
- guideline SBOC
- uspstf preventive services

## Público-Alvo

Clínicos gerais

Médicos do pronto atendimento

Especialistas em outras áreas

Serviços de triagem ou atenção básica

## Stack Tecnológica

Linguagem: Python

Integração com IA: Groq

## Como Rodar o Projeto

- Clone este repositório:

git clone [<URL>](https://github.com/ludmila-sla/hackaton-dev4cancer)

- Crie e ative o ambiente virtual:

python3 -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

- Instale as dependências:

pip install -r requirements.txt

- Rode a aplicação Flask:

export FLASK_APP=app.py
export FLASK_ENV=development # ativa modo debug
flask run

- Abra o navegador e acesse: http://127.0.0.1:5000/
