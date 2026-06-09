# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- IMPORTS DO ADK  ---
import sys
import asyncio
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InMemoryRunner
from google.genai import types 
from IPython.display import display, Markdown
from kaggle_secrets import UserSecretsClient

# 1. ConfiguraÃ§Ã£o da API Key
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# 2. ConfiguraÃ§Ã£o de Retry 
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

print("âœ… ADK Environment & Retry Config loaded.")


def tool_keyword_matcher(target_text: str, source_text: str) -> float:
    """
    Calculates the percentage of unique words from target_text that appear in source_text.
    Useful to check if the resume covers the job description keywords.
    
    Args:
        target_text: The Job Description text.
        source_text: The Resume text.
    Returns:
        float: The match percentage (0.0 to 100.0).
    """
    # LÃ³gica de limpeza simples
    set_target = set(target_text.lower().replace(',', '').split())
    set_source = set(source_text.lower().replace(',', '').split())
    
    # Remove stopwords bÃ¡sicas
    stopwords = {'e', 'de', 'do', 'da', 'para','o', 'a','que', 'se','por', 'mas', 'and', 'the', 'to', 'of'}
    set_target = set_target - stopwords
    
    if not set_target: return 0.0
    
    intersection = set_target.intersection(set_source)
    match_score = (len(intersection) / len(set_target)) * 100
    
    return round(match_score, 2)

print("âœ… Tool defined.")


# --- AGENT CREATION ---

# Agent 1: Analyst (Analista)
agent_analyst = LlmAgent(
    name="analyst_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
VocÃª Ã© um Recrutador TÃ©cnico.
Seu objetivo:
1. Receber uma Vaga e 3. Extrair as 5 principais skills da vaga.
um CurrÃ­culo.
2. USAR A FERRAMENTA 'tool_keyword_matcher' para calcular o match.
4. Retornar um resumo com o Score calculado e as Skills extraÃ­das.
""",
    tools=[tool_keyword_matcher] 
)

# Agent 2: Writer (Redator)
agent_writer = LlmAgent(
    name="writer_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
VocÃª Ã© um Especialista em ATS.
Receba o input do Analista (Score + Skills).
Reescreva o Resumo e a ExperiÃªncia Profissional do currÃ­culo original para incluir as skills listadas.
Seja persuasivo e use verbos de aÃ§Ã£o.
"""
)

# Agent 3: Coach (Treinador)
agent_coach = LlmAgent(
    name="coach_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
VocÃª Ã© um Treinador de Entrevistas.
Com base no currÃ­culo otimizado e nas 'HistÃ³rias STAR' do usuÃ¡rio (que estarÃ£o no prompt),
crie uma pergunta difÃ­cil e sugira uma resposta ideal usando uma das histÃ³rias.
Pode reescrever a histÃ³ria, para ajustar as keywords ou skill necessÃ¡rias, mas sem inventar
"""
)

# --- SEQUENTIAL AGENT ---
recruiter_team = SequentialAgent(
    name="recruiter_team",
    sub_agents=[agent_analyst, agent_writer, agent_coach]
)

print("âœ… Agents and Sequential Agent created.")


#In put data (Preencha aqui)
# Job Description
vaga_texto = """
DESCRIÃ‡ÃƒO DA VAGA
O futuro que vocÃª busca, a velocidade que vocÃª espera.
O Acelera Ã© o programa de formaÃ§Ã£o de Gerentes Comerciais do ItaÃº, com oportunidades para atuar como Gerente de NegÃ³cios no ItaÃº Empresas.
Buscamos profissionais determinados e corajosos, prontos para fazer a diferenÃ§a nas empresas que impulsionam a economia do paÃ­s.
Nosso objetivo Ã© transformar profissionais em protagonistas, preparando-os para compreender o momento de vida de cada cliente e oferecer uma assessoria que faÃ§a a diferenÃ§a. Afinal, clientes do ItaÃº Empresas tÃªm mais chances de crescer.

RESPONSABILIDADES E ATRIBUIÃ‡Ã•ES
GestÃ£o da carteira de clientes PJ (Pessoa JurÃ­dica), atuando como consultor(a) para decisÃµes financeiras;
Assessoria ao cliente nas decisÃµes importantes;
RealizaÃ§Ã£o da gestÃ£o e acompanhamento de crÃ©dito e riscos da sua carteira;
ParticipaÃ§Ã£o em discussÃµes e construÃ§Ãµes nos comitÃªs de crÃ©dito;
Desenvolvimento de um calendÃ¡rio estratÃ©gico de visitas aos clientes da sua carteira;
Relacionamento com stakeholders de outras Ã¡reas para oferecer as melhores soluÃ§Ãµes;
Acompanhamento do mercado financeiro e das tendÃªncias bancÃ¡rias.
Para apoiar essa jornada, Ã© preciso ter sede de aprendizado, coragem para ir alÃ©m e vontade de se desafiar sempre, jÃ¡ que aqui no Acelera vocÃª irÃ¡ adquirir conhecimentos todos os dias.
REQUISITOS E QUALIFICAÃ‡Ã•ES
GraduaÃ§Ã£o completa em cursos das Ã¡reas de exatas ou humanas;
ExperiÃªncia prÃ©via em Ã¡reas comerciais e no desenvolvimento de relacionamento com clientes;
Disponibilidade para atuar em algumas das cidades do programa;
DIFERENCIAIS
PÃ³s-graduaÃ§Ã£o ou especializaÃ§Ã£o;
ExperiÃªncia com clientes PJ (Pessoa JurÃ­dica).
"""
#Curriculo (curriculum)
curriculo_texto = """
RESUMO PROFISSIONAL
Mais de 6 anos na Ã¡rea Comercial. Atuando no gerenciamento, mapeamento, preparaÃ§Ã£o e realizaÃ§Ã£o dos procedimentos pertinentes.
Desenvolvimento e implementaÃ§Ã£o de estratÃ©gias para aperfeiÃ§oamento de fluxos, incremento de receita, fortalecimento da presenÃ§a de marca e expansÃ£o de participaÃ§Ã£o no mercado-alvo.
LideranÃ§a de equipe multidisciplinar, aplicaÃ§Ã£o de treinamento e avaliaÃ§Ã£o de performance.
ElaboraÃ§Ã£o de relatÃ³rios gerenciais; anÃ¡lise de indicadores (Kpiâ€™s); metas e mÃ©tricas.
VivÃªncia no alinhamento multisetorial de stakeholders dos segmentos de negÃ³cios, marketing, trademarketing, logÃ­stica, operaÃ§Ãµes e produtos.
Expertise Comercial, abrangendo inauguraÃ§Ã£o, operaÃ§Ã£o e gestÃ£o de lojas em redes nacionais e internacionais, otimizaÃ§Ã£o de parÃ¢metros como ticket mÃ©dio, conversÃ£o, giro de estoque, margem bruta e NPS, e coordenaÃ§Ã£o de atÃ© 50 colaboradores.
Especialista no planejamento e execuÃ§Ã£o de vendas, incluindo exame de tendÃªncias de mercado, definiÃ§Ã£o de planos de posicionamento de produtos, administraÃ§Ã£o de portfÃ³lio, efetivaÃ§Ã£o de campanhas e integraÃ§Ã£o de canais.
ExperiÃªncia na reestruturaÃ§Ã£o de unidades deficitÃ¡rias e expansÃ£o de mercado em cidades do interior, com diagnÃ³stico operacional, adaptaÃ§Ã£o de mix de produtos e formaÃ§Ã£o de profissionais locais.
Habilidade na aplicaÃ§Ã£o de tÃ¡ticas de marketing digital e trade marketing, com gestÃ£o de conteÃºdo e mÃ­dias pagas, estudo de desempenho, melhoria de pontos de venda e alinhamento entre vendedores de campo e online.

EXPERIÃŠNCIAS PROFISSIONAIS
PUMA GROUP
Cargo: Gerente Comercial (Senior Store Manager) (04/2023 a 09/2025)
AtribuiÃ§Ãµes: InauguraÃ§Ã£o e operaÃ§Ã£o de lojas no conceito 'Full Price' no Brasil. TransformaÃ§Ã£o de loja deficitÃ¡ria em unidade lucrativa, com destaque para maior SSS de Retail de 2024. Monitoramento de KPIs crÃ­ticos, incluindo conversÃ£o, ticket mÃ©dio, giro de estoque e margem bruta. ImplementaÃ§Ã£o de planos de aÃ§Ã£o para maximizaÃ§Ã£o de resultados comerciais.OtimizaÃ§Ã£o de processos de atendimento e controle de estoque para melhoria da experiÃªncia do cliente.
GRUPO SBF
Cargo Final: Coordenador de Projetos (12/2021 a 04/2023)
AtribuiÃ§Ãµes: CoordenaÃ§Ã£o de projeto de expansÃ£o de mercado em cidades do interior, com desenvolvimento de modelo de negÃ³cios inovador. GestÃ£o de loja laboratÃ³rio e operaÃ§Ã£o de novas unidades para validaÃ§Ã£o de hipÃ³teses de negÃ³cio e testes A/B. Alinhamento multidisciplinar de stakeholders das Ã¡reas de operaÃ§Ãµes, marketing, trademarketing, logÃ­stica, comercial e produtos.
Metodologias usadas: SCRUM
Cargo Inicial: Especialista Retail Concept (05/2021 a 11/2021)
AtribuiÃ§Ãµes: LideranÃ§a de projeto de digitalizaÃ§Ã£o pÃ³s-pandemia em mais de 200 lojas. EstruturaÃ§Ã£o e implementaÃ§Ã£o de processo de marketing digital descentralizado, com treinamentos para unidades de negÃ³cio. GestÃ£o de budget para campanhas de mÃ­dia paga segmentadas, integrando canais online e offline. PadronizaÃ§Ã£o do atendimento e processo de vendas via Whatsapp para as lojas. Metodologias usadas: OKR
CENTAURO
Cargo Final: Gerente Comercial III (02/2020 a 05/2021)
AtribuiÃ§Ãµes: OperaÃ§Ã£o da 3Âª maior loja em faturamento nacional, com foco em indicadores de vendas, SSS, NPS, conversÃ£o e perdas. LideranÃ§a de equipe de 52 colaboradores, incluindo supervisores e vendedores. CoordenaÃ§Ã£o de reabertura pÃ³s-reforma durante perÃ­odo pandÃªmico, com adaptaÃ§Ã£o de processos de atendimento.
Cargo: Gerente Comercial II (02/2019 a 01/2020)
AtribuiÃ§Ãµes: OperaÃ§Ã£o da 2Âª maior loja em faturamento regional, com acompanhamento de KPIs de vendas e performance. LideranÃ§a de equipe de 32 colaboradores, incluindo supervisores e vendedores. ImplementaÃ§Ã£o de melhorias operacionais para Ship From Store.
Cargo Inicial: Trainee de OperaÃ§Ãµes (02/2018 a 02/2019)
AtribuiÃ§Ãµes: RotaÃ§Ã£o estratÃ©gica por todas as funÃ§Ãµes operacionais, com experiÃªncia prÃ¡tica em estoque, vendas, supervisÃ£o, atendimento e gerÃªncia. Apoio na mobilizaÃ§Ã£o e reabertura de unidades, incluindo Shopping MetrÃ´ TatuapÃ© e Iguatemi Porto Alegre. Desenvolvimento de projeto de onboarding de novos colaboradores, padronizando processo para mais de 200 lojas. Cobertura de fÃ©rias de supervisores e gerentes, com foco em versatilidade operacional.
"""
#HistÃ³rias STAR (Star Situation)
historias_star = """
1 - Turnaround operacional Puma TatuapÃ©
SituaÃ§Ã£o:
A loja Puma TatuapÃ© estava operando com deficit e nÃ£o batia as metas mensais. Amoral do time estava extremamente baixa pois haviam atingido meta apenas uma vez nosÃºltimos 12 meses. Para termos expansÃ£o das lojas Full Price no Brasil, o primeiro passo eraalcanÃ§ar breakeven e depois atingir as metas de margem operacional estabelecidas pelo global.A situaÃ§Ã£o era crÃ­tica tanto para a estratÃ©gia de crescimento da marca quanto para a motivaÃ§Ã£oda equipe.
Tarefa:
Trazer lucratividade para a loja, reestabelecer a confianÃ§a da equipe e estabelecer ummodelo operacional sustentÃ¡vel que pudesse ser replicado nas futuras expansÃµes.
AÃ§Ã£o:
Inicialmente me conectei com o time para entender profundamente suas necessidades epercepÃ§Ãµes sobre a loja - esse passo foi fundamental para gerar confianÃ§a e identificarproblemas que nÃ£o apareciam nos nÃºmeros. Conduzi anÃ¡lise atravÃ©s de Business Intelligencedas vendas e perfis comportamentais dos vendedores. Identifiquei as causas-raiz: mix deprodutos desalinhado com o perfil local, gaps de conhecimento tÃ©cnico da equipe sobre osprodutos, processos ineficientes no atendimento e estoque completamente desorganizado.Implementei aÃ§Ãµes estruturadas: rebalanceamento do mix em parceria com o Planner da lojaentendendo as diferenÃ§as do pÃºblico local, programa intensivo de treinamento de produtos feitodo time para o time (peer-to-peer), sistema de metas individuais com gamificaÃ§Ã£o mesmo nÃ£osendo comissionados para resgatar a motivaÃ§Ã£o, reorganizaÃ§Ã£o completa do estoque e dos
processos de recebimento e entrega, alÃ©m de ajustes na setorizaÃ§Ã£o da equipe em loja paramelhor cobertura.
Resultado:
Transformei a loja deficitÃ¡ria no Maior Same-Store Sales de 2024 da rede, atingindobreakeven em 4 meses e superando as metas de margem operacional do global. A moral dotime foi completamente restaurada - passamos de 1 meta atingida em 12 meses paraatingimento consistente mensal. A loja tornou-se benchmark interno e o modelo foi utilizadocomo base para planejamento das prÃ³ximas expansÃµes Full Price no Brasil.

2 - SituaÃ§Ã£o:
Como Especialista Retail Concept no Grupo SBF, identificamos que o processo demarketing digital descentralizado - onde cada loja gerenciava seu prÃ³prio budget e criavaconteÃºdo para redes sociais - estava completamente inconsistente. TÃ­nhamos um desvio padrÃ£oaltÃ­ssimo nos resultados: algumas lojas faziam fotos simples e de baixa qualidade, outrascriavam vÃ­deos longos demais, e o esforÃ§o era desproporcional aos resultados. Mais de 200lojas operavam sem padrÃ£o algum.
Tarefa:
Criar um procedimento padronizado para as lojas visando maximizaÃ§Ã£o de resultadoscom mÃ­nimo esforÃ§o e investimento, garantindo consistÃªncia na comunicaÃ§Ã£o da marca.
AÃ§Ã£o:
Realizei anÃ¡lise das melhores prÃ¡ticas internas, identifiquei as lojas com melhoresresultados e busquei benchmarks externos de empresas que faziam marketing descentralizadosimilar. Desenvolvi um SOP completo especificando tipo de conteÃºdo (fotos vs vÃ­deos), formato,datas ideais para postagem e valores por tipo de campanha. Implementei um piloto nas lojascom piores resultados e menor consistÃªncia para validar a metodologia. Criei materiais detreinamento, incluindo vÃ­deos tutoriais explicando as melhores prÃ¡ticas desde a criaÃ§Ã£o doconteÃºdo atÃ© a gestÃ£o do budget.
Resultado:
AlcanÃ§amos 100% de consistÃªncia nas postagens das lojas piloto com resultados200% melhores que o perÃ­odo anterior. ApÃ³s padronizaÃ§Ã£o e documentaÃ§Ã£o do processo paratoda a rede, conseguimos que 3% do share total de vendas viesse dessa plataforma apÃ³s 3 meses de implementaÃ§Ã£o - um resultado expressivo para o canal.

"""

# Prompt
prompt_inicial = f"""
Aqui estÃ£o os dados para o processo:

VAGA: {vaga_texto}
CURRÃ�CULO: {curriculo_texto}
HISTÃ“RIAS STAR: {historias_star}

Comece o processo:
1. Analista: Analise e use a tool.
2. Redator: Reescreva o CV.
3. Coach: Prepare a entrevista.
"""

# --- RUNNER EXECUTION ---

runner = InMemoryRunner(agent=recruiter_team)

print("--- ğŸ�� Starting RecruiterAI Pipeline ---")

# 4. EXECUÃ‡ÃƒO
# O mÃ©todo .run() retorna um objeto Response
response_events = await runner.run_debug(prompt_inicial)

# 5. EXIBINDO RESULTADO FORMATADO
texto_limpo = ""

for event in response_events:
    # Verificamos se o evento tem conteÃºdo (content)
    if hasattr(event, 'content') and event.content:
        if hasattr(event.content, 'role') and event.content.role == 'model':
            if hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        texto_limpo += part.text + "\n\n---\n\n"

# 4. EXIBINDO O RESULTADO FINAL
if texto_limpo:
    display(Markdown(f"# ğŸš€ Resultado Final Consolidado\n\n{texto_limpo}"))
else:
    # Fallback: Se a limpeza falhar, mostra o objeto bruto para nÃ£o ficarmos sem nada
    print("Aviso: NÃ£o foi possÃ­vel limpar o texto automaticamente. Exibindo bruto:")
    print(response_events)




