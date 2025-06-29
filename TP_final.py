import pulp
import networkx as nx
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass


# === CONFIGURAÇÃO E ESTRUTURAS SIMPLIFICADAS ===

@dataclass
class Config:
    """Configuração centralizada de parâmetros"""
    # Caminhos
    pasta_dados: str = r"c:\Users\tastc\Desktop\Trabalhos_UFMG\Otimização Combinatoria\Data"
    
    # Solver
    limite_tempo: int = 300
    mostrar_msgs_solver: bool = False
    multiplicador_tempo_ub: int = 2
    
    # Subgradiente
    lambda_inicial: float = 2.0
    max_iteracoes: int = 1000
    tolerancia: float = 1e-4
    precisao_gap: float = 1e-3
    intervalo_impressao: int = 10


@dataclass
class Resultado:
    """Resultado simplificado de uma instância"""
    instancia: str
    num_nos: int = 0
    num_arestas: int = 0
    num_terminais: int = 0
    fonte: int = 0
    terminais: Set[int] = None
    
    # Bounds principais
    upper_bound: Optional[float] = None
    lower_bound_u0: Optional[float] = None
    lower_bound_subgradiente: Optional[float] = None
    
    # Métricas essenciais
    gap_relativo_final: Optional[float] = None
    melhoria_lower_bound: Optional[float] = None
    
    # Performance
    tempo_total: float = 0.0
    iteracoes_subgradiente: int = 0
    convergiu: bool = False
    status: str = "Unknown"
    erro: Optional[str] = None

    def __post_init__(self):
        if self.terminais is None:
            self.terminais = set()
    
    def calcular_metricas(self):
        """Calcula métricas derivadas"""
        if self.upper_bound and self.lower_bound_subgradiente:
            gap_abs = self.upper_bound - self.lower_bound_subgradiente
            self.gap_relativo_final = (gap_abs / self.upper_bound) * 100 if self.upper_bound > 0 else 0
        
        if self.lower_bound_u0 and self.lower_bound_subgradiente:
            self.melhoria_lower_bound = self.lower_bound_subgradiente - self.lower_bound_u0


# === CLASSES BASE SIMPLIFICADAS ===

class ModeloBase(ABC):
    """Classe base simplificada para modelos de otimização"""
    
    def __init__(self, grafo: nx.DiGraph, custos: Dict[Tuple[int, int], float], 
                 fonte: int, terminais: Set[int]):
        self.grafo = grafo
        self.custos = custos
        self.fonte = fonte
        self.terminais = terminais
        self.commodities = terminais - {fonte}
        self.nos = set(grafo.nodes())
        self.arestas = set(grafo.edges())
        self.x_vars = {}
        self.f_vars = {}
        self.modelo = None
        self.status = None
        self.valor_objetivo = None
        
    @abstractmethod
    def criar_modelo(self): pass
        
    @abstractmethod
    def resolver(self, **kwargs): pass
    
    def _criar_variaveis(self):
        """Cria todas as variáveis necessárias"""
        # Variáveis x_ij (binárias)
        self.x_vars = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat='Binary') 
                       for (i, j) in self.arestas}
        
        # Variáveis f_ij^k (contínuas)
        self.f_vars = {}
        for k in self.commodities:
            for (i, j) in self.arestas:
                self.f_vars[(i, j, k)] = pulp.LpVariable(f"f_{i}_{j}_{k}", lowBound=0, cat='Continuous')
    
    def _criar_funcao_objetivo(self):
        """Função objetivo: minimizar custo das arestas selecionadas"""
        objetivo = pulp.lpSum([self.custos[(i, j)] * self.x_vars[(i, j)] for (i, j) in self.arestas])
        self.modelo += objetivo
    
    def _adicionar_restricoes_fluxo(self):
        """Adiciona restrições de conservação de fluxo"""
        for k in self.commodities:
            for i in self.nos:
                fluxo_saida = pulp.lpSum([self.f_vars[(i, j, k)] for j in self.nos if (i, j) in self.arestas])
                fluxo_entrada = pulp.lpSum([self.f_vars[(j, i, k)] for j in self.nos if (j, i) in self.arestas])
                
                if i == k:
                    # Demanda no destino
                    self.modelo += (fluxo_saida - fluxo_entrada == -1, f"Demanda_{k}_{i}")
                elif i == self.fonte:
                    # Oferta na fonte
                    self.modelo += (fluxo_saida - fluxo_entrada == 1, f"Oferta_{i}_{k}")
                else:
                    # Conservação
                    self.modelo += (fluxo_saida - fluxo_entrada == 0, f"Conservacao_{k}_{i}")
    
    def _adicionar_restricoes_acoplamento(self):
        """Adiciona restrições de acoplamento: f_ij^k <= x_ij"""
        for k in self.commodities:
            for (i, j) in self.arestas:
                self.modelo += (self.f_vars[(i, j, k)] <= self.x_vars[(i, j)], f"Acoplamento_{k}_{i}_{j}")
    
    def _extrair_solucoes(self):
        """Extrai soluções de ambas as variáveis"""
        solucao_x = {(i, j): int(var.varValue) if var.varValue is not None else 0 
                    for (i, j), var in self.x_vars.items()}
        
        solucao_f = {}
        for k in self.commodities:
            solucao_f[k] = {}
            for (i, j) in self.arestas:
                valor = self.f_vars[(i, j, k)].varValue
                solucao_f[k][(i, j)] = valor if valor is not None else 0.0
        return solucao_x, solucao_f
    
    def obter_arestas_solucao(self, solucao_x=None) -> Set[Tuple[int, int]]:
        """Retorna arestas da solução (x_ij = 1)"""
        if solucao_x is None:
            if self.status != pulp.LpStatusOptimal:
                return set()
            solucao_x, _ = self._extrair_solucoes()
        return {(i, j) for (i, j), valor in solucao_x.items() if valor == 1}


class SubproblemaBase(ABC):
    """Classe base para subproblemas do Lagrangeano"""
    
    def __init__(self, multiplicadores: Dict[Tuple[Tuple[int, int], int], float]):
        self.multiplicadores = multiplicadores
        
    @abstractmethod
    def resolver(self) -> Union[Dict, Tuple]:
        """Resolve o subproblema"""
        pass
        
    @abstractmethod
    def calcular_contribuicao_objetivo(self) -> float:
        """Calcula contribuição para função objetivo"""
        pass
    
    def _calcular_custo_reduzido(self, aresta: Tuple[int, int], custos: Dict[Tuple[int, int], float], 
                                commodities: Set[int]) -> float:
        """Método utilitário para calcular custo reduzido de uma aresta"""
        custo_reduzido = custos[aresta]
        for k in commodities:
            if (aresta, k) in self.multiplicadores:
                custo_reduzido -= self.multiplicadores[(aresta, k)]
        return custo_reduzido


# === SUBPROBLEMAS LAGRANGEANOS ===

class SubproblemaSelecaoArestas(SubproblemaBase):
    """
    Primeiro subproblema do Lagrangeano: Seleção de arestas
    Resolve para cada aresta (i,j) ∈ A fixando x_ij = 1 quando 
    c_ij - Σ_k u_ij^k < 0
    """
    
    def __init__(self, custos: Dict[Tuple[int, int], float], 
                 multiplicadores: Dict[Tuple[Tuple[int, int], int], float],
                 arestas: Set[Tuple[int, int]], 
                 commodities: Set[int]):
        super().__init__(multiplicadores)
        self.custos = custos
        self.arestas = arestas
        self.commodities = commodities
        self.solucao_x = {}
    
    def resolver(self) -> Dict[Tuple[int, int], int]:
        """
        Resolve o subproblema de seleção de arestas.
        Para cada aresta, verifica se o custo reduzido é negativo.
        """
        self.solucao_x = {}
        
        for aresta in self.arestas:
            custo_reduzido = self._calcular_custo_reduzido(aresta, self.custos, self.commodities)
            self.solucao_x[aresta] = 1 if custo_reduzido < 0 else 0
        
        return self.solucao_x
    
    def calcular_contribuicao_objetivo(self) -> float:
        """Calcula a contribuição deste subproblema para a função objetivo Lagrangeana."""
        contribuicao = 0.0
        
        for aresta in self.arestas:
            custo_reduzido = self._calcular_custo_reduzido(aresta, self.custos, self.commodities)
            if custo_reduzido < 0:
                contribuicao += custo_reduzido
        
        return contribuicao


class SubproblemaFluxoCommodity(SubproblemaBase):
    """
    Segundo subproblema do Lagrangeano: Fluxo de cada commodity
    Para cada k ∈ K\{s}, encontra o caminho de custo mínimo de s até k,
    onde o custo de cada aresta é u_ij^k
    """
    
    def __init__(self, grafo: nx.DiGraph, 
                 multiplicadores: Dict[Tuple[Tuple[int, int], int], float],
                 fonte: int, commodities: Set[int]):
        super().__init__(multiplicadores)
        self.grafo = grafo
        self.fonte = fonte
        self.commodities = commodities
        self.solucoes_fluxo = {}
    
    def _criar_grafo_commodity(self, k: int) -> nx.DiGraph:
        """Cria grafo auxiliar com custos u_ij^k para uma commodity específica"""
        grafo_aux = self.grafo.copy()
        
        for i, j in grafo_aux.edges():
            weight = self.multiplicadores.get(((i, j), k), 0.0)
            grafo_aux[i][j]['weight'] = weight
        
        return grafo_aux
    
    def _calcular_fluxo_caminho(self, caminho: List[int]) -> Dict[Tuple[int, int], float]:
        """Converte um caminho em um dicionário de fluxos"""
        fluxo = {aresta: 0.0 for aresta in self.grafo.edges()}
        
        for i in range(len(caminho) - 1):
            aresta = (caminho[i], caminho[i + 1])
            fluxo[aresta] = 1.0
        
        return fluxo
    
    def resolver_commodity(self, k: int) -> Dict[Tuple[int, int], float]:
        """Resolve o subproblema de fluxo para uma commodity específica k."""
        grafo_aux = self._criar_grafo_commodity(k)
        
        try:
            caminho = nx.shortest_path(grafo_aux, source=self.fonte, target=k, weight='weight')
            return self._calcular_fluxo_caminho(caminho)
        except nx.NetworkXNoPath:
            return {aresta: 0.0 for aresta in self.grafo.edges()}
    
    def resolver(self) -> Dict[int, Dict[Tuple[int, int], float]]:
        """Resolve o subproblema de fluxo para todas as commodities."""
        self.solucoes_fluxo = {}
        
        for k in self.commodities:
            self.solucoes_fluxo[k] = self.resolver_commodity(k)
        
        return self.solucoes_fluxo
      # Método de compatibilidade
    def resolver_todas_commodities(self) -> Dict[int, Dict[Tuple[int, int], float]]:
        """Método de compatibilidade - chama resolver()"""
        return self.resolver()
    
    def calcular_contribuicao_objetivo(self) -> float:
        """Calcula a contribuição deste subproblema para a função objetivo Lagrangeana."""
        contribuicao = 0.0
        
        for k in self.commodities:
            if k not in self.solucoes_fluxo:
                continue
                
            for aresta, fluxo in self.solucoes_fluxo[k].items():
                if (aresta, k) in self.multiplicadores and fluxo > 0:
                    contribuicao += self.multiplicadores[(aresta, k)] * fluxo
        
        return contribuicao


class SubproblemaLagrangeano:
    """
    Classe principal que coordena os dois subproblemas do Lagrangeano
    """
    
    def __init__(self, grafo: nx.DiGraph, custos: Dict[Tuple[int, int], float],
                 fonte: int, commodities: Set[int]):
        """
        Args:
            grafo: Grafo direcionado NetworkX
            custos: Dicionário {(i,j): c_ij} com custos das arestas
            fonte: Nó fonte (s)
            commodities: Conjunto de commodities K \ {s}
        """
        self.grafo = grafo
        self.custos = custos
        self.fonte = fonte
        self.commodities = commodities
        self.arestas = set(grafo.edges())
        
        # Inicializa os subproblemas
        self.subproblema_arestas = None
        self.subproblema_fluxo = None
    
    def resolver(self, multiplicadores: Dict[Tuple[Tuple[int, int], int], float]) -> Tuple[float, Dict, Dict]:
        """
        Resolve o subproblema Lagrangeano completo para dados multiplicadores.
        
        Args:
            multiplicadores: Dicionário {((i,j), k): u_ij^k} com multiplicadores Lagrangeanos
            
        Returns:
            Tuple com (valor_objetivo, solucao_x, solucoes_fluxo)
        """
        # Cria e resolve o subproblema de seleção de arestas
        self.subproblema_arestas = SubproblemaSelecaoArestas(
            self.custos, multiplicadores, self.arestas, self.commodities
        )
        solucao_x = self.subproblema_arestas.resolver()
        
        # Cria e resolve o subproblema de fluxo
        self.subproblema_fluxo = SubproblemaFluxoCommodity(
            self.grafo, multiplicadores, self.fonte, self.commodities
        )
        solucoes_fluxo = self.subproblema_fluxo.resolver_todas_commodities()
        
        # Calcula o valor objetivo total
        valor_objetivo = (self.subproblema_arestas.calcular_contribuicao_objetivo() + 
                         self.subproblema_fluxo.calcular_contribuicao_objetivo())
        
        return valor_objetivo, solucao_x, solucoes_fluxo


class FormulacaoMulticommodity(ModeloBase):
    """Formulação Multicommodity simplificada para Steiner Tree"""
    
    def criar_modelo(self):
        """Cria o modelo completo"""
        self.modelo = pulp.LpProblem("Steiner_Tree", pulp.LpMinimize)
        self._criar_variaveis()
        self._criar_funcao_objetivo()
        self._adicionar_restricoes_fluxo()
        self._adicionar_restricoes_acoplamento()
    
    def resolver(self, time_limit=None, msg=False, solver=None):
        """Resolve o modelo"""
        if self.modelo is None:
            self.criar_modelo()
        
        # Configura solver
        if solver is None:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=msg) if time_limit else pulp.PULP_CBC_CMD(msg=msg)
        
        self.modelo.solve(solver)
        self.status = self.modelo.status
        
        if self.status == pulp.LpStatusOptimal:
            self.valor_objetivo = pulp.value(self.modelo.objective)
            return self.status, self.valor_objetivo, *self._extrair_solucoes()
        
        return self.status, None, None, None


def carregar_instancia_steiner(caminho_arquivo: str) -> Tuple[nx.DiGraph, Dict[Tuple[int, int], float], int, Set[int]]:
    """Carrega uma instância do problema de Steiner Tree"""
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
    
    # Lê estrutura básica
    num_nos, num_arcos = map(int, linhas[0].strip().split())
    grafo = nx.DiGraph()
    custos = {}
    grafo.add_nodes_from(range(1, num_nos + 1))
    
    # Lê arestas (grafo não orientado -> ambas direções)
    for i in range(1, num_arcos + 1):
        no_i, no_j, custo = map(lambda x: int(x) if x.isdigit() or (x[0] == '-' and x[1:].isdigit()) else float(x), 
                                linhas[i].strip().split())
        grafo.add_edge(no_i, no_j)
        grafo.add_edge(no_j, no_i)
        custos[(no_i, no_j)] = custos[(no_j, no_i)] = custo
    
    # Lê terminais
    linha_terminais_idx = num_arcos + 1
    num_terminais = int(linhas[linha_terminais_idx].strip())
    terminais = []
    linha_atual = linha_terminais_idx + 1
    
    while linha_atual < len(linhas) and len(terminais) < num_terminais:
        linha = linhas[linha_atual].strip()
        if linha:
            terminais.extend(map(int, linha.split()))
        linha_atual += 1
    
    terminais = set(terminais[:num_terminais])
    if len(terminais) != num_terminais:
        raise ValueError(f"Terminais: esperado {num_terminais}, encontrado {len(terminais)}")
    
    fonte = min(terminais)
    print(f"Carregada: {num_nos} nós, {num_arcos} arestas, {len(terminais)} terminais, fonte={fonte}")
    return grafo, custos, fonte, terminais


class ExperimentoSteinerTree:
    """Classe simplificada para experimentos"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.resultados = []
    
    def _criar_multiplicadores_zero(self, grafo, commodities):
        """Cria multiplicadores zerados"""
        return {((i, j), k): 0.0 for (i, j) in grafo.edges() for k in commodities}
    
    def _resolver_upper_bound(self, grafo, custos, fonte, terminais):
        """Resolve upper bound com configuração simplificada"""
        modelo_ub = FormulacaoMulticommodity(grafo, custos, fonte, terminais)
        tempo_inicio = time.time()
        
        status_ub, valor_ub, sol_x_ub, sol_f_ub = modelo_ub.resolver(
            time_limit=self.config.limite_tempo, msg=self.config.mostrar_msgs_solver
        )
        tempo_ub = time.time() - tempo_inicio
        
        # Tenta com mais tempo se necessário
        if status_ub != pulp.LpStatusOptimal and self.config.limite_tempo < 600:
            tempo_inicio = time.time()
            status_ub, valor_ub, sol_x_ub, sol_f_ub = modelo_ub.resolver(
                time_limit=self.config.limite_tempo * self.config.multiplicador_tempo_ub, 
                msg=self.config.mostrar_msgs_solver
            )
            tempo_ub += time.time() - tempo_inicio
        
        if status_ub == pulp.LpStatusOptimal:
            print(f"✓ Upper Bound: {valor_ub} (tempo: {tempo_ub:.2f}s)")
            return valor_ub, tempo_ub, 'Optimal'
        else:
            # Tenta heurística
            ub_heur = self._calcular_upper_bound_heuristico(grafo, custos, fonte, terminais)
            if ub_heur:
                print(f"✓ Upper Bound heurístico: {ub_heur}")
                return ub_heur, tempo_ub, 'Heuristic'
            else:
                print(f"✗ Upper Bound não encontrado")
                return None, tempo_ub, 'Failed'
    
    def _resolver_bounds(self, grafo, custos, fonte, commodities, z_star=None):
        """Resolve lower bounds de forma unificada"""
        # Lower bound inicial (u=0)
        multiplicadores_zero = self._criar_multiplicadores_zero(grafo, commodities)
        subproblema = SubproblemaLagrangeano(grafo, custos, fonte, commodities)
        
        tempo_inicio = time.time()
        valor_u0, _, _ = subproblema.resolver(multiplicadores_zero)
        tempo_u0 = time.time() - tempo_inicio
        
        print(f"✓ Lower Bound (u=0): {valor_u0}")
        
        # Subgradiente
        metodo_subgradiente = MetodoSubgradiente(
            subproblema, 
            limite_superior_inicial=z_star,
            lambda_inicial=self.config.lambda_inicial,
            max_iteracoes=self.config.max_iteracoes,
            tolerancia=self.config.tolerancia,
            precisao_gap=self.config.precisao_gap,
            intervalo_impressao=self.config.intervalo_impressao
        )
        
        tempo_inicio = time.time()
        resultados_subgrad = metodo_subgradiente.resolver()
        tempo_subgrad = time.time() - tempo_inicio
        
        valor_subgrad = resultados_subgrad['melhor_valor_dual']
        print(f"✓ Lower Bound (subgradiente): {valor_subgrad} (iter: {resultados_subgrad['iteracoes_executadas']})")
        
        return {
            'lower_bound_u0': valor_u0,
            'lower_bound_subgradiente': valor_subgrad,
            'tempo_lower_bound': tempo_u0,
            'tempo_subgradiente': tempo_subgrad,
            'iteracoes_subgradiente': resultados_subgrad['iteracoes_executadas'],
            'convergiu': resultados_subgrad['convergiu']
        }
    def _calcular_gaps(self, valor_ub, valor_lag, valor_subgrad, status_ub="Optimal"):
        """Calcula gaps entre upper e lower bounds"""
        if valor_ub is None:
            print(f"\n--- COMPARAÇÃO DE BOUNDS ---")
            print(f"Upper Bound: Não disponível")
            print(f"Lower Bound (u=0): {valor_lag}")
            print(f"Lower Bound (Subgradiente): {valor_subgrad}")
            print(f"Gap: Não pode ser calculado sem Upper Bound")
            print(f"Melhoria do Lower Bound: {valor_subgrad - valor_lag:.4f}")
            
            return {
                'melhoria_lower_bound': valor_subgrad - valor_lag
            }
        
        gap_absoluto_u0 = valor_ub - valor_lag
        gap_relativo_u0 = (gap_absoluto_u0 / valor_ub) * 100 if valor_ub > 0 else 0
        
        gap_absoluto_subgrad = valor_ub - valor_subgrad
        gap_relativo_subgrad = (gap_absoluto_subgrad / valor_ub) * 100 if valor_ub > 0 else 0
        
        # Indicador de qualidade do upper bound
        ub_tipo = "ótimo" if status_ub == "Optimal" else "heurístico"
        
        print(f"\n--- COMPARAÇÃO DE BOUNDS ---")
        print(f"Upper Bound ({ub_tipo}): {valor_ub}")
        print(f"Lower Bound (u=0): {valor_lag}")
        print(f"Lower Bound (Subgradiente): {valor_subgrad}")
        print(f"Gap Absoluto (u=0): {gap_absoluto_u0}")
        print(f"Gap Relativo (u=0): {gap_relativo_u0:.2f}%")
        print(f"Gap Absoluto (Subgradiente): {gap_absoluto_subgrad}")
        print(f"Gap Relativo (Subgradiente): {gap_relativo_subgrad:.2f}%")
        print(f"Melhoria do Lower Bound: {valor_subgrad - valor_lag:.4f}")
        
        if status_ub != "Optimal":
            print(f"Nota: Gap pode ser superestimado (upper bound {ub_tipo})")
        
        return {
            'gap_absoluto_u0': gap_absoluto_u0,
            'gap_relativo_u0': gap_relativo_u0,
            'gap_absoluto_subgrad': gap_absoluto_subgrad,
            'gap_relativo_subgrad': gap_relativo_subgrad,
            'melhoria_lower_bound': valor_subgrad - valor_lag,
            'tipo_upper_bound': status_ub
        }
    
    def executar_instancia(self, nome_arquivo: str) -> Resultado:
        """Executa uma instância específica de forma simplificada"""
        caminho_completo = f"{self.config.pasta_dados}\\{nome_arquivo}"
        
        print(f"\n{'='*60}")
        print(f"EXECUTANDO: {nome_arquivo}")
        print(f"{'='*60}")
        
        try:
            # Carrega instância
            grafo, custos, fonte, terminais = carregar_instancia_steiner(caminho_completo)
            commodities = terminais - {fonte}
            
            # Cria resultado base
            resultado = Resultado(
                instancia=nome_arquivo,
                num_nos=grafo.number_of_nodes(),
                num_arestas=grafo.number_of_edges() // 2,
                num_terminais=len(terminais),
                fonte=fonte,
                terminais=terminais
            )
            
            tempo_total_inicio = time.time()
            
            # 1. Upper bound
            ub, tempo_ub, status_ub = self._resolver_upper_bound(grafo, custos, fonte, terminais)
            resultado.upper_bound = ub
            resultado.status = status_ub
            
            # 2. Lower bounds
            bounds_result = self._resolver_bounds(grafo, custos, fonte, commodities, ub)
            resultado.lower_bound_u0 = bounds_result['lower_bound_u0']
            resultado.lower_bound_subgradiente = bounds_result['lower_bound_subgradiente']
            resultado.iteracoes_subgradiente = bounds_result['iteracoes_subgradiente']
            resultado.convergiu = bounds_result['convergiu']
            
            # 3. Calcula métricas finais
            resultado.tempo_total = time.time() - tempo_total_inicio
            resultado.calcular_metricas()
            
            # Imprime resumo
            print(f"\n--- RESUMO ---")
            if resultado.upper_bound:
                print(f"UB: {resultado.upper_bound}")
                print(f"LB: {resultado.lower_bound_subgradiente}")
                print(f"Gap: {resultado.gap_relativo_final:.2f}%")
            else:
                print(f"LB: {resultado.lower_bound_subgradiente}")
                print(f"Melhoria: {resultado.melhoria_lower_bound:.4f}")
            print(f"Tempo total: {resultado.tempo_total:.2f}s")
            
            self.resultados.append(resultado)
            return resultado
            
        except Exception as e:
            print(f"✗ Erro: {str(e)}")
            resultado_erro = Resultado(instancia=nome_arquivo, erro=str(e))
            self.resultados.append(resultado_erro)
            return resultado_erro
    
    def executar_serie(self, serie: str = 'steinb', instancias: List[int] = None):
        """Executa uma série de instâncias"""
        if instancias is None:
            # Auto-detecta instâncias
            try:
                arquivos = os.listdir(self.config.pasta_dados)
                instancias = []
                for arquivo in arquivos:
                    if arquivo.startswith(serie) and arquivo.endswith('.txt'):
                        try:
                            num = int(arquivo.replace(serie, '').replace('.txt', ''))
                            instancias.append(num)
                        except ValueError:
                            continue
                instancias.sort()
            except Exception:
                instancias = []
        
        print(f"\n{'#'*60}")
        print(f"SÉRIE: {serie.upper()} - {len(instancias)} instâncias")
        print(f"{'#'*60}")
        
        for num_inst in instancias:
            nome_arquivo = f"{serie}{num_inst}.txt"
            caminho = f"{self.config.pasta_dados}\\{nome_arquivo}"
            if os.path.exists(caminho):
                self.executar_instancia(nome_arquivo)
            else:
                print(f"ARQUIVO {nome_arquivo} NÃO ENCONTRADO")
    
    def gerar_relatorio(self):
        """Gera relatório simplificado"""
        print(f"\n{'#'*80}")
        print(f"RELATÓRIO FINAL - {len(self.resultados)} instâncias")
        print(f"{'#'*80}")
        
        if not self.resultados:
            print("Nenhum resultado disponível.")
            return
        
        # Cabeçalho da tabela
        print(f"{'Instância':<12} {'Nós':<5} {'Term':<5} {'UB':<8} {'LB':<8} {'Gap%':<8} {'Iter':<6} {'Tempo':<7}")
        print("-" * 80)
        
        # Dados de cada instância
        for resultado in self.resultados:
            if resultado.erro:
                print(f"{resultado.instancia:<12} ERRO: {resultado.erro[:50]}")
                continue
            
            gap = f"{resultado.gap_relativo_final:.2f}" if resultado.gap_relativo_final else "N/A"
            ub = f"{resultado.upper_bound:.1f}" if resultado.upper_bound else "N/A"
            lb = f"{resultado.lower_bound_subgradiente:.1f}" if resultado.lower_bound_subgradiente else "N/A"
            
            print(f"{resultado.instancia:<12} {resultado.num_nos:<5} {resultado.num_terminais:<5} "
                  f"{ub:<8} {lb:<8} {gap:<8} {resultado.iteracoes_subgradiente:<6} {resultado.tempo_total:<7.1f}")
        
        # Estatísticas
        resultados_validos = [r for r in self.resultados if not r.erro and r.gap_relativo_final is not None]
        if resultados_validos:
            gaps = [r.gap_relativo_final for r in resultados_validos]
            print(f"\n--- ESTATÍSTICAS ---")
            print(f"Instâncias resolvidas: {len(resultados_validos)}")
            print(f"Gap médio: {sum(gaps)/len(gaps):.2f}%")
            print(f"Gap mínimo: {min(gaps):.2f}%")
            print(f"Gap máximo: {max(gaps):.2f}%")
    
    def _calcular_upper_bound_heuristico(self, grafo, custos, fonte, terminais):
        """Upper bound heurístico usando MST"""
        try:
            import networkx as nx
            grafo_nao_dir = nx.Graph()
            
            for (i, j), custo in custos.items():
                if not grafo_nao_dir.has_edge(i, j):
                    grafo_nao_dir.add_edge(i, j, weight=custo)
            
            subgrafo = grafo_nao_dir.subgraph(terminais)
            
            if not nx.is_connected(subgrafo):
                grafo_terminais = nx.Graph()
                grafo_terminais.add_nodes_from(terminais)
                
                for i, t1 in enumerate(terminais):
                    for j, t2 in enumerate(terminais):
                        if i < j:
                            try:
                                peso = nx.shortest_path_length(grafo_nao_dir, t1, t2, weight='weight')
                                grafo_terminais.add_edge(t1, t2, weight=peso)      
                            except nx.NetworkXNoPath:
                                return None
                
                mst = nx.minimum_spanning_tree(grafo_terminais, weight='weight')
            else:
                mst = nx.minimum_spanning_tree(subgrafo, weight='weight')
            
            return sum(data['weight'] for _, _, data in mst.edges(data=True))
        except Exception:
            return None
    
    def gerar_graficos_convergencia(self):
        """Gera gráficos de convergência para qualquer instância disponível"""
        print(f"\n{'#'*60}")
        print("GERAÇÃO DE GRÁFICOS DE CONVERGÊNCIA")
        print(f"{'#'*60}")
        
        # Lista todas as instâncias disponíveis na pasta
        try:
            arquivos_txt = [f for f in os.listdir(self.config.pasta_dados) if f.endswith('.txt')]
            if not arquivos_txt:
                print("Nenhuma instância (.txt) encontrada na pasta de dados.")
                return
            
            arquivos_txt.sort()
            print(f"\nInstâncias disponíveis ({len(arquivos_txt)}):")
            
            for i, arquivo in enumerate(arquivos_txt):
                print(f"{i+1:2d}. {arquivo}")
            
            print("\nOpções:")
            print("- Digite o número de uma instância específica")
            print("- Digite 'todas' para gerar gráficos de todas as instâncias")
            print("- Digite 'steinb' ou 'steinc' para uma série específica")
            
            escolha = input("\nSua escolha: ").strip().lower()
            
            if escolha == 'todas':
                # Gera gráfico para todas as instâncias
                print(f"\nGerando gráficos para {len(arquivos_txt)} instâncias...")
                for arquivo in arquivos_txt:
                    print(f"\n{'='*50}")
                    self._executar_analise_convergencia_detalhada(arquivo)
                    
            elif escolha in ['steinb', 'steinc']:
                # Filtra instâncias da série
                serie_arquivos = [f for f in arquivos_txt if f.startswith(escolha)]
                if serie_arquivos:
                    print(f"\nGerando gráficos para {len(serie_arquivos)} instâncias da série {escolha.upper()}...")
                    for arquivo in serie_arquivos:
                        print(f"\n{'='*50}")
                        self._executar_analise_convergencia_detalhada(arquivo)
                else:
                    print(f"Nenhuma instância da série {escolha.upper()} encontrada.")
                    
            else:
                # Instância específica por número
                try:
                    idx = int(escolha) - 1
                    if 0 <= idx < len(arquivos_txt):
                        arquivo = arquivos_txt[idx]
                        print(f"\nGerando gráfico para: {arquivo}")
                        self._executar_analise_convergencia_detalhada(arquivo)
                    else:
                        print("Número inválido.")
                except ValueError:
                    print("Entrada inválida. Digite um número, 'todas', 'steinb' ou 'steinc'.")
                    
        except Exception as e:
            print(f"Erro ao listar arquivos: {e}")
            return
    
    def _executar_analise_convergencia_detalhada(self, nome_arquivo: str):
        """Executa análise detalhada de convergência para uma instância"""
        print(f"\n--- Análise de Convergência: {nome_arquivo} ---")
        
        try:
            # Carrega instância
            caminho = f"{self.config.pasta_dados}\\{nome_arquivo}"
            if not os.path.exists(caminho):
                print(f"Arquivo {caminho} não encontrado.")
                return
            
            grafo, custos, fonte, terminais = carregar_instancia_steiner(caminho)
            commodities = list(terminais)
            
            # Resolve upper bound
            valor_ub, _, _ = self._resolver_upper_bound(grafo, custos, fonte, terminais)
            
            # Cria subproblema
            subproblema = SubproblemaLagrangeano(grafo, custos, fonte, commodities)
            
            # Executa subgradiente com histórico
            metodo_historico = MetodoSubgradienteComHistorico(
                subproblema,
                limite_superior_inicial=valor_ub,                lambda_inicial=self.config.lambda_inicial,
                max_iteracoes=self.config.max_iteracoes,
                tolerancia=self.config.tolerancia,
                precisao_gap=self.config.precisao_gap,
                intervalo_impressao=self.config.intervalo_impressao
            )
            
            print("Executando método do subgradiente com histórico...")
            print("Modo: ITERAÇÕES INFINITAS até gap de 1.0% (convergência por gap)")
            resultados = metodo_historico.resolver(convergir_para_gap=True, gap_objetivo=0.01)
            
            # Gera gráfico
            nome_grafico = f"convergencia_{nome_arquivo.replace('.txt', '')}.png"
            metodo_historico.gerar_grafico_convergencia(salvar_arquivo=nome_grafico)
            
            print(f"✓ Gráfico salvo: {nome_grafico}")
            
        except Exception as e:
            print(f"Erro na análise de convergência: {e}")
    
    def executar_instancia_com_grafico(self, nome_arquivo: str):
        """Executa uma instância e gera gráfico de convergência"""
        # Executa a instância normalmente
        self.executar_instancia(nome_arquivo)
        
        # Gera gráfico para esta instância
        self._executar_analise_convergencia_detalhada(nome_arquivo)


class MetodoSubgradiente:
    """
    Implementação do método do subgradiente para atualizar multiplicadores Lagrangeanos.
    
    Fórmula: u^{k+1} = u^k + t_k * (Ax^k - b)    onde:
    - (Ax^k - b) é o subgradiente calculado a partir da violação das restrições
    - t_k é o tamanho do passo calculado dinamicamente baseado no λ FIXO
    - λ permanece constante durante toda a execução (não é ajustado automaticamente)
    """
    
    def __init__(self, subproblema_lagrangeano: SubproblemaLagrangeano,
                 limite_superior_inicial: float = None, lambda_inicial: float = 2.0,
                 max_iteracoes: int = 10000, tolerancia: float = 1e-3,
                 precisao_gap: float = 1e-3, intervalo_impressao: int = 10):
        """
        Args:
            subproblema_lagrangeano: Instância do subproblema Lagrangeano
            limite_superior_inicial: Estimativa inicial do valor ótimo Z*
            lambda_inicial: Parâmetro λ fixo (entre 0 e 2) - não será alterado durante a execução
            max_iteracoes: Número máximo de iterações            tolerancia: Tolerância para convergência baseada na melhoria do valor dual
            precisao_gap: Precisão do gap relativo para parada (padrão: 0.001 = 0.1%)
            intervalo_impressao: Intervalo de iterações para impressão do progresso
        """
        self.subproblema = subproblema_lagrangeano
        self.Z_star = limite_superior_inicial
        self.lambda_k = lambda_inicial
        self.max_iteracoes = max_iteracoes
        self.tolerancia = tolerancia
        self.precisao_gap = precisao_gap
        self.intervalo_impressao = intervalo_impressao
        
        # Histórico do algoritmo
        self.historico = {
            'iteracao': [],
            'valor_dual': [],
            'lambda': [],
            'tamanho_passo': [],
            'norma_subgradiente': [],
            'gap_estimado': []
        }
          # Controle de melhoria
        self.melhor_valor_dual = float('-inf')
        self.melhores_multiplicadores = None
        
    def calcular_subgradiente(self, solucao_x: Dict[Tuple[int, int], int], 
                             solucao_f: Dict[int, Dict[Tuple[int, int], float]]) -> Dict[Tuple[Tuple[int, int], int], float]:
        """
        Calcula o subgradiente (Ax^k - b) baseado na violação das restrições dualizadas.
        
        Para a restrição dualizada: f_ij^k <= x_ij
        O subgradiente é: g_ij^k = f_ij^k - x_ij
        
        Args:
            solucao_x: Solução das variáveis x_ij
            solucao_f: Solução das variáveis f_ij^k
            
        Returns:
            Subgradiente para cada (aresta, commodity)
        """
        subgradiente = {}
        
        for k in self.subproblema.commodities:
            for aresta in self.subproblema.arestas:
                # g_ij^k = f_ij^k - x_ij (violação da restrição de acoplamento)
                f_valor = solucao_f.get(k, {}).get(aresta, 0.0)
                x_valor = solucao_x.get(aresta, 0)
                
                subgradiente[(aresta, k)] = f_valor - x_valor
        
        return subgradiente
    
    def calcular_norma_subgradiente(self, subgradiente: Dict[Tuple[Tuple[int, int], int], float]) -> float:
        """
        Calcula a norma euclidiana ao quadrado do subgradiente.
        
        Args:
            subgradiente: Vetor subgradiente
            
        Returns:
            ||subgradiente||²        """
        return sum(valor**2 for valor in subgradiente.values())
    
    def calcular_tamanho_passo(self, valor_dual_atual: float, norma_subgradiente: float) -> float:
        """
        Calcula o tamanho do passo usando a fórmula:
        t_k = λ * (Z* - Z_D(u^k)) / ||Ax^k - b||²
        
        Onde λ é mantido fixo durante toda a execução.
        
        Args:
            valor_dual_atual: Z_D(u^k)
            norma_subgradiente: ||Ax^k - b||²
            
        Returns:
            Tamanho do passo t_k
        """
        if norma_subgradiente == 0:
            return 0.0
        
        if self.Z_star is None:
            # Se não temos estimativa de Z*, usa passo baseado em λ fixo e iteração
            iteracao_atual = len(self.historico['iteracao']) + 1
            return self.lambda_k / iteracao_atual
        
        numerador = self.Z_star - valor_dual_atual
        
        # Fórmula padrão com λ fixo
        return max(0, self.lambda_k * numerador / norma_subgradiente)
        
        # Garante que o passo seja positivo e limitado, mas proporcional a λ
          
    
    def atualizar_multiplicadores(self, multiplicadores_atuais: Dict[Tuple[Tuple[int, int], int], float],
                                 subgradiente: Dict[Tuple[Tuple[int, int], int], float],
                                 tamanho_passo: float) -> Dict[Tuple[Tuple[int, int], int], float]:
        """
        Atualiza os multiplicadores usando: u^{k+1} = max(0, u^k + t_k * subgradiente)
        
        Args:
            multiplicadores_atuais: u^k
            subgradiente: Direção de atualização
            tamanho_passo: t_k
            
        Returns:
            Novos multiplicadores u^{k+1}
        """
        novos_multiplicadores = {}
        
        for chave in multiplicadores_atuais:
            valor_atual = multiplicadores_atuais[chave]
            gradiente = subgradiente.get(chave, 0.0)
              # u^{k+1} = max(0, u^k + t_k * g^k)
            novo_valor = valor_atual + tamanho_passo * gradiente
            novos_multiplicadores[chave] = max(0.0, novo_valor)
        
        return novos_multiplicadores
    
    def verificar_convergencia(self, valor_dual_atual: float) -> bool:
        """
        Verifica se o algoritmo convergiu.
        
        Args:
            valor_dual_atual: Valor atual da função dual
            
        Returns:
            True se convergiu, False caso contrário
        """
        if len(self.historico['valor_dual']) < 2:
            return False
        
        # Atualiza melhor valor se houver melhoria
        if valor_dual_atual > self.melhor_valor_dual + self.tolerancia:
            self.melhor_valor_dual = valor_dual_atual
        
        # Critério de parada: gap pequeno (se temos Z*)
        if self.Z_star is not None:
            gap_relativo = abs(self.Z_star - valor_dual_atual) / max(abs(self.Z_star), 1)
            if gap_relativo < self.precisao_gap:
                return True
        
        # Nunca para por falta de melhorias - continua sempre até atingir max_iteracoes ou precisao_gap
        return False
    
        # λ permanece constante durante toda a execução    
    def resolver(self, multiplicadores_iniciais: Dict[Tuple[Tuple[int, int], int], float] = None,
                convergir_para_precisao: bool = False) -> Dict:
        """
        Executa o método do subgradiente para resolver o problema dual Lagrangeano.
        
        Args:
            multiplicadores_iniciais: Multiplicadores iniciais (se None, usa zeros)
            convergir_para_precisao: Se True, executa até atingir precisao_gap (iterações "infinitas")
            
        Returns:
            Dicionário com resultados do algoritmo
        """
        # Inicializa multiplicadores se não fornecidos
        if multiplicadores_iniciais is None:
            multiplicadores_atuais = {}
            for aresta in self.subproblema.arestas:
                for k in self.subproblema.commodities:
                    multiplicadores_atuais[(aresta, k)] = 0.0
        else:
            multiplicadores_atuais = multiplicadores_iniciais.copy()
        
        print(f"=== MÉTODO DO SUBGRADIENTE ===")
        if convergir_para_precisao:
            print(f"Modo: Convergir para precisão {self.precisao_gap:.1e}")
            print(f"Max iterações: Infinitas (até atingir precisão)")
        else:
            print(f"Max iterações: {self.max_iteracoes}")
        print(f"λ inicial: {self.lambda_k}")
        print(f"Z* estimado: {self.Z_star}")
        print()
        
        iteracao = 0
        max_iter_loop = float('inf') if convergir_para_precisao else self.max_iteracoes
        
        while iteracao < max_iter_loop:
            # Resolve subproblema Lagrangeano
            valor_dual, sol_x, sol_f = self.subproblema.resolver(multiplicadores_atuais)
            
            # Calcula subgradiente
            subgradiente = self.calcular_subgradiente(sol_x, sol_f)
            norma_subgradiente = self.calcular_norma_subgradiente(subgradiente)
            
            # Calcula tamanho do passo
            tamanho_passo = self.calcular_tamanho_passo(valor_dual, norma_subgradiente)
            
            # Atualiza multiplicadores
            multiplicadores_atuais = self.atualizar_multiplicadores(
                multiplicadores_atuais, subgradiente, tamanho_passo
            )
            
            # Calcula gap estimado
            gap_estimado = (self.Z_star - valor_dual) if self.Z_star else None
            
            # Armazena histórico
            self.historico['iteracao'].append(iteracao)
            self.historico['valor_dual'].append(valor_dual)
            self.historico['lambda'].append(self.lambda_k)
            self.historico['tamanho_passo'].append(tamanho_passo)
            self.historico['norma_subgradiente'].append(norma_subgradiente)
            self.historico['gap_estimado'].append(gap_estimado)
            
            # Salva melhores multiplicadores
            if valor_dual > self.melhor_valor_dual:
                self.melhor_valor_dual = valor_dual
                self.melhores_multiplicadores = multiplicadores_atuais.copy()              # Imprime progresso
            if (iteracao % self.intervalo_impressao == 0 or 
                iteracao < min(10, self.intervalo_impressao) or 
                iteracao == self.max_iteracoes - 1):
                gap_str = f"{gap_estimado:.4f}" if gap_estimado else "N/A"
                gap_pct = f"{(gap_estimado/self.Z_star)*100:.2f}%" if gap_estimado and self.Z_star else "N/A"
                print(f"Iter {iteracao:3d}: Z_D = {valor_dual:8.4f}, "
                      f"||g|| = {norma_subgradiente:8.4f}, "
                      f"t_k = {tamanho_passo:6.4f}, "
                      f"λ = {self.lambda_k:5.3f}, "
                      f"Gap = {gap_str} ({gap_pct})")              # Verifica convergência
            if self.verificar_convergencia(valor_dual):
                print(f"\nConvergência atingida na iteração {iteracao}")
                break
            
            # Condição especial para convergir para precisão
            if (convergir_para_precisao and self.Z_star is not None and 
                gap_estimado is not None and gap_estimado <= self.precisao_gap * self.Z_star):
                print(f"\nPrecisão desejada atingida na iteração {iteracao}")
                print(f"Gap: {gap_estimado:.6f} <= {self.precisao_gap * self.Z_star:.6f}")
                break
            
            # λ é mantido fixo durante toda a execução
            # Apenas o tamanho do passo é ajustado dinamicamente
            
            # Incrementa iteração
            iteracao += 1
          # Preparar resultados
        gap_final = (self.Z_star - self.melhor_valor_dual) if self.Z_star else None
        convergiu_por_gap = (gap_final is not None and 
                            gap_final <= self.precisao_gap * self.Z_star if self.Z_star else False)
        
        resultados = {
            'iteracoes_executadas': len(self.historico['iteracao']),
            'melhor_valor_dual': self.melhor_valor_dual,
            'melhores_multiplicadores': self.melhores_multiplicadores,
            'multiplicadores_finais': multiplicadores_atuais,
            'convergiu': convergiu_por_gap,
            'historico': self.historico,
            'gap_final': gap_final
        }
        
        print(f"\nResultados finais:")
        print(f"Melhor valor dual: {self.melhor_valor_dual:.6f}")
        if self.Z_star:
            print(f"Gap final: {resultados['gap_final']:.6f}")
        print(f"Iterações executadas: {resultados['iteracoes_executadas']}")
        
        return resultados
    
    def gerar_grafico_convergencia(self, salvar_arquivo=None):
        """
        Gera gráfico de convergência do método do subgradiente.
        
        Args:
            salvar_arquivo: Caminho para salvar o gráfico (opcional)
        """
        if not self.historico['iteracao']:
            print("Nenhum histórico disponível para plotar.")
            return
        
        # Configuração do gráfico
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Valor Dual vs Iterações
        plt.subplot(2, 2, 1)
        plt.plot(self.historico['iteracao'], self.historico['valor_dual'], 
                'b-', linewidth=2, label='Lower Bound (Dual)')
        
        # Linha do Upper Bound (se disponível)
        if self.Z_star is not None:
            plt.axhline(y=self.Z_star, color='r', linestyle='--', linewidth=2, 
                       label=f'Upper Bound = {self.Z_star:.4f}')
        
        plt.xlabel('Iteração')
        plt.ylabel('Valor Objetivo')
        plt.title('Convergência do Lower Bound')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Gap vs Iterações (se Z* disponível)
        if self.Z_star is not None and self.historico['gap_estimado']:
            plt.subplot(2, 2, 2)
            gaps_pct = [(gap/self.Z_star)*100 if gap and self.Z_star > 0 else 0 
                       for gap in self.historico['gap_estimado']]
            plt.plot(self.historico['iteracao'], gaps_pct, 
                    'g-', linewidth=2, label='Gap Relativo (%)')
            
            # Linha de precisão desejada
            plt.axhline(y=(self.precisao_gap*100), color='orange', linestyle=':', 
                       linewidth=2, label=f'Meta = {self.precisao_gap*100:.1f}%')
            
            plt.xlabel('Iteração')
            plt.ylabel('Gap Relativo (%)')
            plt.title('Evolução do Gap')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Escala logarítmica para melhor visualização
        
        # Subplot 3: Tamanho do Passo vs Iterações
        plt.subplot(2, 2, 3)
        plt.plot(self.historico['iteracao'], self.historico['tamanho_passo'], 
                'purple', linewidth=2, label='Tamanho do Passo')
        plt.xlabel('Iteração')
        plt.ylabel('Tamanho do Passo')
        plt.title('Evolução do Tamanho do Passo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Norma do Subgradiente vs Iterações
        plt.subplot(2, 2, 4)
        plt.plot(self.historico['iteracao'], self.historico['norma_subgradiente'], 
                'orange', linewidth=2, label='||Subgradiente||²')
        plt.xlabel('Iteração')
        plt.ylabel('Norma do Subgradiente')
        plt.title('Evolução da Norma do Subgradiente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Escala logarítmica
        
        # Ajustar layout
        plt.tight_layout()
        
        # Título geral
        if self.Z_star is not None:
            gap_final = (self.Z_star - self.melhor_valor_dual) / self.Z_star * 100
            plt.suptitle(f'Convergência do Método do Subgradiente\n'
                        f'λ = {self.lambda_k}, Gap Final = {gap_final:.3f}%', 
                        fontsize=14, y=0.98)
        else:
            plt.suptitle(f'Convergência do Método do Subgradiente\n'
                        f'λ = {self.lambda_k}, Melhor LB = {self.melhor_valor_dual:.4f}', 
                        fontsize=14, y=0.98)
        
        # Salvar ou mostrar
        if salvar_arquivo:
            plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar_arquivo}")
        else:
            plt.show()
        
        plt.close()
    
    def executar_analise_convergencia(self, nome_arquivo: str, 
                                     lambda_inicial: float = 2.0,
                                     precisao_gap: float = 0.001,
                                     max_iteracoes: int = 1000,
                                     intervalo_impressao: int = 50,
                                     salvar_grafico: str = None):
        """
        Executa análise de convergência com gráfico para uma instância específica.
        
        Args:
            nome_arquivo: Nome do arquivo da instância
            lambda_inicial: Valor do parâmetro λ (fixo durante toda execução)
            precisao_gap: Precisão do gap para parada (padrão: 0.001 = 0.1%)
            max_iteracoes: Máximo de iterações
            intervalo_impressao: Intervalo para impressão do progresso
            salvar_grafico: Caminho para salvar gráfico (opcional)
        """
        caminho_completo = f"{self.pasta_dados}\\{nome_arquivo}"
        
        print(f"\n{'='*80}")
        print(f"ANÁLISE DE CONVERGÊNCIA: {nome_arquivo}")
        print(f"λ = {lambda_inicial}, Precisão Gap = {precisao_gap:.1e}")
        print(f"{'='*80}")
        
        try:
            # Carrega a instância
            grafo, custos, fonte, terminais = carregar_instancia_steiner(caminho_completo)
            commodities = terminais - {fonte}
            
            print(f"Instância: {grafo.number_of_nodes()} nós, {grafo.number_of_edges()//2} arestas, {len(terminais)} terminais")
            
            # 1. Calcula Upper Bound
            print(f"\n--- CALCULANDO UPPER BOUND ---")
            resultado_ub = self._resolver_upper_bound(grafo, custos, fonte, terminais, 600, False)
            upper_bound = resultado_ub.get('upper_bound')
            
            if upper_bound is None:
                print("⚠️ Sem upper bound disponível - análise limitada")
                return None
            
            print(f"✓ Upper Bound: {upper_bound} ({resultado_ub.get('status_upper_bound')})")
            
            # 2. Cria subproblema
            multiplicadores_zero = self._criar_multiplicadores_zero(grafo, commodities)
            subproblema = SubproblemaLagrangeano(grafo, custos, fonte, commodities)
            
            # 3. Executa método do subgradiente com convergência para precisão
            metodo_subgradiente = MetodoSubgradiente(
                subproblema, 
                limite_superior_inicial=upper_bound,
                lambda_inicial=lambda_inicial,
                max_iteracoes=max_iteracoes,
                tolerancia=1e-6,
                precisao_gap=precisao_gap,
                intervalo_impressao=intervalo_impressao
            )
            
            print(f"\n--- EXECUTANDO SUBGRADIENTE ATÉ CONVERGÊNCIA ---")
            import time
            tempo_inicio = time.time()
            
            # Executa até convergência (modo infinito)
            resultados = metodo_subgradiente.resolver(
                multiplicadores_iniciais=multiplicadores_zero,
                convergir_para_precisao=True
            )
            
            tempo_total = time.time() - tempo_inicio
            
            # 4. Gera gráfico de convergência
            print(f"\n--- GERANDO GRÁFICO DE CONVERGÊNCIA ---")
            nome_grafico = salvar_grafico or f"convergencia_{nome_arquivo.replace('.txt', '')}.png"
            metodo_subgradiente.gerar_grafico_convergencia(nome_grafico)
            
            # 5. Relatório final
            gap_final = resultados.get('gap_final', 0)
            gap_pct = (gap_final / upper_bound * 100) if upper_bound > 0 else 0
            
            print(f"\n--- RESULTADOS DA ANÁLISE DE CONVERGÊNCIA ---")
            print(f"Iterações executadas: {resultados['iteracoes_executadas']}")
            print(f"Tempo total: {tempo_total:.2f}s")
            print(f"Upper Bound: {upper_bound}")
            print(f"Lower Bound final: {resultados['melhor_valor_dual']:.6f}")
            print(f"Gap final: {gap_final:.6f} ({gap_pct:.3f}%)")
            print(f"Convergiu: {'✓' if resultados['convergiu'] else '✗'}")
            print(f"λ utilizado: {lambda_inicial}")
            print(f"Gráfico salvo: {nome_grafico}")
            
            return {
                'instancia': nome_arquivo,
                'iteracoes': resultados['iteracoes_executadas'],
                'tempo_total': tempo_total,
                'upper_bound': upper_bound,
                'lower_bound_final': resultados['melhor_valor_dual'],
                'gap_final': gap_final,
                'gap_pct': gap_pct,
                'convergiu': resultados['convergiu'],
                'lambda_usado': lambda_inicial,
                'arquivo_grafico': nome_grafico
            }
            
        except Exception as e:
            print(f"✗ Erro na análise de convergência: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def gerar_graficos_convergencia(self, salvar_em: str = "graficos_convergencia"):
        """Gera gráficos de convergência para todas as instâncias executadas"""
        import matplotlib.pyplot as plt
        import os
        
        if not self.resultados:
            print("Nenhum resultado disponível para gerar gráficos.")
            return
        
        # Cria diretório para salvar os gráficos
        if not os.path.exists(salvar_em):
            os.makedirs(salvar_em)
        
        print(f"\nGerando gráficos de convergência...")
        print(f"Diretório: {salvar_em}")
        
        graficos_gerados = 0
        
        for resultado in self.resultados:
            if resultado.erro:
                continue
                
            # Executa análise de convergência para esta instância
            dados_convergencia = self._executar_analise_convergencia_detalhada(resultado.instancia)
            
            if dados_convergencia:
                nome_arquivo = f"{salvar_em}/{resultado.instancia.replace('.txt', '')}_convergencia.png"
                self._criar_grafico_convergencia(dados_convergencia, nome_arquivo, resultado)
                graficos_gerados += 1
        
        print(f"✓ {graficos_gerados} gráficos gerados em '{salvar_em}'")
    
    def _executar_analise_convergencia_detalhada(self, nome_arquivo: str):
        """Executa análise de convergência detalhada para uma instância específica"""
        try:
            caminho_completo = f"{self.config.pasta_dados}\\{nome_arquivo}"
            
            # Carrega instância
            grafo, custos, fonte, terminais = carregar_instancia_steiner(caminho_completo)
            commodities = terminais - {fonte}
            
            # Calcula upper bound
            ub, _, _ = self._resolver_upper_bound(grafo, custos, fonte, terminais)
            
            # Prepara subproblema
            multiplicadores_zero = self._criar_multiplicadores_zero(grafo, commodities)
            subproblema = SubproblemaLagrangeano(grafo, custos, fonte, commodities)
            
            # Executa método do subgradiente com coleta de dados
            metodo_subgradiente = MetodoSubgradienteComHistorico(
                subproblema,
                limite_superior_inicial=ub,
                lambda_inicial=self.config.lambda_inicial,
                max_iteracoes=self.config.max_iteracoes * 2,  # Mais iterações para gráfico
                tolerancia=self.config.tolerancia,
                precisao_gap=self.config.precisao_gap,
                intervalo_impressao=1000  # Sem impressão para não poluir saída
            )
            
            resultados = metodo_subgradiente.resolver(convergir_para_gap=True, gap_objetivo=0.01)
            
            return {
                'iteracoes': resultados.get('historico_iteracoes', []),
                'valores_duais': resultados.get('historico_valores_duais', []),
                'upper_bound': ub,
                'melhor_valor_dual': resultados.get('melhor_valor_dual'),
                'convergiu': resultados.get('convergiu', False)
            }
            
        except Exception as e:
            print(f"Erro ao gerar dados de convergência para {nome_arquivo}: {e}")
            return None
    
    def _criar_grafico_convergencia(self, dados, nome_arquivo: str, resultado: Resultado):
        """Cria gráfico individual de convergência"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        iteracoes = dados['iteracoes']
        valores_duais = dados['valores_duais']
        upper_bound = dados['upper_bound']
        
        # Linha principal: evolução do lower bound
        plt.plot(iteracoes, valores_duais, 'b-', linewidth=2, label='Lower Bound (Relaxação)')
        
        # Linha do upper bound (constante)
        if upper_bound:
            plt.axhline(y=upper_bound, color='r', linestyle='--', linewidth=2, label=f'Upper Bound = {upper_bound:.2f}')
            
            # Área entre bounds (gap)
            plt.fill_between(iteracoes, valores_duais, upper_bound, alpha=0.2, color='yellow', label='Gap')
        
        # Configurações do gráfico
        plt.xlabel('Iterações', fontsize=12)
        plt.ylabel('Valor Objetivo', fontsize=12)
        plt.title(f'Convergência - {resultado.instancia}\n'
                 f'Nós: {resultado.num_nos}, Terminais: {resultado.num_terminais}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Anotações importantes
        if dados['melhor_valor_dual'] and upper_bound:
            gap_final = ((upper_bound - dados['melhor_valor_dual']) / upper_bound) * 100
            plt.text(0.02, 0.98, f'Gap Final: {gap_final:.2f}%', 
                    transform=plt.gca().transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    verticalalignment='top')
        
        # Marca o melhor valor encontrado
        if valores_duais:
            melhor_idx = valores_duais.index(max(valores_duais))
            plt.plot(iteracoes[melhor_idx], valores_duais[melhor_idx], 'go', markersize=8, 
                    label=f'Melhor LB = {max(valores_duais):.2f}')
        
        plt.tight_layout()
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {nome_arquivo}")
    
    def executar_instancia_com_grafico(self, nome_arquivo: str) -> Resultado:
        """Executa uma instância e gera seu gráfico de convergência"""
        resultado = self.executar_instancia(nome_arquivo)
        
        if not resultado.erro:
            print(f"\nGerando gráfico de convergência...")
            dados_convergencia = self._executar_analise_convergencia_detalhada(nome_arquivo)
            
            if dados_convergencia:
                nome_grafico = f"{nome_arquivo.replace('.txt', '')}_convergencia.png"
                self._criar_grafico_convergencia(dados_convergencia, nome_grafico, resultado)
                print(f"✓ Gráfico salvo: {nome_grafico}")
        
        return resultado


class MetodoSubgradienteComHistorico(MetodoSubgradiente):
    """Versão do MetodoSubgradiente que coleta histórico para gráficos"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.historico_iteracoes = []
        self.historico_valores_duais = []
    
    def resolver(self, convergir_para_gap: bool = True, gap_objetivo: float = 0.01) -> Dict:
        """Resolve o subproblema coletando histórico de convergência
        
        Args:
            convergir_para_gap: Se True, executa até atingir gap_objetivo (iterações infinitas)
            gap_objetivo: Gap relativo objetivo para parada (padrão: 0.01 = 1%)
        """
        print(f"Iniciando método do subgradiente com coleta de histórico...")
        print(f"Upper bound inicial: {self.Z_star}")
        
        if convergir_para_gap:
            print(f"Modo: ITERAÇÕES INFINITAS até gap de {gap_objetivo*100:.1f}%")
            print(f"Máximo teórico de iterações: SEM LIMITE")
        else:
            print(f"Máximo de iterações: {self.max_iteracoes}")
        
        print(f"λ (lambda): {self.lambda_k}")
        print(f"Tolerância: {self.tolerancia}")
        print()
        
        # Inicialização correta dos multiplicadores
        multiplicadores_atuais = {}
        for aresta in self.subproblema.arestas:
            for commodity in self.subproblema.commodities:
                multiplicadores_atuais[(aresta, commodity)] = 0.0
        
        self.melhor_valor_dual = float('-inf')
        melhores_multiplicadores = multiplicadores_atuais.copy()
        self.z_ub = self.Z_star
        
        
        print("Iter | Lower Bound |    Gap (%)  | Subgrad Norm | Step Size")
        print("-" * 60)
        
        iteracao = 0
        # Loop infinito quando convergir_para_gap = True
        while True:
            # Resolve subproblema atual
            valor_dual, sol_x, sol_f = self.subproblema.resolver(multiplicadores_atuais)
            
            # Coleta dados para histórico - SEMPRE coletamos o valor atual da iteração
            self.historico_iteracoes.append(iteracao + 1)
            self.historico_valores_duais.append(valor_dual)
            
            # Atualiza melhor valor
            if valor_dual > self.melhor_valor_dual:
                self.melhor_valor_dual = valor_dual
                melhores_multiplicadores = multiplicadores_atuais.copy()

            
            # Calcula gap atual
            gap_absoluto = (self.z_ub - valor_dual) if self.z_ub else float('inf')
            gap_relativo = (gap_absoluto / self.z_ub * 100) if self.z_ub and self.z_ub > 0 else float('inf')
            
            # Calcula subgradiente e atualiza multiplicadores
            subgradiente = self.calcular_subgradiente(sol_x, sol_f)
            norma_subgradiente_quadrada = sum(val**2 for val in subgradiente.values())
            norma_subgradiente = norma_subgradiente_quadrada**0.5
            
            # Calcula tamanho do passo
            if norma_subgradiente_quadrada > self.tolerancia:
                diferenca_ub = (self.z_ub - valor_dual) if self.z_ub else 1.0
                tamanho_passo = (self.lambda_k * diferenca_ub) / norma_subgradiente_quadrada
            else:
                tamanho_passo = 0.0
              # Mostra progresso a cada 10 iterações ou nos primeiros/últimos
            if (iteracao % 10 == 0 or iteracao < 5):
                print(f"{iteracao+1:4d} | {valor_dual:11.4f} | {gap_relativo:9.3f} | "
                      f"{norma_subgradiente:12.6f} | {tamanho_passo:9.6f}")
            
            # Verifica critérios de parada
            if convergir_para_gap and self.z_ub and valor_dual:
                if gap_relativo <= gap_objetivo * 100:  # gap_objetivo está em decimal
                    print(f"\n✓ Convergiu: Gap {gap_relativo:.4f}% <= {gap_objetivo*100:.4f}%")
                    convergiu = True
                    break
            elif not convergir_para_gap and self.z_ub and valor_dual:
                if gap_relativo <= self.precisao_gap * 100:  # precisao_gap está em decimal
                    print(f"\n✓ Convergiu: Gap {gap_relativo:.4f}% <= {self.precisao_gap*100:.4f}%")
                    convergiu = True
                    break
            
            # Critério de parada por número máximo de iterações (apenas se não for modo infinito)
            if not convergir_para_gap and iteracao >= self.max_iteracoes - 1:
                print(f"\n⚠ Atingiu máximo de iterações ({self.max_iteracoes})")
                convergiu = False
                break
            

            # Parada por subgradiente muito pequeno
            if norma_subgradiente < self.tolerancia:
                print(f"\n✓ Convergiu: Norma do subgradiente {norma_subgradiente:.8f} < {self.tolerancia}")
                convergiu = True
                break
            
            # Atualiza multiplicadores para próxima iteração
            for chave in multiplicadores_atuais:
                if chave in subgradiente:
                    multiplicadores_atuais[chave] = max(0.0, 
                        multiplicadores_atuais[chave] + tamanho_passo * subgradiente[chave])
            
            # Incrementa contador de iteração
            iteracao += 1
            
            # Segurança: evita loop infinito real (limite muito alto para modo infinito)
            if convergir_para_gap and iteracao >= 10000:
                print(f"\n⚠ Atingiu limite de segurança (10000 iterações)")
                convergiu = False
                break
        
        print()
        print(f"Resultado final:")
        print(f"  Melhor lower bound: {self.melhor_valor_dual:.6f}")
        if self.z_ub:
            gap_final = self.z_ub - self.melhor_valor_dual
            gap_final_pct = (gap_final / self.z_ub) * 100
            print(f"  Upper bound: {self.z_ub:.6f}")
            print(f"  Gap final: {gap_final:.6f} ({gap_final_pct:.3f}%)")
        print(f"  Iterações executadas: {len(self.historico_iteracoes)}")
        print(f"  Convergiu: {'Sim' if convergiu else 'Não'}")
        
        return {
            'melhor_valor_dual': self.melhor_valor_dual,
            'iteracoes_executadas': len(self.historico_iteracoes),
            'convergiu': convergiu,
            'melhores_multiplicadores': melhores_multiplicadores,
            'historico_iteracoes': self.historico_iteracoes,
            'historico_valores_duais': self.historico_valores_duais
        }
    
    def gerar_grafico_convergencia(self, salvar_arquivo=None):
        """Gera gráfico simples de convergência do método do subgradiente"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Erro: matplotlib não está instalado. Use: pip install matplotlib")
            return
        
        if not self.historico_iteracoes:
            print("Nenhum histórico disponível para plotar.")
            return
        
        print(f"Gerando gráfico com {len(self.historico_iteracoes)} pontos de convergência...")
        
        # Configuração do gráfico único
        plt.figure(figsize=(12, 8))
        
        # Gráfico principal: Lower Bound vs Upper Bound
        plt.plot(self.historico_iteracoes, self.historico_valores_duais, 
                'b-', linewidth=2, label='Lower Bound (Relaxação Lagrangeana)', marker='o', markersize=3)
        
        # Linha do Upper Bound (se disponível)
        if self.z_ub is not None:
            plt.axhline(y=self.z_ub, color='red', linestyle='--', linewidth=3, 
                       label=f'Upper Bound = {self.z_ub:.2f}')
            
            # Área entre bounds (gap)
            plt.fill_between(self.historico_iteracoes, self.historico_valores_duais, self.z_ub, 
                            alpha=0.2, color='lightcoral', label='Gap')
        
        # Configurações do gráfico
        plt.xlabel('Iterações', fontsize=14)
        plt.ylabel('Valor Objetivo', fontsize=14)
        plt.title('Convergência da Relaxação Lagrangeana', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Anotações importantes
        if self.z_ub is not None and self.historico_valores_duais:
            gap_inicial = self.z_ub - self.historico_valores_duais[0]
            gap_final = self.z_ub - self.melhor_valor_dual
            gap_final_pct = (gap_final / self.z_ub) * 100 if self.z_ub > 0 else 0
            
            # Texto com informações do gap
            info_text = f'Gap Inicial: {gap_inicial:.2f}\n'
            info_text += f'Gap Final: {gap_final:.2f} ({gap_final_pct:.2f}%)\n'
            info_text += f'Melhoria: {gap_inicial - gap_final:.2f}\n'
            info_text += f'Iterações: {len(self.historico_iteracoes)}\n'
            info_text += f'LB Inicial: {self.historico_valores_duais[0]:.2f}\n'
            info_text += f'LB Final: {self.historico_valores_duais[-1]:.2f}'
            
            plt.text(0.02, 0.98, info_text, 
                    transform=plt.gca().transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
        
        # Marca o melhor valor encontrado
        if self.historico_valores_duais:
            melhor_valor = max(self.historico_valores_duais)
            melhor_idx = self.historico_valores_duais.index(melhor_valor)
            plt.plot(self.historico_iteracoes[melhor_idx], melhor_valor, 'go', markersize=10, 
                    label=f'Melhor LB = {melhor_valor:.2f}', markeredgecolor='darkgreen', markeredgewidth=2)
            
            # Também marca o primeiro e último ponto
            plt.plot(self.historico_iteracoes[0], self.historico_valores_duais[0], 
                    'ro', markersize=8, label=f'LB Inicial = {self.historico_valores_duais[0]:.2f}')
            plt.plot(self.historico_iteracoes[-1], self.historico_valores_duais[-1], 
                    'ko', markersize=8, label=f'LB Final = {self.historico_valores_duais[-1]:.2f}')
        
        # Atualiza a legenda
        plt.legend(fontsize=10, loc='best')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar ou mostrar
        if salvar_arquivo:
            plt.savefig(salvar_arquivo, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo em: {salvar_arquivo}")
        else:
            plt.show()
        
        plt.close()
    

def main():
    """Função principal simplificada"""
    config = Config()
    
    if not os.path.exists(config.pasta_dados):
        print(f"Pasta não encontrada: {config.pasta_dados}")
        return
    
    experimento = ExperimentoSteinerTree(config)
    
    while True:
        print("\n1. Instância específica")
        print("2. Série steinb")
        print("3. Série steinc") 
        print("4. Todas as instâncias")
        print("5. Relatório")
        print("6. Gerar gráficos de convergência")
        print("7. Instância específica com gráfico")
        print("8. Sair")
        
        escolha = input("\nOpção (1-8): ").strip()
        
        if escolha == '1':
            arquivo = input("Nome do arquivo: ").strip()
            experimento.executar_instancia(arquivo)
            
        elif escolha == '2':
            experimento.executar_serie('steinb')
            
        elif escolha == '3':
            experimento.executar_serie('steinc')
            
        elif escolha == '4':
            try:
                arquivos = [f for f in os.listdir(config.pasta_dados) if f.endswith('.txt')]
                print(f"Executando {len(arquivos)} instâncias...")
                for arquivo in arquivos:
                    experimento.executar_instancia(arquivo)
            except Exception as e:
                print(f"Erro: {e}")
                
        elif escolha == '5':
            experimento.gerar_relatorio()
            
        elif escolha == '6':
            experimento.gerar_graficos_convergencia()
            
        elif escolha == '7':
            arquivo = input("Nome do arquivo: ").strip()
            experimento.executar_instancia_com_grafico(arquivo)
            
        elif escolha == '8':
            break
            
        else:
            print("Opção inválida.")
    
    print("Obrigado!")


if __name__ == "__main__":
    main()

