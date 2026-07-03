# schrodinger-crank-nicolson
# Solução Numérica da Equação de Schrödinger via Crank-Nicolson

Este repositório contém uma simulação computacional desenvolvida em dupla para resolver numericamente a **Equação de Schrödinger Dependente do Tempo (TDSE)** unidimensional. O foco do projeto é simular a evolução temporal de um pacote de ondas gaussiano colidindo com uma barreira quadrada de potencial, evidenciando os fenômenos de **reflexão quântica** e **tunelamento quântico**.

## 🔬 O Método Numérico: Crank-Nicolson

A equação diferencial parcial que rege o sistema é resolvida através do método de **Crank-Nicolson**, um esquema implícito de diferenças finitas de segunda ordem tanto no espaço quanto no tempo.

Onde mathbf{A} e mathbf{B} são matrizes tridiagonais complexas. O algoritmo utiliza a decomposição LU (`scipy.sparse.linalg.splu`) para resolver o sistema linear a cada passo de tempo com alta eficiência computacional.

## 🚀 Funcionalidades da Simulação

1. **Evolução Estável:** Discretização espacial e temporal rigorosa garantindo a estabilidade unitária da evolução do pacote de onda.
2. **Análise Quantitativa:** Extração automática dos coeficientes físicos de **Reflexão R** e **Transmissão/Tunelamento T**, validando matematicamente a conservação da probabilidade total.
3. **Estilo Científico:** Utilização opcional do pacote `scienceplots` para renderização de gráficos estáticos em qualidade de publicação acadêmica.
4. **Visualização Dinâmica:** Geração automatizada de uma animação `.gif` mostrando a densidade de probabilidade se chocando e dividindo ao passar pela barreira.

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **NumPy:** Vetorização e manipulação de arrays complexos.
* **SciPy (`sparse`):** Construção de matrizes esparsas tridiagonais e resolução por fatoração LU.
* **Matplotlib:** Criação dos instantâneos e da animação dinâmica via `FuncAnimation`.

## 💻 Como Executar

O script foi preparado para rodar perfeitamente em ambientes interativos como o **Google Colab**.

1. Crie um arquivo chamado `schrodinger.py`.
2. Execute o código para calcular a evolução e gerar a análise quantitativa no terminal.
3. O arquivo animado `projeto_schrodinger_final.gif` será gerado automaticamente no seu diretório para visualização.

## 👥 Autores

Projeto desenvolvido em colaboração por:
* **Ana Carla** - [anacarla-acma](https://github.com/anacarla-acma)
* **Aleks Gil Carvalho** - [ALEKSGILLACERDA](https://github.com/ALEKSGILLACERDA)
