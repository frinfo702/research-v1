# 機械学習の基礎的導出まとめ

## 目次

1. [回帰 (Regression)](#1-回帰-regression)
2. [分類 (Classification)](#2-分類-classification)
3. [クラスタリング (Clustering)](#3-クラスタリング-clustering)
4. [深層学習 (Deep Learning)](#4-深層学習-deep-learning)

---

## 1. 回帰 (Regression)

### 1.1 線形回帰 (Linear Regression)

#### モデル

入力 $\mathbf{x} \in \mathbb{R}^d$、出力 $y \in \mathbb{R}$ に対し：

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

バイアス項 $b$ を $\mathbf{w}$ に吸収させるため $\mathbf{x} \leftarrow [\mathbf{x}; 1]$ とすれば：

$$
\hat{y} = \mathbf{w}^\top \mathbf{x}
$$

#### 最小二乗法 (Ordinary Least Squares)

二乗誤差を最小化する：

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
= \frac{1}{2} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2
$$

ここで $\mathbf{X} \in \mathbb{R}^{n \times d}$ は各行が $\mathbf{x}_i^\top$ の計画行列。

勾配を計算する：

$$
\nabla_{\mathbf{w}} \mathcal{L} = -\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{w})
$$

最適性条件 $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$ より **正規方程式 (normal equation)**：

$$
\mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}
$$

$\mathbf{X}^\top \mathbf{X}$ が正則なら閉形式解：

$$
\boxed{\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}
$$

#### 幾何学的解釈

$\mathbf{X}\mathbf{w}$ は $\mathbf{X}$ の列が張る部分空間への射影であり、$(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top$ は射影行列である。残差 $\mathbf{y} - \mathbf{X}\mathbf{w}$ はこの部分空間と直交する。

#### 確率的解釈 (MLE)

$y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i,\; \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ と仮定する：

$$
p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right)
$$

対数尤度：

$$
\log p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
$$

MLE は二乗誤差最小化と等価。

### 1.2 リッジ回帰 (Ridge Regression)

#### 正則化

過学習を防ぐため $\mathbf{w}$ の L2 ノルムにペナルティ：

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

勾配：

$$
\nabla_{\mathbf{w}} \mathcal{L} = -\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{w}) + \lambda \mathbf{w}
$$

最適性条件より：

$$
(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{X}^\top \mathbf{y}
$$

$$
\boxed{\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}}
$$

$\lambda > 0$ により $\mathbf{X}^\top \mathbf{X}$ が特異でも逆行列が存在する。

#### MAP推定としての解釈

事前分布 $\mathbf{w} \sim \mathcal{N}(0, \lambda^{-1} \mathbf{I})$ を仮定した MAP推定と等価。

#### リッジ vs 通常のOLS

- OLS: 不偏だが高分散（$\mathbf{X}^\top \mathbf{X}$ の小さい固有値方向で大きな重み）
- リッジ: バイアスが入るが分散を削減（固有値に $\lambda$ を加算して縮小）

### 1.3 ロジスティック回帰 (Logistic Regression)

分類にも使うが、回帰の一般化線形モデル (GLM) としてここに含める。

#### モデル

確率 $p \in (0,1)$ と線形予測子の関係：

$$
\log\frac{p}{1-p} = \mathbf{w}^\top \mathbf{x}
$$

これより：

$$
p = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}
$$

ここで $\sigma(\cdot)$ はシグモイド関数。

#### シグモイドの導出

ベルヌーイ分布の自然パラメータ $\eta$ を用いると：

$$
p(y \mid \eta) = \exp(\eta y - A(\eta)),\; A(\eta) = \log(1 + e^\eta)
$$

平均 $ \mathbb{E}[y] = \frac{\partial A}{\partial \eta} = \sigma(\eta) $ となり、リンク関数はロジット $\eta = \log(p/(1-p))$。

#### 勾配法による学習

対数尤度（$y \in \{0,1\}$）：

$$
\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \left[ y_i \log \sigma(\mathbf{w}^\top \mathbf{x}_i) + (1-y_i) \log (1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)) \right]
$$

1サンプルあたりの勾配：

$$
\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}} = -[y_i - \sigma(\mathbf{w}^\top \mathbf{x}_i)] \mathbf{x}_i
$$

これは線形回帰の残差 $y_i - \hat{y}_i$ と同じ形。ただし $\hat{y}_i$ が確率になった点が異なる。

---

## 2. 分類 (Classification)

### 2.1 ソフトマックス回帰 (Softmax Regression / Multinomial Logistic Regression)

$K$ クラス分類：$p(y=k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x})}{\sum_{j=1}^K \exp(\mathbf{w}_j^\top \mathbf{x})}$

対数尤度：

$$
\mathcal{L}(\{\mathbf{w}_k\}) = -\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \frac{\exp(\mathbf{w}_k^\top \mathbf{x}_i)}{\sum_j \exp(\mathbf{w}_j^\top \mathbf{x}_i)}
$$

勾配：

$$
\frac{\partial \mathcal{L}_i}{\partial \mathbf{w}_k} = -[y_{ik} - p(y=k \mid \mathbf{x}_i)] \mathbf{x}_i
$$

ソフトマックス関数の Jacobian：

$$
\frac{\partial p_j}{\partial a_k} = p_j (\delta_{jk} - p_k),\quad a_k = \mathbf{w}_k^\top \mathbf{x}
$$

この Jacobian により、損失の勾配が上記のように簡潔になる（クロスエントロピー + ソフトマックスの相性の良さ）。

### 2.2 サポートベクターマシン (SVM)

#### ハードマージンSVM（線形分離可能）

マージン最大化：超平面 $\mathbf{w}^\top \mathbf{x} + b = 0$ と最も近いデータ点の距離を最大化。

点 $\mathbf{x}_i$ から超平面への距離：$\frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|}$

マージン最大化の主問題：

$$
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1,\; \forall i
$$

#### ラグランジュ双対問題

ラグランジアン：

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1],\quad \alpha_i \geq 0
$$

KKT条件より $\mathbf{w}$ と $b$ で微分して0：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = 0 \quad\Rightarrow\quad \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \quad\Rightarrow\quad \sum_i \alpha_i y_i = 0
$$

双対問題：

$$
\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \quad \text{s.t.} \quad \alpha_i \geq 0,\; \sum_i \alpha_i y_i = 0
$$

分類関数は $\alpha_i > 0$ のサポートベクターのみで決まる：

$$
f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b
$$

#### カーネルトリック

双対問題は内積 $\mathbf{x}_i^\top \mathbf{x}_j$ のみに依存。これをカーネル関数 $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ に置き換えることで、高次元特徴空間への暗黙的な写像が可能：

- 多項式カーネル: $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top \mathbf{z} + c)^d$
- RBFカーネル: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$

### 2.3 ナイーブベイズ (Naive Bayes)

#### 生成モデル

クラス $k$ に対し、特徴が独立という強い仮定（条件付き独立性）：

$$
p(\mathbf{x} \mid y=k) = \prod_{j=1}^d p(x_j \mid y=k)
$$

ベイズの定理より：

$$
p(y=k \mid \mathbf{x}) = \frac{p(y=k) \prod_j p(x_j \mid y=k)}{\sum_{k'} p(y=k') \prod_j p(x_j \mid y=k')}
$$

#### 様々なバリアント

| タイプ      | $p(x_j \mid y)$                                   | 用途                |
| ----------- | ------------------------------------------------- | ------------------- |
| Gaussian    | $\mathcal{N}(\mu_{jk}, \sigma_{jk}^2)$            | 実数値特徴          |
| Multinomial | $\frac{\theta_{jk}^{x_j}}{x_j!} e^{-\theta_{jk}}$ | カウント特徴（BoW） |
| Bernoulli   | $\theta_{jk}^{x_j} (1-\theta_{jk})^{1-x_j}$       | 二値特徴            |

#### 対数確率による推論

実装上は確率の積がアンダーフローするため対数で計算：

$$
\hat{y} = \arg\max_k \left[ \log p(y=k) + \sum_{j=1}^d \log p(x_j \mid y=k) \right]
$$

### 2.4 決定木 (Decision Tree)

#### 不純度指標

ノード $m$ におけるクラス分布 $p_{mk}$ に対し：

**エントロピー**：

$$
H(m) = -\sum_{k=1}^K p_{mk} \log_2 p_{mk}
$$

**ジニ不純度**：

$$
G(m) = \sum_{k=1}^K p_{mk}(1 - p_{mk}) = 1 - \sum_{k=1}^K p_{mk}^2
$$

**分類誤差**：

$$
E(m) = 1 - \max_k p_{mk}
$$

エントロピーとジニ不純度は微分可能で木の成長に敏感。分類誤差は他の木との組み合わせに有用。

#### 情報利得 (Information Gain)

分割前の不純度 $I(\text{parent})$、分割後（子ノード $j$ のサンプル数 $N_j$）：

$$
IG = I(\text{parent}) - \sum_{j} \frac{N_j}{N} I(\text{child}_j)
$$

各特徴・閾値で IG を最大化する分割を選択。

---

## 3. クラスタリング (Clustering)

### 3.1 K-Means

#### アルゴリズム

$K$ 個のクラスタ中心 $\{\boldsymbol{\mu}_k\}_{k=1}^K$ と各点の割当 $r_{ik} \in \{0,1\}$（1-of-K）：

目的関数：

$$
J(\{\boldsymbol{\mu}_k\}, \{r_{ik}\}) = \sum_{i=1}^n \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

#### 2ステップ最適化 (Block Coordinate Descent)

**E-step（割当）**：各点を最も近い中心に割り当て

$$
r_{ik} = \begin{cases}
1 & \text{if } k = \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2 \\
0 & \text{otherwise}
\end{cases}
$$

**M-step（更新）**：割当られた点の平均で中心を更新

$$
\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i: r_{ik}=1} \mathbf{x}_i,\quad N_k = \sum_i r_{ik}
$$

#### 収束性

- 各ステップで目的関数は単調減少
- 有限回の割当パターンしかないため局所最適に収束
- 大域最適は保証されない（初期値依存）

#### EM アルゴリズムとの関係

K-Means は GMM の分散 $\sigma^2 \to 0$ 極限に対応（ハード割当）。

### 3.2 混合ガウスモデル (Gaussian Mixture Model)

#### モデル

$K$ 個のガウス分布の重み付き和：

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \,\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

ここで $\sum_k \pi_k = 1,\; \pi_k \geq 0$。

#### 対数尤度

$$
\log p(\mathbf{X} \mid \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

和の中に log があるため閉形式で解けない → EM アルゴリズム。

#### EM アルゴリズムの導出

**負担率 (responsibility)** を導入：

$$
\gamma_{ik} = p(z_i = k \mid \mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

**E-step**: 現在のパラメータで $\gamma_{ik}$ を計算。

**M-step**: 負担率を固定してパラメータを更新：

$$
\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} \mathbf{x}_i,\quad N_k = \sum_i \gamma_{ik}
$$

$$
\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^\top
$$

$$
\pi_k^{\text{new}} = \frac{N_k}{n}
$$

#### EMの一般形

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \parallel p)
$$

ここで：

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \log \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})}
$$

- E-step: $q$ を現在の $\boldsymbol{\theta}$ で最大化（$q = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ とすると KL=0）
- M-step: $\boldsymbol{\theta}$ を固定した $q$ で $\mathcal{L}$ を最大化

### 3.3 階層的クラスタリング

**凝集型 (Agglomerative)**：各点をクラスタとして開始し、最も類似したペアを統合。

**リンケージ基準**（クラスタ $A, B$間の距離）：

- 単連結 (Single): $\min_{i \in A, j \in B} d(\mathbf{x}_i, \mathbf{x}_j)$
- 完全連結 (Complete): $\max_{i \in A, j \in B} d(\mathbf{x}_i, \mathbf{x}_j)$
- 平均連結 (Average): $\frac{1}{|A||B|} \sum_{i \in A} \sum_{j \in B} d(\mathbf{x}_i, \mathbf{x}_j)$
- Ward法: $\frac{|A||B|}{|A|+|B|} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$（統合による分散増分最小化）

デンドログラムにより $K$ を事後的に決定可能。

### 3.4 DBSCAN

#### 定義

- **核心点 (core point)**: 半径 $\varepsilon$ 内に minPts 以上の点がある点
- **到達可能 (reachable)**: 核心点を経由して連続的に到達できる
- **クラスタ**: 核心点の密度連結な極大集合
- **ノイズ**: どのクラスタにも属さない点

#### 利点

- $K$ の事前指定不要
- 任意形状のクラスタを発見
- 外れ値を検出

#### 注意点

- $\varepsilon$ と minPts に敏感
- 密度が不均一なデータに弱い
- $O(n^2)$ の計算量（kd-tree で高速化可能）

---

## 4. 深層学習 (Deep Learning)

### 4.1 多層パーセプトロン (MLP) と誤差逆伝播

#### 順伝播 (Forward Pass)

$L$ 層のネットワーク。第 $l$ 層の出力を $\mathbf{h}^{(l)}$ とする：

$$
\mathbf{a}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{h}^{(l)} = f^{(l)}(\mathbf{a}^{(l)})
$$

ここで $f$ は活性化関数、$\mathbf{h}^{(0)} = \mathbf{x}$（入力）。

#### 連鎖律 (Chain Rule) と逆伝播

損失 $L$ に対する各パラメータの勾配を計算する。

出力層 $L$ における誤差：

$$
\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(L)}} \odot f'^{(L)}(\mathbf{a}^{(L)})
$$

ここで $\odot$ は要素積。例：MSE損失 + 線形出力なら $\boldsymbol{\delta}^{(L)} = \mathbf{h}^{(L)} - \mathbf{y}$。

第 $l$ 層の誤差の逆伝播：

$$
\boldsymbol{\delta}^{(l)} = \left( (\mathbf{W}^{(l+1)})^\top \boldsymbol{\delta}^{(l+1)} \right) \odot f'^{(l)}(\mathbf{a}^{(l)})
$$

勾配：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{h}^{(l-1)})^\top
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
$$

#### 微分の自動微分との関係

誤差逆伝播は連鎖律の効率的な実装であり、計算グラフ上のノード数に対して $O(1)$ 回の順伝播 + $O(1)$ 回の逆伝播で全勾配を計算できる。

#### 活性化関数の微分

| 関数 $f(a)$            | 導関数 $f'(a)$                                        |
| ---------------------- | ----------------------------------------------------- |
| Sigmoid $\sigma(a)$    | $\sigma(a)(1 - \sigma(a))$                            |
| Tanh $\tanh(a)$        | $1 - \tanh^2(a)$                                      |
| ReLU $\max(0, a)$      | $\begin{cases} 1 & a > 0 \\ 0 & a \leq 0 \end{cases}$ |
| GELU $a \cdot \Phi(a)$ | $\Phi(a) + a \cdot \phi(a)$                           |

#### 消失勾配と爆発勾配

- **消失**: シグモイド/tanh の飽和領域で勾配が0近くなり、浅い層に伝播しない
- **爆発**: $\|\mathbf{W}^{(l)}\| > 1$ で指数関数的に勾配が増大
- 対策: ReLU, BatchNorm, ResNet（スキップ接続）, 勾配クリッピング

### 4.2 畳み込みニューラルネットワーク (CNN)

#### 畳み込み演算の微分

2D 畳み込み $h = x * w$ において：

$$
h[m, n] = \sum_{i} \sum_{j} x[m+i, n+j] \, w[i, j]
$$

逆伝播時の勾配：

$$
\frac{\partial \mathcal{L}}{\partial w[i, j]} = \sum_{m} \sum_{n} \frac{\partial \mathcal{L}}{\partial h[m, n]} \, x[m+i, n+j]
$$

$$
\frac{\partial \mathcal{L}}{\partial x[m, n]} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial h[m-i, n-j]} \, w[i, j]
$$

つまり、入力に対する勾配はカーネルを180度回転した $w[-i, -j]$ との畳み込みになる。

#### プーリングの逆伝播

**Max Pooling**: 順伝播時に最大値の位置 (max index) を記憶。逆伝播ではその位置のみ勾素を通過：

$$
\frac{\partial \mathcal{L}}{\partial x_{ij}} = \begin{cases}
\frac{\partial \mathcal{L}}{\partial h_{mn}} & \text{if } (i,j) \text{ is max in pooling window} \\
0 & \text{otherwise}
\end{cases}
$$

**Average Pooling**: 勾配をウィンドウ内で均等に分配。

### 4.3 Transformer / Attention

#### Scaled Dot-Product Attention

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

$\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{n \times d_k}, \mathbf{V} \in \mathbb{R}^{n \times d_v}$。

#### なぜ $\sqrt{d_k}$ でスケールするか

$\mathbf{Q}$ と $\mathbf{K}$ の要素が平均0、分散1の独立な確率変数の場合、内積 $q^\top k = \sum_{j=1}^{d_k} q_j k_j$ は：

$$
\mathbb{E}[q^\top k] = 0,\quad \text{Var}(q^\top k) = d_k
$$

$d_k$ が大きいと分散が大きくなり、softmax の入力が極端な値になって勾配が消失する。$\sqrt{d_k}$ で割ることで分散を $1$ に正規化。

#### Multi-Head Attention

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
$$

$$
\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V)
$$

各 head が異なる部分空間で attention を計算することで、多様な関係を捉える。

#### Positional Encoding (Sinusoidal)

位置 $pos$、次元 $i$ に対するエンコーディング：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

異なる周波数の波で位置を表現。線形な相対位置関係が表現可能：

$$
PE_{pos+\Delta}^\top PE_{pos} = \sum_i \cos\left(\frac{\Delta}{10000^{2i/d}}\right)
$$

これは相対位置 $\Delta$ のみに依存する。

### 4.4 損失関数

#### 平均二乗誤差 (MSE)

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

勾配：$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(y_i - \hat{y}_i)$

#### クロスエントロピー誤差

二値分類：

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]
$$

多クラス分類：

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}
$$

#### ヒンジ損失 (SVM)

$$
\mathcal{L}_{\text{hinge}} = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i f(\mathbf{x}_i))
$$

微分（劣微分）：

$$
\frac{\partial \mathcal{L}_i}{\partial f(\mathbf{x}_i)} = \begin{cases}
0 & \text{if } y_i f(\mathbf{x}_i) \geq 1 \\
-y_i & \text{otherwise}
\end{cases}
$$

#### 対数損失（ロジスティック損失）

$$
\mathcal{L}_{\text{log}} = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i f(\mathbf{x}_i)))
$$

ヒンジ損失の微分可能な近似。

### 4.5 バッチ正規化 (Batch Normalization)

#### アルゴリズム

ミニバッチ $\mathcal{B} = \{\mathbf{x}_1, \dots, \mathbf{x}_m\}$ に対し：

$$
\boldsymbol{\mu}_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m \mathbf{x}_i
$$

$$
\boldsymbol{\sigma}_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B})^2
$$

$$
\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \varepsilon}}
$$

$$
\mathbf{y}_i = \boldsymbol{\gamma} \odot \hat{\mathbf{x}}_i + \boldsymbol{\beta}
$$

#### 逆伝播の導出

$\hat{x}_i$ は $\boldsymbol{\mu}_\mathcal{B}$ と $\boldsymbol{\sigma}_\mathcal{B}$ を通じてバッチ全体に依存するため、連鎖律が複雑になる。

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_i} = \frac{1}{m} \boldsymbol{\gamma} \odot \left[ m \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{x}}_i} - \sum_j \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{x}}_j} - \hat{\mathbf{x}}_i \sum_j \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{x}}_j} \odot \hat{\mathbf{x}}_j \right] \odot \frac{1}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \varepsilon}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\gamma}} = \sum_i \frac{\partial \mathcal{L}}{\partial \mathbf{y}_i} \odot \hat{\mathbf{x}}_i
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \sum_i \frac{\partial \mathcal{L}}{\partial \mathbf{y}_i}
$$

#### なぜ有効か

- 各層の入力分布を安定化（内部共変量シフトの低減）
- 学習率を大きく取れる（勾配のスケールが安定）
- 正則化効果（ミニバッチの統計量によるノイズ）

### 4.6 ドロップアウト (Dropout)

#### アルゴリズム

学習時：各ニューロンを確率 $p$ で保持、$1-p$ で0にする。

$$
\mathbf{h}_{\text{train}} = \mathbf{m} \odot \mathbf{h},\quad m_i \sim \text{Bernoulli}(p)
$$

推論時：全てのニューロンを使い、重みを $p$ 倍する（期待値の一致）。

$$
\mathbf{h}_{\text{test}} = p \cdot \mathbf{h}
$$

あるいは **inverted dropout**（学習時に $1/p$ でスケール、推論時はそのまま）：

$$
\mathbf{h}_{\text{train}} = \frac{1}{p} \mathbf{m} \odot \mathbf{h}
$$

#### バギングとの関係

Dropout は $2^n$ 個のサブネットワークの重み共有アンサンブルと解釈できる。推論時の重み倍率は、これらサブネットワークの出力の幾何平均に相当。

### 4.7 最適化アルゴリズム

#### 確率的勾配降下法 (SGD)

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \nabla \mathcal{L}_i(\mathbf{w}_t)
$$

#### モーメンタム (Momentum)

$$
\mathbf{v}_{t+1} = \mu \mathbf{v}_t + \nabla \mathcal{L}_i(\mathbf{w}_t)
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}
$$

物理的解釈：慣性項により、勾配方向が安定する。

#### Nesterov Accelerated Gradient (NAG)

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}_i(\mathbf{w}_t + \mu \mathbf{v}_t) + \mu \mathbf{v}_t
$$

未来の位置で勾配を評価することで、モーメンタムより早い収束。

#### AdaGrad

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \varepsilon}} \odot \nabla \mathcal{L}_i(\mathbf{w}_t),\quad \mathbf{G}_t = \sum_{\tau=1}^t (\nabla \mathcal{L}_\tau)^2
$$

頻出パラメータの学習率は減衰し、稀なパラメータは大きめの更新が残る。

#### RMSProp

$$
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta) (\nabla \mathcal{L}_i)^2
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \varepsilon}} \odot \nabla \mathcal{L}_i
$$

AdaGrad の学習率単調減少問題を改善。

#### Adam (Adaptive Moment Estimation)

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla \mathcal{L}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla \mathcal{L}_t)^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t},\quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon}
$$

- モーメンタム（1次モーメント） + RMSProp（2次モーメント）
- バイアス補正により初期ステップの偏りを補正
- デフォルト: $\beta_1 = 0.9,\ \beta_2 = 0.999,\ \varepsilon = 10^{-8}$

#### 各手法の収束性比較

| 手法     | 適応的LR | モーメンタム | 収束保証（凸）  |
| -------- | -------- | ------------ | --------------- |
| SGD      | なし     | なし         | $O(1/\sqrt{T})$ |
| Momentum | なし     | あり         | $O(1/T)$        |
| AdaGrad  | あり     | なし         | $O(1/\sqrt{T})$ |
| RMSProp  | あり     | なし         | $O(1/\sqrt{T})$ |
| Adam     | あり     | あり         | $O(1/\sqrt{T})$ |

### 4.8 正則化と汎化

#### L1正則化 (Lasso)

$$
\mathcal{L}_{\text{L1}} = \mathcal{L}_{\text{data}} + \lambda \|\mathbf{w}\|_1
$$

- スパースな解を誘導（$w_i = 0$ が増える）
- 劣勾配: $\partial |w_i| = \text{sign}(w_i)$
- 原点で微分不可能 → proximal gradient が有効

#### L2正則化との比較（幾何学的解釈）

L1: 菱形の制約領域 → 軸上で解が疎になりやすい
L2: 球状の制約領域 → 全ての重みが均等に縮小

#### Early Stopping

検証損失が増加し始めた時点で学習を停止。SGD の逐次更新がリッジ回帰に似た正則化効果を持つことが知られている（線形モデルの場合）。

---

## Appendix: 重要確率分布

### ベルヌーイ分布

$$
p(y \mid \mu) = \mu^y (1-\mu)^{1-y},\quad y \in \{0,1\}
$$

$$
\mathbb{E}[y] = \mu,\quad \text{Var}[y] = \mu(1-\mu)
$$

### カテゴリカル分布

$$
p(\mathbf{y} \mid \boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{y_k},\quad y_k \in \{0,1\},\; \sum_k y_k = 1
$$

### ガウス分布

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

### 指数型分布族

$$
p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \exp\left(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})\right)
$$

- $\boldsymbol{\eta}$: 自然パラメータ
- $T(\mathbf{x})$: 十分統計量
- $A(\boldsymbol{\eta})$: 対数分配関数

ガウス、ベルヌーイ、カテゴリカル、ポアソン、ガンマなど多くの分布が指数型分布族に含まれる。

---

> **参考文献**
>
> - Bishop, "Pattern Recognition and Machine Learning", 2006
> - Goodfellow et al., "Deep Learning", 2016
> - Murphy, "Probabilistic Machine Learning", 2022
