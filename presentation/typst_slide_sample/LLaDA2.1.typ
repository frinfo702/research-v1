#set page(paper: "presentation-16-9", margin: (x: 2.5em, y: 2em))
#set text(font: "Hiragino Kaku Gothic ProN", size: 17pt, lang: "ja")
#set list(indent: 1em, spacing: 0.5em)
#set par(leading: 0.7em)

// ── タイトル ──────────────────────────────────────────────────
#align(center + horizon)[
  = LLaDA2.1: Speeding Up Text Diffusion via Token Editing

  Tiwei Bie et al.
]
#pagebreak()

// ── 問題設定 ─────────────────────────────────────────────────
= 問題設定

- LLM推論の主流は *Autoregressive (AR)* デコーディング：1トークンずつ逐次生成
- *Discrete Diffusion LM (dLLM)* はマスクを並列に予測することで高速化を目指す
- しかし dLLM は *速度と品質のトレードオフ* を抱えている
  - ステップ数を増やすと品質は AR 並みだが速度優位が消える
  - ステップを減らすと品質が劣化する

*このトレードオフを根本から解消できるか？*
#pagebreak()

// ── モチベーション ────────────────────────────────────────────
= モチベーション

*LLaDA2.0 で採用された M2T デコーディングの 3 つの問題*

+ *Exposure Bias*：訓練はきれいなマスク列に対して行われるが、推論では部分的に予測済みのノイジーな列を受け取る → エラーが蓄積する
+ *Stuttering（n-gram 反復）*：位置ごとに独立にサンプリングするため、同じトークンが繰り返されやすい
+ *修正不可能*：一度配置したトークンは後から訂正できない

→ 既存手法（ReMDM, Prophet など）は訓練なしのパッチ適用にとどまり根本解決になっていない
#pagebreak()

// ── 先行研究 ─────────────────────────────────────────────────
= 先行研究

*共通の課題：一度置いたトークンを書き換える学習の目的関数が存在しない*

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  [*手法*], [*概要*], [*課題*],
  [LLaDA (2025.02)],    [dLLM 8B、AR と同等品質],     [ステップ数 ≈ 応答長が必要],
  [LLaDA2.0 (2025.12)], [100B MoE に拡張],            [M2T のまま、速度問題は未解決],
  [ReMDM / Prophet],    [低信頼度トークンの再マスク], [訓練なしパッチ、100B 未検証],
)
#pagebreak()

// ── 新規性 ───────────────────────────────────────────────────
= 新規性

確定済みトークンを推論中に書き直す *Token Editing* のための目的関数を導入

+ *Draft-and-Edit パラダイム*：M2T と T2T を同一ステップで実行
+ *Mixture of M2T and T2T*：editing を学習させる新しい目的関数と Multi-turn Forward (MTF) データ拡張
+ *ELBO-based Block-level Policy Optimization (EBPO)*：dLLM の 100B スケールで初めて成功した RL フレームワーク
#pagebreak()

// ── アイデア ─────────────────────────────────────────────────
= アイデア：Token Editing

- 従来の M2T デコーディング：`[MASK]` → トークン（一方向・不可逆）
- LLaDA2.1 のデコーディング：各ステップで *2 種類の更新集合* を動的に決定
  - *Unmasking Set $Gamma_t$*：信頼度 $tau_"mask"$ を超えた `[MASK]` 位置を確定
  - *Editing Set $Delta_t$*：すでに確定済みだが、モデルの予測が現トークンと異なり、かつ信頼度 $tau_"edit"$ を超えた位置を上書き

$ "更新位置" = Gamma_t union Delta_t $

#align(center)[#image("assets/llada2_editable_mask_to_token_modes_overview.png")]
#pagebreak()

// ── 提案手法：訓練 ────────────────────────────────────────────
= 提案手法：訓練

二つのストリームを組み合わせた目的関数

- *M2T ストリーム*：従来通り、マスクから正しいトークンを生成（ドラフト作成能力）
- *T2T ストリーム*：入力から正しいトークンを復元（自己訂正能力）

*Multi-turn Forward (MTF) データ拡張*
- 訓練中に「中間編集状態」のシーケンスを生成し、多様な (ドラフト, 修正目標) ペアを学習
- 実際の推論過程に近い状況に対処できるようにする

→ アーキテクチャの変更は不要。訓練目的の変更だけで実現

#align(center)[#image("assets/llada2_stage1_cpt_pipeline_overview.png")]
#pagebreak()

// ── 提案手法：推論と RL ───────────────────────────────────────
= 提案手法：推論モードと RL

- *S Mode（Speedy）*：$tau_"mask"$ 低め（積極的展開）→ 高スループット、T2T 訂正に依存。コード生成・数学向き
- *Q Mode（Quality）*：$tau_"mask"$ 高め（慎重展開）→ 高品質、スループットは中程度。汎用・instruction following 向き

*Multiple Block Editing (MBE)*：後続ブロックを参照して前のブロックを遡及修正 → 整合性向上

*ELBO-based Block-level Policy Optimization (EBPO)*
- dLLM の系列対数尤度は計算困難 → ELBO 近似 + Block-Causal Masking で勾配推定を計算可能に
- 100B 規模で RL を初めて成功させた
#pagebreak()

// ── 実験設定 ─────────────────────────────────────────────────
= 実験設定

- *モデル*：LLaDA2.1-Mini (16B MoE) と LLaDA2.1-Flash (100B MoE)
- *ベンチマーク*：33 タスク
  - 推論：GPQA, DROP, MATH
  - コード：HumanEval+, BigCodeBench, LiveCodeBench
  - 知識・常識：TriviaQA, PIQA, HellaSwag, C-Eval
- *比較対象*：LLaDA2.0-Mini/Flash（直接の前身）、Ling、Qwen3（AR モデル）
- *主要メトリクス*
  - *TPS*（Tokens Per Second）：実時間スループット
  - *TPF*（Tokens Per Forward pass）：1 回の forward あたりの生成トークン数（AR では 1）
#pagebreak()

// ── 実験結果：Flash ───────────────────────────────────────────
= 実験結果：Flash

#table(
  columns: (1fr, auto, auto),
  inset: 8pt,
  [*モデル*], [*平均スコア*], [*TPF*],
  [LLaDA2.0-Flash],          [72.43], [3.08],
  [LLaDA2.1-Flash Q Mode],   [73.54], [3.64],
  [LLaDA2.1-Flash S Mode],   [72.34], [5.93],
)

Q Mode は品質・速度の両面で LLaDA2.0-Flash を上回る

#align(center)[#image("assets/llada2_1_flash_benchmark_performance_table_1.png")]
#pagebreak()

// ── 実験結果：Mini ────────────────────────────────────────────
= 実験結果：Mini

#table(
  columns: (1fr, auto, auto),
  inset: 8pt,
  [*モデル*], [*平均スコア*], [*TPF*],
  [LLaDA2.0-Mini],           [63.39], [2.60],
  [LLaDA2.1-Mini Q Mode],    [63.90], [3.12],
  [LLaDA2.1-Mini S Mode],    [62.07], [5.34],
)

Mini 系列でも Q Mode は LLaDA2.0-Mini より高品質・高 TPF

#image("assets/llada2_0_mini_benchmark_performance_table_2.png")
#pagebreak()

// ── 実験結果：スループット ────────────────────────────────────
= 実験結果：スループット

*LLaDA2.1-Flash (100B, S Mode + 量子化) の TPS*
// - HumanEval+：891.74 TPS
// - BigCodeBench-Full：801.48 TPS
// - LiveCodeBench：663.39 TPS

// *LLaDA2.1-Mini (16B, S Mode + 量子化) は最大 1586.93 TPS*

#align(center)[#image("assets/llada2_mini_series_tokens_per_second_chart.png")]
#pagebreak()

// ── アブレーション ────────────────────────────────────────────
= アブレーション

+ *MBE の効果*：推論・コードタスクでスコアが一貫して向上。スループットは微減
+ *MTF データ拡張の効果*：除去すると S Mode の品質が顕著に低下 → 多様な編集シナリオの学習が必要
+ *T2T ストリームの効果*：除去すると S Mode で大幅な品質劣化 → 自己訂正能力の学習が不可欠
+ *量子化の影響*：スループット向上、コード・数学は許容範囲内。open-ended タスクは影響大
+ *EBPO の効果*：RL 無しでは推論・instruction following スコアが低下

#align(center)[#image("assets/llada2_1_mbe_performance_comparison_table_4.png")]
#pagebreak()

// ── 制限 ─────────────────────────────────────────────────────
= 制限

+ *Stuttering は残る*：T2T 編集で軽減するが、並列サンプリング由来の n-gram 繰り返しを完全には排除できない
+ *KV キャッシュ非対応*：トークンが変化し続けるため、AR モデルのような KV キャッシュ再利用ができない。高 TPF でカバーするが、1 ステップのコストは高いまま
+ *Exposure bias は部分的な解消にとどまる*：T2T 訓練で緩和されるが、完全な解消には至らない
+ *RL と editing の統合*：現在の EBPO は生成全体に対して報酬を与えるが、「訂正操作そのもの」への報酬設計は今後の課題
#pagebreak()

// ── まとめ ───────────────────────────────────────────────────
= まとめ

Token Editing（推論中のトークン上書き）を訓練目的に組み込むことで、100B 規模の dLLM において速度と品質のトレードオフを改善した

- 確定済みトークンの書き直しを M2T + T2T を合わせた目的関数で学習
- 高速並列生成＋自己訂正が可能。KV キャッシュ不可、stuttering 残存
- コード・数学では S Mode が有効。open-ended では Q Mode 推奨
- 再現性のために $Gamma_t$, $Delta_t$ の定義（閾値 $tau_"mask"$, $tau_"edit"$）、MTF 拡張、EBPO 勾配推定の調査が必要
