#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: none,
  footer: [GLA-CLIP presented by Kenichiro Goto],
)

#set text(font: "Hiragino Kaku Gothic ProN", size: 18pt, lang: "en")
#set par(leading: 0.8em)
#set list(indent: 1.1em, spacing: 0.38em)
#set math.equation(numbering: "(1)")
#show raw: set text(font: "Geist Mono", size: 13pt)
#show heading.where(level: 1): set text(size: 27pt, weight: "bold")
#show heading.where(level: 2): set text(size: 22pt, weight: "bold")

#let ink = rgb("#111111")
#let accent = rgb("#1d4ed8")
#let muted = rgb("#5f6368")

#let slide-head(title) = [
  #text(size: 22pt, weight: "bold", fill: ink)[#title]
  #v(0.45em)
]


#title-slide[
  = Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation

  ByeongCheol Lee, Hyun Seok Seong, Sangeek Hyun, Gilhan Park, WonJun Moon, Jae-Pil Heo
]

#slide[
  #slide-head([要点 (ref: 1)])

  CLIP ベースの training-free#footnote[すでに訓練済みのモデル（e.g. CLIPやLLM）のパラメータは凍結し、
    プロンプトやコンテキスト、外部のルール（e.g. 検索・ルールベースの処理）を工夫して性能を上げるアプローチ] Open-Vocabulary Semantic Segmentation (OVSS) では、高解像度画像を扱うため overlapping sliding-window#footnote[画像を重なりを持たせて切り出す。windowは1枚のサブ画像] 推論が不可欠

  しかし、window の独立処理により大域的な文脈が失われ、同一物体に対するラベルの不一致や境界における grid artifact#footnote[@fig:comparison-of-segmentation-consistency-near-widnow-boundaries 参照] が発生する
  本論文は、この精度の低下が「学習不足」ではなく「推論時の receptive field の不足」に起因すると指摘している

  これらに対処するためGLA-CLIPでは以下を導入する

  - global key/value: 参照先を画像全体の window に拡張
  - proxy anchor: local query に内在する window bias を軽減
  - dynamic normalization: object scale に応じて global attention を適応的に調整

  GLA-CLIPは既存の training-free OVSS に対し、再学習なしで window 間の推論の整合性を向上させる拡張手法
]

#slide[
  #slide-head([Window 境界の不整合 (ref: 3.1.2)])

  CLIP は低解像度画像で事前学習されているため、大域的な画像空間は十分獲得しているが、高解像度での dense prediction#footnote[ピクセル単位でのタスク（e.g. セグメンテーション）] には不向きである。
  そのため、既存の training-free OVSS は overlapping crop ごとに独立して推論を行い、後から結果を統合する。

  - 画像全体を俯瞰できないため、同一物体でも window ごとに異なるラベルが付与されやすい
  - 特に大きな物体 (large object) や背景 (stuff category) において文脈分断の影響が顕著
  - 本論文では、この推論の不整合を Boundary Error Rate (BER) という指標で定量化

  重なり領域の単純な平均化では解決できない、window 境界における根本的な意味のズレが課題

  #figure(
    image("images/comparison-of-segmentation-consistency-near-widnow-boundaries.png", height: 80%),
    caption: [BER の低下と grid artifact の抑制により、window 境界の不整合が主要課題だと示している。],
  )<fig:comparison-of-segmentation-consistency-near-widnow-boundaries>
]

#slide[
  #slide-head([既存法のギャップ (ref: 2.2)])

  既存法でも sliding-window は用いられるが、window 間での意味的な情報の共有が不十分である。
  処理 token を増やしても、query 自体が局所的な window の特徴から生成される限り、attention は window 内部に偏ってしまう。

  - 局所的な精度は高くても、大きな物体や背景において大域的な文脈が途切れやすい
  - overlapping merge#footnote[複数のwindowに含まれる画素に対する結果の統合処理] は境界の logit を平滑化するのみで、根本的な意味のズレは補正できない
    - 結局そのwindowごとのセグメンテーション結果は変化していない
  - 結果として、BER の悪化、小物体の推論の不安定化、grid artifact の残存といった問題が生じる

  根本的な原因は text-image alignment の不足ではなく、(attentionの) query が局所的な情報のみで構成されている点にある
]

#slide[
  #slide-head([GLA-CLIP の全体像 (ref: 3)])

  GLA-CLIP は新しい backbone#footnote[モデルの基盤となる特徴抽出器 (e.g. CLIP, DINO)] を学習するのではなく、既存の training-free OVSS に対する推論時の補正モジュールとして機能する。

  1. Key-Value Extension: query は局所的なまま、参照先の key/value を全 window へ拡張
  2. Proxy Anchor: local query を画像全体で安定した代表点へ置換し、window bias を軽減
  3. Dynamic Normalization: object scale に応じて attention の閾値とスケールを動的に調整

  単なる global context の追加に留まらず、推論時の bias とスケールのミスマッチを同時に補正するアーキテクチャ

  #figure(
    image("images/overview-of-gla-clip-framework.png", height: 80%),
    caption: [Overview: local query に対して global key/value を参照し、proxy anchor と dynamic normalization で attention を安定化する],
  )
]

#slide[
  #slide-head([Key-Value Extension (ref: 3.2)])

  local window から生成した query は維持しつつ、key と value には全 window から集めた情報を適用する(query自体はlocalしか考慮しないが、返答としては全体を知ったものを用意しておく)
  これにより、現在の window の query が画像全体の token を参照可能となり、receptive field のみが拡張される。

  - 大きな物体や背景を、複数 window にまたがって大域的に捉えることが可能
  - baseline モデルを再学習することなく、scene-level の手掛かりを導入できる
  - ただし、query 自体は局所的であるため、この段階ではまだ inner-window への偏りが残存する

]

#slide[
  #slide-head([Key-Value Extention. 数式])

  $ A_"ext" = Q K_"global"^T $
  $ F_"visual" = "Proj"(A_"ext" V_"global") $

  - $Q$ は現在 window の query、$K_"global"$ と $V_"global"$ は全 window から集めた token。
  - 参照先だけを拡張するので既存 pipeline に差し込みやすい
  - $V_"global"$: 全windowから集めたvalue(情報本体)。各パッチの視覚特徴が入っている
  - $A_"ext"$: query, (global) key の類似度から作成したattention。どのパッチをどれくらい使うか
  - $"proj"(dot)$: attentionを使って、valueを重み付きで集約し1つの特徴にまとめた後、線形変換をして特徴の次元や表現を整える
]

#slide[
  #slide-head([Proxy Anchor (ref: 3.3)])

  global key/value を参照可能にしても、query が局所特徴量に由来する限り、現在の window 内の類似 token を過大評価しがちである。
  本論文ではこれを「window bias」と定義し、query を意味的に安定した代表点 (Proxy Anchor) へと置換する。

  - 各 query に対し、全 window から cosine 類似度の高い token 群を抽出
  - それらの平均を反復的に計算し、新たな proxy anchor を生成
  - proxy anchor を基に attention を再計算し、空間的近さよりも意味的な類似度を重視させる

]

#slide[
  #slide-head([Proxy Anchor 数式])

  より意味的に安定したProxy Anchorの作成（以降queryとして使う）

  $ P_i^(0) = { j | Q_i^(0) K_j^T > rho } $
  $ Q_i^(t) = 1 / abs(P_i^(t-1)) sum_(j in P_i^(t-1)) K_j $

  式(3)はi番目のqueryに対して似ているtokenのインデックスの集合
  - $Q_i^0$: i番目のquery(初期状態)
  - $K_j$: j番目のkey(他のパッチの特徴)
  - $rho$: 閾値

  式4は式3で集めた類似tokenの平均をとる
  - $P_i^(t-1)$: 前ステップで集めた類似トークンのインデックスの集合
  - $|P_i^(t-1)|$: その数（集合の大きさ）
  - $Sigma K_j$: 特徴の和

]

#slide[
  #slide-head([Dynamic Normalization (ref: 3.4)])

  global token の増加は大きな物体の認識には有利だが、小物体では positive token が少なくノイズに埋没しやすくなる。
  これを防ぐため、query ごとに attention の閾値とスケールを動的に変化させ、global context の影響度を調整する。

  - $u$: window 数 $L$ に応じて閾値を引き上げ、global token 増加に伴うノイズを抑制
  - $w_i$: scaling valueとして定義。高信頼度 token 数 $|P_i|$ が少ないほど大きな値を取り、小物体の特徴を保護
  - データセット固有の手動チューニングを排除し、object scale に応じた適応的に処理

  $
    "Attn"_i = bold(w)_i (bold(S)_"proxy" - bold(u)/(N L) Sigma_(j=1,dots, N L) [bold(S)_"proxy"]_(i j))
  $

  $ bold(u) = 1 + lambda_1 log(1 + L) $
  $ bold(w)_i = 1 + lambda_2 / (|P_i|) $

  global context の過剰な導入による小物体の認識精度低下を、query 単位で効果的に防ぐ手法
]

#slide[
  #figure(
    image("images/visualization-of-attention-maps-for-an-anchor-query-token.png"),
    caption: [Key-Value Extension だけでは attention が散り、proxy anchor と dynamic normalization を入れると意味的に一貫した領域へ集中する。],
  )
]

#slide[
  #slide-head([主結果 (ref: 4.2)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      8つの Semantic Segmentation ベンチマークにて、mean Intersection-over-Union (mIoU) #footnote[semantic segmentationの代表的な評価指標。特定クラスについて予測と正解の$"積集合"/"和集合"$を計算しその平均を取る] を用いて評価を実施。
      メインの実験設定として、重みを固定した CLIP ViT-B/16 および DINO ViT-B/8 を使用している。

      - CLIP-DINOiser ベース: 平均 *+1.6* mIoU の改善
      - ClearCLIP への統合: *+1.2* mIoU の改善
      - ProxyCLIP への統合: *+0.6* mIoU の改善
      - SCLIP への統合: *+1.6* mIoU の改善
      - 手動チューニングなしで平均 *44.0*、データセット特化チューニングありで *44.3* を達成

      手動チューニングなしで複数の既存手法に安定した精度向上をもたらす、高い汎用性と実用性が特長
    ],
    [
      #figure(
        image("images/open-vocabulary-semantic-segmentation-results.png", height: 70%),
        caption: [Table 1: GLA adaptation は ClearCLIP, ProxyCLIP, SCLIP など複数の既存法に一貫して追加利得を与える。],
      )
    ],
  )
]

#slide[
  #slide-head([質的比較 (ref: 4.2)])

  #grid(
    columns: (0.92fr, 1.08fr),
    gutter: 1em,
    [
      @fig:proxyclip において、Pascal VOC, COCO-Stuff, Cityscapes いずれのデータセットでも、window を跨ぐ同一領域に対して一貫したラベル付与が確認できる。

      - ProxyCLIP や CASS と比較して境界が自然であり、window 間の不自然な段差が顕著に低減
      - 特に stuff category (背景) や大きな物体において、視覚的な改善が明確
      - 定量的な BER の改善が、定性的な視覚的一貫性の向上として現れている

      単なる mIoU のスコア向上に留まらず、推論結果における window artifact を視覚的にも大きく削減
    ],
    [
      #figure(
        image("images/qualitative-results-among-proxyclip.png"),
        caption: [Pascal VOC, COCO-Stuff, Cityscapes で、ProxyCLIP や CASS より window 境界の不整合が少ない。],
      )<fig:proxyclip>
    ],
  )
]

#slide[
  #slide-head([Ablation (ref: 5.1)])

  #grid(
    columns: (0.9fr, 1.1fr),
    gutter: 1em,
    [
      Baseline モデル (30.8) に対し、Key-Value Extension と Dynamic Normalization の追加で *43.1*、Full model で *44.0* へと向上。
      精度改善の最大の要因は global context の導入だが、残り2つのモジュールが推論の不安定性を効果的に補っている。

      - Key-Value Extension の寄与が最も大きく、従来法の課題が context 欠落にあることを裏付け
      - Proxy anchor による inner-window bias の補正がさらなる精度向上に寄与
      - Dynamic normalization はデータセット固有のパラメータチューニングを不要にする役割を果たす
      - backbone を変更しても改善効果が維持されるため、attention 設計そのものの有効性が示された

      提案する3つの要素は独立して機能し、それぞれ Context 拡張、Bias 補正、Scale 補正の役割を担う
    ],
    [
      #figure(
        image("images/ablation-study-of-eatch-component-in-our-method.png"),
        caption: [Table 2: full model が最良で、proxy anchor と dynamic normalization の両方に追加利得がある。],
      )
    ],
  )
]

#slide[
  #slide-head([限界と要点 (ref: 6)])

  GLA-CLIPはCLIP-basedモデルで問題視される局所性に対してのひとつのアプローチ

  その有効性は、単に global context を導入した点にあるのではなく、
  推論段階において local query 固有の bias や object scale の差異までも同時に補正した点


  - 強み: 既存の training-free OVSS への容易な後付けが可能であり、8つのベンチマークで一貫した改善を達成
  - 弱み: 全 window の token を参照するため、計算量およびメモリコストが増大する
  - 今後の課題: 推論負荷を抑えつつ、高品質な token をいかに効率的に選別するか

]
