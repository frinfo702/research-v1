#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
  header: none,
  footer: [Open-Vocabulary Segmentation Survey (presented by Kenichiro Goto)],
)

#set text(font: "Hiragino Kaku Gothic ProN", size: 17pt, lang: "ja")
#set par(leading: 0.72em)
#set list(indent: 1em, spacing: 0.34em)
#show raw: set text(font: "Geist Mono", size: 13pt)
#show heading.where(level: 1): set text(size: 25pt, weight: "bold")
#show heading.where(level: 2): set text(size: 20pt, weight: "bold")

#let ink = rgb("#14213d")
#let gleen = rgb("#0b6e4f")
#let accent-soft = rgb("#e6f4ef")
#let orange = rgb("#c76b3a")
#let accent-2-soft = rgb("#f9eee7")
#let dark-blue = rgb("#355c7d")
#let accent-3-soft = rgb("#eaf0f6")
#let muted = rgb("#5c6773")
#let skayblue = rgb("A3C4EF")

#let section-slide(title, subtitle: none) = slide[
  #align(center + horizon)[
    #text(size: 28pt, weight: "bold", fill: ink)[#title]
    #v(0.8em)
    #if subtitle != none [
      #text(size: 14pt, fill: muted)[#subtitle]
    ]
  ]
]

#let slide-head(title) = [
  #text(size: 22pt, weight: "bold", fill: ink)[#title]
  #v(0.45em)
]

#let card(title, body, fill: white, stroke: luma(215), title-fill: ink) = rect(
  width: 100%,
  inset: 0.9em,
  radius: 10pt,
  fill: fill,
  stroke: stroke,
)[
  #text(weight: "bold", fill: title-fill)[#title]
  #v(0.35em)
  #body
]

#let method-card(title, signal, gist, examples, fill) = rect(
  width: 100%,
  inset: 0.75em,
  radius: 10pt,
  fill: fill,
  stroke: none,
)[
  #text(weight: "bold", fill: ink)[#title]
  #v(0.25em)
  #text(size: 13pt, fill: gleen)[#signal]
  #v(0.35em)
  #text(size: 14pt)[#gist]
  #v(0.4em)
  #text(size: 12pt, fill: muted)[例: #examples]
]

#title-slide[
  = A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future

  Chaoyang Zhu, Long Chen, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) submission / arXiv v2, 2024

  #v(0.9em)
]

#slide[
  #slide-head([Open-Vocabulary Detection (OVD) / Open-Vocabulary Segmentation (OVS) を弱教師信号で整理する (ref: 1)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    card(
      [サーベイの主張],
      [
        - OVD/OVS を 弱い教師信号を許可するかとその使い方で分類する
          - 弱い教師信号：不完全・不正確・ノイズを含むラベルや教師情報
        - taxonomy は detection だけでなく semantic / instance / panoptic segmentation にも共通
        - さらに 3D / video にも拡張できる
      ],
      fill: accent-soft,
      stroke: gleen.lighten(35%),
      title-fill: gleen,
    ),
    card(
      [発表内容],
      [
        - *OVS* を中心に整理する
          - 個人的に興味がある
        - Zero-Shot Detection (ZSD) / Zero-Shot Segmentation (ZSS) は「どこから OVS に接続したか」を押さえる程度に扱う
        - タスクごとの研究の成熟度の差も確認する
      ],
      fill: accent-2-soft,
      stroke: orange.lighten(35%),
      title-fill: orange,
    ),
  )

  #v(0.8em)
]

#section-slide(
  [OVSの理解 (ref: 2)],
  subtitle: [タスク定義と Vision-Language Model (VLM) をどう使っているのか],
)

#slide[
  #slide-head([OVS は base-only 学習から base+novel 推論へ拡張する設定へ (ref: 2.1)])

  #text(weight: "bold", fill: gleen)[問題設定]
  #v(0.3em)
  - 学習では *base class* のみアノテーションあり、推論では *base + novel* を識別する
    - base class: 人間が高精度にアノテーションした明示的なクラス
    - novel class: 学習時には人間による教師データがなく、擬似ラベルや弱い教師信号で補う初見クラス
  - したがって OVS は「弱教師信号で novel をどこまで扱えるか」を問う設定だと言える

  #v(0.6em)
]
#slide[
  #slide-head([主な3種のタスク (ref: 2.1)])
  #table(
    columns: (1.3fr, 1.8fr, 1.3fr),
    inset: 7pt,
    stroke: luma(220),
    [*task*], [*何を分けるか*], [*個体を区別するか*],
    [*Open-Vocabulary Semantic Segmentation (OVSS)*], [テキスト記述に基づいて全ピクセルへ semantic label を付与], [しない。犬が3匹でもすべて `dog`],
    [*Open-Vocabulary Instance Segmentation (OVIS)*], [各インスタンスごとに mask を分けて識別], [する。`dog1`, `dog2`, `dog3` のように扱う],
    [*Open-Vocabulary Panoptic Segmentation (OVPS)*], [semantic と instance を統合して全ピクセルをラベル付け], [個体(e.g. 人間)は区別し、非個体（e.g. 道路、空）は区別しない],
  )

  #v(0.7em)
  #image("images/semantic_vs_instance_vs_panoptic.webp", width: 100%)
]

#slide[
  #slide-head([なぜ ZSS より OVS が発展したか (ref: 2.1)])

  #text(weight: "bold", fill: dark-blue)[背景]
  #v(0.3em)
  - 初期の *Zero-Shot Segmentation (ZSS)* は Word2Vec / GloVe / BERT など固定 text embedding の性能に強く依存した
  - そのため語彙表現と視覚表現の整合が弱く、密な予測に必要な識別力も不足しやすかった
    - 密な予測: ピクセル単位でのクラスタリング（セグメンテーション）

  #v(0.6em)
  #text(weight: "bold", fill: gleen)[OVS が伸びた理由]
  #v(0.3em)
  - *text-image pairs* のような弱教師信号を使える
   - これは明示的なクラスラベル、ピクセルごとのラベル、バウンディングボックスなどがついているわけではないので教師信号としては弱い
  - *Vision-Language Models (VLMs)*、特に *Contrastive Language-Image Pre-training (CLIP)* をそのまま土台にできる
  - その結果、text embedding が視覚表現と事前学習段階で整合し、novel class への汎化が大きく改善した

  #v(0.9em)
]


#slide[
  #slide-head([OVSにおけるVLMの役割 (ref: 2.4)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.1em,
    card(
      [text encoder],
      [
        - class name / prompt を埋め込みへ変換し *frozen classifier* として使う
        - 例: `a photo of a dog`, class description, attribute prompt
        - 語彙を閉じずに推論できる
      ],
      fill: skayblue,
      stroke: gleen.lighten(35%),
    ),
    card(
      [image encoder],
      [
        - CLIP / Segment Anything Model (SAM) / self-distillation with no labels (DINO), T2I diffusion models  などを 特徴抽出器, teacher, adapter の土台にする
        - ただし image-level 学習済み表現をpixel-levelに転移させるので *局所識別の弱さ* と *domain gap* が残る
      ],
      fill: accent-2-soft,
      stroke: orange.lighten(35%),
      title-fill: orange,
    ),
  )

]

#slide()[
  #slide-head([VLMsのtext encoder, image encoderとしての使用例 (ref: 2.4)])
  #image("images/VLMs-encoder-example.png")
]

#section-slide(
  [taxonomy: 弱教師信号を使うかどうかとその使い方 (ref: 1)],
)

#slide[
  #slide-head([Zero-Shot (ZS) 2 系統と Open-Vocabulary (OV) 4 系統 (ref: 1)])

  #grid(
    columns: (0.8fr, 1.2fr),
    gutter: 1em,
    card(
      [Fig. 1],
      [
        - 横軸: 設定と課題
        - 縦軸: 方法
        - 左の 2 列が *ZS*, 右側が *OV*
        - OVS は semantic / instance / panoptic に分かれるが、同じ分類
      ],
      fill: accent-soft,
      stroke: gleen.lighten(35%),
      title-fill: gleen,
    ),
    [
      #align(center)[
        #image("images/proposed-taxonomy.png", width: 100%)
      ]
    ],
  )
]

#slide[
  #slide-head([OVSでよく使われる手法: pseudo-labeling, Knowledge Distillation (KD), transfer learning (ref: 1)])

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
      #card(
        [OVS で主に見る],
        [
          - *Pseudo-labeling*: teacher が mask / label を作る
          - *Knowledge distillation*: teacher 空間へ student を蒸留
          - *Transfer learning*: VLM を feature extractor や adapter の土台として直接使う
          - Fig. 2.a\~cはZSSの発想
        ],
        fill: accent-2-soft,
        stroke: orange.lighten(35%),
        title-fill: orange,
      )
      #v(0.7em)
    ],
    [
      #align(center)[
        #image("images/methodology-a-through-c.png")
      ]
    ],
    [
      #image("images/methodology-d-through-f.png")
    ]
  )

  #v(0.5em)
]

#section-slide(
  [OVS の実態 (ref: 6)],
  subtitle: [OVSS / OVIS / OVPS で成熟度が大きく異なる],
)

#slide[
  #slide-head([OVS の主戦場は ZSSとは異なる 4つのOV特有領域 (ref: 6)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.1em,
    card(
      [ZSS から受け継いだ発想],
      [
        - class embedding を classifier へ差し込む
        - novel feature synthesis で初見カテゴリの不足を補完
        - ただし密な予測タスクでは *背景との混同* が深刻
          - 初見の物体を、認識済みの物体ではない→背景として認識してしまう
      ],
      fill: white,
    ),
    card(
      [OVS で追加されたもの],
      [
        - image-caption から region-word alignment を掘る
        - CLIP / SAM / DINO などのVLMsの獲得した表現を teacher か backbone として使う
        - prompt, adapter, partial fine-tuning で open-vocabulary を保ったまま局所化を改善
      ],
      fill: accent-soft,
      stroke: gleen.lighten(35%),
      title-fill: gleen,
    ),
  )
]

#slide[
  #slide-head([4つのOV特有領域の違い: region-word 対応をどこまで明示化するか (ref: 6)])

  #table(
    columns: (1.15fr, 1.25fr, 1.6fr),
    inset: 8pt,
    stroke: luma(220),
    [*route*], [*目的*], [*典型的な失敗*],
    [Region-aware], [captionを教師信号にして新しい語彙へ触れる], [正しい region-word 対応が曖昧],
    [Pseudo-labeling], [mask / region 対応を明示的に与える], [pseudo mask のノイズと novel class 名を既知クラスに分類してしまう],
    [Knowledge distillation], [teacher 空間へ student 空間を寄せる], [teacher 自体が密な予測タスクに最適ではない],
    [Transfer Learning], [VLM を直接密な予測に適応させる], [patch-level 識別と境界品質が弱い],
  )
]

#slide[
  #slide-head([OVSSの変遷: region-aware → transfer learning (ref: 6.1)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(
      [OVSS の整理],
      [
        - *Region-aware*: OpenSeg, GroupViT, ViL-Seg, SegCLIP
        - *Pseudo-labeling*: Zabari et al. の relevance map ベース
        - *Knowledge distillation*: GKC, SAM-CLIP, ZeroSeg
        - *Transfer learning*: LSeg, MaskCLIP, OVSeg, CAT-Seg, SAN, TagCLIP
      ],
      fill: accent-3-soft,
      stroke: dark-blue.lighten(35%),
      title-fill: dark-blue,
    ),
    card(
      [傾向],
      [
        - 初期は caption-level 学習だけで dense mask を出す工夫が主流
        - その後は *CLIP image encoder をどう残して局所性を補うか* が中心的課題
        - fine-tuning時、 全部やるよりも *prompt / adapter / selective tuning* が安定
      ],
      fill: white,
    ),
  )
]

#slide[
  #slide-head([OVSS の中心的課題: CLIP を壊さず局所性だけ補う (ref: 6.1.4)])

  #grid(
    columns: (1.05fr, 0.95fr),
    gutter: 1em,
    [
      #align(center)[
        #image("images/framework-for-transfer-learning-based-models.png", height: 68%)
      ]
    ],
    card(
      [図],
      [
        - *frozen VLMs-IE as feature extractor*
        - *fine-tuning VLMs-IE*
        - *learning prompts or adapters*
        - この3系統が transfer learning の主要な選択肢
      ],
      fill: accent-3-soft,
      stroke: dark-blue.lighten(35%),
      title-fill: dark-blue,
    ),
  )
]

#slide[
  #slide-head([OVSS は 3種のタスクの中で最もベンチマークが成熟 (ref: 6.1)])

  #table(
    columns: (1.1fr, auto, auto),
    inset: 7pt,
    stroke: luma(220),
    [*代表結果*], [*dataset / metric*], [*値*],
    [SLIC], [ADE20K A-150 mean Intersection over Union (mIoU)], [36.6],
    [CAT-Seg], [Pascal Context-59 mIoU], [62.0],
    [SCAN], [Pascal VOC-20 mIoU], [97.0],
    [TagCLIP], [generalized Pascal VOC harmonic mean (HM)], [89.2],
    [MAFT], [generalized COCO Stuff HM], [46.5],
  )

  #v(0.55em)
  #text(size: 13pt, fill: muted)[表 13-14 から抜粋。評価設定が異なるので絶対比較よりも「OVSS が最も成熟している」点が重要。]
]

#slide[
  #slide-head([OVIS / OVPS は unified modeling が進む一方で性能はまだ不安定 (ref: 6.2, 6.3)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(
      [OVIS],
      [
        - *Region-aware*: CGG, D2 Zero
        - *Pseudo-labeling*: XPM, Mask-free OVIS, MosaicFusion
        - *KD*: OV-SAM
        - 重点は *pseudo mask noise* と 初見オブジェクトと背景の分離
      ],
      fill: accent-2-soft,
      stroke: orange.lighten(35%),
      title-fill: orange,
    ),
    card(
      [OVPS],
      [
        - *Region-aware*: Uni-OVSeg, X-Decoder, APE
        - *KD*: PADing
        - *Transfer*: FC-CLIP, FreeSeg, OPSNet, ODISE, OMG-Seg
        - thing / stuff を一緒に扱うため、分類だけでなくmaskをどう総合的に取り扱うかが難しい
      ],
      fill: accent-soft,
      stroke: gleen.lighten(35%),
      title-fill: gleen,
    ),
  )
]

#slide[
  #slide-head([OVIS / OVPS は benchmark と手法系譜が分散 (ref: 6.2, 6.3)])

  #table(
    columns: (1.1fr, 1.15fr, auto),
    inset: 7pt,
    stroke: luma(220),
    [*結果*], [*dataset / metric*], [*値*],
    [XPM], [OVIS COCO Average Precision (AP)], [21.6],
    [Mask-free OVIS], [OVIS OpenImages AP], [25.8],
    [APE], [OVPS ADE20K Panoptic Quality (PQ)], [26.1],
    [OPSNet], [OVPS ADE20K PQ], [17.7],
  )

  #v(0.55em)
  #text(size: 13pt, fill: muted)[OVSS に比べると OVIS / OVPS は benchmark と手法系譜がまだ分散している。]
]

#section-slide(
  [3D / videoでの応用: image-domain foundation models (ref: 7)],
  subtitle: [2D VLM を介した open-vocabulary 化が主流],
)

#slide[
  #slide-head([3D / video の open-vocabulary 化は 2D VLM を介した移送で進んでいる (ref: 7.1, 7.2)])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    card(
      [3D scene understanding],
      [
        - 点群とテキストを対応づけた大規模データの作成は現実的に実現不可能
        - そのため *2D detector / CLIP / SAM* などVLMsを中間表現として使う
        - OV-3DET, FM-OV3D, CoDA は 2D box や CLIP featureなどに行列を用いて線形変換する(3D→2D)
        - OpenScene は 点群の特徴量を CLIPの潜在空間へ蒸留
      ],
      fill: accent-3-soft,
      stroke: dark-blue.lighten(35%),
      title-fill: dark-blue,
    ),
    card(
      [video understanding],
      [
        - 課題は segmentation に *tracking* が加わること
        - OV2Seg, OpenVIS は frame-level proposal と tracking memory を組み合わせる
        - 研究量はまだ少なく、dataset / protocol も発展途上
      ],
      fill: white,
    ),
  )

  #v(0.7em)
  #text(size: 14pt, fill: muted)[2Dのtaxonomy は 3D / video にも適用できるが、実際の実装は image-domain foundation models への依存が強い。]
]

#section-slide(
  [ボトルネック: bias, alignment, VLM adaptation (ref: 8)],
  subtitle: [サーベイ中で示されている今後],
)

#slide[
  #slide-head([主要課題: base-class bias と region-word 対応の不安定さ (ref: 8.1)])

  #card(
    [Challenges],
    [
      - *Base-class overfitting*: 擬似でないラベルの質と量が擬似ラベルより高い
      - *Novel vs background confusion*: 初見の物体が背景と判定されてしまう
      - *Correct region-word correspondence*: caption supervision は weak で曖昧
      - *Large VLM adaptation*: CLIP の image-level 表現を dense / masked crops に適応しづらい
      - *Inference speed and evaluation*: 重い backbone と厳しすぎる open-vocab 評価
    ],
    fill: accent-2-soft,
    stroke: orange.lighten(35%),
    title-fill: orange,
  )
]

#slide[
  #slide-head([今後: タスクの統合とfoundation modelsの組み合わせ (ref: 8.2)])

  #card(
    [Future directions#footnote[⭐︎: 個人的に興味あり]],
    [
      - ⭐︎detection/segmentation 以外の open-vocabulary perception へ拡張
      - *OVD と OVS の統合*、さらに 2D / 3D をまたぐ foundational model へ
      - *multimodal Large Language Model (MLLM)* を perception interface として使う
      - ⭐︎CLIP / SAM / DINO / diffusion など foundation models の組み合わせ
      - *real-time OVD/OVS* の実現
    ],
    fill: accent-soft,
    stroke: gleen.lighten(35%),
    title-fill: gleen,
  )
]

#slide[
  #slide-head([まとめ (ref: 9)])
    - OVS は「弱教師信号と VLM を使って閉じていない表現(open-vocabulary)の分類タスクを密な予測タスク（ピクセル単位の予測）に適用する」ということをしている
    - 実際の主流は transfer learning 系で、OVSS が最も成熟している。OVIS / OVPS は性能はまだ不安定
    - 今後の中心課題は region-word 対応、base/novel bias、そして CLIP / SAM / DINO / MLLM をどう組み合わせるか、など

  #v(1em)
  #align(center + horizon)[
    #text(size: 21pt, weight: "bold", fill: ink)[Questions?]
  ]
]
