#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-common(frozen-counters: (figure,)),
)
#show raw.where(block: true): set text(size: 11pt, font: "Geist Mono")
#show figure.where(kind: raw): it => {
  show figure.caption: set text(size: 10pt, fill: luma(120))
  it
}
#show figure.where(kind: table): it => {
  show figure.caption: set text(size: 12pt, fill: luma(120))
  it
}
#set page(background: image("images/presentation_background.png"))

#let dummy-text(words: 20) = {
  let base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
  let arr = base.split(" ")
  let end = calc.min(words, arr.len())
  arr.slice(0, end).join(" ") + "."
}

// title
#slide[
  #set align(center + horizon)
  #text(size: 36pt, weight: "bold")[ov-seg 進捗]
  #v(1em)
  #text(size: 18pt, fill: luma(100))[全体ゼミ]
  #v(2em)
  #text(size: 14pt)[Kenichiro Goto • 20260427]
]

// section title
#slide[
  #set align(center + horizon)
  #text(size: 32pt, weight: "bold", fill: rgb("#0F4C81"))[Section Title]
  #v(0.8em)
  #text(size: 16pt, fill: luma(100))[Section subtitle or brief overview]
]

#slide[
  == 問題設定: Open-Vocabulary Segmentation
  #columns(2, gutter: 2em)[
    Close-Vocabulary Segumentationは、事前に決められたクラス（例：車・人・木）だけをラベルづけする

    Open-Vocabulary

    - 画像中の領域を「テキストによる自由な説明（例：『赤いオープンカー』『大型の白い雲』）」にマッチさせたい
    - そのテキストクラスは訓練時には見ていなくてもいい

    #colbreak()
    #figure(
      image(
        "images/The-segmentation-results-comparison-of-the-closevocabulary-and-open-vocabulary.png",
        width: 100%,
        height: 70%,
        fit: "cover",
      ),
      caption: "The segmentation results comparison of the closevocabulary and open-vocabulary segmentation. #footnote[https://www.researchgate.net/figure/The-segmentation-results-comparison-of-the-closevocabulary-and-open-vocabulary_fig5_390405782",
    )
  ]
]

#slide[
  == OV-Seg
  #columns(2, gutter: 2em)[
    === 主流アプローチのボトルネック
    1. マスク生成（クラス非依存）
    2. マスク領域のテキスト分類
      - 事前学習済みのVLM（例：CLIP）で、マスクされた領域画像を入力し、テキストラベルを割り当てる。

    $arrow$ CLIPは”自然な全体画像”が前提なので背景が抜け落ちたマスク付き画像を上手く扱えない
    #colbreak()
    #figure(
      image(
        "images/overview-exsiting-method.png",
        width: 100%,
        height: 70%,
        fit: "cover",
      ),
      caption: "OV-Seg以前に主流だった2段階アプローチ",
    )
  ]
]

#slide[
  == 提案手法1：Mask-adapted CLIP
  #columns(2, gutter: 2em)[
    - CLIP を*マスク付き画像に強いモデル*に変える
    - 既存のキャプションデータセットを用い、CLIP を使ってマスク領域の画像とキャプション中の名詞をマッチングさせたペアを作成し擬似訓練データを自動作成
      - マスク情報が載ったデータを作成できる
    - それらを使ってCLIPをfine-tuning
    - ノイズは多いが語彙が非常に多様なデータを用いることができる
    #colbreak()
    #figure(
      image(
        "images/how-mask-former-works.png",
        width: 100%,
        height: 70%,
        fit: "cover",
      ),
      caption: "",
    )
  ]
]

#slide[
  == 提案手法2: Mask prompt tuning
  - マスク画像は背景がなく、切り取られた領域だけであるため、*どのピクセルが背景か*を教えてやると CLIP がうまく動作しやすくなる
  - CLIP の重みを一切変更せずにプロンプト側のテンプレートを調整する
  #v(0.5em)
  #figure(
    image(
      "images/prompt-mask-tuning.png",
      width: 100%,
      height: 70%,
      fit: "cover",
    ),
    caption: "",
  )
]

#slide()[
  == 追試：実験設定

  #set text(size: 12pt)

  共通設定：Train = COCO-Stuff-171　　Test = ADE20K-150（val）　　Batch = 32　　Max iter = 120,000　　WarmupPolyLR（warmup 1,500 iter）　　Grad clip = 0.01

  #v(0.4em)

  #table(
    columns: (1.6fr, 1fr, 1fr),
    stroke: 0.5pt + luma(200),
    inset: (x: 7pt, y: 5pt),
    table.header([*項目*], [*Config A*], [*Config B*]),
    [Backbone], [Swin-B（22K, 384×384）], [ResNet-101 DeepLab],
    [CLIP], [ViT-L/14], [ViT-B/16],
    [Embedding dim], [768], [512],
    [Crop size], [640×640], [512×512],
    [Base LR], [`6e-5`], [`2e-4`],
    [Backbone LR mult], [1.0], [0.1],
    [Weight decay], [0.01], [`1e-4`],
    [Mask THR], [0.4], [0.5],
  )
]

#slide[
  == 追試: 実際のセグメンテーション結果
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #image("images/0.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text(weight: "bold")['Oculus' 'Ululele']
    ],
    [
      #image("images/1.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text(weight: "bold")['VR headset' 'Ululele']
    ],
  )
]

#slide[
  == 追試: 実際のセグメンテーション
  #grid(
    columns: (1fr, 1fr),
    rows: (1fr, 1fr),
    gutter: 1em,
    [
      #figure(
        image("images/dogs-cover.jpg", width: 100%, height: 100%, fit: "cover"),
        caption: "元画像",
      )
    ],
    [
      #figure(
        image("images/2.png", width: 100%, height: 100%, fit: "cover"),
        caption: "'dog'",
      )
    ],

    [
      #figure(
        image("images/3.png", width: 100%, height: 100%, fit: "cover"),
        caption: "'Labrador Retriever' など多くの犬種名を入力",
      )

    ],
    [
      #figure(
        image("images/7.png", width: 100%, height: 100%, fit: "cover"),
        caption: "'dogs'",
      )
    ],
  )
  #place(bottom + center, dy: -1em)[
    'dog', 'dogs'間に結果の違いはみられない。また犬種も精度は高くないように見える
  ]
]

#slide[
  == 追試：実際のセグメンテーション
  #columns(2, gutter: 2em)[
    #figure(
      image("images/hamsters.jpg", width: 100%, height: 70%, fit: "cover"),
      caption: "元画像",
    )
    #colbreak()
    #figure(
      image(
        "images/6.png",
        width: 100%,
        height: 70%,
        fit: "cover",
      ),
      caption: "'djungarian' 'campbell' 'hybrid'",
    )
  ]
  #text("こちらも種類は的外れ。classnameはモデルの制約上あまり長くできなかった")
]

#slide[
  == 追試: 実際のセグメンテーション
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    [
      #image("images/person-holding-black-umbrella.webp", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text("元画像")
    ],
    [
      #image("images/4.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text("'person holding a blue umbrella'")
    ],
    [
      #image("images/5.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text("'person holding a black umbrella'")
    ],
  )
  人にセグメンテーションして欲しかったが傘が対象になっている。また色の違いも認識できていない
]

#slide[
  == まとめ

  #columns(2, gutter: 2em)[
    === 知見
    - Mask-adapted CLIP + Mask Prompt Tuning の追試を実施
    - "dog", "person" など一般的なクラスは概ね検出可能
    - 細かい種別（犬種・ハムスター品種）の識別は不正確
    - 色・属性（青い傘 vs 黒い傘）の違いも捉えきれていない
    - マスク領域のみの分類では文脈・背景情報の欠落が限界に

    #colbreak()

    === 今後
    - 今回見つかった弱点を解決している既存モデルの適用を考える
    - OV-Segの論文と実装の対応づけを完了させる
    - 各モジュールの論文を見る

  ]
]
