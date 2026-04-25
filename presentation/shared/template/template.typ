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

// 指定ワード数のダミーテキストを安全に生成
#let dummy-text(words: 20) = {
  let base = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
  let arr = base.split(" ")
  let end = calc.min(words, arr.len())
  arr.slice(0, end).join(" ") + "."
}

// ============================================
// 1. タイトルスライド
// ============================================
#slide[
  #set align(center + horizon)
  #text(size: 36pt, weight: "bold")[Presentation Title]
  #v(1em)
  #text(size: 18pt, fill: luma(100))[Subtitle / Description]
  #v(2em)
  #text(size: 14pt)[Author Name • Date • Event]
]

// ============================================
// 2. セクション区切り（章立て）
// ============================================
#slide[
  #set align(center + horizon)
  #text(size: 32pt, weight: "bold", fill: rgb("#0F4C81"))[Section Title]
  #v(0.8em)
  #text(size: 16pt, fill: luma(100))[Section subtitle or brief overview]
]

// ============================================
// 3. 左右2カラム：テキスト左 + 画像右
// ============================================
#slide[
  == Layout: Text Left | Image Right
  #columns(2, gutter: 2em)[
    #dummy-text(words: 25)
    #v(1em)
    - Bullet point A
    - Bullet point B
    - Bullet point C
    #colbreak()
    #figure(
      image("images/example.png", width: 100%, height: 70%, fit: "cover"),
      caption: "caption",
    )
  ]
]

// ============================================
// 4. 左右2カラム：画像左 + テキスト右
// ============================================
#slide[
  == Layout: Image Left | Text Right
  #columns(2, gutter: 2em)[
    #figure(
      image("images/example.png", width: 100%, height: 65%, fit: "cover"),
      caption: "Caption",
    )
    #colbreak()
    #text(weight: "bold")[Key Points]
    #v(0.5em)
    #dummy-text(words: 15)
    #v(0.8em)
    1. First item
    2. Second item
    3. Third item
  ]
]

// ============================================
// 5. 上下分割：画像上 + テキスト下
// ============================================
#slide[
  == Layout: Image Top | Text Bottom
  #figure(
    align(center, image("images/example.png", width: 85%, height: 45%, fit: "cover")),
    caption: "Overview Diagram",
  )
  #v(0.8em)
  #v(1em)
  #columns(2, gutter: 2em)[
    #dummy-text(words: 12)
    #colbreak()
    #dummy-text(words: 12)
  ]
]

// ============================================
// 6. 3カラムレイアウト
// ============================================
#slide[
  == Layout: Three Columns
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1.5em,
    [
      #image("images/example.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text(weight: "bold")[Column A]
      #dummy-text(words: 10)
    ],
    [
      #image("images/example.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text(weight: "bold")[Column B]
      #dummy-text(words: 10)
    ],
    [
      #image("images/example.png", width: 100%, height: 40%, fit: "cover")
      #v(0.5em)
      #text(weight: "bold")[Column C]
      #dummy-text(words: 10)
    ],
  )
]

// ============================================
// 7. フルスクリーン画像 + テキストオーバーレイ
// ============================================
#slide[
  #box(width: 100%, height: 100%, inset: 0pt, align(center + horizon)[
    #image("images/example.png", width: 100%, height: 100%, fit: "cover")
  ])
  #place(bottom + left, dx: 2em, dy: -2em)[
    #box(fill: rgb("#000000AA"), inset: 1em, radius: 4pt)[
      #text(size: 20pt, weight: "bold", fill: white)[Overlay Title]
      #v(0.3em)
      #text(size: 12pt, fill: luma(220))[Description text over full-bleed image]
    ]
  ]
]

// ============================================
// 8. 箇条書きメイン（テキスト中心）
// ============================================
#slide[
  == Main Points
  #v(1em)
  #columns(2, gutter: 3em)[
    #text(size: 14pt)[
      + First major point with detailed explanation
      + Second point covering methodology
      + Third point about results
    ]
    #colbreak()
    #text(size: 14pt)[
      + Fourth point regarding future work
      + Fifth point for discussion
      + Summary takeaway
    ]
  ]
]

// ============================================
// 9. コード + 説明（2カラム）
// ============================================
#slide[
  == Code & Explanation
  #columns(2, gutter: 2em)[
    #figure(
      ```python
      def hello():
          print("Typst + Touying")
          return True
      ```,
      caption: "exaple code",
    )
    #colbreak()
    #text(weight: "bold")[How it works]
    #v(0.5em)
    #dummy-text(words: 18)
    #v(0.8em)
    - Step 1: Initialize
    - Step 2: Process
    - Step 3: Output
  ]
]

// ============================================
// 10. 画像グリッド（2×2）
// ============================================
#slide[
  == Image Grid (2×2)
  #grid(
    columns: (1fr, 1fr),
    rows: (1fr, 1fr),
    gutter: 1em,
    image("images/example.png", width: 100%, height: 100%, fit: "cover"),
    image("images/example.png", width: 100%, height: 100%, fit: "cover"),

    image("images/example.png", width: 100%, height: 100%, fit: "cover"),
    image("images/example.png", width: 100%, height: 100%, fit: "cover"),
  )
  #place(bottom + center, dy: -1em)[
    #text(size: 10pt, fill: luma(100))[Figure 4: Comparison Results]
  ]
]

// ============================================
// 11. 引用・強調スライド
// ============================================
#slide[
  #set align(center + horizon)
  #text(size: 28pt, weight: "bold", style: "italic")[
    "This is a powerful quote or core message to emphasize."
  ]
  #v(1.5em)
  #text(size: 14pt, fill: luma(100))[— Author, Source, Year]
]
