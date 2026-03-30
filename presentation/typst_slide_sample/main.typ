#import "@preview/touying:0.5.3": *
#import themes.simple: *

#show: simple-theme.with(
  aspect-ratio: "16-9",
  footer: [Touying 文法リファレンス],
)
#set text(font: "Noto Sans JP")
#show heading.where(level: 1): set text(size: 18pt, weight: "bold")
#show heading.where(level: 2): set text(size: 16pt, weight: "semibold")
#show heading.where(level: 3): set text(size: 14pt, weight: "medium")
#show heading.where(level: 4): set text(size: 12pt, weight: "medium", style: "italic")
#show raw: set text(font: "Geist Mono")

// ============================================================
// 1. タイトルスライド
// ============================================================
#title-slide[
  = Touying 文法リファレンス

  テキスト・画像・コード・レイアウト 完全網羅 #sym.dot 2026年
]

// ============================================================
// 2. テキスト書式
// ============================================================
#slide[
  == テキスト書式

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *太字 (bold)*、_イタリック_、`インラインコード`

      #underline[下線テキスト]、#strike[取り消し線]

      上付き: x#super[2]、下付き: H#sub[2]O

      #text(fill: red)[赤色テキスト]、
      #text(fill: blue, size: 1.3em)[大きい青]

      #highlight[ハイライト]
    ],
    [
      ```typst
      *太字*、_イタリック_、`インラインコード`

      #underline[下線]、#strike[取り消し線]

      x#super[2]、H#sub[2]O

      #text(fill: red)[赤色テキスト]
      #text(fill: blue, size: 1.3em)[大きい]

      #highlight[ハイライト]
      ```
    ]
  )
]

// ============================================================
// 3. 見出し・リスト
// ============================================================
#slide[
  == 見出し・リスト

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *箇条書き（unordered）*
      - 項目 A
        - ネスト A-1
        - ネスト A-2
      - 項目 B

      *番号付き（ordered）*
      + ステップ 1
      + ステップ 2
        + サブステップ
      + ステップ 3

      *用語リスト*
      / Typst: モダンな組版言語
      / Touying: スライド作成パッケージ
    ],
    [
      ```typst
      - 項目 A
        - ネスト A-1
      - 項目 B

      + ステップ 1
      + ステップ 2
        + サブステップ

      / Typst: モダンな組版言語
      / Touying: スライド作成パッケージ
      ```
    ]
  )
]

// ============================================================
// 4. コードブロック
// ============================================================
#slide[
  == コードブロック

  *Python の例（シンタックスハイライト）*

  ```python
  def fibonacci(n: int) -> list[int]:
      """フィボナッチ数列を生成する"""
      result = [0, 1]
      for i in range(2, n):
          result.append(result[-1] + result[-2])
      return result

  print(fibonacci(10))
  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  ```
]

#slide[
  == コードブロック（複数言語）

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *Rust*
      ```rust
      fn main() {
          let msg = "Hello, Typst!";
          println!("{}", msg);
      }
      ```

      *Shell*
      ```bash
      $ typst compile main.typ
      $ typst watch main.typ
      ```
    ],
    [
      *JavaScript*
      ```js
      const greet = (name) => {
        return `Hello, ${name}!`;
      };
      console.log(greet("World"));
      ```

      *SQL*
      ```sql
      SELECT name, age
      FROM users
      WHERE age >= 20
      ORDER BY name;
      ```
    ]
  )
]

// ============================================================
// 5. 画像・Figure
// ============================================================
#slide[
  == 画像の挿入

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *`#image` で直接挿入*

      #image("assets/example.png", width: 80%)

      ```typst
      #image("assets/example.png", width: 80%)
      ```
    ],
    [
      *`#figure` でキャプション付き*

      #figure(
        image("assets/example.png", width: 60%),
        caption: [サンプル画像],
      )

      ```typst
      #figure(
        image("path/to/img.png", width: 60%),
        caption: [キャプション],
      )
      ```
    ]
  )
]

// ============================================================
// 6. グリッド・レイアウト
// ============================================================
#slide[
  == グリッドレイアウト

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
      #rect(fill: red.lighten(60%), width: 100%, inset: 1em, radius: 6pt)[
        *列 1*\
        `1fr` で均等分割
      ]
    ],
    [
      #rect(fill: green.lighten(60%), width: 100%, inset: 1em, radius: 6pt)[
        *列 2*\
        `1fr` で均等分割
      ]
    ],
    [
      #rect(fill: blue.lighten(60%), width: 100%, inset: 1em, radius: 6pt)[
        *列 3*\
        `1fr` で均等分割
      ]
    ],
  )

  #v(0.5em)

  ```typst
  #grid(
    columns: (1fr, 1fr, 1fr),  // 列幅指定（固定値も可: 200pt）
    gutter: 1em,               // 隙間
    [列1の内容], [列2の内容], [列3の内容],
  )
  ```
]

// ============================================================
// 7. ボックス・装飾
// ============================================================
#slide[
  == ボックス・装飾

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #rect(
        fill: yellow.lighten(60%),
        stroke: orange,
        inset: 1em,
        radius: 8pt,
        width: 100%,
      )[
        *警告ボックス*\
        `#rect` で背景色・枠線・角丸を指定
      ]

      #block(
        fill: luma(230),
        inset: 0.8em,
        radius: 4pt,
        width: 100%,
      )[
        `#block` はインライン化しない矩形領域
      ]
    ],
    [
      ```typst
      #rect(
        fill: yellow.lighten(60%),
        stroke: orange,
        inset: 1em,
        radius: 8pt,
        width: 100%,
      )[内容]

      #block(
        fill: luma(230),
        inset: 0.8em,
        radius: 4pt,
      )[内容]
      ```
    ]
  )
]

// ============================================================
// 8. 数式
// ============================================================
#slide[
  == 数式

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *インライン数式*: $E = m c^2$、$a^2 + b^2 = c^2$

      *ブロック数式*

      $ sum_(k=1)^n k = (n(n+1))/2 $

      $ integral_(-oo)^(oo) e^(-x^2) d x = sqrt(pi) $

      $ mat(a, b; c, d) vec(x, y) = vec(a x + b y, c x + d y) $
    ],
    [
      ```typst
      // インライン
      $E = m c^2$

      // ブロック
      $ sum_(k=1)^n k = (n(n+1))/2 $

      $ integral_(-oo)^(oo)
        e^(-x^2) d x = sqrt(pi) $

      $ mat(a, b; c, d) vec(x, y)
        = vec(a x + b y, c x + d y) $
      ```
    ]
  )
]

// ============================================================
// 9. テーブル
// ============================================================
#slide[
  == テーブル

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      #table(
        columns: (auto, 1fr, auto),
        inset: 0.6em,
        align: (left, left, right),
        table.header(
          [*言語*], [*用途*], [*人気度*],
        ),
        [Python], [AI / データ分析], [★★★★★],
        [Rust],   [システム開発],     [★★★★☆],
        [Typst],  [組版・ドキュメント], [★★★☆☆],
      )
    ],
    [
      ```typst
      #table(
        columns: (auto, 1fr, auto),
        inset: 0.6em,
        align: (left, left, right),
        table.header(
          [*言語*], [*用途*], [*人気*],
        ),
        [Python], [AI], [★★★★★],
        [Rust], [システム], [★★★★☆],
      )
      ```
    ]
  )
]

// ============================================================
// 10. アニメーション（pause / only / uncover）
// ============================================================
#slide[
  == アニメーション — `#pause`

  ステップごとに内容を表示できます：

  + まず最初のポイントが表示される #pause

  + 次にこのポイントが現れる #pause

  + 最後にこれが表示される

  #pause

  #rect(fill: teal.lighten(70%), inset: 0.8em, radius: 6pt, width: 100%)[
    `#pause` を使うと、その位置でスライドが分割されます。
  ]
]

#slide[
  == アニメーション — `#only` / `#uncover`

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      - `#pause` — 以降を次のステップへ
      - `#only(n)[...]` — n番目のステップのみ表示
      - `#uncover(n)[...]` — n番目で可視化（スペースは保持）

      ```typst
      #only(1)[ステップ1だけ表示]
      #only(2)[ステップ2だけ表示]

      // 範囲指定
      #uncover("2-")[2ステップ目以降]
      #only("1-2")[1〜2ステップ目]
      ```
    ],
    [
      #only(1)[
        #rect(fill: blue.lighten(70%), inset: 0.8em, radius: 6pt, width: 100%)[
          ステップ 1: 最初の状態
        ]
      ]
      #only(2)[
        #rect(fill: green.lighten(70%), inset: 0.8em, radius: 6pt, width: 100%)[
          ステップ 2: 変化した状態
        ]
      ]
    ]
  )
]

// ============================================================
// 11. リンク・引用
// ============================================================
#slide[
  == リンク・引用

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      *リンク*

      #link("https://typst.app")[Typst 公式サイト]

      URL そのまま: #link("https://github.com/touying-typ/touying")

      *ブロック引用*

      #quote(block: true, attribution: [Donald Knuth])[
        Beware of bugs in the above code; \
        I have only proved it correct, not tried it.
      ]
    ],
    [
      ```typst
      #link("https://typst.app")[表示テキスト]

      // URLをそのまま表示しつつリンクにする
      #link("https://example.com")

      #quote(
        block: true,
        attribution: [著者名],
      )[
        引用文をここに書く
      ]
      ```
    ]
  )
]

// ============================================================
// 12. 変数・関数・条件・ループ
// ============================================================
#slide[
  == Typst スクリプト（変数・関数・制御構造）

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #let title = "Typst"
      #let version = 0.12

      変数: *#title* v#version

      #let greet(name) = [こんにちは、*#name* さん！]
      #greet("Touying")

      #let score = 85
      #if score >= 90 [
        優秀
      ] else if score >= 70 [
        良好
      ] else [
        要努力
      ]

      #for i in range(1, 4) [アイテム #i #h(0.5em)]
    ],
    [
      ```typst
      #let title = "Typst"
      #let version = 0.12
      変数: *#title* v#version

      #let greet(name) = [
        こんにちは、*#name* さん！
      ]
      #greet("Touying")

      #let score = 85
      #if score >= 90 [優秀]
      else if score >= 70 [良好]
      else [要努力]

      #for i in range(1, 4) [
        アイテム #i
      ]
      ```
    ]
  )
]

// ============================================================
// 13. show ルール・set ルール
// ============================================================
#slide[
  == `#show` / `#set` ルール

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      `#set` でデフォルト設定を変更:

      ```typst
      // フォント設定
      #set text(font: "Noto Sans JP", size: 12pt)

      // 段落設定
      #set par(justify: true, leading: 0.8em)

      // 見出し番号なし
      #set heading(numbering: none)
      ```

      `#show` で要素を変換:

      ```typst
      // raw テキストのフォント変更
      #show raw: set text(font: "Geist Mono")

      // 強調を赤色に
      #show strong: set text(fill: red)
      ```
    ],
    [
      #rect(fill: luma(240), inset: 0.8em, radius: 6pt, width: 100%)[
        *使いどころ*

        - `#set` → 以降の要素に継続適用
        - `#show` → 特定要素を丸ごと変換
        - スライドの冒頭に書くことでグローバル設定に
      ]
    ]
  )
]

// ============================================================
// 14. 整列・スペーシング
// ============================================================
#slide[
  == 整列・スペーシング

  #grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    [
      #align(left)[← 左寄せ]
      #align(center)[中央寄せ]
      #align(right)[右寄せ →]

      #v(1em)

      #align(center + horizon)[
        #rect(fill: purple.lighten(70%), inset: 1em, radius: 8pt)[
          `center + horizon` で\
          縦横中央に配置
        ]
      ]
    ],
    [
      ```typst
      #align(left)[左寄せ]
      #align(center)[中央寄せ]
      #align(right)[右寄せ]

      #v(1em)   // 縦方向スペース
      #h(1em)   // 横方向スペース

      #align(center + horizon)[
        縦横中央
      ]
      ```
    ]
  )
]

// ============================================================
// 15. まとめ
// ============================================================
#slide[
  == 文法チートシート まとめ

  #table(
    columns: (auto, 1fr),
    inset: 0.55em,
    table.header([*カテゴリ*], [*主な構文*]),
    [テキスト],       [`*太字*` `_斜体_` `` `code` `` `#text(fill:)` `#underline[]` `#highlight[]`],
    [リスト],         [`-` 箇条書き、`+` 番号付き、`/ 用語:` 定義リスト],
    [コード],         [`` ```lang ... ``` `` ブロック、`` `...` `` インライン],
    [画像],           [`#image("path", width:)`, `#figure(image(...), caption:[])`],
    [グリッド],       [`#grid(columns:, gutter:, [...], [...])`],
    [ボックス],       [`#rect(fill:, stroke:, inset:, radius:)`, `#block(...)`],
    [数式],           [`$...$` インライン、`$ ... $` ブロック],
    [テーブル],       [`#table(columns:, table.header(...), [セル], ...)`],
    [アニメーション], [`#pause`, `#only(n)[...]`, `#uncover("n-")[...]`],
    [リンク・引用],   [`#link("url")[text]`, `#quote(block: true, attribution:)[...]`],
    [スクリプト],     [`#let`, `#if`, `#for`, `#show`, `#set`, 関数定義],
    [整列],           [`#align(center + horizon)[...]`, `#v()`, `#h()`],
  )
]
