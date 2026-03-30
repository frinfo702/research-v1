#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

#let talk-title = "Paper Title"
#let presenter = "Presenter Name"
#let venue = "Lab Seminar"
#let talk-date = datetime.today().display("[year]-[month]-[day]")
#let one-line-takeaway = "One sentence that states the core idea."

#title-slide[
  = #talk-title
  #v(0.8em)
  #presenter \
  #venue, #talk-date
]

== Takeaway

#align(center + horizon)[
  emph[#one-line-takeaway]
]

= Problem

== Problem

- Why this problem matters
- What fails in prior work
- What the audience should keep in mind

== Gap

- Limitation 1
- Limitation 2

#speaker-note[
  State the precise failure mode that motivates the method.
]

= Method

== Key Idea

- Core idea in one sentence
- Intuition before formalism

#figure(
  image("fig/key_idea.png", width: 65%),
  caption: [Key idea: replace this with the actual takeaway from the figure.]
)

== Method Details

- Assumptions
- Inputs and outputs
- Why this design works

$ L = sum_i ell(f(x_i), y_i) + lambda R(theta) $

#speaker-note[
  Define symbols only if the audience needs them to follow the next result slide.
]

= Evidence

== Experimental Setup

- Dataset
- Metric
- Baselines

== Main Results

- Main claim
- Quantitative improvement

#figure(
  image("fig/results.png", width: 72%),
  caption: [Main result: summarize what the comparison proves.]
)

== Limitations

- Failure case 1
- Failure case 2
- Scope where the method should not be used

== Takeaway

#align(center + horizon)[
  emph[#one-line-takeaway]

  Questions?
]
