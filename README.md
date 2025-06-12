# fambigen - the Ambigram Generator

![Python](https://img.shields.io/badge/python-3.x-blue.svg)

A command-line tool written in Python to procedurally generate ambigrams from character pairs sourced from any TTF or WOFF font, and compose them into words or phrases. An ambigram is a word or design that can be read from more than one viewpoint, such as when flipped upside-down (180-degree rotational symmetry).

This script takes one or two input words (or phrases), generates the necessary rotationally symmetric glyphs for each character pair, and composes them into a single PNG image.

The project was spawned after I found some late 90's code of mine that created ambigrams using a meticulously hand-drawn fit-to-purpose ambigram font. I've long lost the code and Wayback Machine only archived pieces of it (and it was in perl, with pre-rendered hand-made GIFs) but I wanted that functionality again so I started fambigen in python a few days ago and very quickly decided I did not feel like re-imagining and drawning 676 glyphs to use. I also did not want to use any of the handful of existing ambigram fonts I don't know the origin of but it seems lots of gimmick/meme sites all use the same few fonts for this purpose.

I also wanted to see if I could experiment with ways to programmatically take standard fonts and make them work as ambigrams without all of the weirdness and shoehorning that comes with manually creating such a font. It's still under active development and I'm still experimenting with trying to create a set of rules (and the tech to implement them) to make any font suitable for ambigram use. It was shocking to me to only find two academic papers on the topic of using computers to generate ambigrams; one of them was a manual process using a vector image editing program and the other is a more recent paper on training a Diffusion model to generate ambigrams. Really? Nothing from the folks of SIGGRAPH on the topic, ever? *shrug*

~~Maybe I'll write a paper explaining my method. :-)~~

[I've writtten a paper about this method](https://paperclipmaximizer.ai/fambigen.pdf)

## Examples

`deadbeef`

![deadbeef](https://killsignal.net/deadbeef/deadbeef-deadbeef_Inkfree.ttf_ambigram.png)

`ycombinator-hackernews-ariblk`

![ycombinator-hackernews-ariblk.png](https://github.com/scottvr/fambigen/blob/91c22b352f2aad22b219de4b385bf38ed46bee0f/assets/ycombinator-hackernews!_ariblk.ttf_ambigram.png)

[kitten bundle](https://github.com/scottvr/fambigen/blob/42c489644b62dacf00f2eda971fb4dbf0079153a/assets/kitten-bundle_ambigram.png)


## Features

* **Multiple Generation Strategies**: Implements four distinct strategies for creating ambigram glyphs:
    * `outline`: A purely vector-based method that creates an outline/inline effect.
    * `centerline_trace`: A complex method that generates a centerline skeleton for each character, aligns them, and applies a calligraphic stroke.
* **Multiple Alignment Strategies**: 
    * `centroid`: A simple union based on aligning the geometric centers.
    * `iterative_registration`: Shape overlap maximization.
[    * `principal_axis`: A more stable alignment based on the principal axis of the character shapes.]: #
* **Flexible Ambigram Creation**:
    * Create palindromic ambigrams from a single word (e.g., "level").
    * Create a simple ambigram of a non-palidromic single word that will read the same "upside-down".
    * Create symbiotic ambigrams from two different words of the same length (e.g., "kitten" / "bundle").
* **Command-Line Interface**: All options, including input words, font file, and strategy, can be controlled via command-line arguments for easy scripting and automation.
* **Custom Fonts**: Supports any TrueType Font (`.ttf`), with Arial as a convenient default.
* **Mixed-Case Support**: Correctly generates glyphs from pairs of uppercase and lowercase characters (e.g., 'M' and 'e').
* **Dual Output**: Generates both the individual SVG vector glyphs for each pair and a final composed PNG raster image for the full word.

---

New alignment strategy examples:

`centroid`

![centroid example](https://killsignal.net/deadbeef/GOD-GOD_arial.ttf-centroid_ambigram.png)

`iterative_registration (shape overlap)`

![iterative registration](https://killsignal.net/deadbeef/GOD-GOD_arial.ttf-iterative_registration_ambigram.png)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/scottvr/fambigen
    cd fambigen
    ```
2.  Install the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The script is controlled via command-line arguments.

```bash
python fambigen.py word1 [word2] [--font FONT_PATH] [--strategy STRATEGY_NAME]
```

### Arguments

* `word1`: (Required) The first word, which is read from left to right.
* `word2`: (Optional) The second word, which is read when the image is rotated 180 degrees. If omitted, the script will create a palindromic ambigram of `word1`.
* `-f, --font`: (Optional) The full path to the `.ttf` font file you wish to use. Defaults to Arial.
* `-f2, --font2`: (Optional) The full path to a second `.ttf` font file you wish to use. Defaults to `font`. (You can specify just `font2` if you like and `font` will remain its default of arial unless otherwise specified.)
* `-s, --strategy`: (Optional) The generation strategy to use. Choices are `centroid`, `principal_axis`, `outline`, and `centerline_trace`. Defaults to `outline`.
* `--alignment`: (Optional) The glyph alignment strategy to use. Choices are `centroid` and `iterative_registration (shape_overlap maximization). (`-a c`, or `-a i` for short)
* `--uniform-glyphs`: (Optional) Scale fonts to the same size before merging them.
* `--noambi` : (Optional) Disables ambigrammitization, allowing for font compositing only.

### Examples

**1. Create a simple ambigram of the word "awesome":**
```bash
python fambigen.py awesome
```
* This will generate a `awesome_ambigram.png` file.

**2. Create an ambigram from two different words:**
```bash
python fambigen.py Mary LOVE
```
* This will generate the required mixed-case glyphs (`ML.svg`, `ao.svg`, `rv.svg`, `yE.svg`) and compose them into a `Mary-LOVE_ambigram.png` file.

**3. Use a different font and strategy:**
```bash
python fambigen.py ambigram --font "/path/to/your/font/coolfont.ttf" --strategy centerline_trace
```

**4. Get help and see all options:**
```bash
python fambigen.py --help
```

## How It Works

The script operates in two main stages:

1.  **Stage 1: Glyph Generation**
    * Based on the input word(s), it first determines the unique character pairs required (e.g., for "Mary" and "LOVE", the pairs are `ML`, `ao`, `rv`, `yE`).
    * For each pair, it calls the `generate_ambigram_svg` function using the chosen strategy (`outline` by default).
    * This generates and saves an individual SVG file for each pair into a strategy-specific directory (e.g., `generated_glyphs_outline/ML.svg`).

2.  **Stage 2: Image Composition**
    * After the required SVGs are generated, the `create_ambigram_from_string` function is called.
    * It reads the necessary SVG files in the correct order.
    * It uses the `cairosvg` library to render each SVG into an in-memory PNG image.
    * Finally, it uses the `Pillow` library to stitch these individual glyph images together side-by-side into a single composite PNG file.

## Contributing

It is a fair way from perfect so you issues, fixes, enhancements, etc are welcome! If you'd like to add a new strategy, improve an existing one, or add features, please feel free to:

1.  Fork the repository.
2.  Create a new branch for your feature.
3.  Commit your changes.
4.  Push to the branch 
5.  Open a Pull Request.

## License

This project is licensed under the MIT Licenset push
