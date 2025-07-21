## Generate Training Data

Run `pretokenize.py` to prepare the tokenized data.

## Generate Reading Time Data

We use Dundee, Provo, MECO, and Natural Stories. At the time of writing, each corpus is available under the following links (please cite each paper/repo if you use them):

* [Dundee](https://drive.google.com/file/d/1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8)
* [Provo](https://osf.io/sjefs/wiki/home/)
* [MECO](https://osf.io/3527a/files/)
* [Natural Stories](https://github.com/languageMIT/naturalstories/tree/master/naturalstories_RTS)

Run `preprocess_data.py` to merge them into a single file.
