## Generate Training Data

Run `pretokenize.py` to prepare the tokenized data.

## Generate Reading Time Data

We use Dundee, Provo, MECO, and Natural Stories. At the time of writing, each corpus is available under the following links (please cite each paper/repo if you use them):

* [Dundee](https://drive.google.com/file/d/1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8)
    * `dundee/`
* [Provo](https://osf.io/sjefs/wiki/home/)
    * `Provo_Corpus-Eyetracking_Data.csv`
    * `Provo_Corpus-Predictability_Norms.csv`
* [MECO](https://osf.io/3527a/files/)
    * `joint_data_trimmed.csv`
* [Natural Stories](https://github.com/languageMIT/naturalstories/tree/master/naturalstories_RTS)
    * `all_stories.tok`
    * `processed_RTs.tsv`

Place these files under `data/rt_data/` and run `preprocess_data.py` to merge them into a single file.
