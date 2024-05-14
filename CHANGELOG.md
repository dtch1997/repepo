# CHANGELOG



## v0.8.0 (2024-05-07)

### Feature

* feat: more datasets (#164)

* add all xrisk datasets

* make truthfulqa consistent with others

* fix some prompts

* Verify all existing persona prompts

* add persona prompts for Xrisk, sycophancy, tqa

* refactor: make more datasets

* minor

* fix lint

* fix test persona prompt len

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`0f6dff2`](https://github.com/dtch1997/repepo/commit/0f6dff290e3c79ad6fe8dc353b49497e6b89624c))


## v0.7.0 (2024-05-05)

### Feature

* feat: pipeline generation ([`9c69185`](https://github.com/dtch1997/repepo/commit/9c6918531aae505dcbce4fa8ae2c2784064e8486))

### Unknown

* lint: run linters ([`88cf51a`](https://github.com/dtch1997/repepo/commit/88cf51a0bfee85bdc25b5d8d165737d4a8274b22))


## v0.6.0 (2024-04-30)

### Chore

* chore: add note to explain cluster utils ([`a13a503`](https://github.com/dtch1997/repepo/commit/a13a503ae1be728005a564b50949b5cec4aa550e))

* chore: add cluster utils ([`dcd731a`](https://github.com/dtch1997/repepo/commit/dcd731a079ea48d587f5db8736d581f11caeb8fd))

* chore: modify path to include local dependencies ([`4a97734`](https://github.com/dtch1997/repepo/commit/4a97734f8f6b87c3a44d4457c0581e5915a3cc3b))

* chore: update time requirement in qsub ([`d1268c8`](https://github.com/dtch1997/repepo/commit/d1268c8dff7932f2b54c7f6b858663a6bf4f057e))

* chore: delete incorrect test step ([`1f4c2bb`](https://github.com/dtch1997/repepo/commit/1f4c2bbb9aadcd84847bda1171d0d7d90e5e47ba))

* chore: update lockfile ([`40baf38`](https://github.com/dtch1997/repepo/commit/40baf38e5612de4f606a6ffb1a58378f3cab9797))

* chore: update ci (#161)

* update ci

* chore: fix typo

* chore: add py312 test

* update ruff settings

* chore: ruff format

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`4482d27`](https://github.com/dtch1997/repepo/commit/4482d273234612e5fe6ada43df9485d5dd96c4f8))

### Feature

* feat: cross steering results db (#162)

* modify jobscript  to submit array job

* chore: fix typo

* Add sqlite database for results

* implement multiprocessing to make db

* update db

* add notebook for in-distribution-steerability

* add analysis notebook for intra concept variability

* fix ruff format

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`cd16daa`](https://github.com/dtch1997/repepo/commit/cd16daa573fb2fb05a1088ca779a46ea3a388b69))

### Unknown

* categorize persona prompts ([`6dd631f`](https://github.com/dtch1997/repepo/commit/6dd631f582d962e38907fcedfed66c73af879e06))

* adding persona analysis notebook ([`38d3ed4`](https://github.com/dtch1997/repepo/commit/38d3ed4e9971ba2af96bfbbb72ac804075295296))


## v0.5.0 (2024-04-24)

### Chore

* chore: replace hardcoded user id with daniel ([`b1271a6`](https://github.com/dtch1997/repepo/commit/b1271a6a4b98069a35ffe65f62ce1394670296fa))

* chore: fix qsub script ([`5f50c60`](https://github.com/dtch1997/repepo/commit/5f50c60050b55ac5b96508c55ac1e14ed0ee756c))

* chore: cleanup dead code ([`9e615ff`](https://github.com/dtch1997/repepo/commit/9e615ff75f26d6692cf6966371cc00a3b98228f6))

### Feature

* feat: slim eval results and allowing multipler multipliers in cross-steering (#159)

* adding slim eval results and allowing multipler multipliers in cross-steering

* fixing linting

* fixing typing ([`64d94aa`](https://github.com/dtch1997/repepo/commit/64d94aaf86e0782513c9f80888999bddc3e348f8))

* feat: add more persona prompts (#160)

* add more persona prompts

* add persona prompts, tests

* Modify persona_generalization to work with more prompts

* Add script to run persona generalization

* fix: translation constants

* Add test for variables

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`1cdf847`](https://github.com/dtch1997/repepo/commit/1cdf847c4b78e2a7cab7099594bee6a3c567c4da))

* feat: steerability metrics ([`f9e22cc`](https://github.com/dtch1997/repepo/commit/f9e22ccb0252963041854a8e26fe5295dd2f1b6c))

### Fix

* fix: replace global var ([`d91a75f`](https://github.com/dtch1997/repepo/commit/d91a75f691144d940376812fc19a11a5b7476434))

### Unknown

* update test fixture ([`53d06ab`](https://github.com/dtch1997/repepo/commit/53d06ab494cf42a0e159ca5c96360458bb91cbba))

* text: add dataset fixture ([`1b10ef4`](https://github.com/dtch1997/repepo/commit/1b10ef4d0b21d1e9fd040e74fe6d746e9477d341))

* set layer to 13 ([`f75473d`](https://github.com/dtch1997/repepo/commit/f75473d1f6db0a9c246040fafdc3320ac29cacfe))

* fixing linting ([`cdb5494`](https://github.com/dtch1997/repepo/commit/cdb54943e5c94a2a78b493cb8153839da4422179))

* allow customizing metric name in persona plots ([`d379a6d`](https://github.com/dtch1997/repepo/commit/d379a6de2cbee15471e08e5ee02c84e349d72c47))

* save persona generalization results individually ([`de8bbd0`](https://github.com/dtch1997/repepo/commit/de8bbd0571b2cf6b05a8f3f1c816ed33ca47b71b))

* Faster eval (#153)

* hopefully improving eval speed

* more device issues

* reformatting ([`03a6dfc`](https://github.com/dtch1997/repepo/commit/03a6dfcc9b306c128c97ef0631acd897d55993fe))

* tweaking cross steering ([`2babe64`](https://github.com/dtch1997/repepo/commit/2babe6449c3f9ceeba4295256640c2870f0e9f7e))


## v0.4.0 (2024-04-12)

### Chore

* chore: modify coderabbit config to reduce verbosity ([`f13fc43`](https://github.com/dtch1997/repepo/commit/f13fc43102aaef88aa7b15640c97e897f0c4fe45))

* chore: ignore png files ([`5c9ddc8`](https://github.com/dtch1997/repepo/commit/5c9ddc8bb414b9081d905251a41501e9c68706c6))

### Feature

* feat: improve steering experiments utils (#147)

* Add statsmodels

* Add notebook ow/ results on choosing steerability metric

* feat: add saving, loading for SVs

* Finish initial study on aggregation method

* rename

* fix: use train_completion_template

* Update lockfile

* Remove system prompt for config for backwards-compatibility

* feat: improve logging of missing steering configs

* Update notebooks

* chore: remove failing py311 ci run

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`f1487a9`](https://github.com/dtch1997/repepo/commit/f1487a9546560ebff60c8104e8884ce24ce828e4))

* feat: add functions to compute logit statistics (#145)

* Add functions to compute logit statistics

* Make logit statistics optional

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`c447864`](https://github.com/dtch1997/repepo/commit/c447864dd4e6a484e0ac0f3a69c2e1e0a3b97994))

### Refactor

* refactor: experiments (#141)

* Add concept metrics calculation

* fix: concept metrics

* Add unit test for metrics

* feat: layer-wise steering metrics

* update config fields

* update experiments code

* minor

* refactor: experiments code

* refactor: experiments code

* Test datasets exist before running

* fix: database

* add method to get config, fix delete_table

* changes

* more changes

* Fix bug in experiment path

* Add sweeps

* WIP

* Fix tests

* Fix tests

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`ee5f190`](https://github.com/dtch1997/repepo/commit/ee5f1902b1e929c3f60cf58ce569b271a6502c73))

### Unknown

* updating persona generalization (#151)

* updating persona generalization

* temporarily disabling test due to cpu/cuda issue on ci ([`ef91f2f`](https://github.com/dtch1997/repepo/commit/ef91f2f38c7d7c686317e383f5d749934ea388dd))

* Add evaluate_generalization.py notebook ([`4a25847`](https://github.com/dtch1997/repepo/commit/4a258472e295ef8aa3ce90d925fc1ac380c5fc58))

* minor fixes ([`506ff11`](https://github.com/dtch1997/repepo/commit/506ff11fa416ee79010091405a7d8ab208a67733))

* WIP: Persona cross steering (#150)

* setting up cross-evaluation experiments

* improving progress reporting in experiments

* adding option to normalize steering magnitude to baseline

* tweaking params

* fixing nested progress

* updating persona evals

* passing eval params through persona experiment

* setting up script for persona generalization experiments

* more debugging output

* updating test

* fixing typing

* make datasets as part of experiments script

* fixing eval dataset selection

* fixing eval

* adding cross steering plots

* shorten labels in cross-steering plots

* WIP adding plotting helpers

* refactoring plotting code

* adding more plotting options

* adding more content to plots

* outptting more info in graphs ([`352df94`](https://github.com/dtch1997/repepo/commit/352df94b656a5e75c3e1cd7c272233bac55507e6))

* Add sft training examples ([`bcf8c2c`](https://github.com/dtch1997/repepo/commit/bcf8c2c36b4635d3e77e6b3b20d26cfb8a679a08))

* Experiments (#146)

* Add sweeps

* WIP experimental code

* Update experiment notebook

* Remove pycache

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`037bea3`](https://github.com/dtch1997/repepo/commit/037bea3b52ed3a7e7bf6e52198ecc26ff8f3553b))

* Add notebooks to run experiments ([`9c662c7`](https://github.com/dtch1997/repepo/commit/9c662c756cf8354e6cef528089a30eb2bca9691a))

* Add fucntion to load sweep results ([`66cb2d7`](https://github.com/dtch1997/repepo/commit/66cb2d7e49ef69d3c9f7429e092aabb6758bacb0))


## v0.3.0 (2024-03-07)

### Feature

* feat: steering experiments (#132)

* Add experimental code

* fix: make_country_capital script

* feat: add code to run steering experiment

* update experiments code

* fix: add --config_path arg

* fix: config yaml parsing

* chore: add more configs

* chore: add even more configs

* refactor: plotting

* feat: add script to run sweep

* fix: do not set completion template by default

* refactor sweeps

* refactor: token concept sweep

* fix: bugbears

* chore: add comments

* fix: steering_index of datasets

* test: steering token index

* updating steering_vectors library version

* evaluate on more layers

* refactor: use steering-vectors code, log instead of print

* chore: fix docstring

* test: training, evaluating steering vectors

* fix: minor

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`8d1bd7d`](https://github.com/dtch1997/repepo/commit/8d1bd7da1160df267f0425fc4e0d7ec039a7c2be))


## v0.2.0 (2024-03-06)

### Feature

* feat: add &#39;prompt&#39; string for bats dataset. ([`83688c1`](https://github.com/dtch1997/repepo/commit/83688c11c59deee6ebbda6b5d7e7ef22cbb555ae))

### Fix

* fix: make_bats function ([`d789eb7`](https://github.com/dtch1997/repepo/commit/d789eb722089110b5ec784cda51324c1102e594f))

### Unknown

* Add BATS dataset (#125)

* Add BATS dataset

* test: add test for make_bats

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`dcb7ccd`](https://github.com/dtch1997/repepo/commit/dcb7ccde8c02b8800ff38fef88a64773df93b6ed))


## v0.1.0 (2024-03-05)

### Chore

* chore: remove pypi publishing from ci ([`6ddd28d`](https://github.com/dtch1997/repepo/commit/6ddd28d151f3a8b7e6edc36d58928de511133755))

### Feature

* feat: refactor types (#123)

* Refactor types

* Delete old experimental code

* Refactor datasets

* Improve dataset split parsing; update make_dataset types

* Update preprocessed datasets save dir

* refactor: delete duplicated files

* Fix pyright errors

* Fix tests

* Fix lint

* chore: re-add tqa raw datasets

* feat: add tests for evaluators

* feat: re-add caa-style prompt

* chore: delete unused code

* refactor: migrate SteeringHook to repepo.core

* test: assert pos, neg prompts the same

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`e600270`](https://github.com/dtch1997/repepo/commit/e600270fc0a032975e16061ef8abb21cf2bb1820))


## v0.0.0 (2024-03-04)

### Chore

* chore: add semantic release to ci ([`56be43f`](https://github.com/dtch1997/repepo/commit/56be43f378c039f61128bc7dac3a17d48a59c707))

### Unknown

* Delete dead code ([`a1da6de`](https://github.com/dtch1997/repepo/commit/a1da6de2425ae93eca17dbd06b8efc04f25be42a))

* translate the main persona datasets from @dtch1997&#39;s work (#108) ([`b0b13a2`](https://github.com/dtch1997/repepo/commit/b0b13a2b7e0feae79f4f1a1bf616b54b97b16b17))

* Translation experiments (#99)

* adding helpers for generating and parsing language from dataset filename

* adding compare_dataset_translations experiment

* adding experiment helpers

* tweaking pirate translation strings

* adding translations for non-persona mwes

* fixing up make mwe helper

* adding a &#39;ctx&#39; pseudo-style

* Revert &#34;adding a &#39;ctx&#39; pseudo-style&#34;

This reverts commit a0058c43cd30ef8558dacc9d2c111f790b984504.

* refactoring to allow using arbitrary dataset variations, insead of the hacky pseudo-language stuff

* fixing using existing results in cross-steering

* adding helpers to calculate jenson-shannon and KL for bernoulli distributions

* using js dist for steering deltas

* adding more tests ([`46927b0`](https://github.com/dtch1997/repepo/commit/46927b0ca2c75bc57c14c868f9bfb5e872399a36))

* adding translated persona MWE variants (#103)

* adding translated persona MWE variants by pre-pending the generation ctx to each example

* formatting translated_strings ([`1a42e96`](https://github.com/dtch1997/repepo/commit/1a42e965e239007a981b57f2e42e3b12c33925bb))

* adding google translate and re-translating persona datasets (#102)

* adding google translate and re-translating persona datasets

* fixing linting

* removing unused test ([`312b4ab`](https://github.com/dtch1997/repepo/commit/312b4ab2cae6c394f1d6d4bf8d24be395ecfed05))

* standardizing dataset naming around language (#100) ([`8fda2f9`](https://github.com/dtch1997/repepo/commit/8fda2f96ca066b2fcaf6bbdd4196aff1c857743c))

* Generalization experiments (#96)

* Add functions to do translation

* Add TQA translate

* Fix key name bug

* WIP

* Add script to generate TQA translated datasets

* update expt name and dataset splits

* Add Llama chat formatter

* Minor fixes in caa_repro

* Add options to print output, save steering vectors

* Set default experiment path by train / test datasets

* Add functionality to print examples

* Add script to plot results

* Add title to plotting code

* Fix pdm lock

* Add (very ugly) function to plot multiple results

Very ugly implementation but it works

* Ignore png files

* Enable translated system prompt

* Add new experiments dir

* Add notebook to analyze TQA vectors

* Add script to download datasets

* Add script to download datasets

* WIP translate

* Add code to extract and save steering vectors

* Update experiments

* Add more dataset names

* Improve dataset inspection

* Modify script to extract all SVs

* Changes to notebooks

* Update readme

* WIP

* Fix download datasets

* Enable 4-bit loading

* WIP

* Visualize pairwise cos similarities

* Inspect dataset s dataframe

* Clustering results

* Fix lint errors

* Add script to extract concept vectors

* WIP

* Refactoring

* Refactoring

* Add script to run all experiments

* Fix bug with results suffix

* Uncomment some lines

* Update README, bash script

* Restore original experiments dir

* Fix lint

* Fix lint

* Add more aggregations

* Fix bug in download

* Ignore html files

* Add test for data preprocessing

* Add tests for preprocessing

* fixing black formatting issues

* fixing typing

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`6680de7`](https://github.com/dtch1997/repepo/commit/6680de7bfbea72ac58886761ac2c0b6e2be56a25))

* Translate mwe and sycophancy (#97)

* importing raw persona MWE datasets from anthropic

* adding translation for mwe persona datasets and translating the first 5

* translating sycophancy datasets

* make_sycophancy_caa parses translations, and adding translations for misc strings

* adding a convenience wrapper to load_translation

* adding a script to make MWE personas datasets

* fix lint formatting

* alternating every 2 samples for MWE, not every 1 ([`ce83a8d`](https://github.com/dtch1997/repepo/commit/ce83a8dd4ce5e2ce39c8ccdd4be1af4128ac0a70))

* translating TQA into styles and languages with gpt4 (#94)

* translating TQA into styles and languages with gpt4

* dont force ascii, its not 1998

* fixing test mocking

* refactoring translations to make supporting more datasets easier ([`d6d241a`](https://github.com/dtch1997/repepo/commit/d6d241a116d6792653b1bb3a05680addfb820a05))

* Caa tqa (#91)

* refactoring formatting and benchmarking to support CAA

* adding a basic test for get_normalized_correct_probs()

* fixing tests

* increasing sft loss threshold to make test less flaky

* adding a TQA CAA dataset / experiment

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`97b1236`](https://github.com/dtch1997/repepo/commit/97b1236e529d21a8436570af5ae21d6f6a050406))

* Refactoring formatting and benchmarking to support CAA (#87)

* refactoring formatting and benchmarking to support CAA

* adding a basic test for get_normalized_correct_probs()

* fixing tests

* increasing sft loss threshold to make test less flaky ([`c98f067`](https://github.com/dtch1997/repepo/commit/c98f0672a6aa240f759120ed1d9c111a60ab3bc4))

* Merge pull request #88 from dtch1997/openai-translators

Add `openai` as dependency, and translators notebook ([`c1fb281`](https://github.com/dtch1997/repepo/commit/c1fb2815a0b0efdc7afcd4e5cfe3011cbaa77499))

* Add `openai` as dependency, and translators notebook

This notebook has a few simple functions for translating inputs and
dataframes using gpt-4. You will need an openai API key to run the code
(obviously). ([`dadf9f2`](https://github.com/dtch1997/repepo/commit/dadf9f2255c0e434c8580b95df2f27ad8fe06e06))

* fixing tests after CAA merge (#85) ([`6efbfaf`](https://github.com/dtch1997/repepo/commit/6efbfafc5dcf5609ebd679d3b969d4e760a3f4ac))

* Caa experiments 2 (#80)

* Add script to generate CAA datasets

* Add correct CAA datasets

* Add gitignore for experiments

* Modify default template

* Add get_normalized_correct_probs function

* Add script to generate vectors

* Add scripts to prompt w/ SV, plot results

* Add notebook to compare our vs their CAA vectors

* Add instructions to reproduce results

* Add plots

* Add evaluator for normalized correct probs

* Skip failing tests

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`cca4b29`](https://github.com/dtch1997/repepo/commit/cca4b29310bd44cd19364b4f77566a8b091c6298))

* Fix failing test ([`b7975b1`](https://github.com/dtch1997/repepo/commit/b7975b1a5eba7cb162b62c1ae27b27c75d3e66f6))

* refactoring prompting/formatting (#77)

* refactoring prompting/formatting

* fixing conflict in tests ([`dca53ac`](https://github.com/dtch1997/repepo/commit/dca53ac654beb6ae4d2026317593d5f8273e9e5c))

* Merge pull request #79 from dtch1997/swap-in-steering-vecs

swapping in steering-vectors lib ([`f86d0d2`](https://github.com/dtch1997/repepo/commit/f86d0d275703c591eee135af22fbcac8745d2477))

* Merge pull request #78 from dtch1997/verify-caa-steering

adding a test to assert our steering is identical to CAA steering ([`a5ea301`](https://github.com/dtch1997/repepo/commit/a5ea3018a56cbfb77e9a93f7b005af96a60bf815))

* swapping in steering-vectors lib ([`8e86019`](https://github.com/dtch1997/repepo/commit/8e86019ac9457ffe1783523745e50e6435d34695))

* adding a test to assert our steering is identical to CAA steering ([`f26b2df`](https://github.com/dtch1997/repepo/commit/f26b2dfdce7cda451153d0edded0720369b6d7fe))

* Verify our code matches CAA (#76)

* adding a llama chat formater and prompter based on CAA

* testing that our reading vectors match CAA reading vectors

* fixing linting

* fixing test ([`aa1dd24`](https://github.com/dtch1997/repepo/commit/aa1dd24966d9e1d696782b56dafda779ad99bd30))

* cleaning up oddities with steering vecs and repe algo (#72) ([`773db50`](https://github.com/dtch1997/repepo/commit/773db50594498e53f02cf969ac155ab0d1178774))

* CAA tweaks / improvements (#70)

* Add bitsandbytes, accelerate

* Hardcode second-last token activation position for steering vectors

* Add notebook diffmerge package for pretty git diffs

* Add note on how to change RepE directions

* Add note on how hooks work

* Add options to decouple reading and control

* fixing tests

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`1487639`](https://github.com/dtch1997/repepo/commit/148763914419a557bbb77482e9e687f2e626c64a))

* CAA base (#69)

* adding a record_activations() function to make it easy to collect model activations

* replacing repe with our own CAA-esque implementation

* only patch generated tokens

* fix generating start index selection

* fixing pyright error ([`55980ea`](https://github.com/dtch1997/repepo/commit/55980eab47f152385f7853b78abf3cf1b50a4b29))

* Add CAA datasets (#68)

* Add CAA datasets

* Update makefile

* Add test for make_ab_prompt

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`c1ae7f1`](https://github.com/dtch1997/repepo/commit/c1ae7f1ea12afad94c30178220586ea379ca08c0))

* Sft hf trainer (#50)

* Working HF trainer script

* customize wandb logging

* Remove unused keys from SFTDataset

* Add unit test for SFT

* Fix import

* Fix lint

* Fix lint (again)

* Fix test

* Fix benchmark, pipeline logic

Update the train_and_evaluate fn to be consistent
with algorithm.run now returning a dict

Fix pyright errors related to GenerationConfig

* Modify ICL to match new semantics

* Fix icl test

* Fix tests on GPU

* Fix device handling in tests

* Fix pyright bugbears

* Fix mutable default error in nested dataclass

* Fix mutable dataclass fields in python 3.11

* Fix nitpicks

* Fix default dataset

* Fix pyright

* Fix icl test to use new Algorithm.run signature

* Improve test cases

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`ec9e914`](https://github.com/dtch1997/repepo/commit/ec9e914b2d6d261409a14138fdbde3d72db52d15))

* Logprobs eval (#62)

* porting dataset handling code from tqa branch

* adding logprob calculation and adding an evaluator for multiple choice questions ([`8d07688`](https://github.com/dtch1997/repepo/commit/8d07688d4403c20b0ccb87ff4943025931e3f5d7))

* allow setting a direction coefficient for repe (#61) ([`2a2305e`](https://github.com/dtch1997/repepo/commit/2a2305ef2a6d092b8a0910aea79bf8178f29fbff))

* multiply direction by sign for reading vectors (#60) ([`308686b`](https://github.com/dtch1997/repepo/commit/308686bb313287eeb7087e5591ef6a9f122662bc))

* interleaving positive and negative prompts to match what the original repe code expects (#59) ([`8eb3b75`](https://github.com/dtch1997/repepo/commit/8eb3b75381bf40f6bd3825daccc0cdde95df8da2))

* limiting ICL max examples to avoid prepending entire dataset (#57) ([`856145e`](https://github.com/dtch1997/repepo/commit/856145eefe08731f7660ceab77046fa2fd55964e))

* Fix device handling in tests (#52)

* Fix device handling in tests

* Fix pyright bugbears

* Set device automatically for RepE pipeline

* Specify device in test

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`f46940a`](https://github.com/dtch1997/repepo/commit/f46940a307527a6645e0a5216a0447984db6c150))

* fixing bug where repe algo is patching in ndarray instead of tensors (#55) ([`7bdc280`](https://github.com/dtch1997/repepo/commit/7bdc280e6f6fe9b82b8c8d686f19f703070d35a9))

* Implementing Repe reading control algorithm (#48)

setting up repe reading algorithm ([`f85b3d7`](https://github.com/dtch1997/repepo/commit/f85b3d7480b0af41d91eeb9ec970e79316a49da1))

* Update lockfile (#49)

* Bump python to 3.10

* Update lockfile

---------

Co-authored-by: dtch1997 &lt;dtch1997@users.noreply.github.com&gt; ([`4ed1698`](https://github.com/dtch1997/repepo/commit/4ed169818713ca3d825c772f03839eed4b74a512))

* updating make_truthfulqa to use mc1 targets (#47) ([`d04ea9f`](https://github.com/dtch1997/repepo/commit/d04ea9fcbf56b91e98f7f6ee1f855533a378d8f7))

* Polishing ModelPatcher layer guessing and fully replacing WrappedReadingVecModel (#40)

* polishing layer guessing and fully replacing with WrappedReadingVecModel with ModelPatcher

* adding a test for pipeline skipping patching ([`fb6ccf5`](https://github.com/dtch1997/repepo/commit/fb6ccf5504703d4f62b5ef9658af86fb29923476))

* Configurable model patching (#33)

* adding configurable model patching

* updating original RepE rep_control_reading_vec.py to add operators

* renaming ModelPatcher.py to model_patcher.py

* adding different patch operations to match paper

* fixing comment doc for model patcher ([`2533e84`](https://github.com/dtch1997/repepo/commit/2533e845860d76e5d66ab579676a729892ed8dbe))

* Implement supervised fine-tuning (#31)

* WIP

* SFT working

* remove duplicate AverageMeter implementation

* Add ability to run on custom splits of dataset

* Remove broken code

* Fix Pyright issues

* Fix Pyright issues, again

* Update example, completion to dataclasses

* fix data generation

* Fix SFT inheritance

* Remove unused string methods

* Seeded random dataset shuffle

* Rename BaseAlgorithm to Algorithm

* fix pyright

* Add test for SFT

* minor

* Fix nit

* Fix tests

* Update default huggingface cache dir

* Modify tokenizer config in conftest

* Abstract away logger

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`1d68251`](https://github.com/dtch1997/repepo/commit/1d6825157bfd77d24e6af39fd51d177419caba8a))

* reload model for every test (#38) ([`3e9438e`](https://github.com/dtch1997/repepo/commit/3e9438e8a3d843ea5c3c2867ef6ec764caedd54f))

* updating README based on new github flow (#30)

* updating README based on new github flow

* Update makefile, readme

* Update README.md

---------

Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt;
Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`62658d2`](https://github.com/dtch1997/repepo/commit/62658d21f37b4a7b358f43a3c14b42dacb6cb726))

* adding Benchmark and Eval classes (#25)

* adding Benchmark and Eval classes

* simplifying benchmarking

* updating snapshot ([`e8dd92b`](https://github.com/dtch1997/repepo/commit/e8dd92bd3c4492d126819dbb51266a90235fc008))

* Relax Python version requirements (#26)

* Relax python requirements

* Update dependencies

* add more python version to CI

* Remove Py3.9 support

* Remove PDM caching

May be causing issues with CI workflow re-using same cache for different python versions
Likely doesn&#39;t result in much speedup

* Change PDM to be installed by pip

* Add libopenblas

* Sudo add libopenblas

* Update lock file

* Only Python3.10

* updating CI for 3.11, and adding pyright

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`bd01efb`](https://github.com/dtch1997/repepo/commit/bd01efbb5acdbb5426053617c532bc1e82f04b44))

* Switch to PDM managed; add pre-commit; run linters (#23)

* Switch to PDM managed; add pre-commit; run linters

* Update CI workflow

* Use PDM in ci workflow

* Fix tests

* Add device-aware testing

* adding snapshot test for test_pipeline

* remove trailing whitespace hook, black does this already

* try removing tensorboard to see if that makes things work?

* try explicitly adding ml-dtypes to dev deps, to see if that helps?

* ...trying to add wrapt explicitly now...

* undoing dep changes

* try adding explicit tensorflow dev dep

* try removing bleurt dep and tensorflow

* fixing CI caching

* use string for python version in CI

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: David Chanin &lt;chanindav@gmail.com&gt; ([`62ad2ed`](https://github.com/dtch1997/repepo/commit/62ad2ed876733acee7fdc2954fbdf45d12b528a7))

* silencing repe litning errors (#21)

* silencing repe litning errors

* adding HF token to CI ([`a49c9b6`](https://github.com/dtch1997/repepo/commit/a49c9b6665d189340d1d1025baafe7b4b75161c6))

* setting up basic pipeline arc, and adding icl algo (#15) ([`97bcbaf`](https://github.com/dtch1997/repepo/commit/97bcbafe65b75e145aee48596de865ab1431832f))

* repe is here ([`4e58c97`](https://github.com/dtch1997/repepo/commit/4e58c975d0a796d81993f31dc41541d6c6df9b58))

* Merge branch &#39;dev&#39; of github.com:dtch1997/repepo into dev ([`ec7743e`](https://github.com/dtch1997/repepo/commit/ec7743e16fca596945485940c5957ed79726f961))

* test ([`783577c`](https://github.com/dtch1997/repepo/commit/783577cb49c05810a7b10d44371150945fe59f51))

* simplifying core classes ([`21522aa`](https://github.com/dtch1997/repepo/commit/21522aa6ffb442dbb1b0e86f7abab3ff7c8a08dd))

* Adding pyright and fixing type errors (#14)

* adding pyright and fixing type errors

* fixing linting

* adding scikit-learn to deps

* fixing types pylance (but not pyright) complains about

* ignore reportPrivateImportUsage

* replacing namedtuple with NamedTuple for better type inference ([`4915268`](https://github.com/dtch1997/repepo/commit/4915268c3497896ddd46e8a79fa4e7013609cc1d))

* working on it ([`ee1c2e7`](https://github.com/dtch1997/repepo/commit/ee1c2e7ce7d667bba81560bb0f58fe4f4a5271ca))

* minor ([`22bcc4f`](https://github.com/dtch1997/repepo/commit/22bcc4fa907ab0ae8a2e5e363c0183f9be4c11e8))

* Merge pull request #13 from dtch1997/ci-linting-tests

CI, linting, and tests ([`7e08718`](https://github.com/dtch1997/repepo/commit/7e0871864867436e82e1c98b4bb4070451670808))

* adding CI workflow for linting and tests ([`8ce39e5`](https://github.com/dtch1997/repepo/commit/8ce39e52b60af6cb421474976fc96a1eb9ed83be))

* installing ruff and black and fixing formatting ([`59b4150`](https://github.com/dtch1997/repepo/commit/59b4150dc889ce52aff2fb5b447ecbf90016324b))

* Register datasets; other minor changes ([`413df15`](https://github.com/dtch1997/repepo/commit/413df15f6f01ff743194e266e7e8b75d6900336f))

* Add accelerate ([`4265238`](https://github.com/dtch1997/repepo/commit/426523822805c24319f853978374ec259eba10e0))

* Improve formatting, printing ([`20d84f9`](https://github.com/dtch1997/repepo/commit/20d84f98b9ea70dcd30190c0367c662289874da7))

* ICL eval pipeline ([`d01bbb5`](https://github.com/dtch1997/repepo/commit/d01bbb5f1fffdb684df51c595a4ca8ea34943c4a))

* Make ICL task vectors in standard format ([`e015a00`](https://github.com/dtch1997/repepo/commit/e015a0090db3641ab4a8491e44189889bc75ccb8))

* Add scripts to make datasets from &#34;In-context Learning Creates Task Vectors&#34; ([`e0adc69`](https://github.com/dtch1997/repepo/commit/e0adc69b954b449a16f639b1e35534568fe6aa85))

* Tests working ([`4038c16`](https://github.com/dtch1997/repepo/commit/4038c1627e82a43dc2e1cf0cac8993cd3789b31e))

* Add algorithms and metrics ([`10bb2f0`](https://github.com/dtch1997/repepo/commit/10bb2f02e01da266dd16c1d45e8e451d8608c1a9))

* Add basic abstractions ([`b919229`](https://github.com/dtch1997/repepo/commit/b9192293491eee8464d2e6ea35bd7e8f8902a49e))

* Merge branch &#39;dev&#39; into main ([`82feb1b`](https://github.com/dtch1997/repepo/commit/82feb1bb11b8abbde89a7a2e38188dc3ab312455))

* Add major changes (#11)

* SFT baseline (#5)

* Update requirements

* Add SFT algorithm

* Add datasets, log dirs to gitignore

* Demonstrate how to configure dataset

* Update README

---------

Co-authored-by: aengusl &lt;aenguslynch@gmail.com&gt;
Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;

* Daniel dev1 (#6)

* WIP simple train script

* Add full training pipeline

* Modify SFT dataset to return reference completions

* Add train_simple baseline

* Add BLEURT, ROUGE scores

* Add WandB logging

* Update README; make lr configurable

* Update requirements

* Enable SFT to be used with HF dataset

* Fix bug in lr scheduling

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;

* Refactor out prompt formatting

* Fix bug

* Aengus dev2 (#10)

* WIP simple train script

* Add full training pipeline

* Modify SFT dataset to return reference completions

* Add train_simple baseline

* Add BLEURT, ROUGE scores

* Add WandB logging

* Update README; make lr configurable

* Update requirements

* Enable SFT to be used with HF dataset

* Fix bug in lr scheduling

* adding repe

* organising

* damn things not being easy

* works

* working but weirdly

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: Daniel Tan &lt;25474937+dtch1997@users.noreply.github.com&gt;

* Integrate repe pipeline (#12)

* Add AmbigPrompt datasets

* Inject project variables into script

* put prompts in another file

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: aengusl &lt;aenguslynch@gmail.com&gt;

---------

Co-authored-by: aengusl &lt;aenguslynch@gmail.com&gt;
Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: Aengus Lynch &lt;37474130+aengusl@users.noreply.github.com&gt; ([`cb451dd`](https://github.com/dtch1997/repepo/commit/cb451dd6f1da4754449a6dccc533ed93393704c6))

* Integrate repe pipeline (#12)

* Add AmbigPrompt datasets

* Inject project variables into script

* put prompts in another file

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: aengusl &lt;aenguslynch@gmail.com&gt; ([`f8fef0d`](https://github.com/dtch1997/repepo/commit/f8fef0dcb3dd0e3ba36674c60947256a39c84ecf))

* Aengus dev2 (#10)

* WIP simple train script

* Add full training pipeline

* Modify SFT dataset to return reference completions

* Add train_simple baseline

* Add BLEURT, ROUGE scores

* Add WandB logging

* Update README; make lr configurable

* Update requirements

* Enable SFT to be used with HF dataset

* Fix bug in lr scheduling

* adding repe

* organising

* damn things not being easy

* works

* working but weirdly

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt;
Co-authored-by: Daniel Tan &lt;25474937+dtch1997@users.noreply.github.com&gt; ([`dac8ece`](https://github.com/dtch1997/repepo/commit/dac8ece2b24280f5bd56a8f107bf6da3aefba231))

* Fix bug ([`13e868f`](https://github.com/dtch1997/repepo/commit/13e868fab3ef09b740d687c037085ab5e4fa800d))

* Refactor out prompt formatting ([`d66f9d8`](https://github.com/dtch1997/repepo/commit/d66f9d841d8c63fb228ac54fff2e218094e87777))

* Daniel dev1 (#6)

* WIP simple train script

* Add full training pipeline

* Modify SFT dataset to return reference completions

* Add train_simple baseline

* Add BLEURT, ROUGE scores

* Add WandB logging

* Update README; make lr configurable

* Update requirements

* Enable SFT to be used with HF dataset

* Fix bug in lr scheduling

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`0fdfec6`](https://github.com/dtch1997/repepo/commit/0fdfec6e7e5ce6678dad760b807c73f7687058a7))

* SFT baseline (#5)

* Update requirements

* Add SFT algorithm

* Add datasets, log dirs to gitignore

* Demonstrate how to configure dataset

* Update README

---------

Co-authored-by: aengusl &lt;aenguslynch@gmail.com&gt;
Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`3abba55`](https://github.com/dtch1997/repepo/commit/3abba55ecb2b99dca1bf977641b048ae5948ff3c))

* Add examples (#2)

* Add HF example for fine-tuning on QA

* Add examples from RepEng repo

* Add AlpacaFarm, datasets reqs

---------

Co-authored-by: Daniel Tan &lt;dtch1997@users.noreply.github.com&gt; ([`573df55`](https://github.com/dtch1997/repepo/commit/573df55ab81bc5cc2d811f74946aa9f8768c4248))

* Update README.md ([`a093981`](https://github.com/dtch1997/repepo/commit/a093981541476524964cecfc1f144da72f05ef3b))

* Initial commit ([`9d16317`](https://github.com/dtch1997/repepo/commit/9d163171b1aeb36ae165adfad3060130649d6a27))
