#!/usr/bin/env bash

# Paths
moses="/private/home/louismartin/dev/ext/mosesdecoder/bin/moses"
moses_scripts="/private/home/louismartin/dev/ext/mosesdecoder/scripts"
easse_scripts="/private/home/louismartin/dev/easse/easse/scripts"
easse_resources="/private/home/louismartin/dev/easse/easse/resources"

# Create a temporary file that will be used in the script
tmp_file=$(mktemp /tmp/easse-postprocess_sys_output.XXXXXX)
cat /dev/stdin > "${tmp_file}"

# 1) Deanonymize the Named Entities
python "${easse_scripts}/deanonymise_ner.py" -m test -aner "${easse_resources}/wikilarge.aner.map.t7"
#> "${tmp_file}"
# 2) Recase
#perl "${moses_scripts}/recaser/recase.perl" --in "${tmp_file}" --model "${easse_resources}/recaser/moses.ini" --moses "${moses}" |
## 3) Detokenize
#perl "${moses_scripts}/tokenizer/normalize-punctuation.perl" -l en |
#perl "${moses_scripts}/tokenizer/detokenizer.perl" -l en -q |
#perl "${moses_scripts}/tokenizer/detokenizer.perl" -l en -penn -q

# Remove the temporary files
rm "${tmp_file}"
