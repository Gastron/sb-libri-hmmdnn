#!/bin/bash

oov_entry="<UNK>"
. path.sh
. parse_options.sh

if [ $# -ne 1 ]; then
	echo "Usage: local/seed_dict.sh <lang_tmp>"
	echo "e.g.: $0 data/local/lang"
	echo
	echo "Create files silence_phones, optional_silence, and extra_questions."
	exit 1	
fi

tmpdir=$1


# These should just match your silence phones:
echo "SIL" > ${tmpdir}/silence_phones.txt
echo "SPN" >> ${tmpdir}/silence_phones.txt
#echo "NSN" >> ${tmpdir}/silence_phones.txt
sort -uo ${tmpdir}/silence_phones.txt{,}
echo "$oov_entry SPN" > ${tmpdir}/lexicon.txt
echo "-pau- SIL" >> ${tmpdir}/lexicon.txt
echo "<unk> SPN" >> ${tmpdir}/lexicon.txt

echo "SIL" > ${tmpdir}/optional_silence.txt

touch $tmpdir/extra_questions.txt

