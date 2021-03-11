import subprocess
import argparse
import os
from multiprocessing import cpu_count

#Script to train bicleaner

#Step 1: Build probabilistic dictionaries

def build_probabilistic_dictionaries(path_bitext, source_language, target_language):
    #Split up the tab-separated bitext, tokenize and lowercase it
    subprocess.run(f"cut -f1 {path_bitext} | /mosesdecoder/moses/scripts/tokenizer/tokenizer.perl -l {source_language} -threads {cpu_count()} -no-escape | /mosesdecoder/moses/scripts/tokenizer/lowercase.perl > /data/{path_bitext}.tok.low.{source_language}", shell=True)
    subprocess.run(f"cut -f2 {path_bitext} | /mosesdecoder/moses/scripts/tokenizer/tokenizer.perl -l {target_language} -threads {cpu_count()} -no-escape | /mosesdecoder/moses/scripts/tokenizer/lowercase.perl > /data/{path_bitext}.tok.low.{target_language}", shell=True)

    #Remove lines (and their corresponding lines), that are empty, too short, too long (80) or violate the 9-1 sentence ratio limit of GIZA++
    subprocess.run(f"/mosesdecoder/moses/scripts/training/clean-corpus-n.perl /data/{path_bitext}.tok.low {source_language} {target_language} /data/{path_bitext}.tok.low.clean 1 80", shell=True)

    #train the moses model
    subprocess.run(f"/mosesdecoder/moses/scripts/training/train-model.perl --alignment grow-diag-final-and --root-dir /data  --corpus /data/{path_bitext}.tok.low.clean -e {source_language}  -f {target_language} --mgiza -mgiza-cpus={cpu_count()} --parallel --first-step 1 --last-step 4 --external-bin-dir /mgiza/mgizapp/bin/", shell=True)    

#Step 2: create word frequency files
def create_word_frequencies(path_bitext, source_language, target_language):
    subprocess.run(f"""cut -f1 {path_bitext} | sacremoses -l {source_language} tokenize -x | awk '{{print tolower($0)}}' | tr ' ' '\n' | LC_ALL=c sort | uniq -c | LC_ALL=c sort -nr | grep -v "[[:space:]]*1" | gzip > /workdir/{source_language.upper()}-{target_language.upper()}/wordfreq-{source_language}.gz""", shell=True)
    subprocess.run(f"""cut -f2 {path_bitext} | sacremoses -l {target_language} tokenize -x | awk '{{print tolower($0)}}' | tr ' ' '\n' | LC_ALL=c sort | uniq -c | LC_ALL=c sort -nr | grep -v "[[:space:]]*1" | gzip > /workdir/{source_language.upper()}-{target_language.upper()}/wordfreq-{target_language}.gz""", shell=True)

#Step 3: prune the dictionaries created by the moses training script
def prune_dicts(source_language, target_language):
    subprocess.run(f"python3 /bicleaner/utils/dict_pruner.py /data/model/lex.e2f /workdir/{source_language.upper()}-{target_language.upper()}/dict-{source_language}.gz -n 10 -g ", shell=True)
    subprocess.run(f"python3 /bicleaner/utils/dict_pruner.py /data/model/lex.f2e /workdir/{source_language.upper()}-{target_language.upper()}/dict-{target_language}.gz -n 10 -g ", shell=True)

def train_bicleaner(path_training_set, source_language, target_language):
    #Run the training script
    os.chdir(f'/workdir/{source_language.upper()}-{target_language.upper()}')
    subprocess.run(f"bicleaner-train ../{path_training_set} --relative_paths --normalize_by_length -s {source_language} -t {target_language} -d dict-{source_language}.gz -D dict-{target_language}.gz -b 1000 -c {source_language}-{target_language}.classifier -f wordfreq-{source_language}.gz -F wordfreq-{target_language}.gz -m {source_language}-{target_language}.yaml --lm_training_file_sl lmtrain.{source_language}-{target_language}.{source_language} --lm_training_file_tl lmtrain.{source_language}-{target_language}.{target_language} --lm_file_sl model.{source_language}-{target_language}.{source_language} --lm_file_tl model.{source_language}-{target_language}.{target_language}", shell=True)

def training_pipeline(path_bitext, source_language, target_language):
    if not os.path.isdir(f"/workdir/{source_language.upper()}-{target_language.upper()}"):
        os.mkdir(f"/workdir/{source_language.upper()}-{target_language.upper()}")
    build_probabilistic_dictionaries(path_bitext=path_bitext, source_language=source_language, target_language=target_language)
    create_word_frequencies(path_bitext=path_bitext, source_language=source_language, target_language=target_language)
    prune_dicts(source_language=source_language, target_language=target_language)
    train_bicleaner(path_training_set=path_bitext, source_language=source_language, target_language=target_language)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="The path to the input bitext.")
    parser.add_argument("--source", type=str, required=True, help="Source language.")
    parser.add_argument("--target", type=str, required=True, help="Target language.")
    args = parser.parse_args()
    os.chdir("/workdir")

    training_pipeline(path_bitext=args.input, source_language=args.source, target_language=args.target)
