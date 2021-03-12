import subprocess
import argparse
import os
from multiprocessing import cpu_count

#Script to train bicleaner

class BicleanerPipeline:
    def __init__(self, path_bigcorpus, source_language, target_language):
        self.path_bigcorpus = path_bigcorpus
        self.source_language = source_language
        self.target_language = target_language
    
    def build_probabilistic_dictionaries(self):
        #Step 1: Build probabilistic dictionaries
        
        #Split up the tab-separated bitext, tokenize and lowercase it
        subprocess.run(f"cut -f1 {self.path_bigcorpus} | /mosesdecoder/moses/scripts/tokenizer/tokenizer.perl -l {self.source_language} -threads {cpu_count()} -no-escape | /mosesdecoder/moses/scripts/tokenizer/lowercase.perl > /data/{self.path_bigcorpus}.tok.low.{self.source_language}", shell=True)
        subprocess.run(f"cut -f2 {self.path_bigcorpus} | /mosesdecoder/moses/scripts/tokenizer/tokenizer.perl -l {self.target_language} -threads {cpu_count()} -no-escape | /mosesdecoder/moses/scripts/tokenizer/lowercase.perl > /data/{self.path_bigcorpus}.tok.low.{self.target_language}", shell=True)

        #Remove lines (and their corresponding lines), that are empty, too short, too long (80) or violate the 9-1 sentence ratio limit of GIZA++
        subprocess.run(f"/mosesdecoder/moses/scripts/training/clean-corpus-n.perl /data/{self.path_bigcorpus}.tok.low {self.source_language} {self.target_language} /data/{self.path_bigcorpus}.tok.low.clean 1 80", shell=True)

        #train the moses model
        subprocess.run(f"/mosesdecoder/moses/scripts/training/train-model.perl --alignment grow-diag-final-and --root-dir /data  --corpus /data/{self.path_bigcorpus}.tok.low.clean -e {self.source_language}  -f {self.target_language} --mgiza -mgiza-cpus={cpu_count()} --parallel --first-step 1 --last-step 4 --external-bin-dir /mgiza/mgizapp/bin/", shell=True)    

    def create_word_frequencies(self):
        #Step 2: create word frequency files

        subprocess.run(f"""cut -f1 {self.path_bigcorpus} | sacremoses -l {self.source_language} tokenize -x | awk '{{print tolower($0)}}' | tr ' ' '\n' | LC_ALL=c sort | uniq -c | LC_ALL=c sort -nr | grep -v "[[:space:]]*1" | gzip > /workdir/{self.source_language}-{self.target_language}/wordfreq-{self.source_language}.gz""", shell=True)
        subprocess.run(f"""cut -f2 {self.path_bigcorpus} | sacremoses -l {self.target_language} tokenize -x | awk '{{print tolower($0)}}' | tr ' ' '\n' | LC_ALL=c sort | uniq -c | LC_ALL=c sort -nr | grep -v "[[:space:]]*1" | gzip > /workdir/{self.source_language}-{self.target_language}/wordfreq-{self.target_language}.gz""", shell=True)

    def prune_dicts(self):
        #Step 3: prune the dictionaries created by the moses training script
        subprocess.run(f"python3 /bicleaner/utils/dict_pruner.py /data/model/lex.e2f /workdir/{self.source_language}-{self.target_language}/dict-{self.source_language}.gz -n 10 -g ", shell=True)
        subprocess.run(f"python3 /bicleaner/utils/dict_pruner.py /data/model/lex.f2e /workdir/{self.source_language}-{self.target_language}/dict-{self.target_language}.gz -n 10 -g ", shell=True)

    def train_bicleaner(self):
        #Run the training script
        os.chdir(f'/workdir/{self.source_language}-{self.target_language}')
        subprocess.run(f"bicleaner-train ../{self.path_bigcorpus} --relative_paths --normalize_by_length -s {self.source_language} -t {self.target_language} -d dict-{self.source_language}.gz -D dict-{self.target_language}.gz -b 1000 -c {self.source_language}-{self.target_language}.classifier -f wordfreq-{self.source_language}.gz -F wordfreq-{self.target_language}.gz -m {self.source_language}-{self.target_language}.yaml --lm_training_file_sl lmtrain.{self.source_language}-{self.target_language}.{self.source_language} --lm_training_file_tl lmtrain.{self.source_language}-{self.target_language}.{self.target_language} --lm_file_sl model.{self.source_language}-{self.target_language}.{self.source_language} --lm_file_tl model.{self.source_language}-{self.target_language}.{self.target_language}", shell=True)

    def start_training_pipeline(self):
        os.makedirs(f"/workdir/{self.source_language}-{self.target_language}", exist_ok=True)
        
        self.build_probabilistic_dictionaries()
        self.create_word_frequencies()
        self.prune_dicts()
        self.train_bicleaner()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="The path to the input bitext.")
    parser.add_argument("--source", type=str, required=True, help="Source language.")
    parser.add_argument("--target", type=str, required=True, help="Target language.")
    args = parser.parse_args()
    os.chdir("/workdir")

    pipeline = BicleanerPipeline(path_bigcorpus=args.input, source_language=args.source, target_language=args.target)
    pipeline.start_training_pipeline()
