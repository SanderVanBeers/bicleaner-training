#The input data is expected to be in the mapped working directory. Intermediate files will be saved in the container directory /data/
#Files relevant to the bicleaner instance will be put in the working directory
while getopts b:c:v:s:t: flag
do
    case "${flag}" in
        b) big_corpus=${OPTARG};;
        c) clean_corpus=${OPTARG};;
        v) working_directory=${OPTARG};;
        s) source_language=${OPTARG};;
        t) target_language=${OPTARG};;
    esac
done
docker run --rm -v $working_directory:/workdir:Z bicleaner/training python3 /src/main.py --big_corpus $big_corpus --clean_corpus $clean_corpus --source $source_language --target $target_language
#Example command:
#bash dcli.sh -b big_corpus.de-fr -c clean_corpus -v /home/sandervanbeers/testbicleanertraining -s de -t fr