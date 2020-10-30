#The input data is expected to be in the mapped working directory. Intermediate files will be saved in the container directory /data/
#Files relevant to the bicleaner instance will be put in the working directory
while getopts i:v:s:t: flag
do
    case "${flag}" in
        i) input_file=${OPTARG};;
        v) working_directory=${OPTARG};;
        s) source_language=${OPTARG};;
        t) target_language=${OPTARG};;
    esac
done
docker run --rm -v $working_directory:/workdir:Z bicleaner/training python3 /src/main.py --input $input_file --source $source_language --target $target_language
#Example command:
#bash dcli.sh -i training_data.de-fr -v /home/sandervanbeers/Desktop/spooktober/testbicleanertraining -s de -t fr