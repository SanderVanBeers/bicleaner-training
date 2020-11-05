# Docker image to automate the training of a bicleaner classifier
## Usage
The docker image can be built using the dbuild.sh script. The Docker image will have the tag _bicleaner/training_

```
bash dbuild.sh
```
The docker image can be run using the dcli.sh script.

```
bash dcli.sh -i name_training_data -v /host/directory/to/bind -s source_language -t target_language
```
The inputfile is expected to be a tab-separated plaintext file containing the source and target segments in different columns.
The training data should be in the directory you bind to the docker container.
The source language and target language are expected to be the language codes.
