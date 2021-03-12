# Docker image to automate the training of a bicleaner classifier
## Usage
The docker image can be built using the dbuild.sh script. The Docker image will have the tag _bicleaner/training_

```
bash dbuild.sh
```
The docker image can be run using the dcli.sh script.

```
bash dcli.sh -b name_big_corpus -c name_clean_corpus -v /host/directory/to/bind -s source_language -t target_language
```
The big corpus and clean corpus are expected to be a tab-separated plaintext file containing the source and target segments in different columns.
The corpora should be in the directory you bind to the docker container.
The source language and target language are expected to be the language codes (ISO 639-1).
