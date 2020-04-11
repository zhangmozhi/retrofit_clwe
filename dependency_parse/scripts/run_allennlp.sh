CONFIG_FILE=allen_configs/parse.jsonnet
allennlp train $CONFIG_FILE -s ./output --include-package custom_allennlp_library
