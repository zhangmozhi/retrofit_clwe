local TRAIN_DATA_PATH="SET_ME";
local DEV_DATA_PATH="SET_ME";
local TEST_DATA_PATH="SET_ME";
local VOCAB_PATH="SET_ME";
local PRETRAINED_WORD_VECTOR="SET_ME";
{
    "random_seed": 1,
    "numpy_seed": 1,
    "pytorch_seed": 1,
    "dataset_reader": {
        "type": "universal_dependencies_multilang_empty_head",
        "languages": ["en"],
        "alternate": false,
        "instances_per_file": 32,
        "is_first_pass_for_vocab": true,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
            },
        },
        "use_language_specific_pos": false
    },
    "vocabulary": {
      "tokens_to_add": {
          "tokens": ["@@UNKNOWN@@"],
      },
      "directory_path": VOCAB_PATH,
      "pretrained_files": {"tokens": PRETRAINED_WORD_VECTOR},
      "min_pretrained_embeddings": {"tokens": 400000},
      "extend": true
    },
    "iterator": {
        "type": "same_language",
        "batch_size": 32,
        "sorting_keys": [["words", "num_tokens"]],
    },
    "model": {
        "type": "biaffine_parser",
        "arc_representation_dim": 512,
        "dropout": 0.33,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "hidden_size": 300,
            "input_size": 350,
            "num_layers": 3
        },
        "pos_tag_embedding": {
            "embedding_dim": 50,
            "vocab_namespace": "pos"
        },
        "tag_representation_dim": 128,
        "text_field_embedder": {
            "tokens": {
              "type": "embedding",
              "pretrained_file": PRETRAINED_WORD_VECTOR,
              "embedding_dim": 300,
              "trainable": false
            },
        }
    },
    "train_data_path": TRAIN_DATA_PATH,
    "validation_data_path": DEV_DATA_PATH,
    "test_data_path": TEST_DATA_PATH,
    "evaluate_on_test": true,
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 40,
        "optimizer": "adam",
        "patience": 10,
        "validation_metric": "+LAS"
    },
    "validation_dataset_reader": {
        "type": "universal_dependencies_multilang_empty_head",
        "languages": ["en", "ja"],
        "alternate": false,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
            },
        },
        "use_language_specific_pos": false
    },
    "validation_iterator": {
        "type": "same_language",
        "sorting_keys": [["words", "num_tokens"]],
        "batch_size": 32
    }
}
