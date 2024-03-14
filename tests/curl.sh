#!/bin/bash -x

curl --silent http://localhost:5430/predictions -X POST \
      -H 'Content-Type: application/json' \
          -d '{"input": {"text": "hello", "src_lang": "eng_Latn", "tgt_lang": "fra_Latn"}}' | python -m json.tool

