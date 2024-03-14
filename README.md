# Simple REST API translation server based on cog

## Dependencies

- [cog](https://github.com/replicate/cog)

## Environment Variables

- `MODEL_ID`: The model ID for the translation model to use. Defaults to "facebook/nllb-200-distilled-600M".

## Build instructions

```bash
cog build -t cog_translation_server
```


## Usage

```bash
docker run -d -p 5430:5000 --gpus all cog_translation_server
./tests/curl.sh
```
Example input:
```json
{
    "input": {
        "src_lang": "eng_Latn",
        "text": "hello",
        "tgt_lang": "fra_Latn"
    }
}
```
Example output:
```json
{
    "completed_at": "2024-03-14T14:02:20.172715+00:00",
    "created_at": null,
    "error": null,
    "id": null,
    "input": {
        "model_id": "facebook/nllb-200-distilled-600M",
        "src_lang": "eng_Latn",
        "text": "hello",
        "tgt_lang": "fra_Latn"
    },
    "logs": "Number of lines in data: 0\n",
    "metrics": {
        "predict_time": 0.787398
    },
    "output": "Je vous salue .",
    "output_file_prefix": null,
    "started_at": "2024-03-14T14:02:19.385317+00:00",
    "status": "succeeded",
    "version": null,
    "webhook": null,
    "webhook_events_filter": [
        "start",
        "output",
        "logs",
        "completed"
    ]
}
```