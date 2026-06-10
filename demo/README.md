# ECG Demo UI

This demo replaces the Gradio interface with a React build served by FastAPI.

## Build the React UI

```powershell
cd demo/frontend
npm install
npm run build
```

The build output is written to:

```text
demo/static_dist/
```

## Run the combined app

From the project root:

```powershell
python demo/app.py
```

Open:

```text
http://127.0.0.1:7860
```

FastAPI also exposes:

```text
/api/health
/api/sample/normal
/api/sample/anomaly
/api/predict
/api/upload
```
