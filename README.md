# Plant Disease Detection

This project loads a trained EfficientNet-based Keras model and predicts plant disease classes from uploaded crop images.

## Project structure

- `app.py` – Flask web app and prediction logic
- `efficientnet_best_model.keras/` – saved Keras model and weights
- `class_names.json` – disease labels used for predictions
- `templates/index.html` – upload page
- `static/style.css` – styling

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open the browser at `http://localhost:5000`.
