- Membuat environment baru

```
Python -m venv env
```

- Mengaktifkan environment

```
\env\Scripts\activate #windows
source venv/Scripts/activate # linux/macos
```

- Mematikan environment

```
deactivate
```

- Menginstall installasi yang dibutuhkan dalam project

```
pip install -r requirements.txt
```

- Menghidupkan environment notebook

```
python -m ipykernel install --user --name=env --display-name "Python env"
```

- Menjalankan mlflow

```
mlflow ui
python Membangun_model/modelling.py
python Membangun_model/modelling_tuning.py
```
