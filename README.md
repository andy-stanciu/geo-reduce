
# geo-reduce

**Project Goal:**  
Evaluate the impact of gradient compression on communication cost and training scalability in multi-GPU Vision Transformer training for our OpenGuessr geolocation model.

**Data Source:**  
[Google Street View — Top 50 U.S. Cities (Kaggle)](https://www.kaggle.com/datasets/pinstripezebra/google-streetview-top-50-us-cities)

---

### To Clone Data

#### 1. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
````

#### 2. Install Dependencies

```bash
pip install kagglehub
```

#### 3. Run Script

```bash
python data_sort.py
```
