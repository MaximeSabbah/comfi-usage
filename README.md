# comfi-usage
Scripts to use and showcase the data from the COMFI dataset

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/MaximeSabbah/comfi-usage.git
cd comfi-usage
```
2. **Virtual environment**
python3 -m venv comfi_env
source comfi_env/bin/activate

3. **Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt

4. **Usage**
To visualise joint center position : python -m scripts.visualization.viz_jcp
To visualise jcp and mocap markers : python -m scripts.visualization.viz_multiple_mks_set
To visualise all data (include markers, joint angles, robot ..) : python -m scripts.visualization.viz_all_data
