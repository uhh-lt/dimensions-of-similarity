import glob
import json
from pathlib import Path

from easynmt import EasyNMT
from tqdm import tqdm

input_dir = "eval_data"

# list of all input files
data_root = "./data"
file_list = glob.glob(f"{data_root}/{input_dir}/*/*.json")

# load NMT model
model = EasyNMT('opus-mt', max_loaded_models=5)

# translate the data
for file in tqdm(file_list, desc="translating..."):
    file_path = Path(file)
    out_file_path = Path(f"{file_path.parent}/{file_path.name}.translated")
    if out_file_path.exists():
        print(f"Skipping file {file_path.name}")
        continue

    # load json
    data = json.load(open(file_path))

    # translate text
    try:
        data["text"] = model.translate(data["text"], target_lang='en')
        data["title"] = model.translate(data["title"], target_lang='en')

        # write json
        with open(out_file_path, "w", encoding="UTF-8") as f:
            json.dump(data, f)
    except OSError as e:
        print(e)
        print(file_path.absolute())

# move translated files
for file in glob.glob(f"{data_root}/{input_dir}/*/*.json.translated"):
    file_path = Path(file)
    target_path = Path(str(file_path.parent).replace(input_dir, f"{input_dir}_translated")) / Path(str(file_path.name).replace(".translated", ""))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.rename(target_path)


# COULD NOT TRANSLATE TRAINING DATA (18)
# pt -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/46/1484351446.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/80/1492573080.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/19/1634325419.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/30/1484007830.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/33/1483859533.json
# sl -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/67/1638491367.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/12/1588378112.json
# sw -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/87/1483897487.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/19/1483802219.json
# arz -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/07/1551829707.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/20/1628389320.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/12/1629884312.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/33/1554319033.json
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/44/1575550344.json
# ug -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/29/1645373429.json
# ku -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/78/1535839378.json
# tt -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/20/1558529820.json
# no -> en
# /home/tfischer/Development/dimensions-of-similarity/data/train_data/15/1484358015.json

# COULD NOT TRANSLATE EVAL DATA (43)
# sl
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/25/1589866025.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/73/1645682073.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/95/1543791595.json
# kn
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/58/1584368258.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/05/1489063505.json
# no
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/41/1516385241.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/55/1591054155.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/11/1551167311.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/88/1616226288.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/23/1510984023.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/02/1562637002.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/96/1487766696.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/71/1559637271.json
# arz
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/41/1587386041.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/79/1586998279.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/35/1608373735.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/19/1530602919.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/11/1615195411.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/96/1558026996.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/49/1491920249.json
# wuu
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/36/1489077936.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/42/1521972142.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/94/1635846794.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/11/1527298211.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/91/1641500591.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/95/1640803995.json
# tt
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/97/1576637597.json
# ro
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/31/1519682431.json
# pt
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/10/1537298010.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/53/1589689553.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/62/1628190762.json
# fy
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/51/1547622251.json
# br
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/87/1625132687.json
# sr
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/63/1502378663.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/86/1540020586.json
# fa
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/35/1613019735.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/81/1639292181.json
# oc
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/00/1553892700.json
# lt
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/21/1501877021.json
# ta
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/54/1626275954.json
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/49/1586616449.json
# gu
# /home/tfischer/Development/dimensions-of-similarity/data/eval_data/81/1596749581.json
